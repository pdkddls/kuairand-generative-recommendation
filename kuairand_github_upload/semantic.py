"""Offline batched item embeddings, fused GAOQ and frozen utility codes."""

import numpy as np
import pandas as pd

from data import STRUCTURED, dump, mapping_digest
from famae import fit_fields
from gaoq import quantize, unit


def batch_embed(texts, name, dim, batch, maxlen, device):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    dtype = torch.float32
    if device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModel.from_pretrained(name, dtype=dtype).eval().to(device)
    vectors = np.empty((len(texts), dim), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(texts), batch):
            if start and start % (batch * 100) == 0:
                print(f"encoded {start}/{len(texts)} items", flush=True)
            inputs = tok(
                texts[start : start + batch],
                padding=True,
                truncation=True,
                max_length=maxlen,
                return_tensors="pt",
            ).to(device)
            hidden = model(**inputs, use_cache=False).last_hidden_state
            if hidden.size(-1) < dim or not inputs.attention_mask[:, -1].all():
                raise ValueError(
                    "Expected left-padded sequences with an attended last token"
                )
            v = torch.nn.functional.normalize(hidden[:, -1, :dim].float(), dim=-1)
            vectors[start : start + len(v)] = v.cpu().numpy()
    if not np.isfinite(vectors).all() or not np.allclose(
        np.linalg.norm(vectors, axis=1), 1, atol=1e-4
    ):
        raise ValueError("Invalid Qwen vectors")
    return vectors, getattr(model.config, "_commit_hash", None)


def build(folder, corpus, cfg, device, model_path=None):
    meta = pd.read_csv(
        folder / "items.csv", dtype={field: "string" for field in ["text", *STRUCTURED]}
    )
    if (
        len(meta) != corpus.n
        or not meta.video_id.is_unique
        or not meta.video_id.is_monotonic_increasing
    ):
        raise ValueError("Catalog order does not match the prepared item numbering")
    inside = corpus.ids > 0
    if not np.array_equal(
        meta.video_id.to_numpy()[corpus.ids[inside] - 1], corpus.e["video_id"][inside]
    ):
        raise ValueError("Catalog video IDs differ from the event mapping")
    content = meta.text.fillna("")
    available = content.str.len().gt(0).to_numpy(dtype=bool)
    rows, texts = pd.factorize(content[available], sort=False)
    distinct_vectors, rev = batch_embed(
        texts.tolist(),
        model_path or cfg["embedding_model"],
        cfg["embedding_dim"],
        cfg["embedding_batch"],
        cfg["text_tokens"],
        device,
    )
    qwen = np.zeros((corpus.n, cfg["embedding_dim"]), dtype=np.float32)
    qwen[available] = distinct_vectors[rows]
    # Keep TIGER's content-only input independent of ReSID's learned field vectors.
    np.save(folder / "item_vectors.npy", qwen)
    fields, field_summary = fit_fields(folder, corpus, meta, cfg, device)
    fused = unit(np.concatenate([unit(fields), qwen], axis=1))
    if not np.isfinite(fused).all() or not np.allclose(
        np.linalg.norm(fused, axis=1), 1, atol=1e-5
    ):
        raise ValueError("Invalid fused item representations")
    np.save(folder / "fused_vectors.npy", fused)
    mapping, state = quantize(fused, cfg)
    mask = (corpus.done < corpus.meta["cutoffs_ms"][0]) & (corpus.ids > 0)
    exp = np.bincount(corpus.ids[mask], minlength=corpus.n + 1)[1:]
    clicks = np.bincount(
        corpus.ids[mask], weights=corpus.e["is_click"][mask], minlength=corpus.n + 1
    )[1:]
    if not exp.sum():
        raise ValueError("No observable warmup CTR labels")
    strength = cfg["ctr_strength"]
    rate = (clicks + strength * clicks.sum() / exp.sum()) / (exp + strength)
    cuts = np.quantile(rate, np.arange(1, 8) / 8)
    buckets = np.searchsorted(cuts, rate, side="right")
    mapping[:, 2] = mapping[:, 2] * 8 + buckets
    if len(np.unique(mapping, axis=0)) != corpus.n or not np.array_equal(
        mapping[:, 2] % 8, buckets
    ):
        raise AssertionError("Non-unique SID or incorrect CTR bucket")
    np.save(folder / "codes.npy", mapping)
    np.savez(folder / "quantizer.npz", **state, bucket_edges=cuts)
    dump(
        folder / "semantic.json",
        {
            "mapping_sha256": mapping_digest(meta.video_id, mapping),
            "sid_method": cfg["sid_method"],
            "famae": field_summary,
            "encoder": str(model_path or cfg["embedding_model"]),
            "revision": rev,
            "items": corpus.n,
            "items_with_content": int(available.sum()),
            "qwen_dim": qwen.shape[1],
            "fused_dim": fused.shape[1],
            "ctr_exposures": int(exp.sum()),
            "unique_sids": len(mapping),
            "occupied_buckets": len(np.unique(buckets)),
            "occupied_prefixes": len(np.unique(mapping[:, :2], axis=0)),
        },
    )
