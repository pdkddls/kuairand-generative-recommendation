"""Field-aware masked reconstruction on pre-catalog-cutoff interaction histories."""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from data import STRUCTURED, clean_values


def tag_sets(values):
    # Set IDs index a lookup only. The shared label IDs index trainable embeddings.
    tokens = clean_values(values.reset_index(drop=True).str.split(",").explode())
    tokens = tokens.str.replace(r"^(\d+)\.0+$", r"\1", regex=True).dropna()
    ids, labels = pd.factorize(tokens, sort=True)
    members = pd.Series(ids + 1, index=tokens.index).groupby(level=0)
    set_ids = np.zeros(len(values), dtype=np.int64)
    codes = {(): 0}
    for row, group in members:
        key = tuple(np.unique(group.to_numpy()))
        set_ids[row] = codes.setdefault(key, len(codes))
    bank = np.zeros((len(codes), max(1, max(map(len, codes)))), dtype=np.int64)
    for key, index in codes.items():
        bank[index, : len(key)] = key
    return set_ids, bank, labels.tolist()


def catalog_fields(meta):
    # Local item numbers start at 1. Value 0 is missing/padding in every field.
    columns = [np.arange(1, len(meta) + 1, dtype=np.int64)]
    names = ["video_id"]
    bank = None
    tag_labels = []
    for name in STRUCTURED:
        if name not in meta:
            continue
        values = clean_values(meta[name])
        if name == "tag":
            ids, tag_bank, tag_labels = tag_sets(values)
            if tag_labels:
                names.append(name)
                columns.append(ids)
                bank = tag_bank
            continue
        if name.endswith("_id") or name == "music_type":
            values = values.str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
        if name == "video_duration":
            duration = pd.to_numeric(values, errors="coerce")
            values = np.floor(
                np.log2(1 + duration.where((duration >= 0) & np.isfinite(duration)))
            ).astype("Int64")
        if values.notna().any():
            names.append(name)
            columns.append(pd.factorize(values, sort=True)[0] + 1)
    matrix = np.stack(columns, axis=1)
    return names, np.pad(matrix, ((1, 0), (0, 0))), bank, tag_labels


class FieldEncoder(nn.Module):
    def __init__(self, sizes, length, cfg, tag_index=None, tag_bank=None):
        super().__init__()
        self.tag_index = tag_index
        bank = torch.as_tensor(tag_bank) if tag_bank is not None else None
        self.register_buffer("tag_bank", bank)
        sizes = list(sizes)
        if bank is not None:
            sizes[tag_index] = int(bank.max())
        d = cfg["famae_dim"]
        self.tables = nn.ModuleList(
            nn.Embedding(size + 1, d, padding_idx=0) for size in sizes
        )
        self.masks = nn.Parameter(torch.randn(len(sizes), d) * 0.02)
        self.position = nn.Embedding(length + 1, d)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d, cfg["famae_heads"], 4 * d, dropout=0.1, batch_first=True
            ),
            cfg["famae_layers"],
            enable_nested_tensor=False,
        )
        self.projections = nn.ModuleList(nn.Linear(d, d) for _ in sizes)

    def fields(self, ids):
        vectors = []
        for j, table in enumerate(self.tables):
            field = ids[..., j]
            if j == self.tag_index:
                sets = self.tag_bank[field.reshape(-1)]
                pooled = F.embedding_bag(sets, table.weight, mode="mean", padding_idx=0)
                vectors.append(pooled.reshape(*field.shape, -1))
            else:
                vectors.append(table(field))
        return torch.stack(vectors, dim=-2)

    def tag_loss(self, hidden, set_ids, table, negatives):
        # Uniform target distribution over this video's distinct nonpadding labels.
        labels = self.tag_bank[set_ids]
        present = labels.ne(0)
        classes = table.num_embeddings - 1
        if classes <= negatives + 1:
            logits = 10 * (hidden @ F.normalize(table.weight[1:], dim=-1).T)
            positive = logits.gather(1, (labels - 1).clamp_min(0))
        else:
            samples = torch.randint(
                1, classes + 1, (len(labels), negatives), device=hidden.device
            )
            collisions = samples[:, :, None].eq(labels[:, None, :]).any(-1)
            candidates = torch.cat([labels, samples], dim=1)
            logits = 10 * (
                F.normalize(table(candidates), dim=-1) * hidden[:, None]
            ).sum(-1)
            positive = logits[:, : labels.size(1)]
            logits = logits.masked_fill(
                torch.cat([~present, collisions], dim=1), -torch.inf
            )
        mean_positive = positive.masked_fill(~present, 0).sum(1) / present.sum(1)
        return (logits.logsumexp(1) - mean_positive).sum()

    def loss(self, histories, targets, fields, negatives):
        hist = self.fields(fields[histories]).sum(-2)
        target = fields[targets]
        present = target.ne(0)
        # Draw a uniform nonempty mask among the available field counts.
        count = (
            torch.rand(len(target), device=target.device) * present.sum(1)
        ).long() + 1
        noise = torch.rand(target.shape, device=target.device).masked_fill(~present, 2)
        mask = noise.argsort(1).argsort(1) < count[:, None]
        target_in = torch.where(mask[..., None], self.masks, self.fields(target)).sum(
            -2
        )
        sequence = torch.cat([hist, target_in[:, None, :]], dim=1)
        pad = F.pad(histories.eq(0), (0, 1), value=False)
        sequence = sequence + self.position.weight
        hidden = self.encoder(sequence, src_key_padding_mask=pad)[:, -1]
        total = hidden.new_zeros(())
        for j, (table, projection) in enumerate(zip(self.tables, self.projections)):
            take = mask[:, j]
            if not take.any():
                continue
            h = F.normalize(projection(hidden[take]), dim=-1)
            truth = target[take, j]
            if j == self.tag_index:
                total = total + self.tag_loss(h, truth, table, negatives)
                continue
            classes = table.num_embeddings - 1
            if classes <= negatives + 1:
                logits = h @ F.normalize(table.weight[1:], dim=-1).T
                labels = truth - 1
            else:
                other = torch.randint(
                    1, classes, (len(truth), negatives), device=h.device
                )
                other += (other >= truth[:, None]).long()
                candidates = torch.cat([truth[:, None], other], dim=1)
                logits = (F.normalize(table(candidates), dim=-1) * h[:, None]).sum(-1)
                labels = torch.zeros_like(truth)
            total = total + F.cross_entropy(10 * logits, labels, reduction="sum")
        return total / mask.sum()


def fit_fields(folder, c, meta, cfg, device):
    torch.manual_seed(cfg["seed"])
    names, lookup, bank, tag_labels = catalog_fields(meta)
    fields = torch.as_tensor(lookup, device=device)
    tag_index = names.index("tag") if bank is not None else None
    model = FieldEncoder(lookup.max(0).tolist(), c.length, cfg, tag_index, bank).to(
        device
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["famae_lr"], weight_decay=1e-5)
    cutoff = c.meta["cutoffs_ms"][0]
    fit = np.flatnonzero((c.done < cutoff) & (c.t < cutoff) & (c.ids > 0) & (c.y == 1))
    if not len(fit):
        raise ValueError("No completed catalog long views before SID cutoff")
    rng = np.random.default_rng(cfg["seed"])
    updates = 0
    for epoch in range(cfg["famae_epochs"]):
        order = rng.permutation(fit)
        for start in range(0, len(order), cfg["famae_batch"]):
            ix = order[start : start + cfg["famae_batch"]]
            events = c.history(ix)
            history = np.where(events >= 0, c.ids[events.clip(min=0)], 0)
            loss = model.loss(
                torch.as_tensor(history, device=device).long(),
                torch.as_tensor(c.ids[ix], device=device).long(),
                fields,
                cfg["famae_negatives"],
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            updates += 1
        print(f"FAMAE epoch={epoch + 1} completed_targets={len(fit)}", flush=True)
    # Export embeddings of unmasked static fields, independent of the requesting user.
    with torch.no_grad():
        vectors = torch.cat(
            [model.fields(batch).flatten(1).cpu() for batch in fields[1:].split(2048)]
        ).numpy()
    torch.save(
        {
            "model": model.state_dict(),
            "fields": names,
            "lookup": fields.cpu(),
            "tag_labels": tag_labels,  # label i has embedding row i+1
            "video_id": torch.as_tensor(meta.video_id.to_numpy().copy()),
        },
        folder / "famae.pt",
    )
    return vectors, {
        "fields": names,
        "targets": len(fit),
        "updates": updates,
        "cutoff_ms_exclusive": cutoff,
        "dim": vectors.shape[1],
    }
