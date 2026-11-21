"""CPU integration check with tiny random Qwen3 weights and synthetic exposures."""

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

from baseline import load_baselines, train_baselines
from data import LOGS, Corpus, prepare
from famae import FieldEncoder, catalog_fields
from gaoq import aligned_ids, balanced_partition
from generator import Generator, Legal, SIDTrie, TemporalCounts, recommend, setup
from main import ROOT, evaluate, metrics, rank, train
from semantic import build


def check_utility_pruning():
    trie = SIDTrie(np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]))
    counts = TemporalCounts(np.array([100]), trie.paths[:1, 1:])
    legal = Legal(trie, counts)
    model = Generator(trie).eval()
    # LM and initial utility are tied. Change only the utility of one SID2 token.
    sid2 = trie.item_tokens[2, 1]
    with torch.no_grad():
        model.utility_bias.weight[sid2] = 4
    common = (model, legal, np.zeros((1, 5), dtype="int64"), 50)
    plain = recommend(*common, {"beam": 2, "utility_decode_weight": 0}, "cpu")
    guided = recommend(*common, {"beam": 2, "utility_decode_weight": 1}, "cpu")
    assert plain.tolist() == [1, 2] and guided.tolist() == [2, 4]

    # Utility alone must update the head while leaving the decoder outside its graph.
    h = torch.ones(1, 128, requires_grad=True)
    model.value(h, torch.tensor([sid2])).sum().backward()
    assert h.grad is None and model.utility_bias.weight.grad[sid2].item() == 1


def check_fields(cfg):
    frame = pd.DataFrame(
        {
            "author_id": ["-1", "-1.00", None, "-124.0", "8", "8.0"],
            "tag": ["12,65,12", "65,12.0,-124.0", "12", "65", None, "-1.0,UNKNOWN"],
            "music_type": [None, -1, -1.0, "-124.00", "unknown", ""],
        }
    )
    names, lookup, bank, labels = catalog_fields(frame)
    assert names == ["video_id", "author_id", "tag"]
    assert lookup[1:, 1].tolist() == [0, 0, 0, 0, 1, 1]
    assert labels == ["12", "65"] and lookup[1, 2] == lookup[2, 2]
    enc = FieldEncoder(lookup.max(0), 2, cfg, 2, bank)
    fields = enc.fields(torch.from_numpy(lookup))[:, 2]
    assert torch.equal(fields[1], fields[2])
    assert torch.allclose(fields[1], (fields[3] + fields[4]) / 2)
    assert torch.all(fields[[0, 5, 6]] == 0)
    # Check the distribution over both true labels and gradient flow to shared rows.
    hidden = torch.nn.functional.normalize(torch.randn(1, cfg["famae_dim"]), dim=-1)
    table = enc.tables[2]
    expected = (
        -(10 * (hidden @ torch.nn.functional.normalize(table.weight[1:], dim=-1).T))
        .log_softmax(1)
        .mean()
    )
    loss = enc.tag_loss(hidden, torch.tensor([lookup[1, 2]]), table, 128)
    assert torch.allclose(loss, expected, atol=1e-5)
    loss.backward()
    assert torch.isfinite(table.weight.grad).all()
    assert (table.weight.grad[1:].norm(dim=1) > 0).all()
    assert catalog_fields(pd.DataFrame({"tag": ["", "-1.0,UNKNOWN"]}))[0] == [
        "video_id"
    ]


def check():
    torch.manual_seed(11)
    torch.set_num_threads(1)
    check_utility_pruning()
    cfg = json.loads((ROOT / "config.json").read_text())
    cfg.update(
        t5={**cfg["t5"], "d_model": 64, "d_ff": 96, "num_layers": 1},
        events=480,
        catalog_cap=20,
        centers=3,
        embedding_dim=16,
        embedding_batch=4,
        famae_epochs=1,
        famae_batch=8,
        famae_negatives=4,
        epochs=1,
        batch=32,
        beam=20,
    )
    check_fields(cfg)
    rng = np.random.default_rng(7)
    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        q = out / "tiny_qwen"
        q.mkdir()
        token = Tokenizer(
            WordLevel(
                {"[UNK]": 0, "[PAD]": 1, **{str(i): i + 2 for i in range(30)}},
                unk_token="[UNK]",
            )
        )
        token.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(
            tokenizer_object=token, unk_token="[UNK]", pad_token="[PAD]"
        ).save_pretrained(q)
        Qwen3Model(
            Qwen3Config(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
            )
        ).save_pretrained(q)
        n = 600
        frame = pd.DataFrame(
            {
                "user_id": np.arange(n) % 10,
                "video_id": rng.integers(0, 28, n),
                "time_ms": (np.arange(n) // 3) * 10000,
                "long_view": rng.integers(0, 2, n),
                "is_click": rng.integers(0, 2, n),
                "play_time_ms": rng.integers(0, 25000, n),
            }
        )
        for i, name in enumerate(LOGS):
            part = frame.iloc[i * 200 : (i + 1) * 200].copy()
            if i == 2:
                part = pd.concat([part, frame.iloc[:1].assign(is_click=1)])
            part.to_csv(out / name, index=False)
        pd.DataFrame(
            {
                "video_id": range(28),
                "author_id": [None if i % 2 == 0 else -1 for i in range(28)],
                "video_type": ["NORMAL" for i in range(28)],
                "tag": [f"{i},65" if i % 3 else "65" for i in range(28)],
                "first_level_category_id": ["-124.00"] * 28,
                "category1_id": ["42.0"] * 28,
                "caption": ["0" if i == 0 else "" for i in range(28)],
            }
        ).to_csv(out / "meta.csv", index=False)
        captions = pd.DataFrame(
            {
                "final_video_id": range(0, 28, 2),
                "caption": list(map(str, range(0, 28, 2))),
            }
        )
        captions.loc[0, "caption"] = " UNKNOWN "  # retain valid primary caption
        captions.iloc[::-1].to_csv(out / "captions.csv", index=False)
        pd.DataFrame(
            {
                "final_video_id": [1, 0],
                "first_level_category_name": ["球类", "运动"],
                "first_level_category_id": [-124.0, 7.0],
                "fourth_level_category_id": [None, 1056.0],
            }
        ).to_csv(out / "categories.csv", index=False)
        prepare(
            out,
            out / "meta.csv",
            out,
            cfg,
            captions=out / "captions.csv",
            categories=out / "categories.csv",
        )
        c = Corpus(out, cfg)
        assert len(c.e) == 480 and c.meta["unique_input_exposures"] == 600
        for i, h in enumerate(c.history(np.arange(len(c.e)))):
            oracle = np.flatnonzero(
                (c.t < c.t[i]) & (c.users == c.users[i]) & (c.ids > 0)
            )[-c.length :]
            assert np.array_equal(h[h >= 0], oracle)
        assert np.all(c.done[c.fit] < c.meta["cutoffs_ms"][1])
        times = np.array([1, 5, 5, 86400001])
        nodes = np.array([[1], [2], [2], [2]])
        counts = TemporalCounts(times, nodes)
        for t in (0, 1, 5, 6, 86400001, 86400002, 2 * 86400001):
            for node in (1, 2, 3):
                assert counts.count(np.int64(node), np.int64(t)) == np.sum(
                    (nodes[:, 0] == node) & (times < t) & (times >= t - 86400000)
                )
        big = TemporalCounts(
            np.array([7, 3_000_000_007]), np.array([[99999], [99999]], dtype="int32")
        )
        assert big.count(np.array([99999], dtype="int32"), 8).item() == 1
        labels, _ = balanced_partition(np.zeros((31, 4)), 5, 3, rng)
        assert np.ptp(np.bincount(labels)) == 1
        axes = np.eye(3)
        assert np.array_equal(
            aligned_ids(axes[[2, 0, 1]] + 7, np.full(3, 7), axes), [2, 0, 1]
        )
        build(out, c, cfg, "cpu", str(q))
        meta = pd.read_csv(out / "items.csv")
        keyed = meta.set_index("video_id")
        assert keyed.at[0, "category1_id"] == 7 and keyed.at[1, "category1_id"] == 42
        assert (
            keyed.at[0, "category4_id"] == 1056 and "caption=0" in keyed.at[0, "text"]
        )
        assert all("_id=" not in text for text in meta.text.dropna())
        assert c.meta["items_with_category_text_only"] == 1
        assert c.meta["items_with_free_text"] + 1 == c.meta["items_with_content"]
        checkpoint = torch.load(out / "famae.pt", weights_only=True)
        assert "author_id" not in checkpoint["fields"]
        assert {"tag", "category1_id", "category4_id"} <= set(checkpoint["fields"])
        assert checkpoint["model"]["tag_bank"].max() == len(checkpoint["tag_labels"])
        qwen = np.load(out / "item_vectors.npy")
        fused = np.load(out / "fused_vectors.npy")
        with_caption = meta.video_id.to_numpy() % 2 == 0
        available = with_caption | (meta.video_id.to_numpy() == 1)
        assert np.array_equal(np.linalg.norm(qwen, axis=1) > 0, available)
        assert all(
            f"caption={vid}" in row
            for vid, row in zip(meta.video_id[with_caption], meta.text[with_caption])
        )
        assert np.allclose(
            fused[available, -qwen.shape[1] :], qwen[available] / np.sqrt(2)
        )
        assert np.allclose(np.linalg.norm(fused[~available], axis=1), 1)
        summary = json.loads((out / "semantic.json").read_text())
        assert summary["unique_sids"] == c.n
        assert summary["famae"]["targets"] == np.sum(
            (c.ids > 0) & (c.y == 1) & (c.done < c.meta["cutoffs_ms"][0])
        )
        codes = np.load(out / "codes.npy")
        assert (
            len(np.unique(np.column_stack((codes[:, :2], codes[:, 2] // 8)), axis=0))
            == c.n
        )
        train(out, c, cfg, "cpu")
        bc = json.loads((ROOT / "baselines.json").read_text())
        bc.update(epochs=1, rq_epochs=1, rq_centers=4, width=32, layers=1, batch=16)
        train_baselines(out, c, bc, 42, "cpu", rank, metrics)
        tiger, hstu = load_baselines(out, c, "cpu")
        for baseline in (tiger, hstu):
            recommended = baseline.predict(c, c.test[0], "cpu")
            assert set(recommended) == set(range(1, c.n + 1))
        with redirect_stdout(StringIO()):
            evaluate(out, c, cfg, "cpu")
        # For a beam covering this small catalog, independently enumerate all leaves.
        legal = setup(out, c)
        tr = legal.trie
        model = Generator(tr, **cfg["t5"]).eval()
        model.load_state_dict(
            torch.load(out / "model.pt", weights_only=True)["state_dict"]
        )
        event = c.test[0]
        t = c.t[event]
        hist = c.packed(np.array([event]), tr.item_tokens)
        with torch.no_grad():
            enc, mask = model.encode(torch.as_tensor(hist))
            scores = []
            for item in range(1, c.n + 1):
                score = 0.0
                for d, node in enumerate(tr.paths[item - 1, :3]):
                    prompt = torch.as_tensor(tr.prompts[node][None, :])
                    h = model.decode(prompt, enc, mask)[:, -1]
                    tokens, children = tr.children[node]
                    idx = np.flatnonzero(tokens == tr.item_tokens[item, d])[0]
                    lm = model.t5.lm_head(h)[0, torch.as_tensor(tokens)]
                    mass = legal.counts.count(children, t) + 0.5 * tr.sizes[children]
                    score += (
                        (lm + torch.as_tensor(np.log(mass), dtype=lm.dtype))
                        .log_softmax(0)[idx]
                        .item()
                    )
                score += (
                    cfg["utility_decode_weight"]
                    * model.value(h, torch.tensor([tokens[idx]])).sigmoid().item()
                )
                scores.append(score)
            expected = np.lexsort((np.arange(1, c.n + 1), -np.array(scores))) + 1
            assert np.array_equal(
                expected[:20], recommend(model, legal, hist[0], t, cfg, "cpu")
            )

        report = json.loads((out / "comparison.json").read_text())
        assert report["hits"]["ReSID–T5"]["@20"] == int(
            np.count_nonzero(
                np.loadtxt(out / "test_ranks.csv", delimiter=",", skiprows=1)[:, 3]
            )
        )
        assert report["queries"] == len(c.test) and list(report["metrics"]) == [
            "TIGER",
            "HSTU",
            "ReSID–T5",
        ]
        assert all(0 <= v <= 1 for m in report["metrics"].values() for v in m.values())
        # A legal but permuted codebook must be rejected before scoring a new checkpoint.
        np.save(out / "codes.npy", codes[::-1])
        try:
            evaluate(out, c, cfg, "cpu")
        except ValueError as exc:
            assert "mapping" in str(exc)
        else:
            raise AssertionError("Accepted a permuted SID mapping")
    print(
        "PASS: temporal split, histories, counts, FAMAE, Qwen fusion, balanced GAOQ, unique SIDs, T5 fit and TIGER/HSTU fitting, checkpoint reload and three-row report"
    )


if __name__ == "__main__":
    check()
