"""KuaiRand-1K batch SID generation, training, and final comparison."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from data import Corpus, dump, mapping_digest, prepare

ROOT = Path(__file__).resolve().parent


def metrics(ranks):
    r = np.asarray(ranks)
    out = {}
    for k in (5, 10, 20):
        hit = (r > 0) & (r <= k)
        out[f"Recall@{k}"] = float(hit.mean())
        out[f"NDCG@{k}"] = float((1 / np.log2(1 + r[hit])).sum() / len(r))
    out["MRR@20"] = float((1 / r[(r > 0) & (r <= 20)]).sum() / len(r))
    return out


def rank(items, target):
    if np.any(items <= 0) or len(np.unique(items)) != len(items):
        raise ValueError("Invalid or duplicate recommended items")
    found = np.flatnonzero(items == target)
    return int(found[0]) + 1 if len(found) else 0


def check_mapping(folder, trie, cfg):
    import pandas as pd

    info = json.loads((folder / "semantic.json").read_text())
    ids = pd.read_csv(folder / "items.csv", usecols=["video_id"]).video_id
    digest = mapping_digest(ids, trie.codes)
    if (
        info.get("sid_method") != cfg["sid_method"]
        or info.get("mapping_sha256") != digest
    ):
        raise ValueError("Prepared SID mapping and encoder manifest do not match")
    return digest


def train(folder, c, cfg, device):
    import torch

    from generator import Generator, batch_loss, recommend, setup

    legal = setup(folder, c)
    digest = check_mapping(folder, legal.trie, cfg)
    torch.manual_seed(cfg["seed"])
    model = Generator(legal.trie, **cfg.get("t5", {})).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    rng = np.random.default_rng(cfg["seed"])
    best = -1
    n_updates = 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        order = rng.permutation(c.fit)
        for start in range(0, len(order), cfg["batch"]):
            ix = order[start : start + cfg["batch"]]
            loss = batch_loss(model, legal, c, ix, cfg, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            opt.step()
            n_updates += 1
            if n_updates % 500 == 0:
                print(
                    f"epoch={epoch} updates={n_updates} fit_rows={start + len(ix)}/{len(order)}",
                    flush=True,
                )
        model.eval()
        ranks = []
        for ix in c.valid:
            hist = c.packed(np.array([ix]), legal.trie.item_tokens)[0]
            ranks.append(
                rank(recommend(model, legal, hist, c.t[ix], cfg, device), c.ids[ix])
            )
        ndcg = metrics(ranks)["NDCG@20"]
        print(f"epoch={epoch} validation_NDCG@20={ndcg:.6f}", flush=True)
        if ndcg > best:
            best, chosen = ndcg, epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "mapping_sha256": digest,
                    "t5": cfg.get("t5", {}),
                    "events_sha256": c.meta.get("events_sha256"),
                },
                folder / "model.tmp",
            )
            os.replace(folder / "model.tmp", folder / "model.pt")
    dump(
        folder / "fit.json",
        {
            "selected_epoch": chosen,
            "validation_ndcg@20": best,
            "updates": n_updates,
            "mapping_sha256": digest,
            "epochs": cfg["epochs"],
            "optimizer_exposures_per_epoch": len(c.fit),
        },
    )


def evaluate(folder, c, cfg, device):
    import torch

    from baseline import load_baselines
    from generator import Generator, recommend, setup

    legal = setup(folder, c)
    model = Generator(legal.trie, **cfg.get("t5", {})).to(device)
    checkpoint = torch.load(folder / "model.pt", map_location=device, weights_only=True)
    if "sid_method" in cfg:
        digest = check_mapping(folder, legal.trie, cfg)
        if checkpoint.get("mapping_sha256") != digest:
            raise ValueError("T5 checkpoint was trained on a different SID mapping")
        if "t5" in checkpoint and checkpoint["t5"] != cfg.get("t5", {}):
            raise ValueError("T5 configuration differs from the training checkpoint")
        if checkpoint.get("events_sha256") is not None and checkpoint[
            "events_sha256"
        ] != c.meta.get("events_sha256"):
            raise ValueError("Exposure dataset differs from the training checkpoint")
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()
    tiger, hstu = load_baselines(folder, c, device)
    labels = ("TIGER", "HSTU", "ReSID–T5")
    rows = []
    for ix in c.test:
        hist = c.packed(np.array([ix]), legal.trie.item_tokens)[0]
        sid_items = recommend(model, legal, hist, c.t[ix], cfg, device)
        rows.append(
            (
                int(ix),
                rank(tiger.predict(c, ix, device), c.ids[ix]),
                rank(hstu.predict(c, ix, device), c.ids[ix]),
                rank(sid_items, c.ids[ix]),
            )
        )
    results = {name: metrics([r[j] for r in rows]) for j, name in enumerate(labels, 1)}
    np.savetxt(
        folder / "test_ranks.csv",
        rows,
        fmt="%d",
        delimiter=",",
        header="event_index,tiger_rank,hstu_rank,final_rank",
        comments="",
    )
    keys = list(results[labels[0]])
    lines = ["| 方案 | " + " | ".join(keys) + " |", "|---" + "|---:" * len(keys) + "|"]
    for name, values in results.items():
        lines.append(
            "| " + name + " | " + " | ".join(f"{values[k]:.2%}" for k in keys) + " |"
        )
    table = "\n".join(lines) + "\n"
    context = (
        f"总曝光 {len(c.e):,} 条，训练窗口可用曝光 {len(c.fit):,} 条，"
        f"训练长看正例 {c.meta['fit_long_views']:,} 条。"
        f"固定目录 {c.n:,} 个视频，测试查询 {len(rows):,} 条，"
        f"其中目录内目标 {c.meta['test']['in_catalog']:,} 条。"
    )
    (folder / "comparison.md").write_text(context + "\n\n" + table, encoding="utf-8")
    dump(
        folder / "comparison.json",
        {
            "status": "completed",
            "dataset_exposures": len(c.e),
            "train_exposures": len(c.fit),
            "train_long_views": c.meta["fit_long_views"],
            "catalog_items": c.n,
            "events_sha256": c.meta.get("events_sha256"),
            "exposures_by_window": c.meta["exposures_by_window"],
            "hits": {
                name: {
                    f"@{k}": sum(0 < row[j] <= k for row in rows) for k in (5, 10, 20)
                }
                for j, name in enumerate(labels, 1)
            },
            "queries": len(rows),
            "in_catalog_queries": c.meta["test"]["in_catalog"],
            "selected_epoch": json.loads((folder / "fit.json").read_text())[
                "selected_epoch"
            ],
            "sid_method": cfg.get("sid_method", "legacy_qwen_residual_lfq_ctr"),
            "baseline_epochs": {
                name: json.loads((folder / name / "fit.json").read_text())[
                    "selected_epoch"
                ]
                for name in labels[:2]
            },
            "metrics": results,
        },
    )
    print(table)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command",
        choices=["prepare", "encode", "train", "baselines", "evaluate", "all"],
    )
    p.add_argument("--run-dir", type=Path, required=True)
    sources = p.add_mutually_exclusive_group()
    sources.add_argument(
        "--events-csv",
        type=Path,
        help="Existing dataset with exactly four million unique exposures",
    )
    sources.add_argument(
        "--logs",
        type=Path,
        help="Directory containing all three original KuaiRand-1K log CSVs",
    )
    p.add_argument(
        "--metadata", type=Path, help="One static item metadata CSV, keyed by video_id"
    )
    p.add_argument(
        "--model-path", help="Local Qwen3-Embedding-4B directory or HF identifier"
    )
    p.add_argument(
        "--captions",
        type=Path,
        help="Official video captions CSV keyed by final_video_id",
    )
    p.add_argument(
        "--categories",
        type=Path,
        help="Official video categories CSV keyed by final_video_id",
    )
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    folder = args.run_dir.resolve()
    cfg = json.loads((ROOT / "config.json").read_text())
    if args.command not in ("prepare", "all") and any(
        (args.events_csv, args.logs, args.metadata, args.captions, args.categories)
    ):
        p.error(
            "Input data options apply to prepare/all; other stages read the saved run"
        )
    if args.model_path and args.command not in ("encode", "all"):
        p.error("--model-path applies to encode/all")
    if args.command in ("prepare", "all") and cfg["events"] != 4000000:
        p.error("This release requires exactly 4,000,000 distinct exposures")
    stages = (
        ["prepare", "encode", "train", "baselines", "evaluate"]
        if args.command == "all"
        else [args.command]
    )
    for stage in stages:
        if stage == "prepare":
            if folder.exists():
                p.error(
                    "prepare requires a new run directory; resume with encode, train or evaluate"
                )
            if not (args.logs or args.events_csv) or not args.metadata:
                p.error("prepare requires --metadata and either --logs or --events-csv")
            folder.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=folder.parent) as tmp:
                stage_dir = Path(tmp) / "data"
                stage_dir.mkdir()
                prepare(
                    args.logs,
                    args.metadata,
                    stage_dir,
                    cfg,
                    args.events_csv,
                    args.captions,
                    args.categories,
                )
                dump(stage_dir / "config.json", cfg)
                dump(
                    stage_dir / "baselines.json",
                    json.loads((ROOT / "baselines.json").read_text()),
                )
                stage_dir.rename(folder)
            continue
        saved = json.loads((folder / "config.json").read_text())
        if saved.get("events") != 4_000_000:
            p.error("Expected a run prepared from 4,000,000 distinct exposures")
        method = saved.get("sid_method")
        if method not in ("famae_qwen_gaoq_ctr_v2", "famae_qwen_gaoq_ctr_v1", None):
            p.error("Unknown SID method in saved run")
        if stage == "encode" and method != "famae_qwen_gaoq_ctr_v2":
            p.error("v2 encoding requires a new prepare run with the original inputs")
        if method is None and stage not in ("baselines", "evaluate"):
            p.error("Legacy runs support baselines and evaluate only")
        cfg = saved
        c = Corpus(folder, cfg)
        if stage == "encode":
            if (folder / "semantic.json").exists():
                p.error("SID stage already completed; continue with train")
            from semantic import build

            build(folder, c, cfg, args.device, args.model_path)
        if stage == "train":
            if (
                not (folder / "semantic.json").is_file()
                or (folder / "fit.json").exists()
            ):
                p.error("train requires a completed SID stage and an untrained run")
            train(folder, c, cfg, args.device)
        if stage == "baselines":
            if not (folder / "semantic.json").is_file():
                p.error("baselines requires a completed encode stage")
            from baseline import train_baselines

            baseline_config = folder / "baselines.json"
            if not baseline_config.exists():
                baseline_config = ROOT / "baselines.json"  # pre-snapshot runs
            train_baselines(
                folder,
                c,
                json.loads(baseline_config.read_text()),
                cfg["seed"],
                args.device,
                rank,
                metrics,
            )
        if stage == "evaluate":
            if not all(
                (folder / name / "fit.json").is_file() for name in ("TIGER", "HSTU")
            ):
                p.error(
                    "run the baselines command for this run directory before evaluate"
                )
            if not (folder / "fit.json").is_file():
                p.error("evaluate requires a completed training run")
            evaluate(folder, c, cfg, args.device)


if __name__ == "__main__":
    main()
