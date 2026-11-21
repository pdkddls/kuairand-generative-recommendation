"""Build a fixed-sized chronological evaluation corpus from original KuaiRand logs."""

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

LOGS = [
    "log_standard_4_08_to_4_21_1k.csv",
    "log_standard_4_22_to_5_08_1k.csv",
    "log_random_4_22_to_5_08_1k.csv",
]
COLS = ["user_id", "video_id", "time_ms", "long_view", "is_click", "play_time_ms"]
DTYPE = [(k, "i8" if k not in ("long_view", "is_click") else "i1") for k in COLS]
LEVELS = ("first", "second", "third", "fourth")
CATEGORY_IDS = {
    f"{level}_level_category_id": f"category{i}_id"
    for i, level in enumerate(LEVELS, start=1)
}
STRUCTURED = [
    "author_id",
    "video_type",
    "upload_type",
    "music_type",
    "tag",
    "video_duration",
    *CATEGORY_IDS.values(),
]
CATEGORIES = [level + "_level_category_name" for level in LEVELS]
CONTENT = [
    "title",
    "description",
    "caption",
    "show_cover_text",
    "category",
    *CATEGORIES,
]


def clean_values(values):
    values = values.astype("string").str.strip()
    missing = values.str.upper().isin(["", "UNKNOWN", "NAN", "NULL", "NONE"])
    return values.mask(missing | values.str.fullmatch(r"-(?:1|124)(?:\.0+)?", na=False))


def category_aliases(frame):
    # Use one column per level, including when the primary table has both spellings.
    for official, alias in CATEGORY_IDS.items():
        if official in frame:
            values = clean_values(frame.pop(official))
            if alias in frame:
                values = values.fillna(clean_values(frame[alias]))
            frame[alias] = values
    return frame


def attach_content(catalog, path, fields):
    header = pd.read_csv(path, nrows=0).columns
    selected = [field for field in fields if field in header]
    if "final_video_id" not in header or not selected:
        raise ValueError(f"Expected final_video_id and content columns in {path}")
    matched = []
    for part in pd.read_csv(
        path,
        usecols=["final_video_id", *selected],
        dtype={field: "string" for field in selected},
        chunksize=100000,
    ):
        matched.append(part[part.final_video_id.isin(catalog.video_id)])
    supplement = category_aliases(pd.concat(matched).set_index("final_video_id"))
    if supplement.index.has_duplicates:
        raise ValueError(f"Duplicate catalog video in {path}")
    for field in supplement:
        values = catalog.video_id.map(clean_values(supplement[field]))
        catalog[field] = (
            values.combine_first(catalog[field]) if field in catalog else values
        )
    return len(supplement)


DAY = 86400000


def mapping_digest(video_ids, codes):
    digest = hashlib.sha256()
    digest.update(np.asarray(video_ids, dtype="<i8").tobytes())
    digest.update(np.asarray(codes, dtype="<i4").tobytes())
    return digest.hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(root, meta, out, cfg, events_csv=None, captions=None, categories=None):
    with sqlite3.connect(out / "stage.sqlite") as db:
        db.execute(
            "CREATE TABLE e(u INTEGER,v INTEGER,t INTEGER,y INTEGER,c INTEGER,p INTEGER,"
            "r INTEGER,PRIMARY KEY(u,v,t)) WITHOUT ROWID"
        )
        raw = 0
        sources = (
            [Path(events_csv)]
            if events_csv is not None
            else [root / name for name in LOGS]
        )
        for src, path in enumerate(sources):
            name = path.name
            header = pd.read_csv(path, nrows=0).columns
            columns = COLS + (["is_rand"] if "is_rand" in header else [])
            for frame in pd.read_csv(path, usecols=columns, chunksize=100000):
                if "is_rand" not in frame:
                    frame["is_rand"] = int(src == 2)
                if (
                    not frame[["long_view", "is_click", "is_rand"]]
                    .isin([0, 1])
                    .all()
                    .all()
                ):
                    raise ValueError(f"Missing or invalid binary feedback in {name}")
                play = frame["play_time_ms"].to_numpy(dtype=float)
                if not np.isfinite(play).all() or (play < 0).any():
                    raise ValueError(f"Invalid play_time_ms in {name}")
                frame["play_time_ms"] = np.ceil(play).astype("int64")
                for key in COLS[:3]:
                    if (
                        frame[key].isna().any()
                        or (frame[key] < 0).any()
                        or (frame[key] % 1 != 0).any()
                    ):
                        raise ValueError(f"Invalid {key} in {name}")
                rows = (
                    frame[COLS + ["is_rand"]]
                    .astype("int64")
                    .itertuples(index=False, name=None)
                )
                db.executemany(
                    "INSERT INTO e VALUES (?,?,?,?,?,?,?) ON CONFLICT(u,v,t) DO UPDATE SET "
                    "y=excluded.y,c=excluded.c,p=excluded.p,r=excluded.r WHERE excluded.r>e.r",
                    rows,
                )
                raw += len(frame)
            db.commit()
            print("read", name, raw, flush=True)
        total = db.execute("SELECT count(*) FROM e").fetchone()[0]
        n = cfg["events"]
        if total < n:
            raise ValueError(f"Need {n} distinct exposures, found {total}")
        if events_csv is not None and total != n:
            raise ValueError(
                f"Supplied dataset contains {total} distinct exposures; expected {n}"
            )
        positions = (
            np.arange(n)
            if total == n
            else np.sort(
                np.random.default_rng(cfg["seed"]).choice(total, n, replace=False)
            )
        )
        events = np.lib.format.open_memmap(out / "events.npy", "w+", DTYPE, (n,))
        cursor = db.execute("SELECT u,v,t,y,c,p FROM e ORDER BY t,u,v")
        start = written = 0
        while rows := cursor.fetchmany(100000):
            stop = start + len(rows)
            end = positions.searchsorted(stop)
            events[written:end] = np.array(rows, dtype=DTYPE)[
                positions[written:end] - start
            ]
            start, written = stop, end
        if written != n:
            raise RuntimeError("Incomplete exposure replay")
        events.flush()
    (out / "stage.sqlite").unlink()
    t = events["time_ms"]
    cuts = [int(t[int(n * x)]) for x in cfg["cut_fractions"]]
    if len(set(cuts)) != 3:
        raise ValueError("Time windows collapsed")
    warm = t < cuts[0]
    counts = pd.Series(events["video_id"][warm]).value_counts().rename("exposures")
    # Only static metadata columns can enter the LLM text. Read in bounded batches.
    fields = pd.read_csv(meta, nrows=0).columns
    use = ["video_id"] + [
        k for k in STRUCTURED + list(CATEGORY_IDS) + CONTENT if k in fields
    ]
    if len(use) < 2 and captions is None and categories is None:
        raise ValueError("Need static item attributes or title/description")
    pieces = []
    for part in pd.read_csv(
        meta, usecols=use, chunksize=100000, dtype={k: "string" for k in use[1:]}
    ):
        pieces.append(part[part.video_id.isin(counts.index)])
    catalog = category_aliases(pd.concat(pieces).set_index("video_id"))
    if catalog.index.has_duplicates:
        raise ValueError("Duplicate video_id in static metadata")
    catalog = catalog.join(counts).reset_index()
    catalog = catalog.sort_values(
        ["exposures", "video_id"], ascending=[False, True]
    ).head(cfg["catalog_cap"])
    catalog = catalog.sort_values("video_id").reset_index(drop=True)
    if len(catalog) < cfg["centers"]:
        raise ValueError("Too few eligible catalog items")
    joins = {}
    for path, wanted in (
        (captions, ["caption", "show_cover_text"]),
        (categories, [*CATEGORIES, *CATEGORY_IDS, *CATEGORY_IDS.values()]),
    ):
        if path is not None:
            joins[str(path.resolve())] = attach_content(catalog, path, wanted)
    content_fields = [field for field in CONTENT if field in catalog]
    text = pd.Series("", index=catalog.index)
    has_free_text = pd.Series(False, index=catalog.index)
    for field in content_fields:
        value = clean_values(catalog[field])
        present = value.notna()
        if field in ("title", "description", "caption", "show_cover_text"):
            has_free_text |= present
        text.loc[present] += field + "=" + value[present] + " | "
    catalog["text"] = text.str.rstrip(" |")
    if not catalog.text.str.len().any():
        raise ValueError(
            "No catalog video has content text. Supply --captions or --categories "
            "with official KuaiRand video IDs, or prejoin text into --metadata."
        )
    catalog.to_csv(out / "items.csv", index=False)
    ids = pd.Index(catalog.video_id).get_indexer(events["video_id"]).astype("int32") + 1
    np.save(out / "items.npy", ids)
    done = t + events["play_time_ms"]
    fit = (t >= cuts[0]) & (t < cuts[1]) & (done < cuts[1]) & (ids > 0)
    if not (fit & (events["long_view"] == 1)).any():
        raise ValueError("Empty positive training set")
    np.save(out / "fit.npy", np.flatnonzero(fit).astype("int32"))
    summary = {
        "input_mode": "prepared_csv" if events_csv is not None else "raw_logs",
        "sampled_after_dedup": bool(total > n),
        "events_sha256": hashlib.sha256(memoryview(events).cast("B")).hexdigest(),
        "distinct_users": len(np.unique(events["user_id"])),
        "input_files": [str(p.resolve()) for p in sources],
        "static_metadata": str(meta.resolve()),
        "input_rows": raw,
        "unique_input_exposures": total,
        "dataset_exposures": n,
        "cutoffs_ms": cuts,
        "catalog_items": len(catalog),
        "text_fields": content_fields,
        "supplement_matched_catalog_items": joins,
        "items_with_content": int(catalog.text.str.len().gt(0).sum()),
        "items_with_free_text": int(has_free_text.sum()),
        "items_with_category_text_only": int(
            (catalog.text.ne("") & ~has_free_text).sum()
        ),
        "distinct_item_texts": catalog.loc[catalog.text.ne(""), "text"].nunique(),
        "fit_exposures": int(fit.sum()),
        "fit_long_views": int(events["long_view"][fit].sum()),
        "exposures_by_window": np.bincount(
            np.searchsorted(cuts, t, side="right"), minlength=4
        ).tolist(),
    }
    for name, eligible in [
        ("valid", (t >= cuts[1]) & (t < cuts[2]) & (done < cuts[2])),
        ("test", t >= cuts[2]),
    ]:
        positive = np.flatnonzero(eligible & (events["long_view"] == 1))
        # Last positive per user; at tied timestamps select greater raw video_id deterministically.
        ix = pd.Series(positive, index=events["user_id"][positive])
        ix = np.sort(ix[~ix.index.duplicated(keep="last")].to_numpy()).astype("int32")
        if not len(ix):
            raise ValueError(f"Empty {name} window")
        np.save(out / f"{name}.npy", ix)
        summary[name] = {"queries": len(ix), "in_catalog": int((ids[ix] > 0).sum())}
    dump(out / "data.json", summary)
    print(json.dumps(summary), flush=True)


class Corpus:
    def __init__(self, folder, cfg):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "data.json").read_text())
        self.e = np.load(self.folder / "events.npy", mmap_mode="r")
        self.ids = np.load(self.folder / "items.npy", mmap_mode="r")
        if len(self.e) != cfg["events"] or self.meta["dataset_exposures"] != len(
            self.e
        ):
            raise ValueError("Dataset/config count mismatch")
        self.t = self.e["time_ms"]
        self.done = self.t + self.e["play_time_ms"]
        self.y = self.e["long_view"]
        self.fit, self.valid, self.test = [
            np.load(self.folder / f"{k}.npy") for k in ("fit", "valid", "test")
        ]
        self.n = self.meta["catalog_items"]
        self.length = cfg["history_length"]
        self.users = self.e["user_id"]
        valid = np.flatnonzero(self.ids > 0)
        order = valid[np.argsort(self.users[valid], kind="stable")]
        users, starts = np.unique(self.users[order], return_index=True)
        self.groups = dict(zip(users, np.split(order, starts[1:])))
        self.history_end = np.zeros(len(self.e), dtype="int32")
        self.starts = {}
        # Global flattened history, with each user's own run of left padding.
        segments = []
        size = 0
        for u, group in self.groups.items():
            self.starts[u] = size + self.length
            size += self.length + len(group)
            segments.extend(
                (np.full(self.length, -1, dtype="int32"), group.astype("int32"))
            )
        self.flat = np.concatenate(segments)
        idx = np.argsort(self.users, kind="stable")
        uu, starts = np.unique(self.users[idx], return_index=True)
        for u, group in zip(uu, np.split(idx, starts[1:])):
            if u in self.groups:
                hist = self.groups[u]
                self.history_end[group] = self.starts[u] + np.searchsorted(
                    self.t[hist], self.t[group], "left"
                )

    def history(self, ix):
        end = self.history_end[ix, None]
        positions = end - np.arange(self.length, 0, -1)
        return np.where(end == 0, -1, self.flat[np.maximum(positions, 0)])

    def packed(self, ix, tokens):
        h = self.history(ix)
        valid = h >= 0
        safe = h.clip(min=0)
        elapsed = self.t[ix, None] - self.t[safe]
        observed = self.done[safe] < self.t[ix, None]
        items = np.where(valid, self.ids[safe], 0)
        feedback = np.where(valid, np.where(observed, 2 + self.y[safe], 1), 0)
        age = np.where(
            valid,
            1
            + np.searchsorted(
                [10000, 60000, 600000, 3600000, 86400000], elapsed, side="right"
            ),
            0,
        )
        return np.concatenate(
            (tokens[items], feedback[..., None], age[..., None]), axis=-1
        )
