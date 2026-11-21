"""TIGER and HSTU adapted to the project's temporal next-long-view task."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import T5Config, T5ForConditionalGeneration

from data import dump


def tensor(values, device):
    return torch.as_tensor(values, dtype=torch.long, device=device)


class ResidualQuantizer(nn.Module):
    def __init__(self, width, latent, centers):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(width, 128), nn.ReLU(), nn.Linear(128, latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 128), nn.ReLU(), nn.Linear(128, width)
        )
        self.books = nn.ModuleList(nn.Embedding(centers, latent) for _ in range(3))

    def forward(self, vectors):
        latent = self.encoder(vectors)
        residual = latent
        quantized = torch.zeros_like(latent)
        penalty = 0.0
        ids = []
        for book in self.books:
            distances = torch.cdist(residual.detach(), book.weight.detach())
            index = distances.argmin(-1)
            code = book(index)
            penalty = (
                penalty
                + F.mse_loss(code, residual.detach())
                + 0.25 * F.mse_loss(code.detach(), residual)
            )
            residual = residual - code.detach()
            quantized = quantized + code
            ids.append(index)
        reconstruction = self.decoder(latent + (quantized - latent).detach())
        return torch.stack(ids, dim=1), F.mse_loss(reconstruction, vectors) + penalty


def make_codes(folder, cfg, seed, device):
    from sklearn.cluster import MiniBatchKMeans

    vectors = torch.as_tensor(np.load(folder / "item_vectors.npy"), device=device)
    centers = min(len(vectors), cfg["rq_centers"])
    model = ResidualQuantizer(vectors.size(1), cfg["rq_latent"], centers).to(device)
    with torch.no_grad():
        residual = model.encoder(vectors).cpu().numpy()
        for book in model.books:
            km = MiniBatchKMeans(
                centers, batch_size=max(centers, 1024), n_init=1, random_state=seed
            )
            labels = km.fit_predict(residual)
            book.weight.copy_(torch.as_tensor(km.cluster_centers_))
            residual = residual - km.cluster_centers_[labels]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(cfg["rq_epochs"]):
        for batch in torch.randperm(len(vectors), device=device).split(cfg["rq_batch"]):
            _, loss = model(vectors[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        codes = (
            torch.cat([model(v)[0] for v in vectors.split(cfg["rq_batch"])])
            .cpu()
            .numpy()
        )
    occupied = Counter()
    suffix = []
    for row in map(tuple, codes):
        suffix.append(occupied[row])
        occupied[row] += 1
    tokens = np.column_stack((codes, suffix)) + np.arange(4) * centers + 1
    return np.vstack((np.zeros((1, 4), dtype=np.int64), tokens))


def history_ids(c, ix):
    events = c.history(ix)
    return np.where(events >= 0, c.ids[events.clip(min=0)], 0)


class Tiger(nn.Module):
    def __init__(self, tokens, cfg):
        super().__init__()
        self.tokens = tokens
        self.beam = cfg["tiger_beam"]
        self.legal_children = defaultdict(set)
        self.items = {}
        for item, path in enumerate(map(tuple, tokens[1:].tolist()), 1):
            for depth, token in enumerate(path):
                self.legal_children[path[:depth]].add(token)
            self.items[path] = item
        if len(self.items) != len(tokens) - 1:
            raise ValueError("TIGER code collision")
        self.legal_children = {
            p: sorted(kids) for p, kids in self.legal_children.items()
        }
        width = cfg["width"]
        self.model = T5ForConditionalGeneration(
            T5Config(
                vocab_size=int(tokens.max()) + 1,
                d_model=width,
                d_kv=width // cfg["heads"],
                d_ff=4 * width,
                num_layers=cfg["layers"],
                num_decoder_layers=cfg["layers"],
                num_heads=cfg["heads"],
                dropout_rate=cfg["dropout"],
                pad_token_id=0,
                decoder_start_token_id=0,
                use_cache=False,
            )
        )

    def history(self, c, ix, device):
        ids = tensor(self.tokens[history_ids(c, ix)].reshape(len(ix), -1), device)
        mask = ids.ne(0)
        mask[:, -1] |= ~mask.any(dim=1)
        return ids, mask

    def loss(self, c, ix, device):
        ids, mask = self.history(c, ix, device)
        labels = tensor(self.tokens[c.ids[ix]], device)
        return self.model(input_ids=ids, attention_mask=mask, labels=labels).loss

    @torch.inference_mode()
    def predict(self, c, event, device):
        ids, mask = self.history(c, np.array([event]), device)
        enc = self.model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        paths = [((), 0.0)]
        for depth in range(4):
            expanded = []
            for start in range(0, len(paths), 32):
                group = paths[start : start + 32]
                prompt = tensor([[0, *p] for p, _ in group], device)
                logits = self.model(
                    encoder_outputs=(enc.expand(len(group), -1, -1),),
                    attention_mask=mask.expand(len(group), -1),
                    decoder_input_ids=prompt,
                ).logits[:, -1]
                logp = logits.log_softmax(-1).cpu().numpy()
                for (prefix, score), row in zip(group, logp):
                    expanded.extend(
                        (prefix + (i,), score + float(row[i]))
                        for i in self.legal_children[prefix]
                    )
            expanded.sort(key=lambda x: (-x[1], x[0]))
            paths = expanded[: 20 if depth == 3 else self.beam]
        return np.array([self.items[p] for p, _ in paths])


class HSTUBlock(nn.Module):
    def __init__(self, width, heads, length, dropout):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.uvqk = nn.Linear(width, width * 4)
        self.position_bias = nn.Embedding(length, heads)
        self.time_bias = nn.Embedding(32, heads)
        self.aggregate_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, times, valid):
        batch, length, width = x.shape
        u, v, q, k = (
            F.silu(self.uvqk(self.norm(x)))
            .reshape(batch, length, 4, self.heads, width // self.heads)
            .unbind(2)
        )
        positions = torch.arange(length, device=x.device)
        distance = positions[:, None] - positions
        buckets = (
            torch.log2(1 + (times[:, :, None] - times[:, None, :]).clamp_min(0) / 1000)
            .long()
            .clamp_max(31)
        )
        similarity = torch.einsum("bihd,bjhd->bijh", q, k)
        weights = F.silu(
            similarity
            + self.position_bias(distance.clamp_min(0))
            + self.time_bias(buckets)
        )
        allowed = (distance >= 0) & valid[:, None, :]
        weights = weights.masked_fill(~allowed[..., None], 0.0) / length
        pooled = torch.einsum("bijh,bjhd->bihd", weights, v).reshape(
            batch, length, width
        )
        return x + self.out(self.drop(u.reshape_as(x) * self.aggregate_norm(pooled)))


class HSTU(nn.Module):
    def __init__(self, c, cfg):
        super().__init__()
        width = cfg["width"]
        self.item = nn.Embedding(c.n + 1, width, padding_idx=0)
        self.query = nn.Parameter(torch.randn(width) * 0.02)
        self.blocks = nn.ModuleList(
            HSTUBlock(width, cfg["heads"], c.length + 1, cfg["dropout"])
            for _ in range(cfg["layers"])
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, c, ix, device):
        h = c.history(ix)
        ids = tensor(np.where(h >= 0, c.ids[h.clip(min=0)], 0), device)
        qtime = c.t[ix, None]
        past = np.where(h >= 0, c.t[h.clip(min=0)], qtime)
        times = tensor(np.concatenate([past, qtime], axis=1), device)
        valid = F.pad(ids.ne(0), (0, 1), value=True)
        x = torch.cat([self.item(ids), self.query.expand(len(ix), 1, -1)], dim=1)
        for block in self.blocks:
            x = block(x, times, valid)
        return F.linear(self.norm(x[:, -1]), self.item.weight[1:]) / x.size(-1) ** 0.5

    def loss(self, c, ix, device):
        return F.cross_entropy(self(c, ix, device), tensor(c.ids[ix] - 1, device))

    @torch.inference_mode()
    def predict(self, c, event, device):
        scores = self(c, np.array([event]), device)[0]
        if not scores.isfinite().all():
            raise FloatingPointError("Nonfinite HSTU scores")
        return (
            (torch.argsort(scores, descending=True, stable=True)[:20] + 1).cpu().numpy()
        )


def train_baselines(folder, c, cfg, seed, device, rank, metrics):
    positive = c.fit[c.y[c.fit] == 1]
    for name in ("TIGER", "HSTU"):
        dest = folder / name
        if dest.exists():
            info = json.loads((dest / "fit.json").read_text())
            if info["config"] != cfg or info["seed"] != seed:
                raise ValueError(f"{name} checkpoint configuration differs")
            continue
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        with TemporaryDirectory(dir=folder) as work:
            stage = Path(work) / name
            stage.mkdir()
            if name == "TIGER":
                tokens = make_codes(folder, cfg, seed, device)
                np.save(stage / "tokens.npy", tokens)
                model = Tiger(tokens, cfg)
            else:
                model = HSTU(c, cfg)
            model.to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
            best = -1.0
            for epoch in range(1, cfg["epochs"] + 1):
                model.train()
                order = rng.permutation(positive)
                for start in range(0, len(order), cfg["batch"]):
                    loss = model.loss(c, order[start : start + cfg["batch"]], device)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    )
                    opt.step()
                model.eval()
                ranks = [
                    rank(model.predict(c, event, device), c.ids[event])
                    for event in c.valid
                ]
                ndcg = metrics(ranks)["NDCG@20"]
                print(f"{name} epoch={epoch} validation_NDCG@20={ndcg:.6f}", flush=True)
                if ndcg > best:
                    best, chosen = ndcg, epoch
                    torch.save(model.state_dict(), stage / "model.pt")
            dump(
                stage / "fit.json",
                {
                    "selected_epoch": chosen,
                    "validation_ndcg@20": best,
                    "fit_positives": len(positive),
                    "config": cfg,
                    "seed": seed,
                },
            )
            stage.rename(dest)


def load_baselines(folder, c, device):
    models = []
    for name in ("TIGER", "HSTU"):
        dest = folder / name
        cfg = json.loads((dest / "fit.json").read_text())["config"]
        model = (
            Tiger(np.load(dest / "tokens.npy"), cfg)
            if name == "TIGER"
            else HSTU(c, cfg)
        )
        model.load_state_dict(
            torch.load(dest / "model.pt", weights_only=True, map_location="cpu")
        )
        models.append(model.to(device).eval())
    return models
