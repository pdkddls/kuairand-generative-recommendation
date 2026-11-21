from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data import DAY


class SIDTrie:
    def __init__(self, codes):
        if codes.ndim != 2 or codes.shape[1] != 3 or len(codes) == 0:
            raise ValueError("Expected a nonempty catalog of three-code ReSID paths")
        if len(np.unique(codes, axis=0)) != len(codes):
            raise ValueError(
                "The ReSID map must give every item a unique three-code path"
            )
        self.codes = codes
        encoded = []
        offset = 1  # Reserve token zero for padding and the decoder start.
        for level in range(3):
            distinct, token_ids = np.unique(codes[:, level], return_inverse=True)
            encoded.append(token_ids + offset)
            offset += len(distinct)
        self.vocab_size = offset
        tokens = np.column_stack(encoded)
        self.item_tokens = np.vstack((np.zeros((1, 3), dtype=np.int64), tokens))

        first, sid1 = np.unique(tokens[:, 0], return_inverse=True)
        second, sid12 = np.unique(tokens[:, :2], axis=0, return_inverse=True)
        self.leaf_start = 1 + len(first) + len(second)
        self.paths = np.column_stack(
            (
                np.zeros(len(codes), dtype=np.int64),
                sid1 + 1,
                sid12 + len(first) + 1,
                self.leaf_start + np.arange(len(codes)),
            )
        )
        self.sizes = np.bincount(
            self.paths.ravel(), minlength=self.leaf_start + len(codes)
        )
        children = defaultdict(dict)
        for path, ids in zip(self.paths.tolist(), tokens.tolist(), strict=True):
            for parent, child, token in zip(path[:-1], path[1:], ids, strict=True):
                children[parent][token] = child
        self.children = [
            (
                np.fromiter(children[node], dtype=np.int64),
                np.fromiter(children[node].values(), dtype=np.int64),
            )
            for node in range(self.leaf_start)
        ]
        self.prompts = [
            np.zeros((1,), dtype=np.int64),
            *[np.array([0, token], dtype=np.int64) for token in first],
            *[np.array([0, *pair], dtype=np.int64) for pair in second],
        ]


class TemporalCounts:
    """Vectorized [t-24h,t) queries by node via packed sorted integer keys."""

    def __init__(self, times, paths):
        paths = np.asarray(paths, dtype=np.int64)
        self.origin = int(times.min())
        self.span = int(times.max()) - self.origin + 2
        if (int(paths.max()) + 1) * self.span >= np.iinfo("int64").max:
            raise ValueError("Time span too large for packed counters")
        self.keys = np.sort(
            (paths * self.span + (times - self.origin)[:, None]).ravel()
        )

    def count(self, nodes, t):
        upper = np.clip(t - self.origin, 0, self.span)
        lower = np.clip(t - DAY - self.origin, 0, self.span)
        offset = np.asarray(nodes, dtype=np.int64) * self.span
        return np.searchsorted(self.keys, offset + upper) - np.searchsorted(
            self.keys, offset + lower
        )


class Legal:
    def __init__(self, trie, counts):
        self.trie, self.counts = trie, counts
        self.degree = np.array([len(c[0]) for c in trie.children])
        self.starts = np.r_[0, self.degree.cumsum()]
        self.tokens = np.concatenate([c[0] for c in trie.children])
        self.nodes = np.concatenate([c[1] for c in trie.children])

    def branches(self, parents, time, device):
        length = self.degree[parents]
        slot = np.arange(length.max())
        index = self.starts[parents, None] + np.minimum(slot, length[:, None] - 1)
        nodes = self.nodes[index]
        bias = np.log(
            self.counts.count(nodes, time[:, None]) + 0.5 * self.trie.sizes[nodes]
        )
        bias[slot >= length[:, None]] = -np.inf
        return (
            nodes,
            torch.as_tensor(self.tokens[index], device=device),
            torch.as_tensor(bias, dtype=torch.float32, device=device),
        )


class Generator(nn.Module):
    def __init__(
        self,
        trie,
        *,
        d_model=128,
        d_ff=512,
        num_heads=4,
        d_kv=32,
        num_layers=2,
        num_decoder_layers=2,
        dropout_rate=0.1,
    ):
        super().__init__()
        from transformers import T5Config, T5ForConditionalGeneration

        cfg = T5Config(
            vocab_size=trie.vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            d_kv=d_kv,
            num_layers=num_layers,
            num_decoder_layers=num_decoder_layers,
            dropout_rate=dropout_rate,
            pad_token_id=0,
            decoder_start_token_id=0,
            eos_token_id=None,
            tie_word_embeddings=False,
        )
        self.t5 = T5ForConditionalGeneration(cfg)
        self.feedback = nn.Embedding(4, d_model, padding_idx=0)
        self.age = nn.Embedding(7, d_model, padding_idx=0)
        self.utility = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 32)
        )
        self.utility_tokens = nn.Embedding(trie.vocab_size, 32)
        self.utility_bias = nn.Embedding(trie.vocab_size, 1)
        for weights in [
            self.t5.lm_head.weight,
            self.feedback.weight,
            self.age.weight,
            self.utility_tokens.weight,
            self.utility_bias.weight,
        ]:
            nn.init.zeros_(weights)

    def encode(self, history):
        tokens = history[:, :, :3].flatten(1)
        mask = tokens.ne(0)
        mask[:, -1] |= ~mask.any(dim=1)
        context = self.feedback(history[:, :, 3]) + self.age(history[:, :, 4])
        x = self.t5.shared(tokens) + context.repeat_interleave(3, dim=1)
        return self.t5.encoder(
            inputs_embeds=x, attention_mask=mask
        ).last_hidden_state, mask

    def decode(self, prompt, encoded, mask):
        return self.t5.decoder(
            input_ids=prompt,
            encoder_hidden_states=encoded,
            encoder_attention_mask=mask,
            use_cache=False,
        ).last_hidden_state

    def value(self, hidden, tokens):
        u = self.utility(hidden.detach())
        return (u * self.utility_tokens(tokens)).sum(-1) + self.utility_bias(
            tokens
        ).squeeze(-1)

    def language(self, hidden, tokens):
        # The untied LM head is linear with no bias. Project just the legal rows.
        return torch.einsum("bd,bkd->bk", hidden, self.t5.lm_head.weight[tokens])


def batch_loss(model, legal, corpus, ix, cfg, device):
    trie = legal.trie
    items = corpus.ids[ix]
    times = corpus.t[ix]
    tokens = torch.as_tensor(trie.item_tokens[items], device=device)
    prompt = F.pad(tokens[:, :-1], (1, 0))
    history = torch.as_tensor(corpus.packed(ix, trie.item_tokens), device=device)
    hidden = model.decode(prompt, *model.encode(history))
    labels = torch.as_tensor(corpus.y[ix].astype(np.float32), device=device)
    utility = F.binary_cross_entropy_with_logits(
        model.value(hidden, tokens), labels[:, None].expand_as(tokens)
    )
    positive = np.flatnonzero(corpus.y[ix])
    generation = hidden.new_zeros(())
    if len(positive):
        ix_pos = torch.as_tensor(positive, device=device)
        for d in range(3):
            parents = trie.paths[items[positive] - 1, d]
            _, legal_tokens, bias = legal.branches(parents, times[positive], device)
            scores = model.language(hidden[ix_pos, d], legal_tokens) + bias
            target = (legal_tokens == tokens[ix_pos, d, None]) & torch.isfinite(bias)
            if not (target.sum(1) == 1).all():
                raise ValueError("Target path alignment failed")
            generation = (
                generation + F.cross_entropy(scores, target.long().argmax(1)) / 3
            )
    return generation + cfg["utility_weight"] * utility


@torch.inference_mode()
def recommend(model, legal, history, time, cfg, device):
    trie = legal.trie
    enc, mask = model.encode(torch.as_tensor(history[None], device=device))
    parents = np.array([0])
    cum = np.zeros(1)
    for depth in range(3):
        nodes_out, prob_out, rank_out = [], [], []
        for start in range(0, len(parents), 32):
            pp = parents[start : start + 32]
            n = len(pp)
            prompt = torch.as_tensor(
                np.stack([trie.prompts[p] for p in pp]), device=device
            )
            h = model.decode(prompt, enc.expand(n, -1, -1), mask.expand(n, -1))[:, -1]
            nodes, tok, bias = legal.branches(pp, np.full(n, time), device)
            lp = F.log_softmax(model.language(h, tok) + bias, dim=-1).cpu().numpy()
            good = np.isfinite(bias.cpu().numpy())
            if not np.isfinite(lp[good]).all():
                raise FloatingPointError("Invalid generation probabilities")
            cumulative = lp + cum[start : start + n, None]
            prediction = model.value(h[:, None, :], tok).sigmoid().cpu().numpy()
            guided = cumulative + cfg["utility_decode_weight"] * prediction
            nodes_out.append(nodes[good])
            prob_out.append(cumulative[good])
            rank_out.append(guided[good])
        nodes, prob, score = (
            np.concatenate(x) for x in (nodes_out, prob_out, rank_out)
        )
        if not np.isfinite(score).all():
            raise FloatingPointError("Invalid guided scores")
        k = cfg["beam"] if depth < 2 else 20
        selected = np.lexsort((nodes, -score))[:k]
        parents, cum = nodes[selected], prob[selected]
    return parents - trie.leaf_start + 1


def setup(folder, corpus):
    trie = SIDTrie(np.load(folder / "codes.npy"))
    if len(trie.codes) != corpus.n:
        raise ValueError("Catalog/SID size mismatch")
    good = (corpus.ids > 0) & (corpus.y == 1)
    counts = TemporalCounts(corpus.done[good], trie.paths[corpus.ids[good] - 1, 1:])
    return Legal(trie, counts)
