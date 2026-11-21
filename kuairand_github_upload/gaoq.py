"""Fixed-catalog GAOQ with balanced hierarchical clustering and shared anchors."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits


def unit(rows):
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norm, 1e-12)


def balanced_partition(x, k, steps, rng):
    """Capacity-constrained Lloyd iterations with a greedy assignment step."""
    n = len(x)
    k = min(n, k)
    if k == n:
        return np.arange(n), x.copy()
    if k == 1:
        return np.zeros(n, dtype=int), x.mean(0, keepdims=True)
    centers = x[rng.choice(n, k, replace=False)].copy()
    quotas = np.full(k, n // k)
    quotas[: n % k] += 1
    previous = None
    for _ in range(steps):
        dist = (centers * centers).sum(1) - 2 * (x @ centers.T)
        pref = np.argsort(dist, axis=1, kind="stable")
        rows = np.arange(n)
        gap = dist[rows, pref[:, 1]] - dist[rows, pref[:, 0]]
        priority = np.argsort(-gap, kind="stable")
        labels = np.empty(n, dtype=int)
        remaining = quotas.copy()
        for row in priority:
            choice = pref[row, 0]
            if remaining[choice] == 0:
                choice = pref[row, np.flatnonzero(remaining[pref[row]])[0]]
            labels[row] = choice
            remaining[choice] -= 1
        if previous is not None and np.array_equal(previous, labels):
            break
        # Sorting once avoids scanning the full vector table separately for each cluster.
        grouped = x[np.argsort(labels, kind="stable")]
        starts = np.r_[0, quotas.cumsum()[:-1]]
        centers = (np.add.reduceat(grouped, starts, axis=0) / quotas[:, None]).astype(
            x.dtype
        )
        previous = labels
    return labels, centers


def anchors(dim, count, rng):
    if count > dim:
        raise ValueError(
            "GAOQ needs at least as many fused dimensions as shared directions"
        )
    q, _ = np.linalg.qr(rng.normal(size=(dim, count)))
    return q.T.astype(np.float32)


def aligned_ids(children, parent, directions):
    similarity = unit(children - parent) @ directions.T
    local, shared = linear_sum_assignment(-similarity)
    labels = np.empty(len(children), dtype=np.int32)
    labels[local] = shared
    return labels


@threadpool_limits.wrap(limits=1, user_api="blas")
def quantize(vectors, cfg):
    rng = np.random.default_rng(cfg["seed"])
    k, steps = cfg["centers"], cfg["gaoq_iterations"]
    roots, first = balanced_partition(vectors, k, steps, rng)
    children_per_root = min(k, int(np.bincount(roots).max()))
    max_leaf = (
        int(np.bincount(roots).max()) + children_per_root - 1
    ) // children_per_root
    second_axes = anchors(vectors.shape[1], children_per_root, rng)
    leaf_axes = anchors(vectors.shape[1], max_leaf, rng)
    codes = np.zeros((len(vectors), 3), dtype=np.int32)
    codes[:, 0] = roots
    second_centers = np.zeros(
        (len(first), children_per_root, vectors.shape[1]), dtype=np.float32
    )
    for root, parent in enumerate(first):
        indices = np.flatnonzero(roots == root)
        group = vectors[indices]
        labels, centers = balanced_partition(group, children_per_root, steps, rng)
        second = aligned_ids(centers, parent, second_axes)
        second_centers[root, second] = centers
        codes[indices, 1] = second[labels]
        for child, center in enumerate(centers):
            members = indices[labels == child]
            # One item per leaf. Match sibling directions injectively to shared leaf IDs.
            codes[members, 2] = aligned_ids(vectors[members], center, leaf_axes)
    return codes, {
        "root_centers": first,
        "child_centers": second_centers,
        "child_anchors": second_axes,
        "leaf_anchors": leaf_axes,
    }
