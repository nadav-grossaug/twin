"""Link-prediction utility of a synthetic graph (plan section 7).

Question: if an analyst trains a link predictor on the synthetic graph, how well does it predict
held-out edges of the REAL graph, compared with a predictor trained on the real graph itself?

Design
------
1. The real edges are split 80/20 (fixed seed) into ``train`` and ``test`` edges. The "real train graph"
   keeps all nodes, the real attributes and only the train edges. Test positives are the held-out edges;
   test negatives are an equal number of uniformly sampled node pairs that are not edges of the FULL
   real graph.
2. Features of a node pair (u, v), all computed from the graph the pair is scored in:
   log(1 + common neighbours), Jaccard, Adamic-Adar, log(1 + degree product), number of equal attribute
   codes over the shared attribute columns, and equality of the homophily attribute.
3. Classifier: L2-regularised logistic regression on standardised features (numpy/scipy L-BFGS, no new
   dependency), trained on balanced positive/negative pairs.
4. Two settings, both evaluated (AUC) on the same real test pairs, with test features always computed
   on the real TRAIN graph:
   - TRTR: training pairs come from the real train graph (features from the real train graph).
   - TSTR: training pairs come from the SYNTHETIC graph (features from the synthetic graph).
   Reported: ``auc_tstr``, ``auc_trtr`` and ``auc_ratio = auc_tstr / auc_trtr``.

Why node ids are never compared: synthetic nodes are fresh, unaligned labels, so "is edge (i, j) of the
synthetic graph also a real edge" is meaningless (and an alignment would itself be a privacy leak). The
classifier only sees structural and attribute-match features, whose meaning transfers between graphs;
the model is trained on the synthetic graph's feature semantics and applied to real node pairs.

Leakage control: for a training positive (an edge of the training graph) the edge itself is removed from
the degrees used in its features (common neighbours and Adamic-Adar are unaffected by the edge u-v itself),
so positives are scored as if unseen. Test edges are never part of the TRTR training graph. Training
negatives may by chance coincide with a held-out test edge (about 0.2% of pairs); this only adds label
noise to the training set and is not removed, so no information about test positives is used.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.stats import rankdata

from graph_aim.eval.evaluate import adjacency, degrees
from graph_aim.transfer.types import AttributedGraph

FEATURE_NAMES = ["log_cn", "jaccard", "adamic_adar", "log_degprod", "attr_matches", "homophily_match"]
_CHUNK = 20000


# ----------------------------------------------------------------------------------------- sampling
def _keys(u: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    return lo.astype(np.int64) * n + hi


def _sample_non_edges(edges: np.ndarray, n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """``(<=k, 2)`` unique pairs u < v that are not in ``edges``."""
    if n < 2 or k <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    ek = np.sort(_keys(edges[:, 0], edges[:, 1], n)) if len(edges) else np.zeros(0, dtype=np.int64)
    total = n * (n - 1) // 2
    k = int(min(k, max(total - len(ek), 0)))
    got = np.zeros(0, dtype=np.int64)
    for _ in range(50):
        if len(got) >= k:
            break
        cand = rng.integers(0, n, size=(2 * (k - len(got)) + 16, 2))
        cand = cand[cand[:, 0] != cand[:, 1]]
        ck = np.unique(_keys(cand[:, 0], cand[:, 1], n))
        ck = ck[~np.isin(ck, ek)]
        got = np.unique(np.concatenate([got, ck]))
    got = rng.permutation(got)[:k]
    return np.stack([got // n, got % n], axis=1).astype(np.int64)


# ----------------------------------------------------------------------------------------- features
class _FeatureGraph:
    """Precomputed adjacency, degrees and attribute codes of the graph pairs are scored in."""

    def __init__(self, g: AttributedGraph, cols: list[str], hom_col: str | None):
        self.n = g.n
        self.adj = adjacency(g).tocsr()
        self.deg = degrees(g).astype(np.float64)
        w = np.zeros(g.n)
        big = self.deg > 1
        w[big] = 1.0 / np.log(self.deg[big])
        self.adj_w = self.adj.multiply(w[None, :]).tocsr()  # column j scaled by 1 / log(deg j)
        self.codes = (
            np.stack([g.attrs[c].to_numpy(dtype=np.int64) for c in cols], axis=1)
            if cols
            else np.zeros((g.n, 0), dtype=np.int64)
        )
        self.hom = g.attrs[hom_col].to_numpy(dtype=np.int64) if hom_col else None

    def features(self, pairs: np.ndarray, is_edge: bool) -> np.ndarray:
        """Feature matrix for ``pairs``. ``is_edge`` marks pairs that are edges of this graph (their own
        edge is removed from the degrees)."""
        out = np.zeros((len(pairs), len(FEATURE_NAMES)))
        for s in range(0, len(pairs), _CHUNK):
            p = pairs[s : s + _CHUNK]
            u, v = p[:, 0], p[:, 1]
            au, av = self.adj[u], self.adj[v]
            cn = np.asarray(au.multiply(av).sum(axis=1)).ravel()
            aa = np.asarray(au.multiply(self.adj_w[v]).sum(axis=1)).ravel()
            adj_off = 1.0 if is_edge else 0.0
            du, dv = self.deg[u] - adj_off, self.deg[v] - adj_off
            union = du + dv - cn
            jac = np.divide(cn, union, out=np.zeros(len(cn)), where=union > 0)
            match = (self.codes[u] == self.codes[v]).sum(axis=1) if self.codes.shape[1] else np.zeros(len(u))
            hom = (self.hom[u] == self.hom[v]).astype(float) if self.hom is not None else np.zeros(len(u))
            out[s : s + _CHUNK] = np.stack(
                [np.log1p(cn), jac, aa, np.log1p(du * dv), match, hom], axis=1
            )
        return out


def _training_set(
    fg: _FeatureGraph, edges: np.ndarray, rng: np.random.Generator, max_pos: int
) -> tuple[np.ndarray, np.ndarray]:
    """Balanced (features, labels) from a graph's own edges and sampled non-edges."""
    if len(edges) > max_pos:
        edges = edges[rng.choice(len(edges), size=max_pos, replace=False)]
    neg = _sample_non_edges(_all_edges(fg), fg.n, len(edges), rng)
    x = np.vstack([fg.features(edges, True), fg.features(neg, False)])
    y = np.concatenate([np.ones(len(edges)), np.zeros(len(neg))])
    return x, y


def _all_edges(fg: _FeatureGraph) -> np.ndarray:
    coo = sp.triu(fg.adj, k=1).tocoo()
    return np.stack([coo.row, coo.col], axis=1).astype(np.int64)


# ---------------------------------------------------------------------------------------- classifier
def _fit_logreg(x: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-12] = 1.0
    z = (x - mu) / sd
    d = z.shape[1]

    def loss(wb):
        w, b = wb[:d], wb[d]
        s = z @ w + b
        # log(1 + exp(s)) - y s, stable
        ll = np.logaddexp(0.0, s) - y * s
        p = 1.0 / (1.0 + np.exp(-np.clip(s, -500, 500)))
        r = p - y
        grad = np.concatenate([z.T @ r / len(y) + l2 * w, [r.mean()]])
        return ll.mean() + 0.5 * l2 * float(w @ w), grad

    res = minimize(loss, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": 200})
    return res.x[:d], float(res.x[d]), mu, sd


def _score(model, x: np.ndarray) -> np.ndarray:
    w, b, mu, sd = model
    return ((x - mu) / sd) @ w + b


def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ----------------------------------------------------------------------------------------- the task
def _split_edges(real: AttributedGraph, seed: int, test_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    e = np.asarray(real.edges, dtype=np.int64).reshape(-1, 2)
    perm = np.random.default_rng(seed).permutation(len(e))
    n_test = int(round(test_frac * len(e)))
    return e[perm[n_test:]], e[perm[:n_test]]


class LinkPredictionTask:
    """Real-graph side of the utility task: split, test pairs and the TRTR reference (computed once).

    Build one per real graph and reuse it for every synthetic graph.
    """

    def __init__(
        self,
        real: AttributedGraph,
        seed: int = 0,
        homophily_attrs: list[str] | None = None,
        max_train_pos: int = 50000,
    ):
        self.real, self.seed, self.max_train_pos = real, seed, max_train_pos
        self.cols = list(real.attrs.columns)
        h = [c for c in (homophily_attrs or self.cols[:1]) if c in self.cols]
        self.hom_col = h[0] if h else None
        train_e, self.test_pos = _split_edges(real, seed)
        self.train_graph = AttributedGraph(real.n, train_e, real.attrs, real.domains)
        rng = np.random.default_rng([seed, 1])
        self.test_neg = _sample_non_edges(np.asarray(real.edges, dtype=np.int64).reshape(-1, 2), real.n, len(self.test_pos), rng)
        self.test_y = np.concatenate([np.ones(len(self.test_pos)), np.zeros(len(self.test_neg))])
        self.real_fg = _FeatureGraph(self.train_graph, self.cols, self.hom_col)
        self.test_x = np.vstack([self.real_fg.features(self.test_pos, False), self.real_fg.features(self.test_neg, False)])
        x, y = _training_set(self.real_fg, train_e, np.random.default_rng([seed, 2]), max_train_pos)
        self.auc_trtr = _auc(_score(_fit_logreg(x, y), self.test_x), self.test_y)

    def evaluate(self, syn: AttributedGraph) -> dict:
        cols = [c for c in self.cols if c in syn.attrs.columns]
        hom = self.hom_col if self.hom_col in cols else None
        # attribute features of the real test graph use all real columns; restrict both to shared columns
        if cols != self.cols or hom != self.hom_col:
            fg_real = _FeatureGraph(self.train_graph, cols, hom)
            test_x = np.vstack([fg_real.features(self.test_pos, False), fg_real.features(self.test_neg, False)])
        else:
            test_x = self.test_x
        se = np.asarray(syn.edges, dtype=np.int64).reshape(-1, 2)
        if syn.n < 2 or len(se) == 0:
            auc = 0.5
        else:
            fg_syn = _FeatureGraph(syn, cols, hom)
            x, y = _training_set(fg_syn, se, np.random.default_rng([self.seed, 3]), self.max_train_pos)
            auc = _auc(_score(_fit_logreg(x, y), test_x), self.test_y)
        return {
            "auc_tstr": auc,
            "auc_trtr": self.auc_trtr,
            "auc_ratio": auc / self.auc_trtr if self.auc_trtr else float("nan"),
            "n_test_pairs": int(len(self.test_y)),
        }


def link_prediction_utility(
    real: AttributedGraph,
    syn: AttributedGraph,
    seed: int = 0,
    homophily_attrs: list[str] | None = None,
    task: LinkPredictionTask | None = None,
) -> dict:
    """AUC of a link predictor trained on ``syn`` (TSTR) and on the real train graph (TRTR), both tested on
    held-out real pairs, plus their ratio. Pass a prebuilt ``task`` to reuse the real-side computation.
    Deterministic for a fixed ``seed``."""
    task = task if task is not None else LinkPredictionTask(real, seed, homophily_attrs)
    return task.evaluate(syn)
