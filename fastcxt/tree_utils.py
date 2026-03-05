"""Utilities for extracting coalescence-order features from tree sequences.

The key insight for O(n log n) scaling: instead of predicting TMRCA for all
n(n-1)/2 pairs, we predict internal node times in the tree.  A tree with n
leaves has O(n) internal nodes.  Any pairwise TMRCA is the time of the LCA
of the two leaves -- a simple O(log n) lookup per pair after node times are
predicted.

Topology encoding:
    For each local tree, we extract a *coalescence rank vector* that describes
    the merge order of samples WITHOUT any timing information.  This is the
    order in which lineages coalesce reading the tree from tips to root.

    Concretely, for an internal node at rank r (0 = first coalescence,
    n-2 = root coalescence in a binary tree), we record which sample indices
    are its children.  This is flattened into a fixed-size per-window feature
    vector.
"""

from __future__ import annotations

import numpy as np

try:
    import tskit
except ImportError:
    tskit = None


FEATS_PER_NODE = 5


def extract_coalescence_order(tree: "tskit.Tree", n_samples: int) -> np.ndarray:
    """Extract the coalescence rank vector from a single tskit Tree.

    Returns an array of shape (n_internal_nodes, FEATS_PER_NODE):
        [rank, min_leaf_left, min_leaf_right, subtree_size_left, subtree_size_right]

    Subtree sizes encode the number of descendant leaves below each child,
    which directly determines expected coalescent waiting times.
    """
    n_internal = n_samples - 1
    if tree.num_roots > 1:
        return np.zeros((n_internal, FEATS_PER_NODE), dtype=np.float32)

    internal_nodes = []
    for node in tree.nodes(order="timeasc"):
        if tree.is_leaf(node):
            continue
        children = tree.children(node)
        if len(children) < 2:
            continue
        left_min = min(tree.leaves(children[0]))
        right_min = min(tree.leaves(children[1]))
        left_size = tree.num_samples(children[0])
        right_size = tree.num_samples(children[1])
        internal_nodes.append((left_min, right_min, left_size, right_size))

    out = np.zeros((n_internal, FEATS_PER_NODE), dtype=np.float32)
    norm_n = max(n_samples - 1, 1)
    for rank, (lm, rm, ls, rs) in enumerate(internal_nodes[:n_internal]):
        out[rank] = [
            rank / max(n_internal - 1, 1),
            lm / norm_n,
            rm / norm_n,
            ls / n_samples,
            rs / n_samples,
        ]
    return out


def extract_topology_features(
    ts: "tskit.TreeSequence",
    n_windows: int = 500,
    window_size: int = 2000,
    max_internal: int = 199,
) -> np.ndarray:
    """Extract per-window tree topology features from a tree sequence.

    For each genomic window, finds the spanning tree and encodes its
    coalescence order into a fixed-size feature vector.

    Parameters
    ----------
    ts : tskit.TreeSequence
    n_windows : number of output windows
    window_size : base pair width of each window
    max_internal : max number of internal nodes to encode (= max_samples - 1)

    Returns
    -------
    features : (n_windows, max_internal * FEATS_PER_NODE)  float32
    """
    feat_dim = max_internal * FEATS_PER_NODE
    features = np.zeros((n_windows, feat_dim), dtype=np.float32)
    n_samples = ts.num_samples

    tree_iter = ts.trees()
    current_tree = next(tree_iter)

    for w in range(n_windows):
        w_mid = w * window_size + window_size // 2
        while current_tree.interval.right <= w_mid:
            try:
                current_tree = next(tree_iter)
            except StopIteration:
                break

        coal_order = extract_coalescence_order(current_tree, n_samples)
        n_nodes = min(coal_order.shape[0], max_internal)
        features[w, :n_nodes * FEATS_PER_NODE] = coal_order[:n_nodes].ravel()

    return features


def extract_node_times(
    ts: "tskit.TreeSequence",
    n_windows: int = 500,
    window_size: int = 2000,
    max_internal: int = 199,
) -> np.ndarray:
    """Extract log(time) of each internal node per window.

    Internal nodes are ordered by increasing time (same ordering as
    ``extract_coalescence_order``).

    Returns
    -------
    node_times : (n_windows, max_internal) float32
    """
    times = np.zeros((n_windows, max_internal), dtype=np.float32)
    tree_iter = ts.trees()
    current_tree = next(tree_iter)

    for w in range(n_windows):
        w_mid = w * window_size + window_size // 2
        while current_tree.interval.right <= w_mid:
            try:
                current_tree = next(tree_iter)
            except StopIteration:
                break

        rank = 0
        for node in current_tree.nodes(order="timeasc"):
            if current_tree.is_leaf(node):
                continue
            if rank >= max_internal:
                break
            times[w, rank] = np.log(max(current_tree.time(node), 1e-10))
            rank += 1

    return times


def lca_lookup(
    ts: "tskit.TreeSequence",
    node_times: np.ndarray,
    sample_a: int,
    sample_b: int,
    n_windows: int = 500,
    window_size: int = 2000,
) -> np.ndarray:
    """Look up pairwise TMRCA from predicted node times via LCA.

    For each window, find the LCA of (sample_a, sample_b) in the local tree
    and return the predicted time for that internal node.

    Parameters
    ----------
    node_times : (n_windows, n_internal) predicted log-times per node
    """
    tmrcas = np.zeros(n_windows, dtype=np.float32)
    tree_iter = ts.trees()
    current_tree = next(tree_iter)

    for w in range(n_windows):
        w_mid = w * window_size + window_size // 2
        while current_tree.interval.right <= w_mid:
            try:
                current_tree = next(tree_iter)
            except StopIteration:
                break
        mrca_node = current_tree.mrca(sample_a, sample_b)
        if mrca_node != tskit.NULL:
            node_idx = _node_to_internal_index(current_tree, mrca_node)
            if node_idx is not None and node_idx < node_times.shape[1]:
                tmrcas[w] = node_times[w, node_idx]

    return tmrcas


def lca_lookup_batch(
    ts: "tskit.TreeSequence",
    node_times: np.ndarray,
    pairs: list[tuple[int, int]],
    n_windows: int = 500,
    window_size: int = 2000,
) -> np.ndarray:
    """Look up TMRCA for multiple pairs in one tree-sequence pass.

    More efficient than calling lca_lookup per pair because we iterate
    through windows only once and resolve all pairs at each position.

    Returns
    -------
    tmrcas : (n_pairs, n_windows) float32
    """
    n_pairs = len(pairs)
    tmrcas = np.zeros((n_pairs, n_windows), dtype=np.float32)
    tree_iter = ts.trees()
    current_tree = next(tree_iter)

    for w in range(n_windows):
        w_mid = w * window_size + window_size // 2
        while current_tree.interval.right <= w_mid:
            try:
                current_tree = next(tree_iter)
            except StopIteration:
                break

        # Cache node_id -> rank mapping for this tree
        rank_map = {}
        rank = 0
        for node in current_tree.nodes(order="timeasc"):
            if current_tree.is_leaf(node):
                continue
            rank_map[node] = rank
            rank += 1

        for p_idx, (sa, sb) in enumerate(pairs):
            mrca_node = current_tree.mrca(sa, sb)
            if mrca_node != tskit.NULL:
                node_idx = rank_map.get(mrca_node)
                if node_idx is not None and node_idx < node_times.shape[1]:
                    tmrcas[p_idx, w] = node_times[w, node_idx]

    return tmrcas


def _node_to_internal_index(tree: "tskit.Tree", node: int) -> int | None:
    """Map a tree-sequence node ID to the rank among internal nodes (time-ordered)."""
    rank = 0
    for n in tree.nodes(order="timeasc"):
        if tree.is_leaf(n):
            continue
        if n == node:
            return rank
        rank += 1
    return None
