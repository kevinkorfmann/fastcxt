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


def extract_coalescence_order(tree: "tskit.Tree", n_samples: int) -> np.ndarray:
    """Extract the coalescence rank vector from a single tskit Tree.

    Returns an array of shape (n_internal_nodes, 3):
        [rank, left_child_sample_set_hash, right_child_sample_set_hash]

    For simplicity, we encode each internal node as:
        (rank, min_sample_below_left, min_sample_below_right)
    where rank is assigned bottom-up (earliest coalescence = 0).
    """
    if tree.num_roots > 1:
        return np.zeros((n_samples - 1, 3), dtype=np.float32)

    internal_nodes = []
    for node in tree.nodes(order="timeasc"):
        if tree.is_leaf(node):
            continue
        children = tree.children(node)
        if len(children) < 2:
            continue
        left_min = min(tree.leaves(children[0]))
        right_min = min(tree.leaves(children[1]))
        internal_nodes.append((left_min, right_min))

    n_internal = n_samples - 1
    out = np.zeros((n_internal, 3), dtype=np.float32)
    for rank, (lm, rm) in enumerate(internal_nodes[:n_internal]):
        out[rank] = [rank / max(n_internal - 1, 1), lm / max(n_samples - 1, 1),
                     rm / max(n_samples - 1, 1)]
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
    features : (n_windows, max_internal * 3)  float32
    """
    feat_dim = max_internal * 3
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
        features[w, :n_nodes * 3] = coal_order[:n_nodes].ravel()

    return features


def node_times_from_pair_predictions(
    ts: "tskit.TreeSequence",
    pair_tmrcas: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    """Recover per-window internal-node times from pairwise TMRCA predictions.

    Given predictions for a subset of pairs, assign times to each internal
    node by averaging the TMRCA of any pair whose LCA is that node.

    Returns
    -------
    node_times : (n_internal_nodes, n_windows) array of predicted node times.
    """
    raise NotImplementedError("Placeholder for tree-aware node-time inference")


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

    This is O(n_windows * log(n)) per pair, replacing a full model forward pass.
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
            if node_idx is not None and node_idx < node_times.shape[0]:
                tmrcas[w] = node_times[node_idx, w]

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
