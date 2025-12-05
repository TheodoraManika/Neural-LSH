from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(
    X: np.ndarray, k: int = 10, metric: str = "euclidean"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Κατασκευή k-NN γράφου για το σύνολο σημείων X.

    Επιστρέφει:
    - rows: indices πηγών (i)
    - cols: indices προορισμών (j)
    - weights: βάρη ακμών (float, αρχικά 1.0 – θα τροποποιηθούν αργότερα)
    """
    n, d = X.shape
    # χρησιμοποιούμε sklearn NearestNeighbors (brute force ή auto)
    nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="auto")
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)

    # Αφαιρούμε την ίδια τη κορυφή αν εμφανίζεται (συνήθως indices[:, 0] == i)
    rows = []
    cols = []
    weights = []
    for i in range(n):
        for j_idx, dist in zip(indices[i], distances[i]):
            if j_idx == i:
                continue
            rows.append(i)
            cols.append(j_idx)
            # αρχικό βάρος 1.0 – δεν βάζουμε 1/dist, γιατί θα χρησιμοποιήσουμε
            # απλά 1 ή 2 για KaHIP (ανάλογα με αμοιβαίους/μονόπλευρους)
            weights.append(1.0)

    return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64), np.array(
        weights, dtype=np.float32
    )


def symmetrize_and_weight(
    rows: np.ndarray, cols: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Μετατρέπει τον κατευθυνόμενο k-NN γράφο σε μη κατευθυνόμενο με βάρη:
    - 2 για αμοιβαίες ακμές (i->j και j->i)
    - 1 για μονόπλευρες.

    Επιστρέφει (rows_u, cols_u, weights_u) για μη κατευθυνόμενο γράφο.
    """
    assert rows.shape == cols.shape
    # Κλειδιά ακμών (min(i,j), max(i,j))
    edge_dict = {}
    for i, j in zip(rows, cols):
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        edge_dict.setdefault((a, b), 0)
        edge_dict[(a, b)] += 1

    und_rows = []
    und_cols = []
    und_w = []
    for (a, b), cnt in edge_dict.items():
        w = 2 if cnt > 1 else 1
        und_rows.append(a)
        und_cols.append(b)
        und_w.append(w)

    return (
        np.array(und_rows, dtype=np.int64),
        np.array(und_cols, dtype=np.int64),
        np.array(und_w, dtype=np.int32),
    )


def to_csr(
    n: int, rows: np.ndarray, cols: np.ndarray, weights: np.ndarray
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Μετατροπή μη κατευθυνόμενου γράφου σε CSR format (KaHIP style).

    Επιστρέφει τις λίστες:
    - vwgt: βάρη κορυφών (εδώ όλα 1)
    - xadj: δείκτες αρχής των γειτόνων για κάθε κορυφή (μήκος n+1)
    - adjncy: λίστα γειτόνων (indices)
    - adjcwgt: αντίστοιχα βάρη ακμών
    """
    vwgt = [1] * n
    adjacency = [[] for _ in range(n)]
    adjw = [[] for _ in range(n)]

    for i, j, w in zip(rows, cols, weights):
        adjacency[i].append(j)
        adjacency[j].append(i)
        adjw[i].append(int(w))
        adjw[j].append(int(w))

    xadj = [0]
    adjncy: list[int] = []
    adjcwgt: list[int] = []
    for i in range(n):
        adjncy.extend(adjacency[i])
        adjcwgt.extend(adjw[i])
        xadj.append(len(adjncy))

    return vwgt, xadj, adjncy, adjcwgt


def save_knn_graph(
    save_path: str | Path,
    rows: np.ndarray,
    cols: np.ndarray,
    und_rows: np.ndarray,
    und_cols: np.ndarray,
    und_w: np.ndarray,
    k: int,
    metric: str = "euclidean",
) -> None:
    """
    Save k-NN graph components to disk.

    Args:
        save_path: directory path where graph will be saved
        rows, cols: directed graph edges
        und_rows, und_cols, und_w: undirected graph edges with weights
        k: k value used for graph
        metric: distance metric used
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save as numpy arrays
    np.save(save_path / "rows.npy", rows)
    np.save(save_path / "cols.npy", cols)
    np.save(save_path / "und_rows.npy", und_rows)
    np.save(save_path / "und_cols.npy", und_cols)
    np.save(save_path / "und_weights.npy", und_w)

    # Save metadata
    meta = {
        "k": int(k),
        "metric": metric,
        "num_edges_directed": int(len(rows)),
        "num_edges_undirected": int(len(und_rows)),
    }
    with open(save_path / "graph_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_knn_graph(load_path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load k-NN graph components from disk.

    Args:
        load_path: directory path where graph is stored

    Returns:
        (rows, cols, und_rows, und_cols, und_w, meta)
    """
    load_path = Path(load_path)

    rows = np.load(load_path / "rows.npy")
    cols = np.load(load_path / "cols.npy")
    und_rows = np.load(load_path / "und_rows.npy")
    und_cols = np.load(load_path / "und_cols.npy")
    und_w = np.load(load_path / "und_weights.npy")

    with open(load_path / "graph_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    return rows, cols, und_rows, und_cols, und_w, meta


__all__ = [
    "build_knn_graph",
    "symmetrize_and_weight",
    "to_csr",
    "save_knn_graph",
    "load_knn_graph",
]


