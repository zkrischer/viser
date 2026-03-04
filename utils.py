import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull

def get_boundary_points(coords, labels, k=16):
    """
    Find boundary points for each class based on neighborhood differences.

    Args:
        coords (np.ndarray or torch.Tensor): (N, D) point coordinates.
        labels (np.ndarray or torch.Tensor): (N,) class labels.
        k (int): number of neighbors to check for boundary detection.

    Returns:
        dict: {class_id: np.ndarray of boundary points}
    """
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm="kd_tree").fit(coords)
    distances, neighbors = nbrs.kneighbors(coords)

    boundary_points = {int(c): [] for c in np.unique(labels)}

    for i in range(coords.shape[0]):
        point_class = labels[i]
        neighbor_classes = labels[neighbors[i, 1:]]  # skip self
        if np.any(neighbor_classes != point_class):
            boundary_points[int(point_class)].append(coords[i])

    # Convert lists to numpy arrays
    boundary_points = {int(c): np.array(points, dtype=np.float64) for c, points in boundary_points.items()}
    return boundary_points


def get_extremity_points(coords, labels):
    """
    Find extremity points (geometric edges) for each class using convex hull.

    Args:
        coords (np.ndarray or torch.Tensor): (N, D) point coordinates (D >= 2).
        labels (np.ndarray or torch.Tensor): (N,) class labels.

    Returns:
        dict: {class_id: np.ndarray of extremity points}
    """
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    extremity_points = {}
    for c in np.unique(labels):
        class_coords = coords[labels == c]
        if class_coords.shape[0] < coords.shape[1] + 1:
            # Not enough points for a hull
            extremity_points[int(c)] = class_coords
            continue

        try:
            hull = ConvexHull(class_coords)
            hull_points = class_coords[hull.vertices]
            extremity_points[int(c)] = hull_points
        except Exception:
            # In case hull computation fails (e.g. collinear points)
            extremity_points[int(c)] = class_coords

    return extremity_points