
from scipy.optimize import linear_sum_assignment
import numpy as np

def entity_distance_match_corners(corners1, corners2):
    """
    Computes the entity distance between two entities E and E'
    represented by their 4 corner coordinates (arrays of shape (4,3)).
    
    The distance is defined as the maximum Euclidean distance between each
    matched pair of corners, where the assignment is chosen via Hungarian matching.
    
    Parameters:
      corners1: np.array of shape (4,3)
      corners2: np.array of shape (4,3)
      
    Returns:
      d: float, the computed entity distance.
    """
    # Compute the 4x4 cost matrix of Euclidean distances.
    # cost[i,j] = distance between corners1[i] and corners2[j]
    cost = np.linalg.norm(corners1[:, None, :] - corners2[None, :, :], axis=2)
    # Solve the assignment problem.
    row_ind, col_ind = linear_sum_assignment(cost)
    # The entity distance is the maximum distance among the matched pairs.
    d = cost[row_ind, col_ind].max()
    return d


def cost_matrix_match_corners(gt_entities, pred_entities):
    """
    Compute the cost matrix for matching entities with a fixed number of corners. 
    For each pair, compute the distance (maximum distance between optimally aligned corners).
    
    Parameters:
        gt_entities: list of np.array of shape (4,3) (ground truth corners)
        pred_entities: list of np.array of shape (4,3) (predicted corners)
    Returns:
        cost_matrix: np.array of shape (n_gt, n_pred) where n_gt and n_pred are the number of ground truth and predicted entities.
    """
    n_gt = len(gt_entities)
    n_pred = len(pred_entities)
    # Build the cost matrix between every ground truth and prediction.
    cost_matrix = np.zeros((n_gt, n_pred))
    for i in range(n_gt):
        for j in range(n_pred):
            cost_matrix[i, j] = entity_distance_match_corners(gt_entities[i], pred_entities[j])
    return cost_matrix