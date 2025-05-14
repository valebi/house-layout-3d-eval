import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import os

import tqdm
import open3d as o3d
from hausdorff_distance import cost_matrix_hausdorff_distance
from corner_matching_distance import cost_matrix_match_corners

from scenes import scenes
from glob import glob
import argparse



def compute_assigment_f1_for_class(gt_entities, pred_entities, cost_matrix, threshold):
    """
    Compute an optimal assignment F1 score for a single class of entities.
    The F1 score is computed based on the number of true positives (TP), false positives (FP),
    and false negatives (FN) using the Hungarian algorithm to find the optimal assignment.
    
    Parameters:
        gt_entities: list of np.array of shape (4,3) (ground truth corners)
        pred_entities: list of np.array of shape (4,3) (predicted corners)
        cost_matrix: np.array of shape (n_gt, n_pred) where n_gt and n_pred are the number of ground truth and predicted entities.
        threshold: float, the distance threshold for considering a match.
      
    Returns:
      f1: float, the F1 score.
    """
    n_gt = len(gt_entities)
    n_pred = len(pred_entities)

    if n_gt == 0 and n_pred == 0:
        return 1.0  # perfect score: both empty
    if n_gt == 0:
        return 0.0
    if n_pred == 0:
        return 0.0
    
    
    # Solve the assignment problem (note: cost_matrix need not be square).
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Count true positives: only those assignments where the cost is below threshold.
    tp = 0
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < threshold:
            tp += 1
    fp = n_pred - tp
    fn = n_gt - tp
    
    # Compute F1 score.
    if (2*tp + fp + fn) == 0:
        return 1.0
    f1 = 2 * tp / (2*tp + fp + fn)
    return f1



def evaluate_scene(gt_scene, pred_scene, thresholds):
    """
    Evaluate the layout estimation for a single scene.
    
    Parameters:
      gt_scene: list of entities (dict with keys "class" and "corners")
      pred_scene: list of entities (same format as gt_scene)
      thresholds: list or array of float thresholds at which to evaluate F1
      
    Returns:
      f1_by_threshold: dict mapping each threshold to the average F1 score over classes for this scene.
      f1_per_class: dict mapping each class to a list of F1 scores (one per threshold)
    """
    # Get the set of classes present in either ground truth or predictions.
    classes = set([e["class"] for e in gt_scene] + [e["class"] for e in pred_scene])
    
    f1_by_threshold = {thr: [] for thr in thresholds}
    f1_per_class = {}
    
    for cs in classes:
        # Extract entities of this class.
        gt_cls = [e["corners"] for e in gt_scene if e["class"] == cs]
        pred_cls = [e["corners"] for e in pred_scene if e["class"] == cs]
        
        if cs not in ["planes", "stairs"]:
            # assume all entities have the same number of corners (4), and match first the corners then the entities
            cost_matrix = cost_matrix_match_corners(gt_cls, pred_cls)
        else:
            # for variable number of corners, use maximal point-to-poylgon distance (Hausdorff distance)
            cost_matrix = cost_matrix_hausdorff_distance(gt_cls, pred_cls)
            
        f1_scores = []
        for thr in thresholds:
            f1 = compute_assigment_f1_for_class(gt_cls, pred_cls, cost_matrix, thr)
            f1_scores.append(f1)
            f1_by_threshold[thr].append(f1)
        f1_per_class[cs] = f1_scores
        
    # Average F1 score per threshold for each class..
    avg_f1_by_threshold = {thr: np.mean(f1_by_threshold[thr]) for thr in thresholds}
    # Also compute the average F1 score across all thresholds and classes.
    overall_avg_f1 = np.mean(list(avg_f1_by_threshold.values()))
    
    return avg_f1_by_threshold, overall_avg_f1, f1_per_class

def evaluate_dataset(gt_dataset, pred_dataset, thresholds):
    """
    Evaluate the layout estimation accuracy over a dataset.
    
    Parameters:
      gt_dataset: dict mapping scene id to list of entities (each entity is a dict with "class" and "corners")
      pred_dataset: dict with the same structure as gt_dataset.
      thresholds: list or array of thresholds at which to evaluate F1.
      
    Returns:
      avg_f1_by_threshold: dict mapping each threshold to the average F1 score over all scenes.
      overall_avg_f1: float, the average of avg_f1_by_threshold over thresholds.
      scene_f1: dict mapping scene id to its per-threshold F1 scores.
    """
    scene_f1 = {}
    all_f1_by_threshold = {thr: [] for thr in thresholds}
    
    for scene_id in gt_dataset.keys():
        gt_scene = gt_dataset[scene_id]
        pred_scene = pred_dataset.get(scene_id, [])
        if len(gt_scene) == 0 and len(pred_scene) == 0:
            avg_f1_by_thr = {thr: 1.0 for thr in thresholds}
        else:
            avg_f1_by_thr, _, _ = evaluate_scene(gt_scene, pred_scene, thresholds)
        scene_f1[scene_id] = avg_f1_by_thr
        for thr in thresholds:
            all_f1_by_threshold[thr].append(avg_f1_by_thr[thr])
    
    avg_f1_by_threshold = {thr: np.mean(all_f1_by_threshold[thr]) for thr in thresholds}
    overall_avg_f1 = np.mean(list(avg_f1_by_threshold.values()))
    return avg_f1_by_threshold, overall_avg_f1, scene_f1



def load_datasets(base_pred_path, base_annots_path, task):
    """
    Load the ground truth and predicted annotations for the specified task.
    Assumes the predictions are stored in base_pred_path/scene_id

    Parameters:
        base_pred_path: str, path to the directory containing the predicted annotations.
        base_annots_path: str, path to the directory containing the ground truth annotations.
        task: str, the task to evaluate (e.g., "windows", "doors", "planes", "stairs").
    
    Returns:
        gt_dataset: dict, mapping scene id to list of ground truth entities (each entity is a dict with "class" and "corners").
        pred_dataset: dict, mapping scene id to list of predicted entities (same format as gt_dataset).
    """
    pred_dataset = {}
    gt_dataset = {}
    n_gt = 0
    for scene in tqdm.tqdm(scenes, desc=f"Collecting scenes"):
        # Load the ground truth and predicted annotations for each scene
        if task in ["windows", "doors"]:
            # parse json gt and predictions (fixed format, four corners)
            gt_path = os.path.join(base_annots_path, task, f"{scene}.json")
            pred_path = os.path.join(base_pred_path, f"{scene}.json")
            with open(gt_path, "r") as f:
                gt_dataset[scene] = json.load(f)
            with open(pred_path, "r") as f:
                pred_dataset[scene] = json.load(f)

            n_gt += len(gt_dataset[scene][task])
            print(f"Loaded {len(gt_dataset[scene][task])} {task} annotations vs. {len(pred_dataset[scene][task])} predictions for scene {scene}")
            gt_dataset[scene] = [{"class": task, "corners" : np.array(d["vertices"])} for d in gt_dataset[scene][task]]
            pred_dataset[scene] = [{"class":  task, "corners" : np.array(d["vertices"])} for d in pred_dataset[scene][task]]
        
        elif task == "planes":
            # parse individual meshes (arbitrary number of vertices / triangles)
            compoments_dir = os.path.join(base_annots_path, "structures", "layouts_split_by_entity", scene)
            pred_path = os.path.join(base_pred_path, scene)
            plane_components_gt = [o3d.io.read_triangle_mesh(f) for f in glob(os.path.join(compoments_dir, "*.ply"))]
            plane_components_pred = [o3d.io.read_triangle_mesh(f) for f in glob(os.path.join(pred_path, "*.ply"))]
            
            plane_components_gt = [p.remove_degenerate_triangles().remove_duplicated_vertices().remove_duplicated_triangles() for p in plane_components_gt]
            plane_components_pred = [p.remove_degenerate_triangles().remove_duplicated_vertices().remove_duplicated_triangles() for p in plane_components_pred]
            
            n_gt += len(plane_components_gt)
            print(f"Loaded {len(plane_components_gt)} ground truth planes and {len(plane_components_pred)} predicted planes for scene {scene}")

            min_area = 0.7 # don't consider very small planes (stair steps, doorframes)
            gt_dataset[scene] = [{"class": "planes", "corners": plane} for plane in plane_components_gt if len(plane.vertices) > 2 and len(plane.triangles) > 1 and not np.isnan(np.sum(plane.vertices)) if plane.get_surface_area() > min_area]
            pred_dataset[scene] = [{"class": "planes", "corners": plane} for plane in plane_components_pred if len(plane.vertices) > 2 and len(plane.triangles) > 1 and not np.isnan(np.sum(plane.vertices)) if plane.get_surface_area() > min_area]
        
        elif task == "stairs":
            # parse stair annotation meshes (arbitrary number of vertices / triangles, not necessarily planar)
            compoments_dir = os.path.join(base_annots_path, "stairs", scene)
            pred_dir = os.path.join(base_pred_path, scene)
            plane_components_gt = [o3d.io.read_triangle_mesh(f) for f in glob(os.path.join(compoments_dir, "stairs_*.ply"))]
            plane_components_pred = [o3d.io.read_triangle_mesh(f) for f in glob(os.path.join(pred_dir, "stair_*.ply"))]
            
            print(f"Loaded {len(plane_components_gt)} ground truth stairs and {len(plane_components_pred)} predicted stairs for scene {scene}")
            
            gt_dataset[scene] = [{"class": "stairs", "corners": plane} for plane in plane_components_gt]
            pred_dataset[scene] = [{"class": "stairs", "corners": plane} for plane in plane_components_pred]
            n_gt += len(gt_dataset[scene])
        else:
            raise ValueError(f"Unknown task: {task}")
        
    print(f"Total number of ground truth entities: {n_gt}")
    return gt_dataset, pred_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate scene layout estimation.")
    parser.add_argument("--task", type=str, default=rf"doors", 
                        help="Which task to evaluate. Options: depth, doors, windows, planes, stairs. Depth stands for layout depth, planes for [walls, ceilings, floors].")
    parser.add_argument("--pred-path", type=str,
                        help="Path to the predictions directory.")
    parser.add_argument("--dataset-root", type=str,
                        help="Path to the base annotations directory.")
    args = parser.parse_args()

    base_pred_path = args.pred_path
    base_annots_path = args.dataset_root
    task = args.task


    if task == "depth":
        # evaluate layout depth (render ground truth and predicted meshes, compute depth error)
        import layout_depth_util
        errors = layout_depth_util.eval_layout_depth(scenes, base_pred_path, base_annots_path)
        print(errors)
        mean = errors.mean(axis=0)
        print(mean)
    else:
        # Compute entity distances for the specified entity type.
        if task == "windows":
            scenes = [s for s in scenes if s not in ["jtcxE69GiFV", "e9zR4mvMWw7", "1LXtFkjw3qL", "5LpN3gDmAk7"]]
            # print(scenes)
            
        # Define a range of thresholds (for example, from 0.1 to 1.0 meters).
        thresholds = np.concatenate([np.linspace(0.1, 0.6, 6), np.linspace(0.6, 1, 3)])

        # Load the ground truth and predicted annotations.
        gt_dataset, pred_dataset = load_datasets(base_pred_path, base_annots_path, task)

        # Evaluate the layout estimation for the entire dataset.
        avg_f1_by_threshold, overall_avg_f1, scene_f1 = evaluate_dataset(gt_dataset, pred_dataset, thresholds)           


        df = pd.DataFrame(scene_f1)
        
        df = df.mean(axis=1) # average across scenes
        df.index = [f"Threshold={thr:.2f}" for thr, score in avg_f1_by_threshold.items()]
        
        print(df)
        print("avg f1:")
        print(df.mean(axis=0))
    




