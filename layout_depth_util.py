import os
import PIL
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import pyrender
import trimesh
import json

def nerfstudio_transforms_to_poses(transforms_file):
    """
    Reads camera poses / intrinsics from nerfstudio transforms json file
    Args:
        transforms_file (str): Path to the nerfstudio transforms json file.
    Returns:
        list: List of dictionaries containing the camera poses and intrinsics.
    """
    with open(transforms_file, "r") as f:
        transforms = json.load(f)
    poses = []
    for transform in transforms["frames"]:
        K = np.array([[transform["fl_x"], 0, transform["cx"]],
                        [0, transform["fl_y"], transform["cy"]],
                        [0, 0, 1]])
        c2w = np.array(transform["transform_matrix"])

        width = transform["w"]
        height = transform["h"]
        poses.append({
            "c2w": c2w,
            "K": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
            "width": width,
            "height": height,
        })
    return poses


def render_images(mesh_path, pose_info, resize_to=None):
    """
    Renders RGB and depth images from a mesh using the provided camera poses.
    Args:
        mesh_path (str): Path to the mesh file.
        pose_info (list): List of dictionaries containing the camera poses and intrinsics.
        resize_to (tuple): Tuple of (width, height) to resize the images to.
    Returns:
        tuple: Tuple containing the rendered RGB images, depth images, masks, and paths.
    """
    # Load the mesh
    mesh = trimesh.load(str(mesh_path))
    
    if isinstance(mesh, trimesh.Trimesh):
        trimeshScene = trimesh.Scene()
        trimeshScene.add_geometry(mesh)
    else:
        trimeshScene = mesh

    W, H = pose_info[0]["width"], pose_info[0]["height"]

    assert all(pose["width"] == W and pose["height"] == H for pose in pose_info), "All poses must have the same width and height"

    
    # Create a renderer
    scene_pyrender = pyrender.Scene.from_trimesh_scene(trimeshScene, bg_color=[1.0, 1.0, 1.0])
    ambient_intensity = 0.8
    scene_pyrender.ambient_light = np.array([ambient_intensity, ambient_intensity, ambient_intensity])
    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)

    
    # Define camera intrinsics
    z_near = 0.05
    z_far = 20.0


    images = []
    depths = []
    paths = []
    masks = []
    # for image_id, image in tqdm(images_paths.items(), f"Rendering depths using the predicted model"):
    for pose in tqdm(pose_info, f"Rendering depths using {mesh_path}"):

        c2w = pose["c2w"]
        fx, fy, cx, cy = pose["K"]
        camera_pyrender = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=z_near, zfar=z_far)
        added_node = scene_pyrender.add(camera_pyrender, pose=c2w)
        scene_pyrender.main_camera_node = added_node
        rgb, depth = renderer.render(scene_pyrender)

        rgb = rgb.astype(np.uint8)
        # Make depth in mm and clip to fit 16-bit image
        depth = (depth.astype(np.float32) * 1000).clip(0, 65535).astype(np.uint16)

        if resize_to is not None:
            rgb = np.asarray(PIL.Image.fromarray(rgb).resize(tuple(resize_to), PIL.Image.BILINEAR))
            depth = np.asarray(PIL.Image.fromarray(depth).resize(tuple(resize_to), PIL.Image.NEAREST))


        depth_mask = (depth > 0) & (depth != 65535)
        
        images.append(rgb)
        depths.append(depth)
        paths.append(pose_info[4])
        masks.append(depth_mask)

    return images, depths, masks, paths


def compute_depth_error(depth_gt, depth_pred, mask_pred, mask_gt, success_ratio_percents=[0.1, 1, 5, 10]):
    """
    Computes the absolute and relative depth error between the predicted and ground truth depth maps.
    Args:
        depth_gt (torch.Tensor): Ground truth depth map.
        depth_pred (torch.Tensor): Predicted depth map.
        mask_pred (torch.Tensor): Mask for the predicted depth map.
        mask_gt (torch.Tensor): Mask for the ground truth depth map.
        success_ratio_percents (list): List of success ratio thresholds in percentage.
    Returns:
        tuple: Tuple containing the absolute error, relative error, success ratios, mask, and per-pixel absolute error.
    """
    depth_gt = depth_gt.int()
    depth_pred = depth_pred.int()
    per_pixel_abs_error = (depth_gt - depth_pred).abs()
    mask = mask_pred & mask_gt
    invalid_pred_but_has_gt = (mask_gt & ~mask_pred)
    per_pixel_abs_error[~mask] = 0
    depth_gt = depth_gt[mask]
    depth_pred = depth_pred[mask]
    abs_diff = (depth_gt - depth_pred).abs()
    rel_diff = (depth_gt - depth_pred).abs() / depth_gt
    success_ratios = torch.stack([torch.concatenate([(abs_diff < p), torch.zeros(invalid_pred_but_has_gt.sum().int(), dtype=bool, device=mask.device)], dim=0) for p in success_ratio_percents])

    return abs_diff, rel_diff, success_ratios, mask, per_pixel_abs_error


def compute_scene_error(pred_mesh_path, gt_mesh_path, poses, device="cpu"):
    """
    Computes the reprojection error of the predicted mesh against the ground truth mesh using the provided poses.
    Args:
        pred_mesh_path (str): Path to the predicted mesh.
        gt_mesh_path (str): Path to the ground truth mesh.
        poses (list): List of dictionaries containing the camera poses and intrinsics in nerfstudio format.
        device (str): Device to use for computation (e.g., "cpu" or "cuda").
    Returns:
        dict: Dictionary containing the mean absolute error, mean relative error, and success ratios.
    """
    
    success_ratio_thresholds_cm = [1, 5, 10]
    success_ratio_thresholds_mm = [p * 10 for p in success_ratio_thresholds_cm]

    
    # shape = np.asarray(depth_path_to_tensor(poses[0]["depth_path"], device)[0].cpu().numpy().shape) // 2
    shape = poses[0]["height"], poses[0]["width"]
    shape = np.array((shape[0], shape[1]), dtype=np.int32) // 2
    _, pred_depths, pred_depth_masks, _ = render_images(pred_mesh_path, poses, resize_to=shape)
    _, gt_depths, gt_depth_masks, _ = render_images(gt_mesh_path, poses, resize_to=shape)
    
    
    abs_errors = []
    rel_errors = []
    success_ratios = []
    n_pixels = []
    n_pixels_all = []

    for i, pose in enumerate(tqdm(poses, f"Computing errors")):
        # load the rendered GT depth
        gt_depth = torch.from_numpy(gt_depths[i].astype(float)).to(device)
        gt_depth_mask = torch.from_numpy(gt_depth_masks[i]).to(device)

        pred_depth = torch.from_numpy(pred_depths[i].astype(float)).to(device)
        pred_depth_mask = torch.from_numpy(pred_depth_masks[i]).to(device)

        # compute the per-image statistics
        abs_err_mm, rel_err, success_ratio, final_mask, _ = compute_depth_error(gt_depth, pred_depth, mask_gt=gt_depth_mask, mask_pred=pred_depth_mask, success_ratio_percents=success_ratio_thresholds_mm)
        abs_err_cm = abs_err_mm / 10.0


        # collect
        abs_errors.append(abs_err_cm.mean().item() if final_mask.sum() > 0 else 0)
        rel_errors.append(rel_err.mean().item() if final_mask.sum() > 0 else 0)
        success_ratios.append(((1.0 *success_ratio).mean(axis=1).cpu().numpy() if final_mask.sum() > 0 else [0] * len(success_ratio)))
        n_pixels.append( final_mask.sum().item())
        n_pixels_all.append(final_mask.numel())

    
    # weighted mean
    mean_abs_error = np.average(abs_errors, weights=n_pixels)
    mean_rel_error = np.average(rel_errors, weights=n_pixels)
    success_ratios = np.stack(success_ratios)
    mean_success_ratio = [np.average(success_ratios[:, i], weights=n_pixels) for i in range(len(success_ratio_thresholds_cm))]
    
    errors = {
        "mean_abs_error": mean_abs_error,
        "mean_rel_error": mean_rel_error,
        "n_pixels_wall": sum(n_pixels),
        "n_pixels_total": sum(n_pixels_all),
        "wall_percentage": sum(n_pixels) / (sum(n_pixels_all) + 1e-16)
    }
    for p, error in zip(success_ratio_thresholds_cm, mean_success_ratio):
        errors[f"success_ratio_{p}"] = error

    return errors




def eval_layout_depth(scenes, prediction_dir, dataset_root, device="cuda"):
    """
    Evaluates the depth error of the predicted meshes against the ground truth meshes for a set of scenes.
    Args:
        prediction_dir (str): Directory containing the predicted meshes.
        dataset_root (str): Root directory of the dataset.
        device (str): Device to use for computation (e.g., "cpu" or "cuda").
    Returns:
        pd.DataFrame: DataFrame containing the evaluation results for each scene.
    """
    errors = {}
    for scene_id in scenes:
        pose_info = nerfstudio_transforms_to_poses(f"{dataset_root}/poses/{scene_id}.json")
        prediction_path = f"{prediction_dir}/{scene_id}.ply"
        gt_path = f"{dataset_root}/structures/{scene_id}.obj"
                
        errors[scene_id] = compute_scene_error(prediction_path, gt_path, pose_info, device=device)

    errors_df = pd.DataFrame(errors).T
    return errors_df



if __name__ == "__main__":
    from scenes import scenes
    errors = eval_layout_depth(scenes, "/mnt/usb_ssd/bieriv/tmp/ours-oneformer-reclassify/tmp_extruded_meshes/", "/mnt/usb_ssd/bieriv/ANNOTATIONS_OURS/", "cuda")
    print(errors)