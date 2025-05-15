import open3d as o3d
import numpy as np
import json
import os
import argparse


def get_rotation_matrix_from_vectors(a, b):
    """
    Computes the rotation matrix that rotates vector a to vector b.
    Both a and b are expected to be non-zero and preferably normalized.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # If vectors are nearly opposite, choose an arbitrary perpendicular axis.
    if c < -0.999999:
        axis = np.cross(a, np.array([1, 0, 0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0, 1, 0]))
        axis = axis / np.linalg.norm(axis)
        return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.pi)
    s = np.linalg.norm(v)
    vx = np.array([[    0, -v[2],  v[1]],
                   [ v[2],     0, -v[0]],
                   [-v[1],  v[0],     0]])
    R = np.eye(3) + vx + vx.dot(vx) * ((1 - c) / (s**2))
    return R


def visualize_layout_with_annotations(layout_file, door_dict, name="doors"):
    """
    Loads the original layout and overlays the door/window annotations.
    Each door is drawn as a closed green rectangle (using its sorted vertices).
    
    Parameters:
        layout_file (str): Path to the original layout OBJ file.
        door_dict (dict): Door annotation dictionary (as produced by annotate_doors_simple).
    """
    layout_mesh = o3d.io.read_triangle_mesh(layout_file)
    layout_mesh.compute_vertex_normals()
    geometries = [layout_mesh]
    
    for door in door_dict.get(name, []):
        door_vertices = door["vertices"]
        # Create a closed loop from the door vertices.
        pts = [np.array(v) for v in door_vertices]
        pts.append(pts[0])  # Close the loop
        pts_np = np.array(pts)
        # Create line connections between consecutive vertices.
        lines = [[i, i+1] for i in range(len(pts)-1)]
        if name == "windows":
            colors = [[1, 0, 0] for _ in lines]  # red for windows
        else:
            colors = [[0, 1, 0] for _ in lines] # green for doors

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts_np),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(line_set)


        if name != "windows":
            # Compute the door center.
            door_center = np.mean(np.array(door_vertices), axis=0)
            
            # Determine arrow size based on door width (distance between first two vertices).
            edge_length = np.linalg.norm(np.array(door_vertices[0]) - np.array(door_vertices[1]))
            arrow_length = edge_length * 0.5
            
            # Create the arrow mesh.
            arrow_mesh = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=arrow_length * 0.05,
                cone_radius=arrow_length * 0.1,
                cylinder_height=arrow_length * 0.8,
                cone_height=arrow_length * 0.2
            )
            arrow_mesh.paint_uniform_color([0, 1, 0])  # green arrow
            
            # Rotate the arrow so that its default direction (+X) aligns with the door's normal.
            default_dir = np.array([0, 0, 1])
            door_normal = np.array(door["normal"])
            door_normal = door_normal / np.linalg.norm(door_normal)  # Ensure normalization.
            R = get_rotation_matrix_from_vectors(default_dir, door_normal)
            arrow_mesh.rotate(R, center=np.zeros(3))
            
            # Position the arrow at the door center.
            arrow_mesh.translate(door_center)
            geometries.append(arrow_mesh)
    
    return geometries


def visualize_layout(layout_file, additional_entities):
    """
    Loads the original layout and first adds doors then windows.
    Parameters:
        layout_file (str): Path to the original layout OBJ file.
        additional_entities (dict): Dictionary containing door and window annotations.
    """
    # Load the original layout mesh.
    original_mesh = o3d.io.read_triangle_mesh(layout_file)
    original_mesh.compute_vertex_normals()
    geometries = [original_mesh]
    if "doors" in additional_entities:
        geometries += visualize_layout_with_annotations(layout_file, additional_entities["doors"], name="doors")
    if "windows" in additional_entities:
        geometries += visualize_layout_with_annotations(layout_file, additional_entities["windows"], name="windows")
    return geometries
    


if __name__ == '__main__':
    # choose one of the scenes in
    # [
    #     "WYY7iVyf5p8", 
    #     'TbHJrupSAjP', 
    #     "2t7WUuJeko7",
    #     "YFuZgdQ5vWj", 
    #     "jtcxE69GiFV", 
    #     "1LXtFkjw3qL",
    #     "5LpN3gDmAk7",
    #     "e9zR4mvMWw7",
    #     "i5noydFURQK",
    #     "HxpKQynjfin",
    #     "JeFG25nYj2p",
    #     "JmbYfDe2QKZ",
    #     "p5wJjkQkbXX",
    #     "r47D5H71a5s",
    #     "S9hNv5qa7GM",
    #     "17DRP5sb8fy"
    # ]

    parser = argparse.ArgumentParser(description="Visualize layout with door/window annotations.")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to the dataset root directory.")
    parser.add_argument("--scene", type=str, required=True, help="Scene ID to visualize.")
    args = parser.parse_args()

    dataset_root = args.dataset_root
    scene = args.scene
    
    # Path to the structures-only layout OBJ file.
    structures_only = os.path.join(dataset_root, "structures", f"{scene}.obj")
    
    
    # load the json window and door annotations
    window_file= fr"{dataset_root}/windows/{scene}.json"
    with open(window_file, "r") as f:
        window_annotations = json.load(f)
    door_file = fr"{dataset_root}/doors/{scene}.json"
    with open(door_file, "r") as f:
        door_annotations = json.load(f)

    # Visualize the original layout with window / door annotations overlaid.
    geometries = visualize_layout(structures_only, {"doors": door_annotations, "windows": window_annotations})
    o3d.visualization.draw_geometries(geometries)


            
