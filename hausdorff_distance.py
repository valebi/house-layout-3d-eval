import numpy as np
import open3d as o3d

def cost_matrix_hausdorff_distance(meshes1, meshes2):
    """
    Compute a P1 x P2 distance matrix between two sets of polygon meshes using
    the exact distance computed via Open3D's raycasting scene with pre-computed scenes.

    For each pair (i, j), where i indexes meshes1 and j indexes meshes2, the distance is defined as:
    
        d(M_i, N_j) = max( max_{v in M_i} d(v, N_j), max_{v in N_j} d(v, M_i) )
    
    Parameters:
    -----------
    meshes1 : list of o3d.geometry.TriangleMesh
        A list of polygon meshes for set 1.
    meshes2 : list of o3d.geometry.TriangleMesh
        A list of polygon meshes for set 2.
        
    Returns:
    --------
    numpy.ndarray
        A P1 x P2 matrix where the (i, j) entry is the computed distance between
        meshes1[i] and meshes2[j].
    """
    
    device = o3d.core.Device("CPU:0")
    
    dtype_f = o3d.core.float32
    dtype_i = o3d.core.int32

    P1 = len(meshes1)
    P2 = len(meshes2)
    D = np.zeros((P1, P2))
    
    # Precompute raycasting scenes for meshes in set 1.
    scenes1 = []
    for i, mesh in enumerate(meshes1):
        # print(f"A Precomputing scene for mesh {i}")
        # tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        mesh_t = o3d.t.geometry.TriangleMesh(device)
        mesh_t.vertex.positions = o3d.core.Tensor(np.ascontiguousarray(mesh.vertices).astype(np.float32), dtype_f, device)
        mesh_t.triangle.indices = o3d.core.Tensor(np.ascontiguousarray(mesh.triangles).astype(np.int32), dtype_i, device)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_t)
        scenes1.append(scene)
    
    # Precompute raycasting scenes for meshes in set 2.
    scenes2 = []
    for i, mesh in enumerate(meshes2):
        # print(f"B Precomputing scene for mesh {i}")
        # if i == 535:
        #     print("Mesh 457")
            # break
        #     print("Mesh 702")
        mesh_t = o3d.t.geometry.TriangleMesh(device)
        mesh_t.vertex.positions = o3d.core.Tensor(np.ascontiguousarray(mesh.vertices).astype(np.float32), dtype_f, device)
        mesh_t.triangle.indices = o3d.core.Tensor(np.ascontiguousarray(mesh.triangles).astype(np.int32), dtype_i, device)

        # tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_t)
        scenes2.append(scene)
    
    # Compute the distance matrix.
    for i in range(P1):
        # Prepare vertices of mesh i from set 1 as query points.
        vertices_i = np.ascontiguousarray(meshes1[i].vertices).astype(np.float32)
        query_points_i = o3d.core.Tensor(vertices_i, dtype=o3d.core.Dtype.Float32)
        
        for j in range(P2):
            # Prepare vertices of mesh j from set 2 as query points.
            vertices_j = np.ascontiguousarray(meshes2[j].vertices).astype(np.float32)
            query_points_j = o3d.core.Tensor(vertices_j, dtype=o3d.core.Dtype.Float32)
            
            # Compute distances from vertices in mesh i to mesh j (using scene from set 2).
            distances_i_to_j = scenes2[j].compute_distance(query_points_i).numpy()
            max_i_to_j = distances_i_to_j.max() if distances_i_to_j.size > 0 else 0
            
            # Compute distances from vertices in mesh j to mesh i (using scene from set 1).
            distances_j_to_i = scenes1[i].compute_distance(query_points_j).numpy()
            max_j_to_i = distances_j_to_i.max() if distances_j_to_i.size > 0 else 0
            
            # Symmetric distance is the maximum of these two directional maximum distances.
            D[i, j] = max(max_i_to_j, max_j_to_i)
    
    return D

# Test:
if __name__ == "__main__":
    # Create example polygon meshes for two sets.
    # Set 1:
    mesh1 = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    mesh2 = o3d.geometry.TriangleMesh.create_box(width=1.5, height=1.5, depth=1.5)
    mesh2.translate((3, 0, 0))  # Translate so they're not overlapping.
    
    # Set 2:
    mesh3 = o3d.geometry.TriangleMesh.create_cone(radius=0.8, height=2.0)
    mesh4 = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
    # mesh3.translate((0, 3, 0))
    # mesh4.translate((3, 3, 0))
    
    meshes1 = [mesh1, mesh2]
    meshes2 = [mesh3, mesh4]
    
    distance_matrix = cost_matrix_hausdorff_distance(meshes1, meshes2)
    print("Distance matrix between the two polygon sets:")
    print(distance_matrix)
