import open3d as o3d
import os
import numpy as np

obj_dir = "/home/chengjing/Desktop/keyboard_typer/assets/keyboards/cleaned/12996/textured_objs"

# List of .obj files to load

# obj_files = [f"original-{i}.obj" for i in range(105)]

obj_files = ["original-106.obj"]

""" # Load and visualize the .obj files
meshes = []
for obj_file in obj_files:
    obj_path = os.path.join(obj_dir, obj_file)
    print(f"Loading {obj_path}...")
    mesh = o3d.io.read_triangle_mesh(obj_path)
    if mesh.is_empty():
        print(f"Warning: {obj_file} could not be loaded or is empty!")
    else:
        meshes.append(mesh)

# Combine all meshes into a single geometry for visualization
combined_mesh = o3d.geometry.TriangleMesh()
for mesh in meshes:
    combined_mesh += mesh

# Visualize the combined geometry
o3d.visualization.draw_geometries([combined_mesh])
 """


def extract_mesh_position(obj_path):
    mesh = o3d.io.read_triangle_mesh(obj_path)
    if mesh.is_empty():
        print(f"Warning: {obj_path} is empty!")
        return None
    vertices = np.asarray(mesh.vertices)  # Get all vertices
    centroid = vertices.mean(axis=0)  # Compute the centroid
    return centroid


# Process all .obj files
key_positions = {}
for obj_file in obj_files:
    obj_path = os.path.join(obj_dir, obj_file)
    print(f"Processing {obj_path}...")
    position = extract_mesh_position(obj_path)
    if position is not None:
        key_positions[obj_file] = position

# Print extracted positions
for key, position in key_positions.items():
    print(f"{key}: {position}")
