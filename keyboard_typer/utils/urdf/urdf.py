import trimesh
import numpy as np
from dataclasses import dataclass
from yourdfpy import URDF
from scipy.spatial.transform import Rotation as R
import open3d as o3d


@dataclass
class KeyboardParserConfig:
    urdf_path: str


class KeyboardParser:
    def __init__(self, config: KeyboardParserConfig):
        self.config = config
        self.urdf = URDF.load(self.config.urdf_path)

    def get_mesh(self):
        """Print details of each link and its visual geometry."""
        for link in self.urdf.robot.links:
            print(f"Link Name: {link.name}")
            for visual in link.visuals:
                if visual.geometry.mesh:
                    print(f"  Visual Mesh Filename: {visual.geometry.mesh.filename}")

    def get_links(self, link_names: list[str]) -> list:
        """Get links by their names."""
        return [link for link in self.urdf.robot.links if link.name in link_names]

    def get_joints(self, joint_names: list[str]) -> list:
        """Get joints by their names."""
        return [joint for joint in self.urdf.robot.joints if joint.name in joint_names]

    def get_merged_mesh(self, link, root_path: str = "") -> trimesh.Trimesh:
        """Merge all meshes of a given link."""
        meshes = [
            trimesh.load(root_path / visual.geometry.mesh.filename)
            for visual in link.visuals
            if visual.geometry.mesh is not None
        ]
        return trimesh.util.concatenate(meshes)

    def get_mesh_center(self, links: list, root_path: str = "", keycap=False) -> np.ndarray:
        """Extract the centroid of all given links."""
        centroids = []
        for link in links:
            mesh = self.get_merged_mesh(link, root_path)
            if keycap:
                centroids.append(
                    [
                        mesh.bounding_box.centroid[0],
                        mesh.bounding_box.centroid[1],
                        mesh.bounds[1][2],
                    ]
                )
            else:
                centroids.append(mesh.bounding_box.centroid)
        return np.array(centroids)

    def get_merged_collision_mesh(self, link, root_path: str = "") -> trimesh.Trimesh:
        if not hasattr(link, "collisions") or not link.collisions:
            return trimesh.Trimesh()

        collision_meshes = []
        for collision in link.collisions:
            if collision.geometry.mesh is not None:
                filename = collision.geometry.mesh.filename
                mesh_path = root_path / filename if root_path else filename

                mesh = trimesh.load(mesh_path, force="mesh")

                collision_meshes.append(mesh)

        if collision_meshes:
            merged = trimesh.util.concatenate(collision_meshes)
            return merged
        else:
            # No valid mesh collisions found
            return trimesh.Trimesh()


def extract_link_centroids(parser, active_joints, root_path: str, base_link_name="link_0"):
    """Extract centroids of active links and the base link."""
    active_links = [joint.child for joint in active_joints]
    active_links = parser.get_links(active_links)

    # Get centroids for active links
    centroids = parser.get_mesh_center(active_links, root_path=root_path)

    # Get base link centroid
    base_link = parser.get_links([base_link_name])[0]
    base_centroids = parser.get_mesh_center([base_link], root_path=root_path)

    # Combine base and active centroids
    return np.concatenate((base_centroids, centroids), axis=0)


def visualize_mesh_and_centroids(mesh, centroids):
    """Visualize mesh and centroids using Open3D."""
    # Create Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

    # Create centroid points
    centroid_points = o3d.geometry.PointCloud()
    centroid_points.points = o3d.utility.Vector3dVector(centroids)
    centroid_points.colors = o3d.utility.Vector3dVector(np.ones_like(centroids) * [1, 0, 0])
    centroid_points.colors[0] = [0, 1, 0]  # Highlight base centroid

    # Add coordinate axes
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # Visualize
    o3d.visualization.draw_geometries([o3d_mesh, centroid_points, axes])


def main():
    # Load URDF and configuration
    urdf_path = (
        "/home/chengjing/Desktop/keyboard_typer/assets/keyboards/simplified/12996/mobility.urdf"
    )
    config = KeyboardParserConfig(urdf_path=urdf_path)
    parser = KeyboardParser(config)

    # Load keyboard data
    from keyboard_typer.utils.keyboard.keyboard import BlackKeyboard
    from keyboard_typer.constants import ASSETS_ROOT

    kb = BlackKeyboard()
    root_path = ASSETS_ROOT / "keyboards/simplified/12996"

    # Extract active joints and links
    active_joints = parser.get_joints(kb.mapping_array)
    centroids = extract_link_centroids(parser, active_joints, root_path=root_path)

    # Adjust centroids for visualization
    rotation_matrix = R.from_euler("xyz", [90, 0, -90], degrees=True).as_matrix()
    centroids = (rotation_matrix @ centroids.T).T

    # Visualize single key mesh and centroids
    child_link_name = active_joints[0].child  # Get the name of the child link
    child_link = parser.get_links([child_link_name])[0]  # Retrieve the link object
    single_key_mesh = parser.get_merged_mesh(child_link, root_path=root_path)
    visualize_mesh_and_centroids(single_key_mesh, centroids)

    __import__("IPython").embed(header="urdf.py:125")


def merge_collision_meshes():
    from pathlib import Path
    from yourdfpy.urdf import Collision, Geometry, Mesh

    # Update this path as necessary
    urdf_path = (
        "/home/chengjing/Desktop/keyboard_typer/assets/keyboards/simplified/12996/mobility.urdf"
    )

    # Create the parser
    config = KeyboardParserConfig(urdf_path=urdf_path)
    parser = KeyboardParser(config)

    # We'll save our new URDF next to the old one with a "_merged" suffix
    urdf_dir = Path(urdf_path).parent
    new_urdf_path = urdf_dir / "mobility_merged.urdf"

    # Create a directory for merged collision meshes if it doesn't exist
    merged_mesh_dir = urdf_dir / "merged_collision_meshes"
    merged_mesh_dir.mkdir(exist_ok=True)

    # Iterate over each link in the URDF
    for link in parser.urdf.robot.links:
        if not hasattr(link, "collisions") or len(link.collisions) == 0:
            continue  # No collisions to merge

        collision_meshes = []

        # Collect all collision meshes for this link
        for collision in link.collisions:
            # Check if collision geometry is a mesh (as opposed to box, cylinder, etc.)
            if collision.geometry.mesh is None:
                continue

            # Full path to the mesh file
            mesh_filename = collision.geometry.mesh.filename
            mesh_path = (urdf_dir / mesh_filename).resolve()

            if not mesh_path.exists():
                print(f"Warning: Collision mesh file {mesh_path} not found.")
                continue

            # Load the mesh
            m = trimesh.load(mesh_path, force="mesh")
            if m.is_empty:
                continue

            # If there's a transform (origin) in the collision, apply it
            # (Assuming your URDF parser sets collision.origin as a 4x4 matrix or None)
            if collision.origin is not None:
                m.apply_transform(collision.origin)

            collision_meshes.append(m)

        if len(collision_meshes) == 0:
            # No valid collision meshes found; skip
            continue

        # Merge all collision meshes for this link
        merged_mesh = trimesh.util.concatenate(collision_meshes)

        # Export the merged mesh to a new file
        merged_mesh_filename = f"merged_collision_meshes/merged_{link.name}.stl"
        merged_mesh_path = urdf_dir / merged_mesh_filename
        merged_mesh.export(merged_mesh_path)

        # Clear old collisions from the link
        link.collisions.clear()

        # Create a single new collision entry referencing our merged mesh
        # Note: If your parser or URDF library uses a different object name, adjust accordingly
        new_collision = Collision(
            name="merged_collision",
            geometry=Geometry(
                mesh=Mesh(filename=merged_mesh_filename)
            ),  # or just merged_mesh_filename
            origin=np.eye(4),
        )
        link.collisions.append(new_collision)

    # Finally, save the updated URDF

    parser.urdf.write_xml_file(new_urdf_path)
    print(f"New URDF with merged collision meshes saved to: {new_urdf_path}")


if __name__ == "__main__":
    merge_collision_meshes()
