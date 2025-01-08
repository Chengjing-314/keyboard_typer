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


if __name__ == "__main__":
    main()
