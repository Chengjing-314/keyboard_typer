from pathlib import Path
import os.path as osp
from typing import List

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.building.ground import build_ground


class TableSceneBuilder(SceneBuilder):

    def build(self):

        builder = self.scene.create_actor_builder()
        model_dir = Path(osp.dirname(__file__)) / "assets"
        table_model_file = str(model_dir / "table.glb")
        scale = 1.75

        table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi))
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0.9196429 / 2]),
            half_size=(2.418 / 2, 1.209 / 2, 0.9196429 / 2),
        )
        builder.add_visual_from_file(
            filename=table_model_file,
            scale=[scale] * 3,
            pose=table_pose,
            material=np.ones(3) * 0.8,
        )
        table = builder.build_kinematic(name="table-workspace")
        aabb = (
            table._objs[0]
            .find_component_by_type(sapien.render.RenderBodyComponent)
            .compute_global_aabb_tight()
        )
        self.table_length = aabb[1, 0] - aabb[0, 0]
        self.table_width = aabb[1, 1] - aabb[0, 1]
        self.table_height = aabb[1, 2] - aabb[0, 2]

        floor_width = 100
        if self.scene.parallel_in_single_scene:
            floor_width = 500
        self.ground = build_ground(
            self.scene, floor_width=floor_width, altitude=-self.table_height
        )
        self.table = table
        self.scene_objects: List[sapien.Entity] = [self.table, self.ground]

    def initialize(self):
        self.table.set_pose(
            sapien.Pose(p=[1.209 / 2 - 0.05, 0, -self.table_height], q=euler2quat(0, 0, 0))
        )
