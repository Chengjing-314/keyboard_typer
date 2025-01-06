"""
TableSceneBuilder that allows setting the table pose.
"""

from typing import Tuple

import sapien
import sapien.render
import torch
from mani_skill.utils.scene_builder.table import TableSceneBuilder as ManiSkillTableSceneBuilder


class TableSceneBuilder(ManiSkillTableSceneBuilder):
    def initialize(
        self,
        env_idx: torch.Tensor,
        table_pos: Tuple[float, float, float] = (0.55, 0.0, -0.9196429),
        table_quat: Tuple[float, float, float, float] = (0.70710678, 0.0, 0.0, 0.70710678),
    ):
        # table_height = 0.9196429
        self.table.set_pose(sapien.Pose(p=table_pos, q=table_quat))
