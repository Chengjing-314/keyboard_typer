import json

import numpy as np
import sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env

from keyboard_typer.constants import ASSETS_ROOT


@register_env("KeyboardMapperEnv")
class KeyboardMapperEnv(BaseEnv):
    urdf_path = f"{ASSETS_ROOT}/keyboards/simplified/12996/mobility.urdf"
    bb_box_path = f"{ASSETS_ROOT}/keyboards/simplified/12996/bounding_box.json"

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_scene(self, options: dict):
        builder = self.scene.create_urdf_loader()
        bb_box_path = KeyboardMapperEnv.bb_box_path
        bbox_max, bbox_min = (
            json.load(open(bb_box_path))["max"],
            json.load(open(bb_box_path))["min"],
        )

        width, length, height = np.array(bbox_max) - np.array(bbox_min)
        builder.scale = 0.45 / max(width, length, height)

        builder.scale = 0.25
        articulation_builders = builder.parse(str(KeyboardMapperEnv.urdf_path))[
            "articulation_builders"
        ]
        builder = articulation_builders[0]
        self.keyboard = builder.build(name="keyboard")

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.15, 0, 0]))


def main():
    import gymnasium as gym

    env = gym.make("KeyboardMapperEnv", render_mode="human")
    env.reset()

    while True:
        env.render_human()


if __name__ == "__main__":
    main()
