# python3 keyboard_typer/utils/keyboard/keyboard.py
from dataclasses import dataclass
import numpy as np
import json
from keyboard_typer.constants import ASSETS_ROOT


@dataclass
class KeyboardConfig:
    keyboard_name: str = "black"
    actuation_point: float = 0.015


class Keyboard:
    def __init__(self, actuation_point=0.015):
        self.actuation_point = actuation_point
        self.mapping = {}
        self.mapping_array = np.array([])

    def get_mapped_key(self, key_name: str):
        """Retrieve the mapped value for a given key name."""
        return self.mapping.get(key_name)

    def simulate_key_presses(self, num_envs: int):
        """
        Simulate key presses for multiple environments.

        Args:
            num_envs (int): Number of environments for which to simulate key presses.

        Returns:
            tuple: Simulated key names and their corresponding indices.
        """
        indices = np.random.choice(len(self.mapping_array), num_envs, replace=True)
        key_names = self.mapping_array[indices]
        return key_names, indices

    def simulate_key_presses_over_time(self, time_steps: int, num_envs: int):
        """
        Simulate key presses over multiple time steps for multiple environments.

        Args:
            time_steps (int): Number of time steps for simulation.
            num_envs (int): Number of environments per time step.

        Returns:
            tuple: Batched key names and indices for all time steps.
        """
        batched_key_names = []
        batched_indices = []
        for _ in range(time_steps):
            key_names, indices = self.simulate_key_presses(num_envs)
            batched_key_names.append(key_names)
            batched_indices.append(indices)
        return batched_key_names, batched_indices


class BlackKeyboard(Keyboard):
    def __init__(self, actuation_point=0.015):
        super().__init__(actuation_point)

        # Key mapping
        self.mapping = self._initialize_mapping()
        self.mapping_array = np.array(list(self.mapping.keys()))

        # Keyboard dimensions and default key positions
        self.bbox_max, self.bbox_min = self._load_bounding_box()
        self.kb_length = max(self.bbox_max - self.bbox_min)
        self.kb_height = min(self.bbox_max - self.bbox_min)
        self._default_key_position = self._load_default_key_positions()
        self._scale = 1.0

    @staticmethod
    def _initialize_mapping():
        """Initialize the key mapping."""
        return {
            "key_8_star": "8",
            "key_7_and": "7",
            "key_6_caret": "6",
            "key_5_percentage": "5",
            "key_4_dollar": "4",
            "key_9_openparenthesis": "9",
            "key_0_closeparenthesis": "0",
            "key_dash_underscore": "-",
            "key_equal_plus": "=",
            "key_backspace": "backspace",
            "key_t": "t",
            "key_y": "y",
            "key_u": "u",
            "key_i": "i",
            "key_o": "o",
            "key_closebracket": "]",
            "key_openbracket": "[",
            "key_p": "p",
            "key_apostrophe": "'",
            "key_semicolon_colon": ";",
            "key_l": "l",
            "key_k": "k",
            "key_j": "j",
            "key_h": "h",
            "key_g": "g",
            "key_f": "f",
            "key_d": "d",
            "key_r": "r",
            "key_3_pound": "3",
            "key_e": "e",
            "key_w": "w",
            "key_2_at": "2",
            "key_q": "q",
            "key_s": "s",
            "key_a": "a",
            "key_z": "z",
            "key_x": "x",
            "key_c": "c",
            "key_v": "v",
            "key_b": "b",
            "key_n": "n",
            "key_m": "m",
            "key_comma_less": ",",
            "key_period_greater": ".",
            "key_enter": "enter",
            "key_backslash": "\\",
            "key_space": " ",
            "key_tab": "tab",
            "key_1_exclamation": "1",
            "key_tilda_backticks": "`",
            "key_esc": "esc",
            "key_left": "left",
            "key_right": "right",
            "key_down": "down",
            "key_up": "up",
            "key_fowardslash_question": "/",
        }

    def simulate_key_presses_word(self, word: str, num_envs: int):
        # TODO: implement word to key mapping
        pass

    @staticmethod
    def _load_bounding_box():
        """Load bounding box data."""
        bb_box_path = f"{ASSETS_ROOT}/keyboards/simplified/12996/bounding_box.json"
        with open(bb_box_path) as f:
            bbox_data = json.load(f)
        return np.array(bbox_data["max"]), np.array(bbox_data["min"])

    @staticmethod
    def _load_default_key_positions():
        """Load default key positions."""
        key_position_path = f"{ASSETS_ROOT}/keyboards/simplified/12996/kb_mesh_centroids_t.npy"
        return np.load(key_position_path)

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        if value <= 0:
            raise ValueError("Scale must be greater than 0")
        self._scale = value
        self.default_key_position = self._default_key_position * self._scale

    @property
    def default_key_position(self):
        return self._default_key_position

    @default_key_position.setter
    def default_key_position(self, new_positions):
        self._default_key_position = new_positions

    def get_scale(self, max_side_length: float):
        """Calculate scale relative to the max side length."""
        return max_side_length / self.kb_length

    def get_default_base_pos(self):
        """Get the default position of the base key."""
        return self.default_key_position[0]

    def get_default_key_pos(self):
        """Get the default positions of all keys except the base."""
        return self.default_key_position[1:]


def main():
    kb = BlackKeyboard()
    print(len(kb.mapping_array))


if __name__ == "__main__":
    main()
