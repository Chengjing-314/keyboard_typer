# python3 keyboard_typer/utils/keyboard/keyboard.py
from dataclasses import dataclass
import numpy as np


@dataclass
class KeyboardConfig:
    keyboard_name:str = "black"
    actuation_point: float = 0.015



class Keyboard:
    def __init__(self, actuation_point=0.015):
       self.actuation_point = actuation_point
       self.mapping = {}     
       self.mapping_array = np.array([])

    def get_key(self, key_name):
        return self.mapping[key_name]

    def generate_key_press(self, num_keys):
        indices = np.random.choice(len(self.mapping_array), num_keys, replace=True)
        key_names = self.mapping_array[indices]
        return key_names, indices 


class BlackKeyboard(Keyboard):
    def __init__(self, actuation_point=0.015):
        super().__init__(actuation_point=actuation_point)

        self.mapping = {
                'key_8_star': '8',
                'key_7_and': '7',
                'key_6_caret': '6',
                'key_5_percentage': '5',
                'key_4_dollar': '4',
                'key_9_openparenthesis': '9',
                'key_0_closeparenthesis': '0',
                'key_dash_underscore': '-',
                'key_equal_plus': '=',
                'key_backspace': 'backspace',
                'key_t': 't',
                'key_y': 'y',
                'key_u': 'u',
                'key_i': 'i',
                'key_o': 'o',
                'key_closebracket': ']',
                'key_openbracket': '[',
                'key_p': 'p',
                'key_apostrophe': "'",
                'key_semicolon_colon': ';',
                'key_l': 'l',
                'key_k': 'k',
                'key_j': 'j',
                'key_h': 'h',
                'key_g': 'g',
                'key_f': 'f',
                'key_d': 'd',
                'key_r': 'r',
                'key_3_pound': '3',
                'key_e': 'e',
                'key_w': 'w',
                'key_2_at': '2',
                'key_q': 'q',
                'key_s': 's',
                'key_a': 'a',
                'key_z': 'z',
                'key_x': 'x',
                'key_c': 'c',
                'key_v': 'v',
                'key_b': 'b',
                'key_n': 'n',
                'key_m': 'm',
                'key_comma_less': ',',
                'key_period_greater': '.',
                'key_enter': 'enter',
                'key_backslash': '\\',
                'key_space': ' ',
                'key_tab': 'tab',
                'key_1_exclamation': '1',
                'key_tilda_backticks': '`',
                'key_esc': 'esc',
                'key_left': 'left',
                'key_right': 'right',
                'key_down': 'down',
                'key_up': 'up',
                'key_fowardslash_question': '/'
                }
        
        self.mapping_array = np.array(list(self.mapping.keys()))

    

def main():
    kb = BlackKeyboard()
    print(len(kb.mapping_array))
    print(kb.generate_key_press(10))
    
if __name__ == "__main__":
    main() 