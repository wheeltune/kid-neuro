import abc
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder

__all__ = [
    "pad_tensor",
    "is_modifier_code",
    "ASCIIKeyEncoder", "CoordKeyEncoder", "OneHotKeyEncoder", "select_key_encoder"
]

#===============================================================================

_KEYBOARD = [
    [ 0, 192, 49, 50, 51, 52, 53, 54, 55,  56,  57,  48, 189, 187,  8,],
    [ 0,   9, 81, 87, 69, 82, 84, 89, 85,  73,  79,  80, 219, 221, 13,],
    [ 0,  20, 65, 83, 68, 70, 71, 72, 74,  75,  76, 186, 222, 220,  0,],
    [16, 192, 90, 88, 67, 86, 66, 78, 77, 188, 190, 191,  16,   0,  0,],
    [17,  18, 91,  0,  0, 32,  0,  0, 93,  18,  17,   0,   0,   0,  0,],
]

#===============================================================================

_MODIFIERS = set([16, 17, 18, 91])
def is_modifier_code(code):
    return code in _MODIFIERS

#===============================================================================

def pad_tensor(tensor, max_length, method='top'):
    content_len = min(len(tensor), max_length)

    if method == 'bottom':
        return nn.functional.pad(tensor[:content_len], (0, 0, max_length - content_len + 1, 0))
    elif method == 'top':
        return nn.functional.pad(tensor[:content_len], (0, 0, 0, max_length - content_len + 1))
    else:
        raise NotImplementedError

#===============================================================================

class KeyEncoder(metaclass=abc.ABCMeta):

    #---------------------------------------------------------------------------
    @abc.abstractmethod
    def transform(self, code):
        raise NotImplemented

#===============================================================================

class ASCIIKeyEncoder(KeyEncoder):

    #---------------------------------------------------------------------------
    def transform(self, code):
        return [code / 255]

#===============================================================================

class CoordKeyEncoder(KeyEncoder):

    #---------------------------------------------------------------------------
    def __init__(self):
        self._coords = self._read_coords(_KEYBOARD)
        self._unknown = set()

    #---------------------------------------------------------------------------
    def transform(self, code):
        if self._coords[code][0] == 0:
            self._unknown.add(code)
        return self._coords[code]

    #---------------------------------------------------------------------------
    def _read_coords(self, keyboard):
        max_len = len(keyboard[0]) - 1

        coords = [[-1, 0, 0] for i in range(256)]
        for i, line in enumerate(keyboard):
            for j, button in enumerate(line):
                if coords[button] == [-1, 0, 0]:
                    coords[button] = [int(is_modifier_code(button)) * 2 - 1,
                                      j / max_len * 2 - 1,
                                      i / max_len * 2 - 1]

        return coords

#===============================================================================

class OneHotKeyEncoder(KeyEncoder):

    #---------------------------------------------------------------------------
    def __init__(self):
        codes = self._read_codes(_KEYBOARD)
        codes = np.array(codes).reshape(-1, 1)
        self._encoder = OneHotEncoder(handle_unknown='ignore').fit(codes)

    #---------------------------------------------------------------------------
    def transform(self, code):
        return self._encoder.transform([[code]]).toarray()[0]

    #---------------------------------------------------------------------------
    def _read_codes(self, keyboard):
        codes = set()
        for i, line in enumerate(keyboard):
            codes.update(line)
        codes.remove(0)
        return sorted(list(codes))

#===============================================================================

def select_key_encoder(name):
    if name == 'ascii':
        return ASCIIKeyEncoder()
    elif name == 'coord':
        return CoordKeyEncoder()
    elif name == 'onehot':
        return OneHotKeyEncoder()
    else:
        raise NotImplementedError

#===============================================================================
