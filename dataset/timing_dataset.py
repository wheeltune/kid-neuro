import collections
import numpy as np
import os
from   pathlib import Path
import pandas as pd
import re
import shutil
from   sklearn.model_selection import train_test_split
import torch
from   torch.utils.data import Dataset
import tqdm

from .keystrokes_dataset import KeystrokesDataset
from .utils import is_modifier_code, select_key_encoder

__all__ = ["TimingDataset"]

#===============================================================================

class TimingSample:

    #---------------------------------------------------------------------------
    def __init__(self, path):
        self.path = path

#===============================================================================

class TimingDataset(Dataset):

    #---------------------------------------------------------------------------
    def __init__(self, path, user_id, mode='train', key_encoder='coord', max_length=10):
        if mode not in ['train', 'test']:
            raise Exception(f"bad mode {mode}")

        self._root_path = Path(path) / str(user_id) / mode
        self._key_encoder = select_key_encoder(key_encoder)
        self._samples = []

        # self._min_length = min_length
        self._max_length = max_length

        for file_path in self._root_path.iterdir():
            self._samples.append(TimingSample(file_path))

    #---------------------------------------------------------------------------
    def _load_record(self, sample):
        record = self._parse_record(sample.path)

        start_pos = 0
        # if len(record[0]) > self._max_length:
        #     start_pos = np.random.randint(len(record[0]) - self._max_length)
        #     record = list(map(lambda _: _[start_pos:], record))

        record[1][0][0] = 0.0

        record = pad_all_arrays(record, self._max_length)
        return list(map(lambda _: torch.Tensor(_).float(), record))

    #---------------------------------------------------------------------------
    def _parse_record(self, record_filename):
        codes = []
        times = []

        with open(record_filename, "r") as record_file:

            m_times = None
            for line in record_file:
                parts = line.rstrip('\n').split(',')
                code, down_time, up_time = map(int, parts)

                codes_i = [int(is_modifier_code(code)) * 2 - 1]
                codes_i.extend(self._key_encoder.transform(code))
                times_i = [down_time, up_time]

                codes.append(codes_i)
                times.append(times_i)

        codes = np.array(codes, dtype='float')
        times = np.array(times, dtype='float')

        times[:, 1] = times[:, 1] - times[:, 0]
        times[1:, 0] = times[1:, 0] - times[:-1, 0]

        times = times / 1000
        return codes, times

    #---------------------------------------------------------------------------
    def __len__(self):
        return len(self._samples)

    #---------------------------------------------------------------------------
    def __getitem__(self, index):
        return self._load_record(self._samples[index])

#===============================================================================
