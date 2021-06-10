import collections
import numpy as np
import os
from   pathlib import Path
import pandas as pd
import random
import re
import shutil
from   sklearn.model_selection import train_test_split
import torch
from   torch.utils.data import Dataset
import tqdm

from .utils import pad_tensor, ASCIIKeyEncoder, CoordKeyEncoder, OneHotKeyEncoder, select_key_encoder

#===============================================================================

__all__ = ["KeystrokesParser", "KeystrokesSplitter", "KeystrokesDataset"]

#===============================================================================

_INPUT_FILE_RE = re.compile(r'^(?P<user_id>\d+)_keystrokes[.]txt$')

#===============================================================================

class _NoneFile:

    #---------------------------------------------------------------------------
    def close(self):
        pass

#===============================================================================

class KeystrokesParser:

    #---------------------------------------------------------------------------
    def parse(self, input_path, output_path):
        input_path = Path(input_path) / "files"
        metadata_file = input_path / "metadata_participants.txt"
        output_path = Path(output_path)

        metadata = self._parse_metadata(metadata_file)

        errors_count = 0
        for user_file in tqdm.tqdm(input_path.glob("*.txt")):
            file_match = _INPUT_FILE_RE.match(user_file.name)
            if file_match == None:
                continue

            user_id = int(file_match.group("user_id"))
            user_meta = metadata.get(user_id, None)
            if user_meta is None:
                continue

            user_output_path = output_path / str(user_id)
            os.makedirs(user_output_path, exist_ok=True)

            with open(user_output_path / "meta.txt", "w") as meta_file:
                meta_file.write(",".join(user_meta))

            with open(user_file, "r") as input_file:
                try:
                    self._parse_file(input_file, user_output_path)
                except Exception as e:
                    errors_count += 1
                    shutil.rmtree(user_output_path)

                    if errors_count % 100 == 0:
                        print("Errors:", errors_count)

        return errors_count

    #---------------------------------------------------------------------------
    def _parse_metadata(self, input_file):
        result = collections.defaultdict(list)
        metadata_df = pd.read_csv(input_file, sep='\t')

        for meta_row in metadata_df.iloc:
            user_id = meta_row["PARTICIPANT_ID"]
            device = meta_row["KEYBOARD_TYPE"]

            result[user_id].append(meta_row["LAYOUT"])

            if device in ["full", "laptop"]:
                result[user_id].append("desktop")
            elif device in ["on-screen"]:
                result[user_id].append("mobile")

        return result

    #---------------------------------------------------------------------------
    def _parse_file(self, input_file, output_path):
        current_id = None
        current_file = _NoneFile()

        for i, line in enumerate(input_file):
            if i == 0:
                continue

            parts = line.split("\t")
            section_id = parts[1]

            if section_id != current_id:
                current_file.close()
                current_file = open(output_path / f"{section_id}.csv", "w")
                current_id = section_id

            current_file.write(parts[-1].strip())
            current_file.write(",")
            current_file.write(str(int(parts[5])))
            current_file.write(",")
            current_file.write(str(int(parts[6])))
            current_file.write("\n")

        current_file.close()

#===============================================================================

class KeystrokesSplitter:

    #---------------------------------------------------------------------------
    def __init__(self, train_size=0.7, max_users=None, meta=None):
        self._search_meta = set(meta) if meta is not None else set()

        self._train_size = train_size
        self._max_users = max_users

    #---------------------------------------------------------------------------
    def split(self, input_path, output_path):
        input_path = Path(input_path)
        output_path = Path(output_path)

        selected_users = list(input_path.iterdir())
        random.shuffle(selected_users)

        num_selected = 0
        for user_folder in tqdm.tqdm(selected_users):
            if self._max_users is not None and num_selected >= self._max_users:
                break

            user_id = str(user_folder.name)

            with open(user_folder / "meta.txt") as meta_file:
                user_meta = set(meta_file.readline().split(','))
                if not self._search_meta.issubset(user_meta):
                    continue

            num_selected += 1

            user_output_path = output_path / str(user_id)
            os.makedirs(user_output_path, exist_ok=True)

            train_output_path = user_output_path / "train"
            test_output_path = user_output_path / "test"
            os.makedirs(train_output_path, exist_ok=True)
            os.makedirs(test_output_path, exist_ok=True)

            records = []
            for user_record in user_folder.glob('*.csv'):
                records.append(user_record.name)

            try:
                train_records, test_records = train_test_split(records, train_size=self._train_size)
            except:
                continue

            for record in train_records:
                shutil.copyfile(input_path / user_id / record, train_output_path / record)

            for record in test_records:
                shutil.copyfile(input_path / user_id / record, test_output_path / record)

#===============================================================================

class KeystrokeSample:

    #---------------------------------------------------------------------------
    def __init__(self, name, label):
        self.name = name
        self.label = label

#===============================================================================

class KeystrokesDataset(Dataset):

    #---------------------------------------------------------------------------
    def __init__(self, path, mode='train', key_encoder='coord', min_length=10, max_length=50, use_augmentation=False):
        if mode not in ['train', 'test']:
            raise Exception(f"bad mode {mode}")

        self._max_length = max_length
        self._min_length = min_length
        self._key_encoder = select_key_encoder(key_encoder)
        self._root_path = Path(path)
        self._mode = mode
        self._use_augmentation = use_augmentation

        self._index_root = self._root_path.parent / f"{self._root_path.name}.index" / self._mode
        if self._index_root.exists():
            shutil.rmtree(self._index_root)
        os.makedirs(self._index_root)

        self._labels = []
        self._samples = []

        for user_folder in tqdm.tqdm(sorted(self._root_path.iterdir())):
            user_id = int(user_folder.name)
            user_label = len(self._labels)
            self._labels.append(user_id)

            user_index_path = self._index_root / str(user_label)
            os.makedirs(user_index_path)

            for record in (user_folder / self._mode).iterdir():
                record_id = int(record.stem)
                
                self._samples.append(KeystrokeSample(record_id, user_label))
                record = self._parse_record(self._samples[-1])

                record_index_path = user_index_path / str(record_id)
                os.makedirs(record_index_path)

                torch.save(record, record_index_path / 'record.pt')

    #---------------------------------------------------------------------------
    def _parse_record_file(self, record_filename):
        codes = []
        times = []

        with open(record_filename, "r") as record_file:

            m_times = None
            for line in record_file:
                parts = line.rstrip('\n').split(',')
                code, down_time, up_time = map(int, parts)

                codes_i = self._key_encoder.transform(code)
                times_i = [down_time, up_time]

                codes.append(codes_i)
                times.append(times_i)

        codes = np.array(codes, dtype='float')
        times = np.array(times, dtype='float')

        time_features = np.zeros((len(codes), 4), dtype='float')
        time_features[1:, 0] = times[1:, 0] - times[:-1, 0]
        time_features[ :, 1] = times[ :, 1] - times[:  , 0]
        time_features[1:, 2] = times[1:, 1] - times[:-1, 0]
        time_features[1:, 3] = times[1:, 1] - times[:-1, 1]
        time_features /= 1000

        codes = torch.Tensor(codes)
        time_features = torch.Tensor(time_features)
        return torch.cat((codes, time_features), axis=1)

    #---------------------------------------------------------------------------
    def _parse_record(self, sample):
        user_id = str(self._labels[sample.label])
        record_filename = self._root_path / user_id / self._mode / f"{sample.name}.csv"
        record = self._parse_record_file(record_filename)
        return record

    #---------------------------------------------------------------------------
    def _load_record_full(self, sample):
        record_index = self._index_root / str(sample.label) / str(sample.name)
        record = torch.load(record_index / 'record.pt')
        return record

    #---------------------------------------------------------------------------
    def _load_record(self, sample):
        record = self._load_record_full(sample)

        if self._use_augmentation:
            record = self._make_augmention(record)

        record = pad_tensor(record, self._max_length, 'bottom')
        return record

    #---------------------------------------------------------------------------
    def _make_augmention(self, record):
        if len(record) > self._min_length:
            start_pos = np.random.randint(len(record) - self._min_length)

            if self._min_length >= self._max_length:
                size = self._max_length
            else:
                size = np.random.randint(self._min_length, self._max_length)
            end_pos = min(len(record), start_pos + size)

            record = record[start_pos:end_pos]
            record[0][-4] = 0.0
            record[0][-2] = 0.0
            record[0][-1] = 0.0

        return record

    #---------------------------------------------------------------------------
    def __len__(self):
        return len(self._samples)

    #---------------------------------------------------------------------------
    def __getitem__(self, index):
        return self._load_record(self._samples[index]), self.get_label(index)

    #---------------------------------------------------------------------------
    def get_item_full(self, index):
        return self._load_record_full(self._samples[index])

    #---------------------------------------------------------------------------
    def get_labels_count(self):
        return len(self._labels)

    #---------------------------------------------------------------------------
    def get_label(self, index):
        return self._samples[index].label

#===============================================================================
