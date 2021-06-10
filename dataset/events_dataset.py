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
from .utils import ASCIIKeyEncoder, CoordKeyEncoder, OneHotKeyEncoder

__all__ = ["EventsDataset"]

#===============================================================================

class EventsDataset(Dataset):

    #---------------------------------------------------------------------------
    def __init__(self, path, encoder, mode='train', key_encoder='ascii'):
        self._item_ds = KeystrokesDataset(path, mode=mode, key_encoder=key_encoder)
        self._hidden = self._calc_hidden(path, encoder, key_encoder)

    #---------------------------------------------------------------------------
    def _calc_hidden(self, path, encoder, key_encoder):
        encoder = encoder.eval()
        item_ds = KeystrokesDataset(path, key_encoder=key_encoder)

        hiddens = [[] for i in range(item_ds.get_labels_count())]
        batch_size = 256
        for i in tqdm.tqdm(range(0, len(item_ds), batch_size)):
            records = []
            for j in range(i, min(len(item_ds), i + batch_size)):
                records.append(item_ds[j][0].unsqueeze(0))

            records = torch.cat(records)
            embeddings = encoder(records)

            for j, embedding in enumerate(embeddings):
                hiddens[item_ds.get_label(i + j)].append(embedding.detach().numpy())

        hiddens = np.array(hiddens)
        return list(map(torch.Tensor, hiddens.mean(axis=1)))

    #---------------------------------------------------------------------------
    def _parse_record(self, record_filename):
        events = []
        targets = []

        with open(record_filename, "r") as record_file:
            for line in record_file:
                parts = line.strip().split(",")

                events.append(self._key_encoder.transform(int(parts[0])))
                targets.append(list(map(lambda _: float(_) / 1000, parts[1:3])))

        return events, targets

    #---------------------------------------------------------------------------
    def __len__(self):
        return len(self._item_ds)

    #---------------------------------------------------------------------------
    def __getitem__(self, index):
        record, label = self._item_ds[index]
        keys = record[:, :-4]
        times = record[:, -4:-2]
        return self._hidden[label], keys, times

#===============================================================================

# class EventsDataset(Dataset):

#     #---------------------------------------------------------------------------
#     def __init__(self, path, encoder, mode='train', key_encoder='ascii'):
#         self._hidden = self._calc_hidden(path, encoder, key_encoder)
#         self._root_path = Path(path)
#         self._key_encoder = self._select_key_encoder(key_encoder)
#         self._max_length = 5
#         self._mode = mode

#         self._labels = []
#         self._samples = []

#         for user_folder in tqdm.tqdm(sorted(self._root_path.iterdir())):
#             user_id = int(user_folder.name)
#             user_label = len(self._labels)
#             self._labels.append(user_id)

#             for record in (user_folder / self._mode).iterdir():
#                 record_id = int(record.stem)
#                 self._samples.append((record_id, user_label))

#     #---------------------------------------------------------------------------
#     def _select_key_encoder(self, key_encoder):
#         if key_encoder == 'ascii':
#             return ASCIIKeyEncoder()
#         elif key_encoder == 'coord':
#             return CoordKeyEncoder()
#         elif key_encoder == 'onehot':
#             return OneHotKeyEncoder()
#         else:
#             raise NotImplementedError

#     #---------------------------------------------------------------------------
#     def _pad_record(self, record):
#         content_len = min(len(record), self._max_length)
#         record = np.pad(record[:content_len], ((self._max_length - content_len, 0), (0, 0)))
#         return record

#     #---------------------------------------------------------------------------
#     def _calc_hidden(self, path, encoder, key_encoder):
#         encoder = encoder.eval()
#         item_ds = KeystrokesDataset(path, key_encoder=key_encoder)

#         hiddens = [[] for i in range(item_ds.get_labels_count())]
#         batch_size = 256
#         for i in tqdm.tqdm(range(0, len(item_ds), batch_size)):
#             records = []
#             for j in range(i, min(len(item_ds), i + batch_size)):
#                 records.append(item_ds[j][0].unsqueeze(0))

#             records = torch.cat(records)
#             embeddings = encoder(records)

#             for j, embedding in enumerate(embeddings):
#                 hiddens[item_ds.get_label(i + j)].append(embedding.detach().numpy())

#         hiddens = np.array(hiddens)
#         return list(map(torch.Tensor, hiddens.mean(axis=1)))

#     #---------------------------------------------------------------------------
#     def _load_item(self, sample):
#         user_id = str(self._labels[sample[1]])
#         record_name = f"{sample[0]}.csv"
#         record_filename = self._root_path / user_id / self._mode / record_name

#         events, targets = self._parse_record(record_filename)
#         i = np.random.randint(0, len(events) - 1)
#         events = self._pad_record(events[max(0, i - self._max_length):i + 1])
#         target = targets[i]

#         return self._hidden[sample[1]], torch.Tensor(events).float(), torch.Tensor(target)

#     #---------------------------------------------------------------------------
#     def _parse_record(self, record_filename):
#         events = []
#         targets = []

#         with open(record_filename, "r") as record_file:
#             for line in record_file:
#                 parts = line.strip().split(",")

#                 events.append(self._key_encoder.transform(int(parts[0])))
#                 targets.append(list(map(lambda _: float(_) / 1000, parts[1:3])))

#         return events, targets

#     #---------------------------------------------------------------------------
#     def __len__(self):
#         return len(self._samples)

#     #---------------------------------------------------------------------------
#     def __getitem__(self, index):
#         return self._load_item(self._samples[index])

#===============================================================================
