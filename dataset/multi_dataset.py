import numpy as np
import torch
from   torch.utils.data import Dataset

__all__ = ["PairwiseDataset", "TripletDataset", "PairwiseLabelsDataset"]

#===============================================================================

class PairwiseLabelsDataset():

    #---------------------------------------------------------------------------
    def __init__(self, items_ds, p_neg=0.95):
        self._items_ds = items_ds
        self._p_neg = p_neg

        self._label_indexes = [[] for i in range(self._items_ds.get_labels_count())]
        for i in range(len(self._items_ds)):
            self._label_indexes[self._items_ds.get_label(i)].append(i)

        self._label_indexes = list(map(np.array, self._label_indexes))

    #---------------------------------------------------------------------------
    def __len__(self):
        return self._items_ds.get_labels_count()

    #---------------------------------------------------------------------------
    def __getitem__(self, index):
        lhs_label = index

        if np.random.uniform() >= self._p_neg:
            rhs_label = lhs_label
        else:
            offset = np.random.randint(0, self._items_ds.get_labels_count() - 1)
            rhs_label = (lhs_label + offset) % self._items_ds.get_labels_count()

        if lhs_label == rhs_label:
            lhs_idx, rhs_idx = np.random.choice(self._label_indexes[lhs_label], 2, replace=False)
        else:
            lhs_idx = np.random.choice(self._label_indexes[lhs_label])
            rhs_idx = np.random.choice(self._label_indexes[rhs_label])

        return self._items_ds[lhs_idx][0], self._items_ds[rhs_idx][0], int(lhs_label == rhs_label)

#===============================================================================

class PairwiseDataset():

    #---------------------------------------------------------------------------
    def __init__(self, items_ds, p_neg=0.95):
        self._items_ds = items_ds
        self._p_neg = p_neg

        self._label_indexes = [[] for i in range(self._items_ds.get_labels_count())]
        for i in range(len(self._items_ds)):
            self._label_indexes[self._items_ds.get_label(i)].append(i)

        self._label_indexes = list(map(np.array, self._label_indexes))

    #---------------------------------------------------------------------------
    def __len__(self):
        return len(self._items_ds)

    #---------------------------------------------------------------------------
    def __getitem__(self, index):
        lhs_idx = index
        lhs_label = self._items_ds.get_label(lhs_idx)

        if np.random.uniform() >= self._p_neg:
            rhs_label = lhs_label
        else:
            offset = np.random.randint(0, self._items_ds.get_labels_count() - 1)
            rhs_label = (lhs_label + offset) % self._items_ds.get_labels_count()

        rhs_idx = np.random.choice(self._label_indexes[rhs_label])
        return self._items_ds[lhs_idx][0], self._items_ds[rhs_idx][0], int(lhs_label == rhs_label)

#===============================================================================

class TripletDataset():

    #---------------------------------------------------------------------------
    def __init__(self, items_ds):
        self._items_ds = items_ds

        self._label_indexes = [[] for i in range(self._items_ds.get_labels_count())]
        for i in range(len(self._items_ds)):
            self._label_indexes[self._items_ds.get_label(i)].append(i)

        self._label_indexes = list(map(np.array, self._label_indexes))

    #---------------------------------------------------------------------------
    def __len__(self):
        return len(self._items_ds)

    #---------------------------------------------------------------------------
    def __getitem__(self, index):
        achore_idx = index
        achore_label = self._items_ds.get_label(achore_idx)

        offset = np.random.randint(0, self._items_ds.get_labels_count() - 1)
        negative_label = (achore_label + offset) % self._items_ds.get_labels_count()

        positive_idx = np.random.choice(self._label_indexes[achore_label])
        negative_idx = np.random.choice(self._label_indexes[negative_label])

        return self._items_ds[achore_idx][0], self._items_ds[positive_idx][0], self._items_ds[negative_idx][0]

#===============================================================================
