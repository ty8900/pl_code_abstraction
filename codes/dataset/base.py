import json
import pickle
import random
import os
from abc import *

from torch.utils.data import Dataset
from tqdm import tqdm

from .util import *


class AbstractDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset = {}
        self.train = []
        self.val = []
        self.test = []
        # self.transform = args.transforms
        self.force_update = args.force_update

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    def get_ds_from_pkl(self):
        if os.path.isfile(self.pklpath) and not self.force_update:
            print('Already preprocessed. Use exist pickle')
            with self.pklpath.open('rb') as f:
                self.dataset = pickle.load(f)
            self.train = self.dataset['train']
            self.val = self.dataset['validation']
            self.test = self.dataset['test']
            return True
        else:
            return False

    def save_ds_to_pkl(self):
        with self.pklpath.open('wb') as f:
            pickle.dump(self.dataset, f)
        self.train = self.dataset['train']
        self.val = self.dataset['validation']
        self.test = self.dataset['test']

    def load_dataset(self):
        """Main model calls dataset with this method."""
        return {
            'train': self.train,
            'validation': self.val,
            'test': self.test
        }

    def __len__(self):
        return len(self.input)
