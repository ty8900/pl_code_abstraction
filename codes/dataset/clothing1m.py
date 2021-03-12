import json
import pickle
import random
import pandas as pd
from pathlib import Path
from PIL import Image

from tqdm import tqdm

from .base import AbstractDataset
from .util import make_onehot_encoding


class Clothing1MDataset(AbstractDataset):

    def __init__(self, args):
        super().__init__(args)
        self.dataset = {}

        # must match to folder name
        self.ds_name = "Clothing1M"
        self.frontpath = f"{args.dataset_path}/{self.ds_name}/dataset/"
        self.kv_names = [self.frontpath + "noisy_label_kv.txt",
                         self.frontpath + "clean_label_kv.txt"]
        self.key_names = [self.frontpath + "noisy_train_key_list.txt",
                          self.frontpath + "clean_val_key_list.txt",
                          self.frontpath + "clean_test_key_list.txt"]
        self.pklpath = Path(f"{args.dataset_path}/{self.ds_name}/{self.ds_name}dataset.pkl")

        print("start dataset build")
        if not self.get_ds_from_pkl():
            self.build()
            self.save_ds_to_pkl()

    @classmethod
    def code(cls):
        return 'clothing1m'

    def build(self):
        """

        build dataset from 2 jsons.

        1. Find preprocessed pickle
        2. if not, create
        3. save pickle

        """
        # open txt : dataframes
        noisy_kv = self.open_txt(self.kv_names[0], sep_type=' ', column_name=['input', 'label'])
        clean_kv = self.open_txt(self.kv_names[1], sep_type=' ', column_name=['input', 'label'])
        train_keys = self.open_txt(self.key_names[0], column_name=['input'])
        val_keys = self.open_txt(self.key_names[1], column_name=['input'])
        test_keys = self.open_txt(self.key_names[2], column_name=['input'])
        # preprocessing
        if self.args.img_preprocessing:
            print("resize imgs")
            for url in tqdm(train_keys['input']):
                img = Image.open(self.frontpath + url)
                img = img.resize((224, 224))
                img.save(self.frontpath + url)
            for url in tqdm(val_keys['input']):
                img = Image.open(self.frontpath + url)
                img = img.resize((224, 224))
                img.save(self.frontpath + url)
            for url in tqdm(test_keys['input']):
                img = Image.open(self.frontpath + url)
                img = img.resize((224, 224))
                img.save(self.frontpath + url)
        # create dataset
        train_dict = pd.merge(noisy_kv, train_keys, on='input').to_dict('index')
        val_dict = pd.merge(clean_kv, val_keys, on='input').to_dict('index')
        test_dict = pd.merge(clean_kv, test_keys, on='input').to_dict('index')
        for i in train_dict:
            train_dict[i]['input'] = self.frontpath + train_dict[i]['input']
            train_dict[i]['label'] = [train_dict[i]['label']]
            train_dict[i]['index'] = [int(i)]
        for i in val_dict:
            val_dict[i]['input'] = self.frontpath + val_dict[i]['input']
            val_dict[i]['label'] = [val_dict[i]['label']]
            val_dict[i]['index'] = [int(i)]
        for i in test_dict:
            test_dict[i]['input'] = self.frontpath + test_dict[i]['input']
            test_dict[i]['label'] = [test_dict[i]['label']]
            test_dict[i]['index'] = [int(i)]
        self.dataset = {
            'train': train_dict,
            'validation': val_dict,
            'test': test_dict
        }
        print('finished dataset')

    @staticmethod
    def open_txt(path, sep_type='\n', column_name=None):
        # todo column_name handling
        print(f"opening {path}...")
        if column_name is None:
            df = df = pd.read_csv(path, sep=sep_type, engine='python')
        else:
            df = pd.read_csv(path, sep=sep_type, engine='python',
                             names=column_name)
        return df

