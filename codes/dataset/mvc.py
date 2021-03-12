import json
import pickle
import random

from PIL import Image
from pathlib import Path
from tqdm import tqdm

from .base import AbstractDataset
from .util import open_json


class MVCDataset(AbstractDataset):

    def __init__(self, args):
        super().__init__(args)
        self.input = []
        self.item = []
        self.itemlen = []
        self.label = []
        # must match to folder name
        self.ds_name = "MVC"
        self.frontpath = f"{args.dataset_path}/{self.ds_name}/"
        self.json_names = [f"{self.frontpath}dataset/attribute_labels.json",
                           f"{self.frontpath}dataset/image_links.json"]
        self.pklpath = Path(f"{self.frontpath}{self.ds_name}dataset.pkl")

        print("start dataset build")
        if not self.get_ds_from_pkl():
            self.build()
            self.save_ds_to_pkl()

    @classmethod
    def code(cls):
        return 'mvc'

    def build(self):
        """

        build dataset from 2 jsons.

        1. Find preprocessed pickle
        2. if not, create
        'input' : image link (/image/~.jpg)
        'itemlen' : item # for each item
        'label' : 264 attr tags  for each item
        3. save pickle

        """
        # open json, len 161,260
        at_json = open_json(self.json_names[0])
        link_json = open_json(self.json_names[1])
        # if need preprocessing, do it
        if self.args.img_preprocessing:
            print("resize imgs")
            for i in tqdm(range(len(link_json))):
                image_url = "image/" + link_json[i]["image_url_4x"].split('/')[-1]
                img = Image.open(image_url)
                img = img.resize((224, 224))
                img.save(image_url)

        # create dataset
        itemlen = 0
        previd = 0
        for i in tqdm(range(len(link_json))):
            image_url = link_json[i]["image_url_4x"].split('/')[-1]
            uid = image_url.split('-')[0]
            if previd != uid:
                self.label.append(list(at_json[i].values())[2:])
                if i != 0:
                    self.itemlen.append(itemlen)
                    itemlen = 0
            self.input.append(f"{self.frontpath}dataset/image/" + image_url)
            previd = uid
            itemlen += 1
        self.itemlen.append(itemlen)
        self.separate()
        self.dataset = {
            'train': self.train,
            'validation': self.val,
            'test': self.test
        }

        print('finished dataset')

    def separate(self):
        """separate dataset to train/validation/test set by paper's method."""
        print("start dataset separating")
        sum = 0
        for i in tqdm(range(len(self.itemlen))):
            il = self.itemlen[i]
            if il < 3:
                sum += il
                continue
            rarr = list(range(sum, sum+il))
            random.shuffle(rarr)
            self.train.append({
                'input': self.input[rarr[0]],
                'label': self.label[i]
            })
            self.val.append({
                'input': self.input[rarr[1]],
                'label': self.label[i]
            })
            for j in range(2, len(rarr)):
                self.test.append({
                    'input': self.input[rarr[j]],
                    'label': self.label[i]
                })
            sum += il
