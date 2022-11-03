from pathlib import Path

import cv2
from datetime import datetime
import numpy as np
import os
import re
import shutil
import torch.utils.data as data
from PIL import Image
from progressbar import progressbar


class CustomConstantTimeDataset_new(data.Dataset):
    def __init__(self, csv_filepath: str, image_root: str, label_root: str, transform=None, phase='train',
                 image_suffix="png", label_suffix="png"):
        self.csv_filepath = csv_filepath
        self.transform = transform
        self.phase = phase
        self.tmp_dir = Path(".tmp").joinpath(datetime.now().strftime('%H%M%S%f'))
        self.fp_set = {'image': None, 'label': None}
        self.images_root = Path(image_root)
        self.labels_root = Path(label_root)
        self.only_images = label_root == ''
        self.img_suffix = image_suffix
        self.label_suffix = label_suffix
        os.makedirs(self.tmp_dir, exist_ok=True)

        with open(self.csv_filepath, 'r') as f:
            self.file_lines = [line.rstrip() for line in f]

        # 画像サイズの取得
        sample_filepath = f"{self.images_root.joinpath(self.file_lines[0].split(',')[0])}.{self.img_suffix}"
        assert Path(sample_filepath).exists(), f"{sample_filepath} is not found."
        self.image_shape = cv2.imread(sample_filepath, 0).shape

        self.fp_set['image'] = np.memmap(
            filename=str(self.tmp_dir.joinpath('image')),
            dtype=np.uint8,
            mode='w+',
            shape=tuple([len(self.file_lines)] + [3] + list(self.image_shape) + [1])
        )
        self.fp_set['label'] = np.memmap(
            filename=str(self.tmp_dir.joinpath('label')),
            dtype=np.int,
            mode='w+',
            shape=tuple([len(self.file_lines)] + [3] + list(self.image_shape))
        )

        print(f"{self.csv_filepath} reading...")
        for f_idx, file_line in progressbar(enumerate(self.file_lines), max_value=len(self.file_lines)):
            for o_idx, orf in enumerate(file_line.split(',')):
                image = cv2.imread(f"{self.images_root.joinpath(orf)}.{self.img_suffix}", 0)
                self.fp_set['image'][f_idx, o_idx, :, :, :] = image[:, :, np.newaxis]

                if self.only_images:
                    continue

                label = Image.open(f"{self.labels_root.joinpath(orf)}.{self.label_suffix}")
                self.fp_set['label'][f_idx, o_idx, :, :] = np.asarray(label)

    def __del__(self):
        print(f"{__file__} exec __del__")
        if self.tmp_dir.exists():
            for filename in self.tmp_dir.glob("*"):
                print(filename)
                os.remove(filename)
            shutil.rmtree('.tmp')

    @staticmethod
    def __numerical_sort__(value):
        """
        ファイル名の数字を元にソートする関数
        :param value:
        :return:
        """
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    def __len__(self):
        return len(self.file_lines)

    def __getitem__(self, index):
        if self.only_images:
            images = self.fp_set['image'][index]
            dataset: dict = self.transform(
                self.phase,
                image1=images[0],
                image2=images[1],
                image3=images[2],
            )
            dataset.setdefault("filename", self.file_lines[index].split(',')[1])
        else:
            images = self.fp_set['image'][index]
            labels = self.fp_set['label'][index]
            dataset: dict = self.transform(
                self.phase,
                image1=images[0],
                image2=images[1],
                image3=images[2],
                label1=labels[0],
                label2=labels[1],
                label3=labels[2],
            )
            dataset.setdefault("filename", self.file_lines[index].split(',')[1])
        return dataset