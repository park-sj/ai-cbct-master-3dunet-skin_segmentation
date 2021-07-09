#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import glob
import imageio
import torch
from skimage.transform import resize

import augment.transforms as transforms
from datasets.utils import ConfigDataset
from unet3d.utils import get_logger

logger = get_logger('PngDataset')

class PngDataset(ConfigDataset):
    def __init__(self, file_path, phase, slice_builder_config, transformer_config, mirror_padding=(0, 32, 32)):
        """
        :param file_path: path to png root directory
        :param phase: 'train' for training, 'val' for validation, 'test' for testing; data augmentation is performed
            only during the 'train' phase
        :param transformer_config: data augmentation configuration
        """
        assert os.path.isdir(file_path), 'Incorrect dataset directory'
#        assert phase in ['train', 'val', 'test']
        assert phase is 'test', 'Dataset for train is not implemented'
        
        self.slice_builder_config = slice_builder_config
        self.phase = phase
        self.file_path = file_path
        self.patients = os.listdir(os.path.join(file_path, phase))
        self.transformer_config = transformer_config
        
    def getImage(self, count):
        if count >= len(self.patients):
            raise StopIteration
        if self.phase == 'test':
            logger.info(f'Loading images from {os.path.join(self.file_path, self.phase, self.patients[count])}')
        patient = os.path.join(self.file_path, self.phase, self.patients[count])
        imageList = glob.glob(patient + '/*.png')
        imageList = sorted(imageList, reverse=True) # 지금 테스트 해 본 데이터 대상으로는 reverse가 맞는데 프로메디우스 데이터에 해봐야한다.
        #imageList = sorted(imageList)

        self.cur_image = None

        for i in imageList:
            if self.cur_image is None:
                #self.cur_image = imageio.imread(i)[:,:,0]
                self.cur_image = imageio.imread(i)
                self.cur_image = np.expand_dims(self.cur_image, 0)
            else:
                #self.cur_image = np.concatenate((self.cur_image, np.expand_dims(imageio.imread(i)[:,:,0], 0)), axis = 0)
                self.cur_image = np.concatenate((self.cur_image, np.expand_dims(imageio.imread(i), 0)), axis=0)
        
        # CT와 똑같은 scale로 만들고 다시 정규화
#        self.cur_image = self.cur_image.astype(np.float32)
#        self.cur_image = (self.cur_image/255)*2000 - 750
#        self.min_value = -750
#        self.max_value = 1250
#        self.cur_image[self.cur_image>self.max_value] = self.max_value
#        self.cur_image[self.cur_image<self.min_value] = self.min_value
#        self.cur_image = resize(self.cur_image.astype(np.float32), (296, 296, 296), anti_aliasing = False)
#        mean = (self.min_value + self.max_value)/2
#        std = (self.max_value - self.min_value)/2
#        transformer = transforms.get_transformer(self.transformer_config, min_value=self.min_value, max_value=self.max_value,
#                                                 mean=mean, std=std)
#        self.raw_transform = transformer.raw_transform()

        # directly 정규화
        self.cur_image = self.cur_image.astype(np.float32)
        self.cur_image = (self.cur_image/255)*2-1
        self.cur_image[:, 0:20, :] = -1 # error of style transfer
        #self.cur_image = self.cur_image[:, 10:, :] # slicing error of style transfer
        self.cur_image = resize(self.cur_image, (296,)*3)
        self.cur_image = np.expand_dims(self.cur_image, 0)

    def __getitem__(self, idx):
        self.getImage(int(idx))
#        image = self.raw_transform(self.cur_image)
        image = self.cur_image
        return torch.from_numpy(image)
        
    def __len__(self):
        return len(self.patients)
        
    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load slice builder config
        slice_builder_config = phase_config['slice_builder']
        # load files to process
        file_paths = phase_config['file_paths']
        # mirror padding conf
        mirror_padding = dataset_config.get('mirror_padding', None)

        return [cls(file_paths[0], phase, slice_builder_config, transformer_config, mirror_padding)]
