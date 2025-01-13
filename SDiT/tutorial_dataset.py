import json
import cv2
import os
import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
class MyDataset(ImageFolder):
    def __init__(self, root, transform):
        data_path = os.path.join(root, "train")
        words_file = os.path.join(root, "words.txt")
        super().__init__(data_path, transform=transform)
        
        # words.txt 파일에서 클래스 번호와 이름을 읽어서 매핑 생성
        self.idx_to_class = {}
        self.class_to_name = {}
        with open(words_file, 'r') as f:
            for line in f:
                parts = line.strip().split(maxsplit=2)
                class_id = parts[0]
                class_num = parts[1]
                class_name = parts[2]
                self.idx_to_class[class_id]=int(class_num)
                self.class_to_name[class_id] = class_name
        
        # 클래스 인덱스에서 클래스 이름으로 매핑
        self.class_idx_to_name = {self.class_to_idx[k]: self.class_to_name[k] for k in self.class_to_idx.keys()}
        self.class_idx_to_num = {self.class_to_idx[k]: self.idx_to_class[k] for k in self.class_to_idx.keys()}

    def __getitem__(self, idx):
        # 부모 클래스의 __getitem__ 호출
        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        
        # 클래스 이름 가져오기
        class_name = self.class_idx_to_name[target]
        label = self.class_idx_to_num[target]

        # 클래스 이름을 출력하거나 추가 정보로 반환
        return dict(jpg = sample, txt = class_name, label = target)