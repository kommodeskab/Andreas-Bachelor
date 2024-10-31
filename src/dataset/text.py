from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import BaseDataset
import random
import torch

class TextDataset(BaseDataset):
    def __init__(self, text : str, size : int = 1000):
        super().__init__()
        self.text = text
        self.size = size
        img_size = (len(text) * 150, 150)
        font_size = 150
        img = Image.new('L', img_size, color = 255)
        d = ImageDraw.Draw(img)
        font = ImageFont.load_default(size=font_size)
        d.text((0, 0), text, font=font, fill=0)
        img = 1 - np.array(img) / 255
        img = img[img.any(axis=1)]
        img = img[:, img.any(axis=0)]
        width, height = img.shape
        indices = np.where(img)
        coordinates = np.stack(indices, axis=1).astype(np.float32)
        coordinates[:, 0] = height - coordinates[:, 0]
        coordinates = coordinates[:, [1, 0]]
        coordinates[:, 0] = 2 * (coordinates[:, 0] - np.min(coordinates[:, 0])) / (np.max(coordinates[:, 0]) - np.min(coordinates[:, 0])) - 1
        coordinates[:, 1] = 2 * (coordinates[:, 1] - np.min(coordinates[:, 1])) / (np.max(coordinates[:, 1]) - np.min(coordinates[:, 1])) - 1
        self.coordinates = torch.tensor(coordinates)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        # randomly sample a point and add some noise
        coor = random.choice(self.coordinates)
        return coor + 0.01 * torch.randn(2)