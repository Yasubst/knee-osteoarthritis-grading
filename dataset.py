import os
import cv2
from torch.utils.data import Dataset

def read_img(path_file):
    xray = cv2.imread(path_file, cv2.IMREAD_GRAYSCALE)
    return xray


class knee_xray_dataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = self.dataset["image_path"].iloc[idx]
        name = os.path.basename(img_path)
        target = int(self.dataset["label"].iloc[idx])

        img = read_img(img_path)

        res = {
            "xray": img,
            "name": name,
            "target": target
        }

        return res