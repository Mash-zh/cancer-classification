import os

from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, imgpath, transform=None):
        self.imgpath = imgpath
        self.transform = transform  # Data transformation functions
        self.img_list = os.listdir(self.imgpath)  # List of file paths or data

    def __len__(self):
        return len(self.img_list)
        # pass

    def __getitem__(self, idx):
        img_path = self.imgpath
        image = Image.open(img_path + '/' + self.img_list[idx]).convert("RGB")
        label = int(self.img_list[idx][-5])
        if self.transform:
            image = self.transform(image)

        return image, label

# train_dataset = CustomDataset('train')
# print(train_dataset.__getitem__(0))
# print(train_dataset.__len__())