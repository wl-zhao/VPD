# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

import random
import importlib
import albumentations as A
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def get_dataset(dataset_name, **kwargs):
    dataset_name = dataset_name.lower()
    dataset_lib = importlib.import_module(
        '.' + dataset_name, package='dataset')

    dataset_abs = getattr(dataset_lib, dataset_name)
    print(dataset_abs)

    return dataset_abs(**kwargs)


class BaseDataset(Dataset):
    def __init__(self, crop_size):
        
        self.count = 0
        
        basic_transform = [
            A.HorizontalFlip(),
            A.RandomCrop(crop_size[0], crop_size[1]),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue()
        ]
        self.basic_transform = basic_transform    
        self.to_tensor = transforms.ToTensor()

    def readTXT(self, txt_path):
        with open(txt_path, 'r') as f:
            listInTXT = [line.strip() for line in f]

        return listInTXT

    def augment_training_data(self, image, depth):
        H, W, C = image.shape

        if self.count % 4 == 0:
            alpha = random.random()
            beta = random.random()
            p = 0.75

            l = int(alpha * W)
            w = int(max((W - alpha * W) * beta * p, 1))

            image[:, l:l+w, 0] = depth[:, l:l+w]
            image[:, l:l+w, 1] = depth[:, l:l+w]
            image[:, l:l+w, 2] = depth[:, l:l+w]

        additional_targets = {'depth': 'mask'}
        aug = A.Compose(transforms=self.basic_transform,
                        additional_targets=additional_targets)
        augmented = aug(image=image, depth=depth)
        image = augmented['image']
        depth = augmented['depth']

        image = self.to_tensor(image)
        depth = self.to_tensor(depth).squeeze()

        self.count += 1

        return image, depth

    def augment_test_data(self, image, depth):
        image = self.to_tensor(image)
        depth = self.to_tensor(depth).squeeze()

        return image, depth

