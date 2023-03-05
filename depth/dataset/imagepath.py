# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

import os
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class imagepath(Dataset):
    # for test only
    def __init__(self, data_path):
        super().__init__()

        self.data_path = data_path
        self.to_tensor = transforms.ToTensor()

        self.filenames_list = [os.path.join(data_path, i) for i in os.listdir(data_path)
                               if i.split('.')[-1] in ['jpg', 'png']]

        print("Dataset : Image Path")
        print("# of images: %d" % (len(self.filenames_list)))

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        batch = {}
        file = self.filenames_list[idx]
        filename = file.split('/')[-1]

        image = cv2.imread(file)  # [H x W x C] and C: BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # input size should be multiple of 32
        h, w, c = image.shape
        new_h, new_w = h // 32 * 32, w // 32 * 32
        image = cv2.resize(image, (new_w, new_h))
        image = self.to_tensor(image)

        batch['image'] = image
        batch['filename'] = filename

        return batch
