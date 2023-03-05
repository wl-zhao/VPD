# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

import os
import cv2

from dataset.base_dataset import BaseDataset
import json

class nyudepthv2(BaseDataset):
    def __init__(self, data_path, filenames_path='./dataset/filenames/',
                 is_train=True, crop_size=(448, 576), scale_size=None):
        super().__init__(crop_size)


        if crop_size[0] > 480:
            scale_size = (int(crop_size[0]*640/480), crop_size[0])

        self.scale_size = scale_size

        self.is_train = is_train
        self.data_path = os.path.join(data_path, 'nyu_depth_v2')

        self.image_path_list = []
        self.depth_path_list = []

        with open('nyu_class_list.json', 'r') as f:
            self.class_list = json.load(f)

        txt_path = os.path.join(filenames_path, 'nyudepthv2')
        if is_train:
            txt_path += '/train_list.txt'
            self.data_path = self.data_path + '/sync'
        else:
            txt_path += '/test_list.txt'
            self.data_path = self.data_path + '/official_splits/test/'
 
        self.filenames_list = self.readTXT(txt_path) # debug
        phase = 'train' if is_train else 'test'
        print("Dataset: NYU Depth V2")
        print("# of %s images: %d" % (phase, len(self.filenames_list)))

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        img_path = self.data_path + self.filenames_list[idx].split(' ')[0]
        gt_path = self.data_path + self.filenames_list[idx].split(' ')[1]
        filename = img_path.split('/')[-2] + '_' + img_path.split('/')[-1]

        class_id = -1
        for i, name in enumerate(self.class_list):
            if name in filename:
                class_id = i
                break
        
        assert class_id >= 0

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')

        # print(image.shape, depth.shape, self.scale_size)

        if self.scale_size:
            image = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))
            depth = cv2.resize(depth, (self.scale_size[0], self.scale_size[1]))
        
        # print(image.shape, depth.shape, self.scale_size)

        if self.is_train:
            image, depth = self.augment_training_data(image, depth)
        else:
            image, depth = self.augment_test_data(image, depth)

        depth = depth / 1000.0  # convert in meters

        return {'image': image, 'depth': depth, 'filename': filename, 'class_id': class_id}
