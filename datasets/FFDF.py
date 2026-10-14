import os

import torch

from .dataset import DeepFakeDataset
from .utils import (
    get_image_from_path,
    get_mask_path_from_img_path,
)
from utils.registries import DATASET_REGISTRY

'''
DATASET:
  DATASET_NAME: FFDF
  ROOT_DIR: /some_where/FF++/face
  TRAIN_INFO_TXT: '/some_where/train_face_c23.txt'
  VAL_INFO_TXT: '/some_where/val_face_c23.txt'
  TEST_INFO_TXT: '/some_where/test_face_c23.txt'
  IMG_SIZE: 380
  SCALE_RATE: 1.0
  ROTATE_ANGLE: 10
  CUTOUT_H: 10
  CUTOUT_W: 10
  COMPRESSION_LOW: 65
  COMPRESSION_HIGH: 80
'''

@DATASET_REGISTRY.register()
class FFDF(DeepFakeDataset):
    def __getitem__(self, idx):
        info_line = self.info_list[idx]
        image_info = info_line.strip('\n').split()
        image_path = image_info[0]
        image_abs_path = os.path.join(self.root_dir, image_path)
        mask_abs_path = get_mask_path_from_img_path(
            self.dataset_name, self.root_dir, image_path
        )
        img, mask = get_image_from_path(
            image_abs_path, mask_abs_path, self.mode, self.dataset_cfg
        )
        img_label_binary = int(image_info[1])

        # 提取 video_id：从文件名中去除 _frame_xxx 后缀
        # 格式：000_003_frame_000_face_0.jpg -> 000_003
        video_id = os.path.basename(image_path).split('_frame_')[0]

        sample = {
            'img': img,
            'bin_label': [int(img_label_binary)],
            'video_id': video_id,
        }

        sample['img'] = torch.FloatTensor(sample['img'])
        sample['bin_label'] = torch.FloatTensor(sample['bin_label'])
        sample['bin_label_onehot'] = self.label_to_one_hot(
            sample['bin_label'], 2
        ).squeeze()
        sample['mask'] = torch.FloatTensor(mask)
        return sample
