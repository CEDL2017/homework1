import glob
import os
import random
import re
from collections import defaultdict
from enum import Enum

import numpy as np
from torch.utils.data import dataset
from torchvision import transforms


class Dataset(dataset.Dataset):

    def __init__(self, path_to_data, mode):
        super().__init__()

        assert isinstance(mode, Dataset.Mode)

        self._mode = mode
        self._path_to_hand_images = []
        for hand in ['Rhand', 'Lhand']:
            self._path_to_hand_images.extend(
                glob.glob('{:s}/frames/{:s}/*/*/{:s}/*.png'.format(path_to_data, mode.value, hand)))
        self._labels = defaultdict(str)

        for environment in ['house', 'lab', 'office']:
            if mode == Dataset.Mode.TRAIN:
                numbers = ['1', '2', '3'] if not environment == 'lab' else ['1', '2', '3', '4']
            else:
                numbers = ['4', '5', '6'] if not environment == 'lab' else ['5', '6', '7', '8']

            for side in ['left', 'right']:
                for num in numbers:
                    self._labels['{:s}/obj/{:s}/{:s}'.format(environment, side, num)] = np.load(
                        os.path.join(path_to_data, 'labels', environment, 'obj_{:s}{:s}.npy'.format(side, num)))

    def __len__(self):
        return len(self._path_to_hand_images)

    def __getitem__(self, index):
        path_to_hand_image = self._path_to_hand_images[index]
        path_to_head_image = re.sub('[L|R]hand', 'head', path_to_hand_image)

        hand_image = transforms.Image.open(path_to_hand_image)
        head_image = transforms.Image.open(path_to_head_image)

        transform = transforms.Compose([
            transforms.Scale(256),
            transforms.RandomCrop(224) if self._mode == Dataset.Mode.TRAIN else transforms.CenterCrop(224),
            Dataset.RandomHorizontalFlip(0.5 if self._mode == Dataset.Mode.TRAIN else 0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        hand_image = transform(hand_image)
        head_image = transform(head_image)

        path_to_hand_image_components = path_to_hand_image.split('/')
        environment = path_to_hand_image_components[-4]

        num = int(path_to_hand_image_components[-3])
        if self._mode == Dataset.Mode.TEST:
            num += 3
            if environment == 'lab':
                num += 1

        hand = path_to_hand_image_components[-2]
        assert hand in ['Lhand', 'Rhand']
        side = 'left' if hand == 'Lhand' else 'right'

        image_index = int(re.match('.*?(\d+)\.png', path_to_hand_image_components[-1]).group(1)) - 1

        label = self._labels['{:s}/obj/{:s}/{:d}'.format(environment, side, num)][image_index]
        label = int(label)

        return hand_image, head_image, label

    class Mode(Enum):
        TRAIN = 'train'
        TEST = 'test'

        @staticmethod
        def from_string(s):
            if s == Dataset.Mode.TRAIN.value:
                return Dataset.Mode.TRAIN
            elif s == Dataset.Mode.TEST.value:
                return Dataset.Mode.TEST
            else:
                raise ValueError()

    class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
        def __init__(self, prob=0.5):
            super().__init__()
            self._prob = prob

        def __call__(self, img):
            if random.random() < self._prob:
                return img.transpose(transforms.Image.FLIP_LEFT_RIGHT)
            return img

if __name__ == '__main__':
    def main():
        import matplotlib.pyplot as plt

        dataset = Dataset('./data', Dataset.Mode.TEST)
        indices = np.random.choice(len(dataset), 6, replace=False)

        plt.figure()
        for i, index in enumerate(indices):
            head_image, hand_image, label = dataset[index]
            plt.subplot(3, 2, i + 1)
            plt.imshow(head_image.permute(1, 2, 0).cpu().numpy())
            print('label = {:d}'.format(label))
        plt.show()

    main()
