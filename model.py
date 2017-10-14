import glob
import os

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional


class Model(nn.Module):
    CHECKPOINT_FILENAME_PATTERN = 'model-{}.pth'

    def __init__(self):
        super().__init__()
        pretrained_net = models.alexnet(pretrained=True)
        self._feature = pretrained_net.features
        self._fc6 = nn.Sequential(
            nn.Linear(256 * 8 * 15, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )
        self._classifier = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 24),
        )

    def forward(self, hand_images, head_images):
        hand_features = self._feature(hand_images)
        hand_features = hand_features.view(-1, 256 * 8 * 15)
        hand_features = self._fc6(hand_features)

        head_feature = self._feature(head_images)
        head_feature = head_feature.view(-1, 256 * 8 * 15)
        head_feature = self._fc6(head_feature)

        features = torch.cat([hand_features, head_feature], dim=1)
        logits = self._classifier(features)
        return logits

    @staticmethod
    def loss(logits, labels):
        cross_entropy = torch.nn.functional.cross_entropy(input=logits, target=labels)
        return cross_entropy

    def save(self, path_to_dir, step, optimizer, maximum=5):
        path_to_models = glob.glob(os.path.join(path_to_dir, Model.CHECKPOINT_FILENAME_PATTERN.format('*')))
        if len(path_to_models) == maximum:
            min_step = min([int(path_to_model.split('/')[-1][6:-4]) for path_to_model in path_to_models])
            path_to_min_step_model = os.path.join(path_to_dir, Model.CHECKPOINT_FILENAME_PATTERN.format(min_step))
            os.remove(path_to_min_step_model)

        checkpoint_filename = os.path.join(path_to_dir, Model.CHECKPOINT_FILENAME_PATTERN.format(step))
        torch.save({
            'step': step,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict()
        }, checkpoint_filename)

        return checkpoint_filename

    def load(self, checkpoint_filename, optimizer=None):
        checkpoint = torch.load(checkpoint_filename)
        step = checkpoint['step']
        self.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        return step


if __name__ == '__main__':
    def main():
        model = Model()

    main()
