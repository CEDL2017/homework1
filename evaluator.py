import torch.nn.functional
import torch.utils.data
from torch.autograd import Variable
from tqdm import *

from dataset import Dataset


class Evaluator(object):
    def __init__(self, path_to_data_dir, mode):
        self._batch_size = 128
        self._dataset = Dataset(path_to_data_dir, mode)
        self._dataloader = torch.utils.data.DataLoader(self._dataset, batch_size=self._batch_size,
                                                       shuffle=False, num_workers=8)

    def evaluate(self, model):
        model.cuda()

        num_hits = 0
        progress_bar = tqdm(total=len(self._dataset))

        for batch_index, (head_images, hand_images, fa_labels, ges_labels, obj_labels) in enumerate(self._dataloader):
            hand_images = Variable(hand_images, volatile=True).cuda()
            head_images = Variable(head_images, volatile=True).cuda()
            obj_labels = obj_labels.cuda()

            _, _, obj_logits = model.eval().forward(hand_images, head_images)
            obj_probabilities = torch.nn.functional.softmax(obj_logits)
            obj_predictions = obj_probabilities.data.max(dim=1)[1]

            num_hits += (obj_predictions == obj_labels).sum()

            progress_bar.update(len(obj_labels))

        accuracy = num_hits / len(self._dataset)
        return accuracy
