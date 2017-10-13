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
                                                       shuffle=False, num_workers=2)

    def evaluate(self, model):
        model.cuda().eval()

        num_hit = 0
        progress_bar = tqdm(total=len(self._dataset))

        for batch_index, (images, labels) in enumerate(self._dataloader):
            images = Variable(images, volatile=True).cuda()
            labels = labels.cuda()

            logits = model.train().forward(images)
            probabilities = torch.nn.functional.softmax(logits)
            predictions = probabilities.data.max(dim=1)[1]

            num_hit += (predictions == labels).sum()

            progress_bar.update(self._batch_size)

        accuracy = num_hit / len(self._dataset)
        return accuracy
