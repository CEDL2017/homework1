import argparse
import os

import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import Dataset
from model import Model


def _train(path_to_data_dir, path_to_logs_dir, path_to_restore_checkpoint_file):
    batch_size = 16

    dataset = Dataset(path_to_data=path_to_data_dir, mode=Dataset.Mode.TRAIN)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    model, step = Model(), 0
    losses = []

    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
    if path_to_restore_checkpoint_file is not None:
        step = model.load(path_to_restore_checkpoint_file, optimizer=optimizer)
        losses = np.load('logs/losses.npy').tolist()
        losses = list(filter(lambda x: x[0] <= step, losses))
        print('Model restored from file: %s' % path_to_restore_checkpoint_file)
    model.cuda()

    num_steps_to_check_loss = 20
    num_steps_to_snapshot = 1000
    num_steps_to_train = 10000
    should_continue = True

    while should_continue:
        for batch_index, (images, labels) in enumerate(dataloader):
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            logits = model.train().forward(images)

            cross_entropy = Model.loss(logits, labels)
            loss = cross_entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % num_steps_to_check_loss == 0:
                losses.extend([[step, loss.data[0]]])
                print('step = {}, loss = {}'.format(step, loss.data[0]))

            if step % num_steps_to_snapshot == 0 or step % num_steps_to_train == 0:
                np.save(os.path.join(path_to_logs_dir, 'losses.npy'), np.array(losses))
                path_to_checkpoint_file = model.save(path_to_dir=path_to_logs_dir,
                                                     step=step, optimizer=optimizer)
                print('=> Model saved to file: %s' % path_to_checkpoint_file)

            if step % num_steps_to_train == 0:
                should_continue = False
                break


if __name__ == '__main__':
    def main(args):
        path_to_data_dir = args.data_dir
        path_to_logs_dir = args.logs_dir
        path_to_restore_checkpoint_file = args.restore_checkpoint

        if not os.path.exists(path_to_logs_dir):
            os.mkdir(path_to_logs_dir)

        print('Start training')
        _train(path_to_data_dir, path_to_logs_dir, path_to_restore_checkpoint_file)
        print('Done')

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='./data', help='path to data directory')
    parser.add_argument('-l', '--logs_dir', default='./logs', help='path to logs directory')
    parser.add_argument('-r', '--restore_checkpoint', default=None, help='path to restore checkpoint file, e.g., ./logs/model-100.pth')

    main(parser.parse_args())
