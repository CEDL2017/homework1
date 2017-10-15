import argparse
import os

import numpy as np
from visdom import Visdom


def _visualize(path_to_logs_dir):
    steps, losses = zip(*np.load(os.path.join(path_to_logs_dir, 'losses.npy')))
    losses_length = len(losses)

    avg_losses = np.cumsum(losses) / np.arange(1, len(losses) + 1)

    window_size = max(min(losses_length // 10, 100), 1)
    batch_avg_losses = np.convolve(losses, np.ones(window_size) / window_size, mode='same')
    batch_avg_losses[:window_size - 1] = np.nan
    batch_avg_losses[-(window_size - 1):] = np.nan

    viz = Visdom()
    viz.line(
        X=np.array(steps).reshape(-1, 1).repeat(3, 1),
        Y=np.column_stack([losses, avg_losses, batch_avg_losses]),
        opts=dict(
            legend=['Loss', 'Average Loss', 'Batch Average Loss (Size=%d)' % window_size]
        )
    )


def main(args):
    path_to_logs_dir = args.logs_dir
    _visualize(path_to_logs_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logs_dir', default='./logs', help='path to logs directory')

    main(parser.parse_args())
