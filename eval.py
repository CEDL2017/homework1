import argparse

from dataset import Dataset
from evaluator import Evaluator
from model import Model


def _eval(path_to_checkpoint_file, path_to_data_dir, mode):
    model = Model()
    model.load(path_to_checkpoint_file)
    model.cuda()

    print('Evaluate {:s} on dataset with mode `{:s}`:'.format(path_to_checkpoint_file, mode.value))

    accuracy = Evaluator(path_to_data_dir, mode).evaluate(model)

    print('=> accuracy:', accuracy)


if __name__ == '__main__':
    def main(args):
        path_to_checkpoint_file = args.checkpoint
        path_to_data_dir = args.data_dir
        mode = Dataset.Mode.from_string(args.mode)

        print('Start evaluating')
        _eval(path_to_checkpoint_file, path_to_data_dir, mode)
        print('Done')

    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='path to evaluate checkpoint file, e.g. ./logs/model-100.pth')
    parser.add_argument('-d', '--data_dir', default='./data', help='path to data directory')
    parser.add_argument('-m', '--mode', default='test', help='mode of dataset, must be one of `train`, `test`')

    main(parser.parse_args())
