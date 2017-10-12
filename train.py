from __future__ import print_function
import numpy as np
import os
from cntk import load_model
from TransferLearning import *


# define base model location and characteristics
base_folder = '.\\'
base_model_file = os.path.join(base_folder, "ResNet50_ImageNet_CNTK.model")
feature_node_name = "features"
last_hidden_node_name = "z.x"
image_height = 224
image_width = 224
num_channels = 3

# define data location and characteristics
train_image_folder = os.path.join(base_folder, "..", "DataSets", "Train")
test_image_folder = os.path.join(base_folder, "..", "DataSets", "Test")
file_endings = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def create_map_file_from_folder(root_folder, class_mapping, include_unknown=False):
    map_file_name = os.path.join(root_folder, "map.txt")
    lines = []
    for class_id in range(0, len(class_mapping)):
        folder = os.path.join(root_folder, class_mapping[class_id])
        if os.path.exists(folder):
            for entry in os.listdir(folder):
                filename = os.path.join(folder, entry)
                if os.path.isfile(filename) and os.path.splitext(filename)[1] in file_endings:
                    lines.append("{0}\t{1}\n".format(filename, class_id))

    if include_unknown:
        for entry in os.listdir(root_folder):
            filename = os.path.join(root_folder, entry)
            if os.path.isfile(filename) and os.path.splitext(filename)[1] in file_endings:
                lines.append("{0}\t-1\n".format(filename))

    lines.sort()
    with open(map_file_name , 'w') as map_file:
        for line in lines:
            map_file.write(line)

    return map_file_name

def create_class_mapping_from_folder(root_folder):
    classes = []
    for _, directories, _ in os.walk(root_folder):
        for directory in directories:
            classes.append(directory)
    classes.sort()
    return np.asarray(classes)

def format_output_line(img_name, true_class, probs, class_mapping, top_n=3):
    class_probs = np.column_stack((probs, class_mapping)).tolist()
    class_probs.sort(key=lambda x: float(x[0]), reverse=True)
    top_n = min(top_n, len(class_mapping)) if top_n > 0 else len(class_mapping)
    true_class_name = class_mapping[true_class] if true_class >= 0 else 'unknown'
    line = '[{"class": "%s", "predictions": {' % true_class_name
    for i in range(0, top_n):
        line = '%s"%s":%.3f, ' % (line, class_probs[i][1], float(class_probs[i][0]))
    line = '%s}, "image": "%s"}]\n' % (line[:-2], img_name.replace('\\', '/').rsplit('/', 1)[1])
    if true_class_name == class_probs[0][1]:
        error = 0
    else:
        error = 1
    return line, error

def train_and_eval(_base_model_file, _train_image_folder, _test_image_folder, _results_file, _new_model_file, testing = False):
    # check for model and data existence
    if not (os.path.exists(_base_model_file) and os.path.exists(_train_image_folder) and os.path.exists(_test_image_folder)):
        print("Please run 'python install_data_and_model.py' first to get the required data and model.")
        exit(0)

    # get class mapping and map files from train and test image folder
    class_mapping = create_class_mapping_from_folder(_train_image_folder)
    train_map_file_head = os.path.join(base_folder, "map_file_head_train.txt")
    train_map_file_hand = os.path.join(base_folder, "map_file_hand_train.txt")
    test_map_file_head = os.path.join(base_folder, "map_file_head_test.txt")
    test_map_file_hand = os.path.join(base_folder, "map_file_hand_test.txt")

    # train
    trained_model = train_model(_base_model_file, feature_node_name, last_hidden_node_name,
                                image_width, image_height, num_channels,
                                len(class_mapping),
                                train_map_file_head, train_map_file_hand,
                                num_epochs=30, freeze=True)

    #trained_model = load_model(os.path.join(base_folder, "Output", "TransferLearning_ResNet18_Dule.model"))
    if not testing:
        trained_model.save(_new_model_file)
        print("Stored trained model at %s" % tl_model_file)

    # evaluate test images
    with open(_results_file, 'w') as output_file:
        with open(test_map_file_head, "r") as input_file_head, open(test_map_file_hand, 'r') as input_file_hand:
            errors = 0
            count = 0
            for line_head, line_hand in zip(input_file_head, input_file_hand):
                tokens_head = line_head.rstrip().split('\t')
                img_file_head = tokens_head[0]
                tokens_hand = line_hand.rstrip().split('\t')
                img_file_hand = tokens_hand[0]
                true_label = int(tokens_hand[1])
                probs = eval_single_image(trained_model, img_file_head, img_file_hand, image_width, image_height)

                formatted_line, error = format_output_line(img_file_hand, true_label, probs, class_mapping)
                output_file.write(formatted_line)
                errors = errors + error
                count = count + 1
                print(str((1-(errors/count))*100))
            output_file.write('Error rate: '+str((1-(errors/count))*100)+"%")

    print("Done. Wrote output to %s" % _results_file)

if __name__ == '__main__':
    try_set_default_device(gpu(0))

    # You can use the following to inspect the base model and determine the desired node names
    node_outputs = get_node_outputs(load_model(base_model_file))
    for out in node_outputs:
        print("{0} {1}".format(out.name, out.shape))

    results_file = os.path.join(base_folder, "Output", "predictions_ResNet50_Dule.txt")
    new_model_file = os.path.join(base_folder, "Output", "TransferLearning_ResNet50_Dule.model")
    train_and_eval(base_model_file, train_image_folder, test_image_folder, results_file, new_model_file)