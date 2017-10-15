TRAIN_DIR=/home/johnson/Desktop/CEDL/homework1/models/train_3
CHECKPOINT_FILE=${TRAIN_DIR}/model.ckpt-9701
DATASET_DIR=/home/johnson/Desktop/CEDL/homework1/data/reformed_test
SLIM_DIR=/home/johnson/Desktop/CEDL/homework1/workspace
RESULTS_DIR=/home/johnson/Desktop/CEDL/homework1/results/train_1

python ${SLIM_DIR}/models/research/slim/eval_image_classifier_CEDLhw1.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=CEDLhw1 \
    --dataset_split_name=test \
    --model_name=inception_resnet_v2 \
    --eval_dir=${RESULTS_DIR} \
    --preprocessing_name=inception_resnet_v2
