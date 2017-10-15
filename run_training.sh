TRAIN_DIR=/home/johnson/Desktop/CEDL/homework1/models/train_4
DATASET_DIR=/home/johnson/Desktop/CEDL/homework1/data/reformed_train
CHECKPOINT_PATH=/home/johnson/Desktop/CEDL/homework1/models/inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt
SLIM_DIR=/home/johnson/Desktop/CEDL/homework1/workspace

python ${SLIM_DIR}/models/research/slim/train_image_classifier_CEDLhw1.py \
        --train_dir=${TRAIN_DIR} \
        --dataset_dir=${DATASET_DIR} \
        --dataset_name=CEDLhw1 \
        --dataset_split_name=train \
        --model_name=inception_resnet_v2 \
        --checkpoint_path=${CHECKPOINT_PATH} \
        --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
        --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
        --batch_size=128 \
        --preprocessing_name=inception_resnet_v2 \
        --train_image_size=224
