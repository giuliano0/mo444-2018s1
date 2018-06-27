# source env.sh, do not execute

export PYTHONPATH=/home/giulianorp2010/tf-models/research:/home/giulianorp2010/tf-models/research/slim

export PIPECFG_PATH=/home/giulianorp2010/mo444/lab5/tf_model/models/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config
export TRAIN_DIR=/home/giulianorp2010/mo444/lab5/tf_model/models/ssd_mobilenet_v2_coco_2018_03_29/train
export EVAL_DIR=/home/giulianorp2010/mo444/lab5/tf_model/models/ssd_mobilenet_v2_coco_2018_03_29/eval

# These are used like so
# For training:
# $ python object_detection train.py --logtostderr --pipeline_config_path=${PIPECFG_PATH} --train_dir=${TRAIN_DIR}
#
# For validation:
# $ python object_detection eval.py --logtostderr --pipeline_config_path=${PIPECFG_PATH} --eval_dir=${EVAL_DIR} --checkpoint_dir=${TRAIN_DIR}

