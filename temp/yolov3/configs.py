#==========================================================================
#
#   Author      : NMD
#   Github      : https://github.com/manhdung20112000/
#   Email       : manhdung20112000@gmail.com
#   File        : configs.py
#   Created on  : 2021-2-25
#   Description : yolov3 configuration file
#
#==========================================================================

#YOLO option
YOLO_TYPE                   = "yolov3" #or yolov4 (not support)
YOLO_FRAMEWORK              = "tf" #or tfr (not support)
YOLO_TRT_QUANTIZE_MODE      = "INT8" # INT8, FP16, FP32
YOLO_SAVE_OUTPUT            = False #True to save it
YOLO_STRIDES                = [8, 16, 32]
YOLO_IOU_LOSS_THRESH        = 0.5
YOLO_ANCHOR_PER_SCALE       = 3
YOLO_MAX_BBOX_PER_SCALE     = 100
YOLO_INPUT_SIZE             = 320
if YOLO_TYPE                == "yolov4":
    YOLO_ANCHORS            = [[[12,  16], [19,   36], [40,   28]],
                               [[36,  75], [76,   55], [72,  146]],
                               [[142,110], [192, 243], [459, 401]]]
if YOLO_TYPE                == "yolov3":
    YOLO_ANCHORS            = [[[10,  13], [16,   30], [33,   23]],
                               [[30,  61], [62,   45], [59,  119]],
                               [[116, 90], [156, 198], [373, 326]]]
#path to file
YOLO_CLASSES                = "model_data/yolov3_custom_names.txt"
YOLO_V3_WEIGHTS             = "model/yolov3_custom.weights"
YOLO_MODEL_NAMES            = "yolov3_custom"