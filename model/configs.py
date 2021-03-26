#==========================================================================
#
#   Author      : NMD
#   Github      : https://github.com/manhdung20112000/
#   Email       : manhdung20112000@gmail.com
#   File        : configs.py
#   Created on  : 2021-3-22
#   Description : config of the model
#
#==========================================================================

# general option
VERSION                     = 1.1
INPUT_SIZE                  = 224
DATA_PATH                   = 'data/'
TEST_DATA_PATH              = 'data/test'
TRAIN_DATA_PATH             = 'data/train'
SAVE_PATH                   = 'runs'
SAVE_WEIGHT                 = True

# dataset option
RANDOM_SEED                 = 42
NUMBER_K_FOLD               = 3
SPLIT_DATASET               = False

# classifier option
CLASSIFIER_BATCH_SIZE       = 64
CLASSIFIER_EPOCHS           = 100
CLASSIFIER_LEARNING_RATE    = 1e-3

# VGG-16 config
NUM_CLASSES                 = 2
CLASSES                     = ['digital', 'analog'] # none of these class is other
VGG_LAYER_1                 = [ [3,64],
                                [64,64], 
                                [3,3], 
                                [1,1], 
                                2, 2]
VGG_LAYER_2                 = [ [64,128],
                                [128,128], 
                                [3,3], 
                                [1,1], 
                                2, 2]
VGG_LAYER_3                 = [ [128,256,256], 
                                [256,256,256], 
                                [3,3,3], 
                                [1,1,1], 
                                2, 2]
VGG_LAYER_4                 = [ [256,512,512], 
                                [512,512,512], 
                                [3,3,3], 
                                [1,1,1], 
                                2, 2]
VGG_LAYER_5                 = [ [512,512,512], 
                                [512,512,512], 
                                [3,3,3], 
                                [1,1,1], 
                                2, 2]
VGG_FC_1                    = [7*7*512, 4096]
VGG_FC_2                    = [4096, 4096]
VGG_FC_3                    = [4096, NUM_CLASSES]
