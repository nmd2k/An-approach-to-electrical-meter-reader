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
VERSION                     = 0.1
INPUT_SIZE                  = 32
DATA_PATH                   = 'data/'
TEST_DATA_PATH              = 'data/test'
TRAIN_DATA_PATH             = 'data/train'
SAVE_PATH                   = 'runs/'

# classifier option
CLASSIFIER_BATCH_SIZE       = 64
CLASSIFIER_EPOCHS           = 100
CLASSIFIER_CLASSES          = ['digital', 'analog'] # none of these class is other
CLASSIFIER_LEARNING_RATE    = 1e-3

# dataset option
