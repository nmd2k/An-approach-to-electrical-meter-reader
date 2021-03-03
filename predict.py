#==========================================================================
#
#   Author      : NMD
#   Github      : https://github.com/manhdung20112000/
#   Email       : manhdung20112000@gmail.com
#   File        : predict.py
#   Created on  : 2021-2-25
#   Description : main function of prediction process
#
#==========================================================================

import os
from yolov3.configs import YOLO_INPUT_SIZE
import cv2
from yolov3.utils import load_model, predict_image, predict_realtime, predict_video

IMAGE_PATH = './sample_test/test.jpg'
VIDEO_PATH = './sample_test/test.mp4'

def main():
    model = load_model()
    image_pred = predict_image(
        model=model, 
        image_path=IMAGE_PATH, 
        output_path='./samepl_test/test_pred.jpg',
        show=True,
    )
    # predict_video(model=model, input=IMAGE_PATH, output='./samepl_test/test_pred.jpg', show=True)
    # predict_realtime(model=model, input=IMAGE_PATH, output='./samepl_test/test_pred.jpg', show=True)
    pass

if __name__ == '__main__':
    main()
