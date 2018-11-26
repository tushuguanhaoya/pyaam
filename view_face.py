#!/usr/bin/env python
# coding: utf-8

from __future__ import division

import sys
import cv2
import argparse
from pyaam.draw import draw_muct_shape
from pyaam.tracker import FaceTracker
from pyaam.detector import FaceDetector

import glob
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', choices=['tracker', 'detector'], help='model name')
    parser.add_argument('--detector', default='data/detector.npz', help='face detector filename')
    return parser.parse_args()



def view_face_tracker():
    tracker = FaceTracker()
    while True:
        img = cv2.imread('jpg/i000qd-fn.jpg', cv2.IMREAD_COLOR)
        tracker.track(img)
        draw_muct_shape(img, tracker.points)
        cv2.imshow('face tracker', img)
        key = cv2.waitKey(10)
        if key == 27:
            break
        elif key == ord('r'):
            tracker.reset()



def view_face_detector(detector_fn):
    detector = FaceDetector.load(detector_fn)

    file_list = glob.glob('jpg/*.jpg')

    index = 0


    while True:

            img = cv2.imread(file_list[index], cv2.IMREAD_COLOR)
            p = detector.detect(img)


            try:
                draw_muct_shape(img, p)
                cv2.imshow('face detector', img)
            except:
                continue



            if cv2.waitKey(0) == 27:
                break
            elif cv2.waitKey(0) == ord('k'):
                index += 1



if __name__ == '__main__':
    args = parse_args()


    if args.model == 'detector':
        view_face_detector(args.detector)

    elif args.model == 'tracker':
        view_face_tracker()
