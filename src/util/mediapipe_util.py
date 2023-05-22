import math

import cv2
import dataclasses
import matplotlib.pyplot as plt
import numpy as np
import ast
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec, _normalize_color



def get_handedness(landmark_results):
    '''

    :param landmark_results:
    :return: ['Both','Left','Right']
    '''
    if landmark_results.multi_handedness is None:
        return None

    if len(landmark_results.multi_handedness) == 2:
        return 'Both'
    elif len(landmark_results.multi_handedness) == 1:
        return landmark_results.multi_handedness[0].classification[0].label
    else:
        return None

def calculate_angle(v1, v2, rad=True):
    cosval = math.acos(np.round(dotproduct(v1, v2) / (length(v1) * length(v2)), decimals=7))

    if not rad:
        cosval = (180 * cosval) / np.pi

    return cosval

def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))

# return cross product magnitude
def crossproduct(v1, v2):
    result = [v1[1] * v2[2] - v1[2] * v2[1],
              v1[2] * v2[0] - v1[0] * v2[2],
              v1[0] * v2[1] - v1[1] * v2[0]]

    return math.sqrt(sum((v1[0] ** 2, v1[1] ** 2, v1[2] ** 2)))

def length(v):
    return math.sqrt(dotproduct(v, v))

def calculate_distance(v1, v2):
    # d = math.sqrt(sum(((v1[0] - v2[0])*frame_width ** 2, (v1[1] - v2[1])*frame_height ** 2, (v1[2] - v2[2]) ** 2)))
    x_diff = (v1[0] - v2[0])
    y_diff = (v1[1] - v2[1])
    d = math.sqrt(x_diff * x_diff + y_diff * y_diff)
    return d
