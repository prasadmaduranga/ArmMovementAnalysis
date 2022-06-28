import argparse
import ast
import math
import os
from _datetime import datetime
import pandas as pd
import mediapipe as mp
import numpy as np
from pathlib import Path
from src.features.Enumerators import HAND_ANGLES, HAND_DISTANCE_PAIRS, BODY_ANGLES, BODY_DISTANCE_PAIRS

import src.util.base_util as base_util

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
NUM_HAND_FEATURES = 21


def extract_hand_features(args):
    '''
    Compute Angles and distances of key joints in the hand. Output features are written to a csv file and stored in output_dir
    Go through all keypoint csv files in the in the input dir.
    :param args: input_dir,output_dir
    :return: None
    '''

    for root, directories, filenames in os.walk(args.input_dir):
        for landmark_file in filenames:
            print(landmark_file)
            image_features = []

            # feature list
            left_hand_angle_headings = ['LEFT_' + e.name for e in HAND_ANGLES]
            right_hand_angle_headings = ['RIGHT_' + e.name for e in HAND_ANGLES]
            body_angle_headings = [e.name for e in BODY_ANGLES]
            left_hand_distance_headings = ['LEFT_' + e.name for e in HAND_DISTANCE_PAIRS]
            right_hand_distance_headings = ['RIGHT_' + e.name for e in HAND_DISTANCE_PAIRS]
            body_distance_headings = [e.name for e in BODY_DISTANCE_PAIRS]

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = base_util.get_file_name()
            if not landmark_file.endswith('.csv') or landmark_file[0] == '.':
                continue

            with open(os.path.join(args.input_dir, landmark_file), 'r') as inputfile:
                with open(os.path.join(args.output_dir, 'features_' + Path(landmark_file).stem + '.csv'),
                          'w') as outputfile:
                    frames = inputfile.readlines()
                    feature_list = [
                                       'ID'] + left_hand_angle_headings + right_hand_angle_headings + body_angle_headings + left_hand_distance_headings + right_hand_distance_headings + body_distance_headings

                    for frame in frames:
                        frame_features = []

                        # Extract frame id
                        frame_id = ast.literal_eval(frame)[0]

                        # if no hands detected, frame will not be processed
                        if frame.find(',') < 0 or not (
                                frame.find('Right') > 0 or frame.find('Left') > 0 or frame.find('Both') > 0):
                            frame_features.extend([frame_id])
                            image_features.append(frame_features)
                            continue

                        handedness = get_handedness(frame)

                        # frame keypoint string
                        frame_keypoint_str = frame[frame.find(',') + 1:].replace(']', '').replace('[', '')

                        keypoint_coordinates = read_hand_landmark_coordinates(frame_keypoint_str.strip())

                        # Calculate angles
                        frame_angles = compute_angles(keypoint_coordinates)

                        # Calculate distances
                        frame_distances = compute_distances(keypoint_coordinates)

                        frame_features.extend([frame_id])
                        frame_features.extend(frame_angles)
                        frame_features.extend(frame_distances)

                        image_features.append(frame_features)

                        # features = features[~np.all(features == 0, axis=1)]

                    df = pd.DataFrame(image_features, columns=feature_list)
                    df.to_csv(outputfile, encoding='utf-8', index=False)


def read_hand_landmark_coordinates(frame_keypoints_record):
    frame_keypoints_record = ast.literal_eval(frame_keypoints_record)

    # remove handeddness param from the tuple
    frame_keypoints_record = frame_keypoints_record[1:]

    keypoint_coordinates = []
    try:
        for point in frame_keypoints_record:
            if point is None:
                keypoint_coordinates.append(None)
                continue
            keypoint_coordinates.append((point['X'], point['Y'], point['Z']))
    except:
        print('error reading landmark points')

    return keypoint_coordinates


def get_handedness(frame_keypoints_record):
    frame_keypoints = ast.literal_eval(frame_keypoints_record)

    if frame_keypoints[1].find('Right') > 0:
        return ['Right']
    elif frame_keypoints[1].find('Left') > 0:
        return ['Left']
    elif frame_keypoints[1].find('Both') > 0:
        return ['Left', 'Right']
    return


def compute_angles(keypoint_coordinates, rad=False):
    hand_landmark_coords = keypoint_coordinates[:42]
    body_landmark_coords = keypoint_coordinates[42:]
    angles = np.zeros(36)
    idx = 0
    feature_offset = 0

    # Calculate angles in the hand
    # 16 angles per each hand
    for hand in ['Left', 'Right']:
        if hand == 'Right':
            feature_offset = NUM_HAND_FEATURES

        for angle in HAND_ANGLES:
            try:
                vertex_1_coords = hand_landmark_coords[angle.value[0].value + feature_offset]
                vertex_2_coords = hand_landmark_coords[angle.value[1].value + feature_offset]
                vertex_3_coords = hand_landmark_coords[angle.value[2].value + feature_offset]

                if vertex_1_coords is None or vertex_2_coords is None or vertex_3_coords is None:
                    angles[idx] = None
                    continue

                vertex_1 = np.array(vertex_1_coords)
                vertex_2 = np.array(vertex_2_coords)
                vertex_3 = np.array(vertex_3_coords)

                V1 = vertex_1 - vertex_2
                V2 = vertex_3 - vertex_2
                a = calculate_angle(V1, V2, rad)
                angles[idx] = round(a, 2)
            except:
                angles[idx] = None
            finally:
                idx += 1

    # Get angles extracted from shoulder and elbow coordinates
    # three angles per each hand (shoulder , elbow, wrist)
    for angle in BODY_ANGLES:
        try:

            vertex_1_coords = body_landmark_coords[angle.value[0].value]
            vertex_2_coords = body_landmark_coords[angle.value[1].value]
            vertex_3_coords = body_landmark_coords[angle.value[2].value]

            if  vertex_1_coords is None or vertex_2_coords is None or vertex_3_coords is None:
                angles[idx] = None
                continue

            vertex_1 = np.array(vertex_1_coords)
            vertex_2 = np.array(vertex_2_coords)
            vertex_3 = np.array(vertex_3_coords)

            V1 = vertex_1 - vertex_2
            V2 = vertex_3 - vertex_2
            a = calculate_angle(V1, V2, rad)
            angles[idx] = round(a, 2)
        except:
            angles[idx] = None
        finally:
            idx += 1

    return angles


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


def calculate_angle(v1, v2, rad=True):
    cosval = math.acos(np.round(dotproduct(v1, v2) / (length(v1) * length(v2)), decimals=7))

    if not rad:
        cosval = (180 * cosval) / np.pi

    return cosval

    # if need to get the cloxkwise angle, uncomment below code
    # sinval = -math.degrees(math.asin(length(np.cross(v1,v2)) / (length(v1) * length(v2))))
    #
    # if not rad:
    #     cosval = (180 * cosval) / np.pi
    #
    # if sinval < 0 and rad:
    #     return 2 * math.pi - cosval
    # elif sinval < 0 and not rad:
    #     return 360 - cosval
    # else:
    #     return cosval


def calculate_distance(v1, v2):
    d = math.sqrt(sum(((v1[0] - v2[0]) ** 2, (v1[1] - v2[1]) ** 2, (v1[2] - v2[2]) ** 2)))

    return d


def compute_distances(keypoint_coordinates):
    hand_landmark_coords = keypoint_coordinates[:42]
    body_landmark_coords = keypoint_coordinates[42:]

    distance = np.zeros(20)
    idx = 0

    feature_offset = 0

    # Calculate angles in the hand
    # 16 angles per each hand
    for hand in ['Left', 'Right']:
        if hand == 'Right':
            feature_offset = NUM_HAND_FEATURES

        for hand_keypoint_pair in HAND_DISTANCE_PAIRS:

            try:

                vertex_1_coords = hand_landmark_coords[hand_keypoint_pair.value[0].value  + feature_offset]
                vertex_2_coords = hand_landmark_coords[hand_keypoint_pair.value[1].value + feature_offset]

                if vertex_1_coords is None or vertex_2_coords is None :
                    distance[idx] = None
                    continue

                vertex_1 = np.array(vertex_1_coords)
                vertex_2 = np.array(vertex_2_coords)

                d = calculate_distance(vertex_1, vertex_2)
                distance[idx] = d
            except:
                distance[idx] = None
            finally:
                idx += 1

    # Get angles extracted from shoulder and elbow coordinates
    # three angles per each hand (shoulder , elbow, wrist)
    for body_distance_pair in BODY_DISTANCE_PAIRS:
        try:

            vertex_1_coords = body_landmark_coords[body_distance_pair.value[0].value]
            vertex_2_coords = body_landmark_coords[body_distance_pair.value[1].value]

            if vertex_1_coords is None or vertex_2_coords is None:
                distance[idx] = None
                continue

            vertex_1 = np.array(vertex_1_coords)
            vertex_2 = np.array(vertex_2_coords)


            if (vertex_1 is None or vertex_2 is None):
                distance[idx] = None
                continue

            d = calculate_distance(vertex_1, vertex_2)
            distance[idx] = d
        except:
            distance[idx] = None
        finally:
            idx += 1

    return distance


def main():
    parser = argparse.ArgumentParser(
        description="Compute features from Hough Points to Human Action Recognition"
    )

    # parser.add_argument("--input_dir", type=str,
    #                     default='../../data/processed/landmarks/imageLandmarks',
    #                     help="Name of directory to where input hand landmark points located")

    parser.add_argument("--input_dir", type=str,
                        default='../../data/processed/landmarks/videoLandmarks',
                        help="Name of directory to where input hand landmark points located")

    parser.add_argument("--output_dir", type=str,
                        default='/data/processed/features/video',
                        help="Name of directory to output hand feature output file")

    args = parser.parse_args()

    print(args)

    extract_hand_features(args)


if __name__ == "__main__":
    main()

# --input_dir=../../data/processed/landmarks/videoLandmarks --output_dir=/data/processed/features/video