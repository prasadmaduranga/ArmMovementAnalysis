import json
import math
import time
import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
from datetime import datetime
from Enumerators import Body, Hand, HAND_ANGLES, HAND_DISTANCE_PAIRS, BODY_ANGLES, BODY_DISTANCE_PAIRS, Upper_Body
import pandas as pd
import src.util.base_util as base_util

# This script extract following features from the video file
# ['VIDEO_ID', 'USER_ID', 'ITERATION', 'FRAME_SQ_NUMBER', 'TIMESTAMP', 'CURRENT_POS_AVI_RATIO', 'HANDEDNESS']
# 3D coordinates of the each joint of the arm + speed of each joint (speed each averaged over 10 consecutive frames)
# for each arm nodes are numbered from 1 to 18 (2 hands x 21 nodes x 4 features)
# 3d joints for shoulder , elbow and wrist joints + averaged speed for each joint (6 nodes)
# upper body : (6 jonts x 4 features)
# Finally the ['SIGN']
# Altogether 200 features

meta_data_path = '../data/sign_language_dataset/meta_data.py'

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
fps = 30

video_pose_key_extractor = mp_pose.Pose(
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

video_hand_key_extractor = mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.5)

column_headers = []
# define column headers
left_hand_joints = ['L_' + e.name for e in HAND_ANGLES]
right_hand_joints = ['R_' + e.name for e in HAND_DISTANCE_PAIRS]
left_body_joint = [e.name for e in BODY_ANGLES]
body_distnace_feature_headings = [e.name for e in BODY_DISTANCE_PAIRS]

column_headers = ['VIDEO_ID', 'USER_ID', 'ITERATION', 'FRAME_SQ_NUMBER', 'TIMESTAMP', 'CURRENT_POS_AVI_RATIO',
                  'HANDEDNESS']

for hand in ['L_', 'R_']:
    for e in Hand:
        for node_feature in ['_X', '_Y', '_Z', '_SPEED']:
            column_headers.append(str(hand) + str(e.name) + str(node_feature))

for e in Upper_Body:
    for node_feature in ['_X', '_Y', '_Z', '_SPEED']:
        column_headers.append(str(e.name) + str(node_feature))

column_headers.append('SIGN')


def extract_hand_landmarks_from_videos(args):
    df = pd.DataFrame(columns=column_headers)
    output_file = os.path.join(args.output_file)

    with mp_pose.Pose(
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose, mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        for root, directories, filenames in os.walk(args.video_input_dir):

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            totalFileCount = len(filenames)
            processedFileCount = 0

            for video_file in filenames:

                if video_file.endswith(".mp4"):

                    video_id = base_util.get_file_name(video_file, False)
                    video_input_stream = cv2.VideoCapture(os.path.join(args.video_input_dir, video_file))
                    video_input_stream.set(cv2.CAP_PROP_FPS, fps)
                    frame_sq_number = 1
                    sign, user_id, iteration = video_id.split('_')
                    print('Progress ', (processedFileCount / totalFileCount) * 100)
                    frame_width = video_input_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
                    frame_height = video_input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    total_frames = video_input_stream.get(cv2.CAP_PROP_FRAME_COUNT)

                    try:
                        while video_input_stream.isOpened():
                            success, image = video_input_stream.read()
                            frame_feature_vector = pd.Series(dtype=object)
                            current_frame = video_input_stream.get(cv2.CAP_PROP_POS_FRAMES)
                            frame_feature_vector = pd.concat([frame_feature_vector,pd.Series([video_id])])
                            frame_feature_vector = pd.concat([frame_feature_vector,pd.Series([user_id])])
                            frame_feature_vector = pd.concat([frame_feature_vector,pd.Series([iteration])])
                            frame_feature_vector = pd.concat([frame_feature_vector, pd.Series([frame_sq_number])])
                            frame_feature_vector = pd.concat([frame_feature_vector, pd.Series([video_input_stream.get(cv2.CAP_PROP_POS_MSEC)])])
                            frame_feature_vector = pd.concat([frame_feature_vector, pd.Series([round(current_frame / total_frames, 4)])])

                            if image is None:
                                break
                            if not success:
                                print("Ignoring empty camera frame.")
                                # If loading a video, use 'break' instead of 'continue'.
                                continue

                            image.flags.writeable = False
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                            # Extract frame keypoints
                            frame_feature_vector = pd.concat(
                                [frame_feature_vector,extract_frame_features(cv2.flip(image, 1)) ])
                            frame_feature_vector = pd.concat(
                                [frame_feature_vector, pd.Series([sign])])

                            frame_feature_list_temp = frame_feature_vector.tolist()

                            try:
                                frame_feature_vector = pd.Series(frame_feature_list_temp, index=column_headers)
                                frame_feature_vector = calculate_speeds(df, frame_feature_vector, frame_width,
                                                                        frame_height)
                                df = pd.concat([df, frame_feature_vector.to_frame().T])
                            except Exception as e:
                                print(f"Error: {e}")
                                continue

                            # Draw the pose annotation on the image.
                            image.flags.writeable = True
                            frame_sq_number = frame_sq_number + 1
                    except Exception as e:
                        print(f"Error: {e} in file {video_file}")
                        continue

                    processedFileCount = processedFileCount + 1
                    video_input_stream.release()
            df.to_csv(output_file, index=False)


# check handedness of the result landmark set
def check_handedness(multi_handedness):
    if multi_handedness is None:
        return None

    if len(multi_handedness) == 2:
        return 'Both'
    elif len(multi_handedness) == 1:
        return multi_handedness[0].classification[0].label
    else:
        return None


def extract_frame_features(image):
    image_features = pd.Series(dtype=object)

    results_hand = video_hand_key_extractor.process(image)
    results_pose = video_pose_key_extractor.process(image)
    handedness = check_handedness(results_hand.multi_handedness)

    if (handedness is not None):
        image_features = pd.concat([image_features,pd.Series([handedness], index=['HANDEDNESS'])])

        # hand keypoint extraction
        # if the frame is single handed and it's the right hand, empty values will be added to
        # fill up left hand key points
        if handedness == 'Right':
            image_features = pd.concat([image_features, pd.Series([np.nan] * len(Hand) * 4)])


        # if both hands present, first it will give landmark of left hand, then right hand
        if results_hand.multi_hand_landmarks:
            for hand_landmarks in results_hand.multi_hand_landmarks:
                image_features = pd.concat([image_features,extract_hand_coordinates(hand_landmarks)])

        # if the frame is single handed and it's the left hand, empty values will be added to
        # fill up right hand key points
        if handedness == 'Left':
            image_features = pd.concat([image_features, pd.Series([np.nan] * len(Hand) * 4)])

    # Body feature extraction
    image_features = pd.concat([image_features, extract_body_coordinates(results_pose.pose_landmarks)])

    return image_features


def extract_hand_coordinates(landmarks):
    coords_series = pd.Series(dtype=object)
    for joint in Hand:

        try:
            joint_coords = landmarks.landmark[joint.value]

            if joint_coords is None:
                coords_series = pd.concat([coords_series,pd.Series([np.nan] * 4)])
            else:
                coords_series = pd.concat([coords_series,pd.Series([round(joint_coords.x, 4), round(joint_coords.y, 4), round(joint_coords.z, 6), np.nan]) ])

        except:
            coords_series = pd.concat([coords_series, pd.Series([np.nan] * 4)])

    return coords_series;


def extract_body_coordinates(landmarks):
    coords_series = pd.Series(dtype=object)
    for joint in Upper_Body:

        try:
            joint_coords = landmarks.landmark[joint.value]

            if joint_coords is None or joint_coords.visibility < 0.5:
                coords_series = pd.concat([coords_series,pd.Series([np.nan] * 4)])
                # coords_series = coords_series.append(pd.Series([np.nan] * 4))
            else:
                coords_series = pd.concat([coords_series, pd.Series([joint_coords.x, joint_coords.y, joint_coords.z, np.nan])])
                # coords_series = coords_series.append(
                #     pd.Series([joint_coords.x, joint_coords.y, joint_coords.z, np.nan]))

        except:
            coords_series = pd.concat([coords_series, pd.Series([np.nan] * 4)])

            # coords_series = coords_series.append(pd.Series([np.nan] * 4))

    return coords_series;


def calculate_speeds(df, frame_feature_vector, frame_width, frame_height):
    # when calculating the speed , two frames which are apart by 5 seq numbers are considered
    frame_seq_number = frame_feature_vector['FRAME_SQ_NUMBER']

    if frame_seq_number == 1:
        return frame_feature_vector

    video_id = frame_feature_vector['VIDEO_ID']
    prev_frame_seq_number = max(1, frame_seq_number - 5)

    prev_feature_vector = df.query('VIDEO_ID == @video_id & FRAME_SQ_NUMBER == @prev_frame_seq_number')

    # handle the null case

    # handle
    for hand in ['L_', 'R_']:
        for e in Hand:
            node_id = hand + e.name
            speed_col = node_id + '_SPEED'

            try:
                x_diff = abs(frame_feature_vector[node_id + '_X'] - prev_feature_vector[node_id + '_X']) * frame_width
                y_diff = abs(frame_feature_vector[node_id + '_Y'] - prev_feature_vector[node_id + '_Y']) * frame_height
                displacement = math.sqrt(x_diff.iloc[0] * x_diff.iloc[0] + y_diff.iloc[0] * y_diff.iloc[0])
                time_gap = frame_feature_vector['TIMESTAMP'] - prev_feature_vector['TIMESTAMP']

                speed = displacement / abs(time_gap.iloc[0])
                frame_feature_vector[speed_col] = round(speed, 4)
            except Exception as e:
                frame_feature_vector[speed_col] = np.nan

    for e in Upper_Body:
        node_id = e.name
        speed_col = node_id + '_SPEED'

        try:
            x_diff = abs(frame_feature_vector[node_id + '_X'] - prev_feature_vector[node_id + '_X']) * frame_width
            y_diff = abs(frame_feature_vector[node_id + '_Y'] - prev_feature_vector[node_id + '_Y']) * frame_height
            displacement = math.sqrt(x_diff.iloc[0] * x_diff.iloc[0] + y_diff.iloc[0] * y_diff.iloc[0])
            time_gap = frame_feature_vector['TIMESTAMP'] - prev_feature_vector['TIMESTAMP']

            speed = displacement / abs(time_gap.iloc[0])
            frame_feature_vector[speed_col] = round(speed, 4)
        except Exception as e:
            frame_feature_vector[speed_col] = np.nan

    return frame_feature_vector


def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


# return cross product magnitude
def crossproduct(v1, v2):
    result = [v1[1] * v2[2] - v1[2] * v2[1],
              v1[2] * v2[0] - v1[0] * v2[2],
              v1[0] * v2[1] - v1[1] * v2[0]]

    return math.sqrt(sum((v1[0] ** 2, v1[1] ** 2, v1[2] ** 2)))


def calculate_distance(v1, v2):
    d = math.sqrt(sum(((v1[0] - v2[0]) ** 2, (v1[1] - v2[1]) ** 2, (v1[2] - v2[2]) ** 2)))

    return d


def length(v):
    return math.sqrt(dotproduct(v, v))


def main():
    parser = argparse.ArgumentParser(
        description="Extract 3D skeleton landmarks from videos"
    )

    parser.add_argument("--video_input_dir", type=str,
                        default='../../data/raw/Sign_Language_Data/All',
                        help="Name of directory to read video data from")

    parser.add_argument("--output_file", type=str,
                        default='../../data/processed/video_graph_features_all_right_hand_signs.csv',
                        help="Name of directory to output extracted feature vector")

    args = parser.parse_args()

    print(args)
    start_time = time.time()
    # code to be timed

    # Extract body landmarks from video and save to file
    if os.path.exists(args.video_input_dir):
        extract_hand_landmarks_from_videos(args)
    end_time = time.time()

    print("Execution time: ", (end_time - start_time))

if __name__ == "__main__":
    main()
#
# python src/features/extract_3D_skeleton_data.py --base_dir=../../data --video_input_dir=/raw/strokeVideo --landmark_output_dir=/processed/landmarks \
# --video_output_dir=/processed/annotatedVideo

# --image_input_dir=../../data/raw/images --image_landmark_output_dir=../../data/processed/landmarks/imageLandmarks
