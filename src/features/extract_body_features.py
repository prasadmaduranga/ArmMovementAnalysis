import json
import math
import time
import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
from datetime import datetime
from Enumerators import Body, Hand, HAND_ANGLES, HAND_DISTANCE_PAIRS, BODY_ANGLES, BODY_DISTANCE_PAIRS
import pandas as pd
import src.util.base_util as base_util

meta_data_path = '../data/sign_language_dataset/meta_data.py'

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

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
df_column_headers = []
left_hand_angle_feature_headings = ['L_' + e.name for e in HAND_ANGLES]
left_hand_distance_feature_headings = ['L_' + e.name for e in HAND_DISTANCE_PAIRS]
right_hand_angle_feature_headings = ['R_' + e.name for e in HAND_ANGLES]
right_hand_distance_feature_headings = ['R_' + e.name for e in HAND_DISTANCE_PAIRS]
body_angle_feature_headings = [e.name for e in BODY_ANGLES]
body_distnace_feature_headings = [e.name for e in BODY_DISTANCE_PAIRS]

column_headers = ['VIDEO_ID', 'USER_ID', 'ITERATION', 'FRAME_SQ_NUMBER',
                  'HANDEDNESS'] + left_hand_angle_feature_headings \
                 + left_hand_distance_feature_headings + right_hand_angle_feature_headings \
                 + right_hand_distance_feature_headings + body_angle_feature_headings + body_distnace_feature_headings + [
                     'SIGN']


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
                    frame_sq_number = 1
                    sign, user_id, iteration = video_id.split('_')
                    frame_width = video_input_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
                    frame_height = video_input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    print('Progress ', (processedFileCount / totalFileCount) * 100)

                    try:
                        while video_input_stream.isOpened():
                            success, image = video_input_stream.read()
                            frame_feature_vector = pd.Series(dtype=object)

                            frame_feature_vector = pd.concat([frame_feature_vector, pd.Series([video_id])])
                            frame_feature_vector = pd.concat([frame_feature_vector, pd.Series([user_id])])
                            frame_feature_vector = pd.concat([frame_feature_vector, pd.Series([iteration])])
                            frame_feature_vector = pd.concat([frame_feature_vector, pd.Series([frame_sq_number])])

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
                                [frame_feature_vector,
                                 extract_frame_features(cv2.flip(image, 1), frame_width, frame_height)])
                            frame_feature_vector = pd.concat(
                                [frame_feature_vector, pd.Series([sign])])
                            frame_feature_list_temp = frame_feature_vector.tolist()

                            try:
                                frame_feature_vector = pd.Series(frame_feature_list_temp, index=column_headers)
                                df = pd.concat([df, frame_feature_vector.to_frame().T])

                            except Exception as e:
                                print(f"Error: {e}")
                                continue

                            # Draw the pose annotation on the image.
                            image.flags.writeable = True
                            frame_sq_number = frame_sq_number + 1
                    except Exception as e:
                        print(f"Error: {e}")
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


def extract_frame_features(image, frame_width, frame_height):
    image_features = pd.Series(dtype=object)

    results_hand = video_hand_key_extractor.process(image)
    results_pose = video_pose_key_extractor.process(image)
    handedness = check_handedness(results_hand.multi_handedness)

    if (handedness is not None):
        image_features = pd.concat([image_features, pd.Series([handedness], index=['HANDEDNESS'])])

        # hand keypoint extraction
        # if the frame is single handed and it's the right hand, empty values will be added to
        # fill up left hand key points
        if handedness == 'Right':
            image_features = pd.concat([image_features, pd.Series([np.nan] * len(HAND_ANGLES))])
            image_features = pd.concat([image_features, pd.Series([np.nan] * len(HAND_DISTANCE_PAIRS))])

        # if both hands present, first it will give landmark of left hand, then right hand
        if results_hand.multi_hand_landmarks:
            for hand_landmarks in results_hand.multi_hand_landmarks:
                image_features = pd.concat([image_features, extract_hand_angles(hand_landmarks)])
                image_features = pd.concat(
                    [image_features, extract_hand_distance_measures(hand_landmarks, frame_width, frame_height)])

        # if the frame is single handed and it's the left hand, empty values will be added to
        # fill up right hand key points
        if handedness == 'Left':
            image_features = pd.concat([image_features, pd.Series([np.nan] * len(HAND_ANGLES))])
            image_features = pd.concat([image_features, pd.Series([np.nan] * len(HAND_DISTANCE_PAIRS))])

    # Body feature extraction
    image_features = pd.concat([image_features, extract_body_angles(results_pose.pose_landmarks)])
    image_features = pd.concat(
        [image_features, extract_body_distance_measures(results_pose.pose_landmarks, frame_width, frame_height)])

    return image_features


# Extract hand angles
def extract_hand_angles(landmarks):
    angle_series = pd.Series(dtype=object)
    for angle in HAND_ANGLES:

        try:
            vertex_1_coords = landmarks.landmark[angle.value[0].value]
            vertex_2_coords = landmarks.landmark[angle.value[1].value]
            vertex_3_coords = landmarks.landmark[angle.value[2].value]

            if vertex_1_coords is None or vertex_2_coords is None or vertex_3_coords is None:
                angle_series = pd.concat([angle_series, pd.Series([np.nan])])
                continue

            vertex_1 = np.array([vertex_1_coords.x, vertex_1_coords.y, vertex_1_coords.z])
            vertex_2 = np.array([vertex_2_coords.x, vertex_2_coords.y, vertex_2_coords.z])
            vertex_3 = np.array([vertex_3_coords.x, vertex_3_coords.y, vertex_3_coords.z])

            V1 = vertex_1 - vertex_2
            V2 = vertex_3 - vertex_2
            a = calculate_angle(V1, V2, False)
            angle_series = pd.concat([angle_series, pd.Series([round(a, 4)])])

        except:
            angle_series = pd.concat([angle_series, pd.Series([np.nan])])

    return angle_series;


# Extract hand distance measures
def extract_hand_distance_measures(landmarks, frame_width, frame_height):
    distance_series = pd.Series(dtype=object)
    for coordinate in HAND_DISTANCE_PAIRS:

        try:
            vertex_1_coords = landmarks.landmark[coordinate.value[0].value]
            vertex_2_coords = landmarks.landmark[coordinate.value[1].value]

            if vertex_1_coords is None or vertex_2_coords is None:
                distance_series = pd.concat([distance_series, pd.Series([np.nan])])
                continue

            vertex_1 = np.array([vertex_1_coords.x, vertex_1_coords.y, vertex_1_coords.z])
            vertex_2 = np.array([vertex_2_coords.x, vertex_2_coords.y, vertex_2_coords.z])

            d = calculate_distance(vertex_1, vertex_2, frame_width, frame_height)
            distance_series = pd.concat([distance_series, pd.Series([round(d, 4)])])
        except:
            distance_series = pd.concat([distance_series, pd.Series([np.nan])])

    return distance_series;


def extract_body_angles(pose_landmarks):
    angle_series = pd.Series(dtype=object)

    for angle in BODY_ANGLES:
        try:

            # if pose_landmarks.landmark[body_landmark.value].visibility < 0.5:
            #     image_features.append(None)
            #     continue
            vertex_1_coords = pose_landmarks.landmark[angle.value[0].value]
            vertex_2_coords = pose_landmarks.landmark[angle.value[1].value]
            vertex_3_coords = pose_landmarks.landmark[angle.value[2].value]

            if vertex_1_coords is None or vertex_2_coords is None or vertex_3_coords is None or \
                    vertex_1_coords.visibility < 0.5 or vertex_2_coords.visibility < 0.5 or vertex_3_coords.visibility < 0.5:
                angle_series = pd.concat([angle_series, pd.Series([np.nan])])
                continue

            vertex_1 = np.array([vertex_1_coords.x, vertex_1_coords.y, vertex_1_coords.z])
            vertex_2 = np.array([vertex_2_coords.x, vertex_2_coords.y, vertex_2_coords.z])
            vertex_3 = np.array([vertex_3_coords.x, vertex_3_coords.y, vertex_3_coords.z])

            V1 = vertex_1 - vertex_2
            V2 = vertex_3 - vertex_2
            a = calculate_angle(V1, V2, False)
            angle_series = pd.concat([angle_series, pd.Series([round(a, 4)])])
        except:
            angle_series = pd.concat([angle_series, pd.Series([np.nan])])

    return angle_series;


def extract_body_distance_measures(pose_landmarks, frame_width, frame_height):
    distance_series = pd.Series(dtype=object)
    for coordinate in BODY_DISTANCE_PAIRS:

        try:
            vertex_1_coords = pose_landmarks.landmark[coordinate.value[0].value]
            vertex_2_coords = pose_landmarks.landmark[coordinate.value[1].value]

            if vertex_1_coords is None or vertex_2_coords is None or vertex_1_coords.visibility < 0.5 or \
                    vertex_2_coords.visibility < 0.5:
                distance_series.concat(pd.Series([np.nan]))
                continue

            vertex_1 = np.array([vertex_1_coords.x, vertex_1_coords.y, vertex_1_coords.z])
            vertex_2 = np.array([vertex_2_coords.x, vertex_2_coords.y, vertex_2_coords.z])

            d = calculate_distance(vertex_1, vertex_2, frame_width, frame_height)
            distance_series = pd.concat([distance_series, pd.Series([round(d, 4)])])

        except:
            distance_series = pd.concat([distance_series, pd.Series([np.nan])])

    return distance_series;


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


def calculate_distance(v1, v2, frame_width, frame_height):
    # d = math.sqrt(sum(((v1[0] - v2[0])*frame_width ** 2, (v1[1] - v2[1])*frame_height ** 2, (v1[2] - v2[2]) ** 2)))
    x_diff = (v1[0] - v2[0]) * frame_width
    y_diff = (v1[1] - v2[1]) * frame_height
    d = math.sqrt(x_diff * x_diff + y_diff * y_diff)
    return d


def main():
    parser = argparse.ArgumentParser(
        description="Extract 3D skeleton landmarks from videos"
    )

    parser.add_argument("--video_input_dir", type=str,
                        default='../../data/raw/Sign_Language_Data/set0',
                        help="Name of directory to read video data from")

    parser.add_argument("--output_file", type=str,
                        default='../../data/processed/video_body_features.csv',
                        help="Name of directory to output extracted feature vector")

    args = parser.parse_args()

    print(args)
    start_time = time.time()
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
