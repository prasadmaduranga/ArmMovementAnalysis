import json

import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
from datetime import datetime
from Enumerators import Body, Hand
import pandas as pd
import src.util.base_util as base_util

BG_COLOR = (192, 192, 192)  # gray

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# to activate video leypoint extraction and hand keypoint extraction
video_landmarks = False
image_landmarks = True

video_pose_key_extractor = mp_pose.Pose(
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

video_hand_key_extractor = mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.5)


# extract image keypoints from image list and write to a csv file
def extract_hand_landmarks_from_images(args):
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose, mp_hands.Hands(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.5) as hands:

        image_keypoints = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        landmark_file = os.path.join(args.image_landmark_output_dir, 'image_landmarks_' + timestamp + '.csv')
        img_index = 0

        for root, directories, filenames in os.walk(args.image_input_dir):

            # feature list
            body_keypoint_headings = [e.name for e in Body]
            left_hand_keypoint_headings = ['L_' + e.name for e in Hand]
            right_hand_keypoint_headings = ['R_' + e.name for e in Hand]
            feature_list = ['IMAGE_ID'] + [
                'HANDEDNESS'] + left_hand_keypoint_headings + right_hand_keypoint_headings + body_keypoint_headings

            for image in filenames:

                # Skip hidden files
                if image[0] == '.':
                    continue

                frame_keypoints = []
                image_id = base_util.get_file_name(image, False)

                frame_keypoints.append(img_index)
                img = cv2.imread(os.path.join(root, image))
                results_hand = hands.process(img)
                results_pose = pose.process(img)

                annotated_image = img.copy()

                handedness = check_handedness(results_hand.multi_handedness)

                if (handedness is not None):

                    frame_keypoints.append("'" + handedness + "'")
                    # hand
                    # if the frame is single handed and it's the right hand, empty values will be added to
                    # fill up left hand key points
                    if handedness == 'Right':
                        for hand_landmark in Hand:
                            frame_keypoints.append(None)

                    # if both hands present, first it will give landmark of left hand, then right hand
                    if results_hand.multi_hand_landmarks:
                        for hand_landmarks in results_hand.multi_hand_landmarks:
                            for data_point in hand_landmarks.landmark:
                                frame_keypoints.append({'X': data_point.x,
                                                        'Y': data_point.y,
                                                        'Z': data_point.z,
                                                        'Visibility': data_point.visibility, })

                    # if the frame is single handed and it's the left hand, empty values will be added to
                    # fill up right hand key points
                    if handedness == 'Left':
                        for hand_landmark in Hand:
                            frame_keypoints.append(None)

                    if results_hand.multi_hand_landmarks:
                        for hand_landmarks in results_hand.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                annotated_image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())

                    # body
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        results_pose.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                    for body_landmark in Body:
                        # Avoid adding invisible joints to the keypoint list
                        if results_pose.pose_landmarks.landmark[body_landmark.value].visibility < 0.5:
                            frame_keypoints.append(None)
                            continue

                        frame_keypoints.append({'X': results_pose.pose_landmarks.landmark[body_landmark.value].x,
                                                'Y': results_pose.pose_landmarks.landmark[body_landmark.value].y,
                                                'Z': results_pose.pose_landmarks.landmark[body_landmark.value].z,
                                                'Visibility': results_pose.pose_landmarks.landmark[
                                                    body_landmark.value].visibility, })

                # cv2.imshow('MediaPipe Pose', cv2.flip(annotated_image, 1))
                image_keypoints.append(frame_keypoints)

                cv2.imwrite(args.annotated_image_output_dir + '/annotated_' + image_id + '.png', annotated_image)

                img_index = img_index + 1
                if cv2.waitKey(33) == ord('a'):
                    continue

            np.savetxt(landmark_file, np.asarray(image_keypoints), delimiter=' ,', fmt='%s')
            # df = pd.DataFrame(image_keypoints, columns=feature_list)
            # df.to_csv(landmark_file, encoding='utf-8', index=False)


def extract_hand_landmarks_from_videos(args):
    with mp_pose.Pose(
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose, mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        for root, directories, filenames in os.walk(args.video_input_dir):

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            landmark_file = os.path.join(args.video_landmark_output_dir, 'video_landmarks_' + timestamp + '.csv')

            for video_file in filenames:

                # skip hidden files
                if video_file[0] == '.':
                    continue

                video_id = base_util.get_file_name(video_file, False)
                video_input_stream = cv2.VideoCapture(os.path.join(args.video_input_dir, video_file))

                if os.path.exists(args.annotated_video_output_dir):
                    video_output = cv2.VideoWriter(
                        os.path.join(args.annotated_video_output_dir, video_file),
                        cv2.VideoWriter_fourcc(*'MP4V'),
                        video_input_stream.get(cv2.CAP_PROP_FPS),
                        (int(video_input_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(video_input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))))

                video_keypoints = []

                while video_input_stream.isOpened():
                    success, image = video_input_stream.read()
                    frame_keypoints = []
                    frame_keypoints.append(video_id)

                    if image is None:
                        break
                    if not success:
                        print("Ignoring empty camera frame.")
                        # If loading a video, use 'break' instead of 'continue'.
                        continue

                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    results_hand = hands.process(image)
                    results_pose = pose.process(image)

                    # Draw the pose annotation on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # hand
                    if results_hand.multi_hand_landmarks:
                        for hand_landmarks in results_hand.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())

                            for data_point in hand_landmarks.landmark:
                                frame_keypoints.append({'X': data_point.x,
                                                        'Y': data_point.y,
                                                        'Z': data_point.z,
                                                        'Visibility': data_point.visibility, })

                    mp_drawing.draw_landmarks(
                        image,
                        results_pose.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                    # Flip the image horizontally for a selfie-view display.
                    if os.path.exists(args.annotated_video_output_dir):
                        video_output.write(cv2.flip(image, 1))

                    video_keypoints.append(frame_keypoints)

                np.savetxt(landmark_file, np.asarray(video_keypoints), delimiter=' ,', fmt='%s')

                video_output.release()
                video_input_stream.release()


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


def extract_frame_keypoints(image):
    video_pose_key_extractor
    video_hand_key_extractor

    results_hand = video_hand_key_extractor.process(image)
    results_pose = video_pose_key_extractor.process(image)


def main():
    parser = argparse.ArgumentParser(
        description="Extract 3D skeleton landmarks from videos"
    )

    parser.add_argument("--image_input_dir", type=str,
                        default='../../data/raw/images',
                        help="Name of directory to read video data from")

    parser.add_argument("--image_landmark_output_dir", type=str,
                        default='../../data/processed/landmarks/imageLandmarks',
                        help="Name of directory to read video data from")

    parser.add_argument("--annotated_image_output_dir", type=str,
                        default='../../data/processed/annotatedImage',
                        help="Name of directory to read video data from")

    parser.add_argument("--video_input_dir", type=str,
                        default='../../data/raw/strokeVideo',
                        help="Name of directory to read video data from")

    parser.add_argument("--video_landmark_output_dir", type=str,
                        default='../../data/processed/landmarks/videoLandmarks',
                        help="Name of directory to output computed landmarks")

    parser.add_argument("--annotated_video_output_dir", type=str,
                        default='../../data/processed/annotatedVideo',
                        help="Name of directory to output computed landmarks")

    args = parser.parse_args()

    print(args)

    # compute_angles_from_body_parts(args)

    # Extract body landmarks from video and save to file
    if os.path.exists(args.video_input_dir):
        extract_hand_landmarks_from_videos(args)

    # Extract hand landmarks from images and save to file
    # if os.path.exists(args.image_input_dir):
    #     extract_hand_landmarks_from_images(args)


if __name__ == "__main__":
    main()
#
# python src/features/extract_3D_skeleton_data.py --base_dir=../../data --video_input_dir=/raw/strokeVideo --landmark_output_dir=/processed/landmarks \
# --video_output_dir=/processed/annotatedVideo
