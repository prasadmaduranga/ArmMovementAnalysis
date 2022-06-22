import json

import cv2
import mediapipe as mp
import numpy as np
import argparse
import os

BG_COLOR = (192, 192, 192)  # gray

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


def extract_body_landmarks(args):
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose, mp_hands.Hands(
        static_image_mode=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        for root, directories, filenames in os.walk(os.path.join(args.base_dir, args.video_input_dir)):
            for video_file in filenames:

                # skip hidden files
                if video_file[0] == '.':
                    continue

                video_file_name = video_file.split('.')[0]
                video_input_stream = cv2.VideoCapture(os.path.join(args.base_dir, args.video_input_dir, video_file))
                video_output = cv2.VideoWriter(os.path.join(args.base_dir, args.video_output_dir, video_file),
                                      cv2.VideoWriter_fourcc(*'MP4V'),
                                      video_input_stream.get(cv2.CAP_PROP_FPS), (int(video_input_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                                  int(video_input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                landmark_file = os.path.join(args.base_dir, args.landmark_output_dir, video_file_name + '_landmarks.csv')
                video_keypoints = []

                while video_input_stream.isOpened():
                    success, image = video_input_stream.read()
                    frame_keypoints = []

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
                    results_pose = pose.process(image)

                    results_hand = hands.process(image)

                    # Draw the pose annotation on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # body
                    mp_drawing.draw_landmarks(
                        image,
                        results_pose.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                    for data_point in results_pose.pose_landmarks.landmark:
                        frame_keypoints.append({'X': data_point.x,
                                                'Y': data_point.y,
                                                'Z': data_point.z,
                                                'Visibility': data_point.visibility, })

                    # hand
                    if results_hand.multi_hand_landmarks:
                      for hand_landmarks in results_hand.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                        frame_keypoints.append({'X': data_point.x,
                                                'Y': data_point.y,
                                                'Z': data_point.z,
                                                'Visibility': data_point.visibility, })

                    # Flip the image horizontally for a selfie-view display.
                    video_output.write(cv2.flip(image, 1))
                    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
                    video_keypoints.append(frame_keypoints)
                np.savetxt(landmark_file, np.asarray(video_keypoints), delimiter=' ,', fmt='%s')
                # with open(landmark_file, 'w') as f:
                #     json.dump(video_keypoints,landmark_file)
                video_output.release()
                video_input_stream.release()


def main():
    parser = argparse.ArgumentParser(
        description="Extract 3D skeleton landmarks from videos"
    )

    parser.add_argument("--base_dir", type=str,
                        default='../../data',
                        help="Name of the base directory")

    parser.add_argument("--video_input_dir", type=str,
                        default='raw/strokeVideo',
                        help="Name of directory to read video data from")

    parser.add_argument("--landmark_output_dir", type=str,
                        default='processed/landmarks',
                        help="Name of directory to output computed landmarks")

    parser.add_argument("--video_output_dir", type=str,
                        default='processed/annotatedVideo',
                        help="Name of directory to output computed landmarks")

    args = parser.parse_args()

    print(args)

    # compute_angles_from_body_parts(args)
    extract_body_landmarks(args)


if __name__ == "__main__":
    main()
#
# python src/features/extract_3D_skeleton_data.py --base_dir=../../data --video_input_dir=/raw/strokeVideo --landmark_output_dir=/processed/landmarks \
# --video_output_dir=/processed/annotatedVideo

