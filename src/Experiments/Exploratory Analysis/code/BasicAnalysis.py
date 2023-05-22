import time

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd

from src.features.Enumerators import HAND_JOINTS, Hand
from src.util.mediapipe_util import get_handedness, calculate_distance
from matplotlib.animation import FuncAnimation
import seaborn as sns
import pandas as df

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

SPEED_FRAME_OFFSET = 1
MAX_SPEED_THRESHOLD = 1500
hand_joint_tracking_coordinates = {enum.name: [] for enum in HAND_JOINTS}
hand_joint_tracking_speed = {enum.name: [] for enum in HAND_JOINTS}
hand_joint_traversed_distance = {enum.name: [] for enum in HAND_JOINTS}

elapsed_time_points = []

# tracking_joint_list = ['L_INDEX_FINGER_TIP','L_PINKY_TIP','L_WRIST']
tracking_joint_list = ['L_INDEX_FINGER_TIP', 'L_WRIST']

# For webcam input:
cap = cv2.VideoCapture(0)
frame_seq_number = 0

with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    start_time = time.time()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        results_hand = hands.process(image)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        current_time = time.time()

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        handedness = get_handedness(results_hand)

        if (handedness is not None):

            hands_list = []
            # hand keypoint extraction
            # if the frame is single handed and it's the right hand, empty values will be added to
            # fill up left hand key points
            if handedness == 'Right':
                hands_list.append('R_')
                for hand_joint in Hand:
                    hand_joint_tracking_coordinates['L_' + hand_joint.name].append(None)

            if handedness == 'Left':
                hands_list.append('L_')
                for hand_joint in Hand:
                    hand_joint_tracking_coordinates['R_' + hand_joint.name].append(None)

            if handedness == 'Both':
                hands_list = ['R_', 'L_']
            # Right,Left
            # if both hands present, first it will give landmark of left hand, then right hand
            i = 0
            if results_hand.multi_hand_landmarks:
                for hand_landmarks in results_hand.multi_hand_landmarks:
                    for data_point, hand_joint in zip(hand_landmarks.landmark, Hand):
                        hand_joint_tracking_coordinates[hands_list[i] + hand_joint.name].append(
                            (data_point.x, data_point.y))
                    i += 1
        else:
            for hand_joint in tracking_joint_list:
                hand_joint_tracking_coordinates[hand_joint].append(None)

        elapsed_time_points.append(current_time - start_time)
        # Flip the image horizontally for a selfie-view display.
        i = 0
        for joint in tracking_joint_list:
            for i in range(len(hand_joint_tracking_coordinates[joint]) - 1):
                if hand_joint_tracking_coordinates[joint][i] is not None and hand_joint_tracking_coordinates[joint][
                    i + 1] is not None:
                    point_1 = (round(hand_joint_tracking_coordinates[joint][i][0] * frame_width),
                               round(hand_joint_tracking_coordinates[joint][i][1] * frame_height))
                    point_2 = (round(hand_joint_tracking_coordinates[joint][i + 1][0] * frame_width),
                               round(hand_joint_tracking_coordinates[joint][i + 1][1] * frame_height))
                    cv2.line(image, point_1, point_2, (255, 0, 0), 2)

        # calculate speed of selected joints and update hand_joint_tracking_speed list
        if handedness is not None:
            for joint in tracking_joint_list:
                speed = 0
                delta_distance = 0
                if frame_seq_number < SPEED_FRAME_OFFSET:
                    hand_joint_tracking_speed[joint].append(0)
                    hand_joint_traversed_distance[joint].append(0)
                    continue
                elif (hand_joint_tracking_coordinates[joint][frame_seq_number] is None) or (
                        hand_joint_tracking_coordinates[joint][frame_seq_number - SPEED_FRAME_OFFSET] is None):
                    speed = 0
                    delta_distance = 0
                else:
                    point_1 = (round(hand_joint_tracking_coordinates[joint][frame_seq_number][0] * frame_width),
                               round(hand_joint_tracking_coordinates[joint][frame_seq_number][1] * frame_height))
                    point_previous = None if hand_joint_tracking_coordinates[joint][frame_seq_number - 1] is None else (
                        round(hand_joint_tracking_coordinates[joint][frame_seq_number - 1][0] * frame_width),
                        round(hand_joint_tracking_coordinates[joint][frame_seq_number - 1][1] * frame_height))

                    point_2 = (round(
                        hand_joint_tracking_coordinates[joint][frame_seq_number - SPEED_FRAME_OFFSET][0] * frame_width),
                               round(hand_joint_tracking_coordinates[joint][frame_seq_number - SPEED_FRAME_OFFSET][
                                         1] * frame_height))

                    d = calculate_distance(point_1, point_2)
                    delta_distance = calculate_distance(point_1, point_previous) if point_previous is not None else 0
                    time_dif = elapsed_time_points[frame_seq_number] - elapsed_time_points[
                        frame_seq_number - SPEED_FRAME_OFFSET]

                    speed = d / time_dif

                total_traversed_distance = hand_joint_traversed_distance[joint][frame_seq_number - 1] + delta_distance

                hand_joint_tracking_speed[joint].append(speed)
                hand_joint_traversed_distance[joint].append(total_traversed_distance)
            frame_seq_number += 1
        else:
            for joint in tracking_joint_list:
                total_traversed_distance = hand_joint_traversed_distance[joint][
                    frame_seq_number - 1] if frame_seq_number != 0 else 0

                hand_joint_tracking_speed[joint].append(0)
                hand_joint_traversed_distance[joint].append(total_traversed_distance)

            frame_seq_number += 1

        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(20) & 0xFF == 27:
            break


def plot_matrices():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.flatten()

    df = pd.DataFrame({joint: hand_joint_tracking_speed[joint] for joint in tracking_joint_list})
    for joint in tracking_joint_list:
        sns.lineplot(x=elapsed_time_points, y=hand_joint_tracking_speed[joint], ax=ax1, label="{0} Speed".format(joint))
        sns.lineplot(x=elapsed_time_points, y=hand_joint_traversed_distance[joint], ax=ax2,
                     label="{0} total distance".format(joint))

        hand_joint_tracking_speed[joint] = [val for val in hand_joint_tracking_speed[joint] if
                                            val < MAX_SPEED_THRESHOLD]
        sns.histplot(x=hand_joint_tracking_speed[joint], ax=ax3, element="poly")
        sns.kdeplot(x=hand_joint_tracking_speed[joint], ax=ax4)

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Speed (px/frame)')
    ax1.set_title('Finger Speed vs Time')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Traversed Distance (px)')
    ax2.set_title('Total Traversed Distance vs Time')

    ax3.set_xlabel('Speed')
    ax3.set_ylabel('Count')
    ax3.set_title('Speed Histogram')

    ax4.set_xlabel('Speed')
    ax4.set_ylabel('Probability Density')
    ax4.set_title('Kernel Density Estimate for Speed')

    print("Elapsed Time     :{:.2f} seconds".format(elapsed_time_points[len(elapsed_time_points) - 1]))
    print("Frame Count      :{0}".format(frame_seq_number))
    print(df.describe())

    plt.show()


def print_movement_distribution_charactoristics():
    print("Elapsed Time     :{:.2f} seconds".format(elapsed_time_points[len(elapsed_time_points) - 1]))
    print("Frame Count      :{0}".format(frame_seq_number))

    # for joint in tracking_joint_list:
    #     hand_joint_tracking_speed[joint]


# code for plotting the standard diviation of titanic dataset
# sns.displot(titanic, x="age", hue="class", kind="kde", fill=True)

cap.release()
plot_matrices()
# print_movement_distribution_charactoristics()
