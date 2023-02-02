import os
import cv2
import mediapipe as mp
from mediapipe.framework.formats import image_frame
from mediapipe.models.hand_landmark import hand_landmark

# Set the path to the directory containing the video files
path = "../../data/raw/Sign_Language_data"

# Iterate over all the files in the directory
for filename in os.listdir(path):
    # Check if the file is a video file
    if filename.endswith(".mp4"):
        # Open the video file
        cap = cv2.VideoCapture(os.path.join(path, filename))

        while True:
            # Read a frame from the video
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to a MediaPipe image_frame
            image_frame = image_frame.ImageFrame(frame)

            # Run the hand landmark model
            output_frame = hand_landmark.main(image_frame)

            # Do something with the output frame
            ...

            # Display the frame
            # cv2.imshow("Frame", output_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # Release the video capture
        cap.release()
cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Compute features from Hough Points to Human Action Recognition"
    )

    # parser.add_argument("--input_dir", type=str,
    #                     default='../../data/processed/landmarks/imageLandmarks',
    #                     help="Name of directory to where input hand landmark points located")

    parser.add_argument("--input_dir", type=str,
                        default='../../data/processed/landmarks/videoLandmarks/Sign_Language_Data/set0/',
                        help="Name of directory to where input hand landmark points located")

    parser.add_argument("--output_dir", type=str,
                        default='/data/processed/features/video/Sign_Language_Data/set0',
                        help="Name of directory to output hand feature output file")

    args = parser.parse_args()

    print(args)

    extract_hand_features(args)
    # filter_data()


if __name__ == "__main__":
    main()

# --input_dir=../../data/processed/landmarks/videoLandmarks --output_dir=/data/processed/features/video