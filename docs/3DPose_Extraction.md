Feature Extraction
======================


----

### Extract skeleton model from video

This command will extract landmarks and write to a csv file. All the video files inside 
the given input directory will be processed. Annotated video file be saved in the given
output video directory and identified key landmark points will be saved to a csv file.


`python src/features/extract_3D_skeleton_data.py --base_dir=../../data --video_input_dir=/raw/strokeVideo --landmark_output_dir=/processed/landmarks \ --annotated_video_output_dir=/processed/annotatedVideo`


