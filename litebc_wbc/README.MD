## This is a folder for the LiteBC_WBC group's work

### Current Work

One of the ideas of the work is to collect the data at a certain good part of a wide cam video and use the data to count how many white blood cells went through that place on the video.

To find a good place for data collection the wide cam video is represented by a single image first. This image is build by accumulating per-pixel motion data and average color of the pixel. This data is weighted by a radial filter to give less weight for the pixel at the edges of the frames where video quality is often compromized. A good part of the video is the brightest spot on an image. 
 The brightest area of the image is an area with greatest amount of change. It is the area where the video is of its best quality. 

The line that 'cuts' the capillary is found in this brightest part of the image. It is a line that goes throgh the brightest pixel of the image that is inside a capillary. The line is perpendicular to the skeleton of the capillary. 

Once the line is found the per-frame data that goes through the line is collected. The idea here is that we expect certain groups of light coloured points being seen going through the line in sequences when the white blood cell is passing through the line. At all other times the white dots are plasma between the red dots. That is the line is mostly dark. Especially at the center.

End of white blood cells is collected from the annotation file that comes with the video. Ground truth is collected from the ground truth annotation file. The data itself is passed to RNN in an attempt to predict the end of blood cells in previously unseen video.

* line_cross.py

Line_cross.py is data collection part of the solution. It is the *primitive* version of the solution. That means that this code does not include any future steps as DN capillary contour prediction, and almost all the algorithms are written from scratch in their basic form (motion detection, color averaging, contour search, skeleton formation, search for a perpendicular line to a skeleton). Only dense optical flow solution is taken fully from [opencv](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html). 

This code goes through all the files stored at /mnt/litebc/patient-data that are annotated and have ground truth for them. Data collected for one video is stored as a single file in pickeled format. xy_data folder has to be created for the data to be stored there.
 
The code collects the following: video_file, video_points, y_points, points_on_line, total_diff, mean_angles, (date2, hgb, wbc)

video_file - name of the file that has been analyzed

video_points - data for the points for the line where the information was be collected

y_points - marks for the end of wbc that went through the line

points_on_line - points (line) where the information is collected

total_diff - final representation of the video as one image  

mean_angles - point-wise direction angles from the dense flow calculations (averages for all the frames)

(date2, hbg, wbc) - results of the lab blood test

* line_cross_gpu.py

video_file - as before

video_points - as before

y_points - as before

sorted_end_frames - y's in a list

points_on_line - get_line function uses the predicted contour instead of thresholding

points_on_line_old - as previously points_on_line

total_diff - as before

bgr - image of dense optical flow

mean_angles - as before

(date2, hgb, wbc) - as before

first_frame - first frame of the stabilzed.avi

capil_mask - predicted capillary mask

capil_contours - contours for the mask

The data collection is done in a parallel mode, that is the code uses multiple CPUs to run video processing jobs in parallel. Hence the code will run much faster on a machine with higher number of CPUs.

* wbcrnn_synth.py

* synthetic_data.py

RNN training using the symplified data that resembles data collected by line_cross.py 

The resulting dataset is very unbalanced. The amout of WBCs is small and they pass through the line seldom. Hence, it may be challenging to find the right settings to train the RNN well. It learns fast to predict everything or as 1 or as 0 depending on the hyperparameters for the loss function.

* review2.ipynb

Visual review of the data

* Utils and Utils_output_examples

input_to_rnn_visualization.py - before assessing the results of the RNN it would be good to look at the input that is provided to the Net in order to understand how good it is, that is, to see whether it is indeed a good input from the the machine should easily learn where the white blood cells are, or whether the input needs further cleaning or exploration

create_test_videos_with_line_and_annotations.py - where the input for RNN seems to be strange it is best to check where in the video the input comes from and why it is like that

opticalflow.py - at times the annotations on the videos look a bit misplaced, one of the filters for well placed annotations can be the direction in which it flows: it should flow to the same direction with the cell it contains, if not, something may not be right

 
### Future Work

It would be nice to concatenate the data that is retrieved from the 'cut' on the capillary with the Line Camera Data. Even simple vector concatenation of the two data inputs may improve the white blood cell count. It is something that is worth checking. 

Having the cut, knowing the direction of the flow at the cut point and the distance from the cut to the line camera will most probably allow to connect the data.

* line_cam.ipynb

### Capillary segmentation

Instance / Semantic segmentation of an image represenation of the video to predict the shape of a capillary


