import os
import cv2
import time
import numpy as np

MODE = "COCO"

# COCO Output Format : 
# Nose – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4, 
# Left Shoulder – 5, Left Elbow – 6, Left Wrist – 7, Right Hip – 8, 
# Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12, 
# LAnkle – 13, Right Eye – 14, Left Eye – 15, Right Ear – 16, 
# Left Ear – 17, Background – 18

# MPII Output Format : 
# Head – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4, 
# Left Shoulder – 5, Left Elbow – 6, Left Wrist – 7, Right Hip – 8, 
# Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12, 
# Left Ankle – 13, Chest – 14, Background – 15

if MODE is "COCO":
    PROTO_FILE = "./../models/pose/coco/pose_deploy_linevec.prototxt"
    WEIGHTS_FILE = "./../models/pose/coco/pose_iter_440000.caffemodel"
    N_POINTS = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE is "MPI" :
    PROTO_FILE = "./../models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    WEIGHTS_FILE = "./../models/pose/mpi/pose_iter_160000.caffemodel"
    N_POINTS = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

input_filepath = "./../data/input/"
input_filename = "test-sample-image-1.jpg"
input_file = os.path.join(input_filepath, input_filename)
file_ext = os.path.splitext(input_file)[1]
output_filepath = "./../data/output/"
output_filename = input_filename[0 : input_filename.rfind(file_ext)]
output_file = os.path.join(output_filepath, output_filename + '-output-skeleton-' + MODE + file_ext)

frame = cv2.imread(input_file)
frame_copy = np.copy(frame)
frame_width = frame.shape[1]
frame_height = frame.shape[0]
probability_threshold = 0.1

dnn_network = cv2.dnn.readNetFromCaffe(PROTO_FILE, WEIGHTS_FILE)

t = time.time()
# input image dimensions for the network
model_input_width = 368
model_input_height = 368
input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (model_input_width, model_input_height), (0, 0, 0), swapRB=False, crop=False)

dnn_network.setInput(input_blob)

output = dnn_network.forward()
print("Time taken by network : {:.3f}".format(time.time() - t))
# Output is 4 dimensions - 
# 1. image ID ( in case you pass more than one image to the network ).
# 2. index of a keypoint. The model produces Confidence Maps and Part Affinity maps which are all concatenated. For COCO model it consists of 57 parts – 18 keypoint confidence Maps + 1 background + 19*2 Part Affinity Maps. Similarly, for MPI, it produces 44 points. We will be using only the first few points which correspond to Keypoints.
# 3. height of the output map.
# 4. width of the output map.

output_height = output.shape[2]
output_weight = output.shape[3]

# Empty list to store the detected keypoints
points = []

for i in range(N_POINTS):
    # check whether each keypoint is present in the image or not
    # confidence map of corresponding body's part.
    probability_map = output[0, i, :, :]

    # get the location of the keypoint by finding the maxima of the confidence map of that keypoint
    # Find global maxima of the probability_map.
    min_value, probability, min_loc, point = cv2.minMaxLoc(probability_map)
    
    # Scale the point to fit on the original image
    x = (frame_width * point[0]) / output_weight
    y = (frame_height * point[1]) / output_height

    # use a threshold to reduce false detections
    # Plot Keypoints
    if probability > probability_threshold : 
        cv2.circle(frame_copy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frame_copy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        # Add the point to the list if the probability is greater than the probability_threshold
        points.append((int(x), int(y)))
    else :
        points.append(None)

# Draw Skeleton
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
        cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

#cv2.imshow('Output-Keypoints', frame_copy)
cv2.imshow('Output-Skeleton' + MODE, frame)

#cv2.imwrite(output_file[0 : output_file.rfind('-output-skeleton-' + MODE + file_ext)] + '-output-keypoints-' + MODE + file_ext, frame_copy)
cv2.imwrite(output_file, frame)

print("Total time taken : {:.3f}".format(time.time() - t))

cv2.waitKey(0)