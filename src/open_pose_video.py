import os
import cv2
import time
import numpy as np

MODE = "MPI"

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

ROTATE_IMAGE = False

input_filepath = "./../data/input/"
input_filename = "test-sample-video-2.mp4"
input_file = os.path.join(input_filepath, input_filename)
input_file_ext = os.path.splitext(input_file)[1]
output_filepath = "./../data/output/"
output_filename = input_filename[0 : input_filename.rfind(input_file_ext)]
output_file_ext = '.avi'
output_file = os.path.join(output_filepath, output_filename + '-output-skeleton-' + MODE + output_file_ext)

cap = cv2.VideoCapture(input_file)
has_frame, frame = cap.read()

vid_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

dnn_network = cv2.dnn.readNetFromCaffe(PROTO_FILE, WEIGHTS_FILE)

# model parameters
model_input_width = 368
model_input_height = 368
probability_threshold = 0.1

frame_num = 0
while cv2.waitKey(1) < 0:
    t = time.time()
    has_frame, frame = cap.read()
    frame_copy = np.copy(frame)
    if ROTATE_IMAGE:
        frame = cv2.transpose(frame)
        frame_copy = cv2.transpose(frame_copy)

    if not has_frame:
        break

    frame_num = frame_num + 1
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (model_input_width, model_input_height), (0, 0, 0), swapRB=False, crop=False)
    dnn_network.setInput(input_blob)
    output = dnn_network.forward()
    # The output is a 4D matrix :
    # 1. image ID ( in case you pass more than one image to the network ).
    # 2. index of a keypoint. The model produces Confidence Maps and Part Affinity maps which are all concatenated. For COCO model it consists of 57 parts – 18 keypoint confidence Maps + 1 background + 19*2 Part Affinity Maps. Similarly, for MPI, it produces 44 points. We will be using only the first few points which correspond to Keypoints.
    # 3. height of the output map.
    # 4. width of the output map.

    output_height = output.shape[2]
    output_width = output.shape[3]
    # Empty list to store the detected keypoints
    feature_points = []

    for i in range(N_POINTS):
        # check whether each keypoint is present in the image or not
        # confidence map of corresponding body's part.
        probability_map = output[0, i, :, :]

        # get the location of the keypoint by finding the maxima of the confidence map of that keypoint
        # Find global maxima of the probability_map.
        min_value, probability, min_loc, point = cv2.minMaxLoc(probability_map)
        
        # Scale the point to fit on the original image
        x = (frame_width * point[0]) / output_width
        y = (frame_height * point[1]) / output_height

        # use a threshold to reduce false detections
        if probability > probability_threshold : 
            cv2.circle(frame_copy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame_copy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            # Add the point to the list if the probability is greater than the probability_threshold
            feature_points.append((int(x), int(y)))
        else :
            feature_points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0] 
        partB = pair[1]

        if feature_points[partA] and feature_points[partB]:
            cv2.line(frame, feature_points[partA], feature_points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, feature_points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, feature_points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    cv2.putText(frame, ("Frame : " + str(frame_num)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    # cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.imshow('Output-Keypoints', frame_copy)
    cv2.imshow('Output-Skeleton-' + MODE, frame)

    vid_writer.write(frame)

vid_writer.release()
cv2.destroyAllWindows()