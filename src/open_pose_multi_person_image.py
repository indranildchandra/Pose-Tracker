import os
import cv2
import time
import numpy as np
from random import randint

MODE = "COCO"

# COCO Output Format : 
# Nose – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4, 
# Left Shoulder – 5, Left Elbow – 6, Left Wrist – 7, Right Hip – 8, 
# Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12, 
# LAnkle – 13, Right Eye – 14, Left Eye – 15, Right Ear – 16, 
# Left Ear – 17, Background – 18

if MODE is "COCO":
    PROTO_FILE = "./../models/pose/coco/pose_deploy_linevec.prototxt"
    WEIGHTS_FILE = "./../models/pose/coco/pose_iter_440000.caffemodel"
    N_POINTS = 18
    POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9], [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16], [0,15], [15,17], [2,17], [5,16] ]
    KEYPOINTS_MAPPING = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
    # index of pafs correspoding to the POSE_PAIRS
    # e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
    MAP_IDx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], [55,56], [37,38], [45,46]]
    COLORS = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255], [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255], [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]

input_filepath = "./../data/input/"
input_filename = "test-sample-image-4.jpg"
input_file = os.path.join(input_filepath, input_filename)
file_ext = os.path.splitext(input_file)[1]
output_filepath = "./../data/output/"
output_filename = input_filename[0 : input_filename.rfind(file_ext)]
output_file = os.path.join(output_filepath, output_filename + '-output-postures-' + MODE + file_ext)


def get_keypoints(probability_map, threshold=0.1):
    map_smoothing_factor = cv2.GaussianBlur(probability_map,(3,3),0,0)
    map_mask = np.int32(map_smoothing_factor>threshold)
    keypoints = []

    #find the blobs
    contours, _ = cv2.findContours(map_mask, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)

    #for each blob find the maxima
    for contour in contours:
        blob_mask = np.zeros(map_mask.shape)
        blob_mask = cv2.fillConvexPoly(blob_mask, contour, 1)
        masked_probability_map = map_smoothing_factor * blob_mask
        _, max_value, _, max_loc = cv2.minMaxLoc(masked_probability_map)
        keypoints.append(max_loc + (probability_map[max_loc[1], max_loc[0]],))

    return keypoints

# Find valid connections between the different joints of a all persons present
def get_valid_pairs(output):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_threshold = 0.1
    conf_threshold = 0.7
    # loop for every POSE_PAIR
    for k in range(len(MAP_IDx)):
        # A->B constitute a limb
        pafA = output[0, MAP_IDx[k][0], :, :]
        pafB = output[0, MAP_IDx[k][1], :, :]
        pafA = cv2.resize(pafA, (frame_width, frame_height))
        pafB = cv2.resize(pafB, (frame_width, frame_height))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid
        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if ( len(np.where(paf_scores > paf_score_threshold)[0]) / n_interp_samples ) > conf_threshold :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
            print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs


# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def get_personwise_keypoints(valid_pairs, invalid_pairs):
    # the last number in each row is the overall score
    personwise_keypoints = -1 * np.ones((0, 19))

    for k in range(len(MAP_IDx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwise_keypoints)):
                    if personwise_keypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwise_keypoints[person_idx][indexB] = partBs[i]
                    personwise_keypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwise_keypoints = np.vstack([personwise_keypoints, row])
    return personwise_keypoints

image = cv2.imread(input_file)

frame_width = image.shape[1]
frame_height = image.shape[0]

t = time.time()
dnn_network = cv2.dnn.readNetFromCaffe(PROTO_FILE, WEIGHTS_FILE)

# Fix the input Height and get the width according to the Aspect Ratio
model_input_height = 368
model_input_width = int((model_input_height/frame_height)*frame_width)

input_blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (model_input_width, model_input_height), (0, 0, 0), swapRB=False, crop=False)

dnn_network.setInput(input_blob)
output = dnn_network.forward()
print("Time Taken in forward pass = {}".format(time.time() - t))

detected_keypoints = []
keypoints_list = np.zeros((0,3))
keypoint_id = 0
probability_threshold = 0.1

for part in range(N_POINTS):
    probability_map = output[0,part,:,:]
    probability_map = cv2.resize(probability_map, (image.shape[1], image.shape[0]))
    keypoints = get_keypoints(probability_map, probability_threshold)
    print("Keypoints - {} : {}".format(KEYPOINTS_MAPPING[part], keypoints))
    keypoints_with_id = []
    for i in range(len(keypoints)):
        keypoints_with_id.append(keypoints[i] + (keypoint_id,))
        keypoints_list = np.vstack([keypoints_list, keypoints[i]])
        keypoint_id += 1

    detected_keypoints.append(keypoints_with_id)

frame_clone = image.copy()
for i in range(N_POINTS):
    for j in range(len(detected_keypoints[i])):
        cv2.circle(frame_clone, detected_keypoints[i][j][0:2], 5, COLORS[i], -1, cv2.LINE_AA)
#cv2.imshow("Output-Keypoints",frame_clone)
#cv2.imwrite(output_file[0 : output_file.rfind('-output-postures-' + MODE + file_ext)] + '-output-features-' + MODE + file_ext, frame_clone)

valid_pairs, invalid_pairs = get_valid_pairs(output)
personwise_keypoints = get_personwise_keypoints(valid_pairs, invalid_pairs)

for i in range(17):
    for n in range(len(personwise_keypoints)):
        index = personwise_keypoints[n][np.array(POSE_PAIRS[i])]
        if -1 in index:
            continue
        B = np.int32(keypoints_list[index.astype(int), 0])
        A = np.int32(keypoints_list[index.astype(int), 1])
        cv2.line(frame_clone, (B[0], A[0]), (B[1], A[1]), COLORS[i], 3, cv2.LINE_AA)

cv2.imshow("Output-Posture" , frame_clone)
cv2.imwrite(output_file, frame_clone)

print("Total time taken : {:.3f}".format(time.time() - t))

cv2.waitKey(0)
