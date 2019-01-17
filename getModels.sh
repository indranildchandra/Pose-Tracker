# ------------------------- BODY, FACE AND HAND MODELS -------------------------
# Downloading body pose (COCO and MPI), face and hand models
OPENPOSE_URL="http://posefs1.perception.cs.cmu.edu/OpenPose"
FACE_FOLDER="/models/face/"
HAND_FOLDER="/models/hand/"
POSE_FOLDER="/models/pose/"
COCO_FOLDER=${POSE_FOLDER}"coco/"
MPI_FOLDER=${POSE_FOLDER}"mpi/"

# ------------------------- POSE MODELS -------------------------
# COCO
# create directory if not exists
if [[ ! -e ${COCO_FOLDER} ]]; then
    mkdir -p ${COCO_FOLDER}
fi
COCO_MODEL=${COCO_FOLDER}"pose_iter_440000.caffemodel"
# download model
wget -c ${OPENPOSE_URL}${COCO_MODEL} -P ${COCO_FOLDER}
# # alternative it will check whether file was fully downloaded or not
# if [ ! -f $COCO_MODEL ]; then
#     wget ${OPENPOSE_URL}$COCO_MODEL -P $COCO_FOLDER
# fi

# MPI
# create directory if not exists
if [[ ! -e ${MPI_FOLDER} ]]; then
    mkdir -p ${MPI_FOLDER}
fi
MPI_MODEL=${MPI_FOLDER}"pose_iter_160000.caffemodel"
# download model
wget -c ${OPENPOSE_URL}${MPI_MODEL} -P ${MPI_FOLDER}
# # alternative it will check whether file was fully downloaded or not
# if [ ! -f $MPI_MODEL ]; then
#     wget ${OPENPOSE_URL}$MPI_MODEL -P $MPI_FOLDER
# fi

# "------------------------- FACE MODELS -------------------------"
# create directory if not exists
if [[ ! -e ${FACE_FOLDER} ]]; then
    mkdir -p ${FACE_FOLDER}
fi
FACE_MODEL=${FACE_FOLDER}"pose_iter_116000.caffemodel"
# download model
wget -c ${OPENPOSE_URL}${FACE_MODEL} -P ${FACE_FOLDER}
# alternative it will check whether file was fully downloaded or not
# if [ ! -f $FACE_MODEL ]; then
#     wget ${OPENPOSE_URL}$FACE_MODEL -P $FACE_FOLDER
# fi

# "------------------------- HAND MODELS -------------------------"
# create directory if not exists
if [[ ! -e ${HAND_FOLDER} ]]; then
    mkdir -p ${HAND_FOLDER}
fi
HAND_MODEL=$HAND_FOLDER"pose_iter_102000.caffemodel"
# download model
wget -c ${OPENPOSE_URL}${HAND_MODEL} -P ${HAND_FOLDER}
# # alternative it will check whether file was fully downloaded or not
# if [ ! -f $HAND_MODEL ]; then
#     wget ${OPENPOSE_URL}$HAND_MODEL -P $HAND_FOLDER
# fi
