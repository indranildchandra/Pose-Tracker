## Steps to run :

1. Download pre-trained models:
	a. In Unix, execute -> $ ./getModels.sh
	b. In Windows, download manually following models in the folders -
		- models/pose/coco -> http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
		- models/pose/mpi -> http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel
		- [not required currently] models/face -> http://posefs1.perception.cs.cmu.edu/OpenPose/models/face/pose_iter_116000.caffemodel
		- [not required currently] models/hand-> http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel
2. Download all Python dependencies using the command -> 
   $ pip install -r requirements.txt