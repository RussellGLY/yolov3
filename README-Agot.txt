Instructions for YOLO experimentation on Agot dataset

Git Branches:

- base: We forked from this commit from the Ultralytics repository
- non-recurrent: for training/testing regular YOLO on the Agot dataset
- master: for training Recurrent YOLO on the Agot dataset


Part 1: Baseline YOLO (non-recurrent)
Switch to the "non-recurrent" branch.


Dataset Formatting:

1. Load each annotated video into CVAT. You may find the file TODO useful for loading the video into CVAT. See "CVAT/CVAT setup instructions.txt" for more information.

2. Export each annotated video in YOLO format

3. Place the unzipped, exported datasets into a single folder accessible by the yolov3 code.

3a. If you want to use the existing paths in yolov3/custom/train.txt and yolov3/custom/valid.txt, then set up the directory structure such that from yolov/custom:
- the path to *.jpg for 'task5' is ../../tasks/tasks/task5/*.jpg
- the path to *.jpg for 'task6' is ../../tasks/task6/obj_train_data/*.jpg

3b. If you want to recreate custom/train.txt and custom/valid.txt, you should put the exported datasets into a single folder, and modify/run custom/format.py. This script recursevly searches the directory specified by "TASKS_DIR" and separates the data into training/validation according to the fraction specified by "TRAIN_SPLIT". The "BATCH_SIZE" of 128 means that every 128 frames are either in training or validation, and not separated.


Training:

Run: python train.py --epochs 30 --batch-size 128 --accumulate 1 --cfg custom/yolov3-spp-custom.cfg --data custom/custom.data

This will train using the initial weights in weights/yolov3-spp-ultralytics.pt. Use the pretrained Agot weights by adding the flag "--weights weights/yolov3-spp-custom.pt". Use no initial weights with the flag "--weights ''". While training, the most recent weights are stored in "weights/last.pt" and the best weights (based on the validation set) are stored in "weights/best.pt". If you want to resume training from where it last left off, add the --resume flag. Other options can be found in yolov3/train.py.


Testing:

This will run the model on the validation set and output metrics.

Run: python test.py --cfg custom/yolov3-spp-custom.cfg --data custom/custom.data --weights weights/yolov3-spp-custom.pt --batch-size 128


Detection:

This runs the model on an input image or video and outputs another image or video with drawn labels and boxes. The output is stored in the "output" directory.

python detect.py --cfg custom/yolov3-spp-custom.cfg --names custom/custom.names --weights weights/yolov3-spp-custom.pt --source <source video or image>


Part 2: Recurrent YOLO
Switch to the "non-recurrent" branch.

TODO