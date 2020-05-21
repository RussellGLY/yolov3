Instructions for YOLO experimentation on Agot dataset

Git Branches:

- base: We forked from this commit from the Ultralytics repository
- non-recurrent: for training/testing regular YOLO on the Agot dataset
- master: for training Recurrent YOLO on the Agot dataset

Prerequisites:

1. Install git-lfs to access the pretrained weights in the weights directory.
2. Run: pip install -U -r requirements.txt
3. See the main README.md file for more information on the YOLO implementation we have forked from.


Part 1: Baseline YOLO (non-recurrent)
Switch to the "non-recurrent" branch.


Dataset Formatting:

1. Load each annotated video into CVAT. See "CVAT/CVAT setup instructions.txt" for more information.

2. Export each annotated video in YOLO format

3. Place the unzipped, exported datasets into a single folder accessible by the yolov3 code.

3a. If you want to use the existing paths in yolov3/custom/train.txt and yolov3/custom/valid.txt, then set up the directory structure such that from yolov/custom:
- the path to *.jpg for 'task5' is ../../tasks/tasks/task5/*.jpg
- the path to *.jpg for 'task6' is ../../tasks/task6/obj_train_data/*.jpg

3b. If you want to recreate custom/train.txt and custom/valid.txt, you should put the exported datasets into a single folder, and modify/run custom/format.py. This script recursevly searches the directory specified by "TASKS_DIR" and separates the data into training/validation according to the fraction specified by "TRAIN_SPLIT". The "BATCH_SIZE" of 128 means that every 128 frames are either in training or validation, and not separated.


Training:

Run: python train.py --epochs 30 --batch-size 128 --accumulate 1 --cfg custom/yolov3-spp-custom.cfg --data custom/custom.data

This will train using the initial weights in weights/yolov3-spp-ultralytics.pt. Use the pretrained Agot weights by adding the flag "--weights weights/yolov3-spp-custom.pt". Use no initial weights with the flag "--weights ''". While training, the most recent weights are stored in "weights/last.pt" and the best weights (based on the validation set) are stored in "weights/best.pt". If you want to resume training from where it last left off, add the "--resume" flag. Other command line options can be found in yolov3/train.py.


Testing:

This will run the model on the validation set and output metrics. You can see our output in the file "Yolo Baseline Results.xlsx".

Run: python test.py --cfg custom/yolov3-spp-custom.cfg --data custom/custom.data --weights weights/yolov3-spp-custom.pt --batch-size 128


Detection:

This runs the model on an input image or video and outputs another image or video with drawn labels and boxes. The output is stored in the "output" directory.

python detect.py --cfg custom/yolov3-spp-custom.cfg --names custom/custom.names --weights weights/yolov3-spp-custom.pt --source <source video or image>


Tips:
- Most of our changes are in the "custom" directory. For more information on what changed, you can diff between the "base" branch and the "nonrecurrent" branch.
- We had to reduce the GLOU loss gain and disabled mixed precision in order to avoid gradients going to infinity. We're not sure why.
- Training should take 15 minutes per epoch on 4 NVIDIA Tesla V100 GPUs. We achieved peak accuracy within 30 epochs.
- Using Apex did not improve training speed.
- Using the '-cache-images' flag did not improve training speed.


Part 2: Recurrent YOLO
Switch to the "non-recurrent" branch.

This branch uses a hybrid shuffling mechanism and a Convolutional LSTM, as described in the paper. The ConvLSTM implementation is in convlstm.py, and incoporated into models.py. The hybrid shuffling preprocessing is done using generate_clips.py, which generates numpy files of train/valid paths. The data loading is done using VideoDataLoader in utils/datasets.py, which makes use of a modified version of the original LoadImagesAndLabels Dataset. See the paper for details on the hybrid shuffling mechanism.

ConvLSTM and Hybrid Shuffling Files:
- convlstm.py: convolutional lstm implementation
- models.py: modified to incoporate convlstm
- utils/datasets.py: added VideoDataLoader and modified LoadImagesAndLabels dataset
- custom/train.npy: image paths for each training clip
- custom/valid.npy: image paths for each validation clip
- custom/custom.data: modified to refer to the above npy files
- custom/generate_clips.py: generate train/valid clips

Model configuration files:
- custom/yolov3-spp-lstm1.cfg: model configuration for single ConvLSTM
- custom/yolov3-spp-lstm2.cfg: model configuration for two stacked ConvLSTMs
- custom/yolov3-spp-lstm3.cfg: model configuration for three stacked ConvLSTMs

Pretrained weight files:
These are weight files for the recurrent architectures, with all weights prior to the ConvLSTM layers copied from the best weights for the nonrecurrent architecture.

- weights/yolov3-spp-lstm1.pt: initial weights for single ConvLSTM
- weights/yolov3-spp-lstm2.pt: initial weights for two stacked ConvLSTM
- weights/yolov3-spp-lstm3.pt: initial weights for three stacked ConvLSTM

- copy_model.py: use this helper script to convert model weights for one Recurrent YOLO architecture to weights for another new architecture, by copying all weights other than those of the ConvLSTM layer. For example, this was used to generate yolov3-spp-lstm2.pt and yolov3-spp-lstm3.pt from yolov3-spp-lstm1.pt.

Training/Testing/Detection:
The training/testing/detection procedure is the same as before, but you'll want to change the model configuration to one of the mentioned cfg files and the initial weights to the matching .pt file.

Notes:
- cannot use multiple GPUs
- takes 2 hours per epoch on a Tesla V100 GPU
- 0% precision/recall/mAP after 30 epochs
- see the paper for potential improvements to this architecture
- check the conv-lstm and hybrid shuffling implementations for bugs
- even trying to overfit on a small subset of data doesn't work; troubling