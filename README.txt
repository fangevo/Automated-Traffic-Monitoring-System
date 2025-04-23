README
1. Team: p22

2. Team members:
   Yangtao Fang
   Taiwei Wu

3. File Structure

    yolov5-master/
    ├── README.txt           # The documentation that explains how the program works.
    ├── export.py            # YOLOv5 official model conversion script, which can convert pt model to onnx model
    ├── iouCommon.py         # IOU Tracker and Kalman filter function definition
    ├── main.py              # Main program of the vehicle flow automatic detection system. Run to start the system.
    ├── train.py             # YOLOv5 training script, run to perform model training
    ├── val.py               # YOLOv5 script for verifying model performance.
    ├── hubconf.py           # YOLOv5 script for defining and managing pre-trained weights and configuration information for the model.
    ├── requirements.txt          # Python dependencies
    ├── model.py                  # Weather recognition Model Architecture
    ├── weather_prediction.py     # Weather recognition function definition
    ├── config.py                 # Weather recognition configuration
    ├── yolov5l.pt                # Pre-trained weights file for the YOLOv5 model
    ├── data/
    │   ├── videos/
    │   │   └── video files...     # The directory where the videos to be tested are stored. The videos are not required to be placed in this folder. You can select the videos to be tested through the GUI.
    │   ├── coco128.yaml           # Training set/validation set path and label settings.
    │   └── Other YOLOv5 default configuration files....
    ├── models/                    # Folder for YOLOv5 model configuration files
    │   ├── models-yolov5m.yaml    # PCustom labels and model architecture
    │   └── Other YOLOv5 default configuration files....
    ├── runs/                      # Folder for storing run results
    │    └── train/                # The folder where each training result is stored
    │          └── exp/             # Folder for storing single training results
    │              ├── weights/     # Folder where trained model weights are stored
    │              │    ├── best.pt  # The best model weight file obtained by training
    │              │    └── last.pt  # The last model weight file obtained by training
    │              └── Some training results data and model performance graph files....
    ├── utils/                       # Folder for YOLOv5 official common function definition file
    │    └── raw data files...
    ├── VOC/                          # Folder for dataset
    │    ├── images/                  # Folder for all images in the training and validation sets
    │    ├── labels/                  # Folder for all label .txt files for training and validation sets
    │    ├── list/                    # Folder for training set and validation set path files
    │    │    ├── train.txt           # A txt file that saves the paths of all training set images.
    │    │    └── val.txt             # A txt file that saves the paths of all validation set images.
    │    ├── txt_dataset/             # Folder for files containing training set and dataset image IDs
    │    │    ├── train.txt           # A txt file that saves IDs of all training set images.
    │    │    ├── val.txt             # A txt file that saves IDs of all validation set images.
    │    │    └── trainval.txt        # A txt file that saves IDs of all dataset images.
    │    ├── xml_dataset/             # Folder for UA-DETRAC dataset xml label files
    │    ├── maketxt.py               # Script for splitting into training and validation sets and generating txt files of dataset image IDs in the txt_dataset folder.
    │    └── voc_label_coco.py        # Script for converting UA-DETRAC dataset xml label files into txt label files suitable for yolo, and saves the txt label files in the labels folder. Generate path indexes of training set and validation set images in the list folder.
    └── weather_model/
        ├── resnet50-11ad3fa6.pth             # ResNet50 pre-trained model
        └── weather-2022-10-14-07-36-57.pth   # Trained weather recognition model weights (open source)

4. Instructions

    (1) Prerequisites

    Tip: It is highly recommended to use a device with a GPU to run this project

    Install Python 3.9.19
    
    Install necessary dependencies using:
        pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
    
    Preparing Data
        Download the dataset: https://drive.google.com/file/d/1qeoyWlAduQxZjBF0_odZEGLL_z2DJIfn/view?usp=drive_link
        Place the images from the dataset in the ./VOC/images/ folder
        Place the label .xml files in the ./VOC/xml_dataset/ folder
        Ensure the filenames match the expected format

    ---

    (2) Running the Code

    Set the working directory to the root directory
    
    Preprocess train and validation Data:

        python VOC/maketxt.py
        This will split the dataset into training and validation sets and generate txt files containing the training and
        validation set image IDs in the VOC/txt_dataset folder.

        python VOC/voc_label_coco.py
        This will convert the .xml label file into a .txt label file. And generate two txt files containing the training
        set and validation set image paths in the VOC/list folder.


    Run the training script:

        python train.py
        This will train the model and save training results to runs/train/exp
        Depending on the number of trainings, the folders will be exp, exp2, exp3...

    Run the traffic flow detection system:

        Modify the model path in line 14 of the main.py file according to the folder where your model weight file is stored.
        The default model weight path is runs/train/exp/weights/best.pt

        python main.py
        This will run the GUI script that integrates all the functions.

        !!! Tip: The illegal lane change detection function is not enabled by default in this project, because it requires
        manual adjustment of ROI for different videos. If you need to enable it, please uncomment lines 223 to 234 and
        lines 252 to 257 of main.py. And modify the ROI settings from lines 33 to 36.

5. Still need to improve:

        Due to lack of time, we directly adopted an open source weather recognition model and did not use our own
        traffic dataset to train the weather model. If there is only a road in the video without obvious weather features
        (such as sky, rain, snow, thunderstorm), wrong weather conditions may be predicted. But this can be solved by
         simply training with our own traffic dataset annotated with weather labels.

         Kalman filter cannot estimate real world speed

        The lane detection function needs to be applied to highway monitoring with fixed viewing angle and straight lane lines.
