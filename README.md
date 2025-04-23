# Automated-Traffic-Monitoring-System
This repository contains the code for a vehicle flow automatic detection system using YOLOv5, incorporating a weather recognition function based on ResNet50.

The project implements basic functionalities including real-time detection of different vehicle types and real-time vehicle counts in both north and south directions. It features a comprehensive GUI that allows manual setting of maximum vehicle flow thresholds for different directions, issuing warnings when thresholds are exceeded. Additionally, the project can automatically adjust thresholds based on recognized weather conditions (maximum vehicle flow: sunny > rainy > snowy). Lastly, there is an incomplete function for detecting illegal lane changes across solid lines.

## Highlights
### Automatic traffic flow threshold based on weather detection
![wechat_2025-04-23_214043_266](https://github.com/user-attachments/assets/e967fc98-02f5-4ad1-9fcb-21f519e0df89)

### Good night detection results
![image](https://github.com/user-attachments/assets/d09ab67b-f49d-4a56-927c-2beaa1bb4167)

### Demo

https://github.com/user-attachments/assets/927cac7a-90dc-4a2b-97a2-5f03d8f07c36



## Instructions

### Prerequisites

* **Tip:** It is highly recommended to use a device with a GPU to run this project.
* Install Python 3.9.19.
* Install necessary dependencies:
    ```bash
    pip install -r requirements.txt -f [https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html)
    ```
* **Preparing Data:**
    * Download the dataset: [Link](https://drive.google.com/file/d/1qeoyWlAduQxZjBF0_odZEGLL_z2DJIfn/view?usp=drive_link)
    * Place the images from the dataset in the `./VOC/images/` folder.
    * Place the label `.xml` files in the `./VOC/xml_dataset/` folder.
    * Ensure the filenames match the expected format.
* **Preparing Model Weight:**
    * Download model weight: [Link](https://drive.google.com/file/d/1kTxbhnib5hB8NNDqd9b-E1PM-6iecrpP/view?usp=sharing)
    * Extract both folders from the zip file to the project's root directory.   
---

### Running the Code

1.  **Set the working directory:** Navigate to the root directory of the project.
    ```bash
    cd Automated-Traffic-Monitoring-System
    ```
2.  **Preprocess Training and Validation Data:**
    * Run `python VOC/maketxt.py`
        * This script splits the dataset into training and validation sets and generates `.txt` files containing the respective image IDs in the `VOC/txt_dataset/` folder.
    * Run `python VOC/voc_label_coco.py`
        * This script converts the `.xml` label files (from UA-DETRAC) into `.txt` label files suitable for YOLO and saves them in the `VOC/labels/` folder.
        * It also generates two `.txt` files (`train.txt` and `val.txt`) in the `VOC/list/` folder, containing the image paths for the training and validation sets.

3.  **Run the Training Script:**
    * Execute `python train.py`
    * This will train the YOLOv5 model. Training results, including model weights (`best.pt`, `last.pt`) and performance graphs, will be saved to `runs/train/exp`. Subsequent training runs will create folders like `exp2`, `exp3`, etc.

4.  **Run the Traffic Flow Detection System:**
    * Modify the model path variable in line 14 of the `main.py` file to point to your trained model weights. The default path is `runs/train/exp/weights/best.pt`.
    * Execute `python main.py`
    * This will launch the GUI for the vehicle flow detection system, integrating all functionalities.

    > **Important Tip:** The illegal lane change detection function is **not enabled by default**. This feature requires manual adjustment of the Region of Interest (ROI) specific to each video. To enable it:
    > * Uncomment lines 223 to 234 in `main.py`.
    > * Uncomment lines 252 to 257 in `main.py`.
    > * Modify the ROI settings defined in lines 33 to 36 of `main.py` according to your video's perspective.

## Potential Improvement

* **Weather Recognition Model:** Due to time constraints, an open-source weather recognition model was used directly (The original link of the weather recognition model is [here](https://github.com/mengxianglong123/weather-recognition)). This model was not trained on our specific traffic dataset. Consequently, if a video frame primarily shows the road without distinct weather features (like clear sky, rain, snow), the weather prediction might be inaccurate. Training the provided weather model architecture (`model.py`) using our traffic dataset annotated with weather labels would resolve this.
* **Kalman Filter Speed Estimation:** The current Kalman filter implementation tracks objects but cannot accurately estimate their real-world speed.
* **Lane Detection Applicability:** The lane detection function is best suited for highway monitoring scenarios where the camera has a fixed viewing angle and the lane lines are relatively straight.
