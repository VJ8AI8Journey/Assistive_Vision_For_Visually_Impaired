Assistive Vision Object Detection System

Bachelor of Engineering Final-Year Project

An assistive computer-vision system developed to help visually impaired users understand their surroundings by detecting objects through a camera and providing simple spatial and spoken guidance.

Safety notice: This is an educational project, not a certified mobility aid. It must not be used as the sole means of navigation.

Project background

I developed this project as my Bachelor of Engineering final-year project. The goal was to explore how real-time object detection could give visually impaired users additional awareness of objects, obstacles and navigation features around them.

The system uses a YOLO-based object detector to process camera frames, identify multiple objects and display bounding boxes with class labels. It then calculates the approximate position of each detection in the frame and can communicate that information through speech.

The project covered the following applied computer-vision workflow:

Define the assistive use case.

Collect and prepare image data.

Organise object classes and annotations.

Train an object-detection model.

Evaluate model predictions.

Run detection through a live camera.

Convert detections into user-friendly guidance.

Problem statement

Visually impaired users may find it difficult to identify objects and obstacles in unfamiliar environments. A camera-based detection system can provide additional environmental information by recognising visible objects and describing where they appear.

This project investigated whether an object-detection pipeline could:

Detect multiple objects in a camera frame

Recognise objects relevant to navigation

Estimate the approximate position of each object

Display bounding boxes, labels and confidence scores

Provide spoken information about detected objects

Operate continuously using a live camera

System workflow

Camera input
     ↓
Frame capture with OpenCV
     ↓
Image preprocessing
     ↓
YOLO object detection
     ↓
Bounding boxes, labels and confidence scores
     ↓
Relative position calculation
     ↓
Visual and spoken guidance

Development process

1. Selecting object classes

The first stage was deciding which objects and navigation features would be useful in an assistive-vision context. Examples included people, vehicles, doors, stairs, crosswalks, traffic lights, potholes and other potential obstacles.

2. Preparing the dataset

Images and their object annotations were organised into training, validation and test sets.

dataset/
├── train/
│   ├── images/
│   └── labels/
├── validation/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/

Each visible object was represented by a bounding box. A YOLO annotation uses the following format:

class_id x_center y_center width height

The bounding-box coordinates are normalised relative to the image dimensions.

3. Validating the data

Before training, the dataset was checked to ensure that:

Images could be opened correctly

Label files matched their images

Bounding-box coordinates were valid

Class identifiers matched the dataset configuration

Training, validation and test data were kept separate

This validation reduced the risk of failed training runs and misleading evaluation results.

4. Training the detector

The project used transfer learning with a pretrained YOLO model. Starting from pretrained weights allowed the model to adapt existing visual features to the selected assistive object classes.

During training, the model learned to predict:

The object class

The bounding-box coordinates

A confidence score

Training-time image augmentation helped the model learn from variations in scale, position, lighting and viewpoint.

5. Evaluating the model

The detector was assessed with standard object-detection metrics:

Precision: the proportion of predicted detections that were correct

Recall: the proportion of labelled objects successfully detected

mAP@0.5: mean average precision at an intersection-over-union threshold of 0.5

mAP@0.5:0.95: mean performance across stricter overlap thresholds

Predicted images were also inspected visually. This was necessary because numerical metrics alone do not show whether the model behaves appropriately in real scenes.

6. Running live detection

OpenCV was used to read frames from a connected camera. Each frame was passed to the trained detector, and the predicted boxes, labels and confidence scores were added to the displayed frame.

Open camera
    ↓
Read frame
    ↓
Run inference
    ↓
Process detections
    ↓
Draw boxes and labels
    ↓
Generate guidance
    ↓
Repeat until stopped

7. Calculating relative position

The centre of each predicted bounding box was compared with regions of the camera frame. This allowed the system to describe an object using positions such as:

Top-left, top-centre or top-right

Centre-left, centre or centre-right

Bottom-left, bottom-centre or bottom-right

Example guidance:

Person at centre
Door at top-left
Car at bottom-right

8. Providing spoken guidance

Text-to-speech functionality converted detected class names and positions into spoken announcements. A speech interval could be used to prevent the same detection from being announced continuously.

Technologies used

Python

YOLOv5

PyTorch

OpenCV

NumPy

Text-to-speech

Git and GitHub

Main features

Live camera processing

Multiple-object detection

Bounding-box visualisation

Class labels and confidence scores

Relative object-position estimation

Spoken object announcements

Configurable detection threshold

CPU or CUDA-compatible inference

Repository structure

assistive-vision-yolov5/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── yolov5_assistive_vision.py
├── configs/
│   └── data.yaml
├── weights/
│   └── best.pt
├── results/
│   ├── results.png
│   ├── confusion_matrix.png
│   ├── PR_curve.png
│   └── F1_curve.png
└── demo/
    └── detection_demo.jpg

The complete dataset, virtual environment, cloned YOLOv5 source, caches and full training-output folders are not committed to this repository.

Installation

1. Clone this repository

git clone https://github.com/YOUR-USERNAME/assistive-vision-yolov5.git
cd assistive-vision-yolov5

2. Clone YOLOv5

git clone https://github.com/ultralytics/yolov5.git

3. Create and activate an environment

python -m venv yolov5-env
.\yolov5-env\Scripts\Activate.ps1

4. Install dependencies

python -m pip install --upgrade pip
python -m pip install -r yolov5\requirements.txt
python -m pip install pyttsx3

Install the appropriate CPU or CUDA-enabled PyTorch package for the computer being used.

Running the application

Display the available arguments:

python yolov5_assistive_vision.py --help

Example:

python yolov5_assistive_vision.py --camera 0 --confidence 0.25

The precise command depends on the checkpoint path and command-line arguments implemented in the included application script.

Recent technical verification

I recently revisited the project to check that its main technical workflow still operated in a current local environment. This was a limited verification exercise, not a new project-development period.

The checks included:

Recreating the Python environment

Confirming CUDA inference on an NVIDIA GPU

Rechecking dataset conversion and model training

Running small indoor-object experiments

Training a one-class Bottle model as a controlled pipeline test

Confirming live-camera detection of one and multiple bottles

These checks verified that the data, training, evaluation and webcam-inference process still worked. They were limited follow-up checks and are not presented as a separate project.

Limitations

Relative frame position does not measure physical distance.

Accuracy depends on lighting, viewpoint, camera quality and object size.

Objects may be missed or incorrectly classified.

Spoken alerts require prioritisation in crowded scenes.

The system is not a replacement for certified accessibility equipment.

Future improvements

Add depth or distance estimation

Improve obstacle-priority rules

Reduce repeated spoken announcements

Collect and manually label a more diverse assistive dataset

Evaluate performance across different environments

Optimise the model for mobile or edge deployment

Conduct structured accessibility-focused testing

What I learned

This project gave me practical experience with:

Object detection

Dataset preparation and annotation formats

Transfer learning

Model training and evaluation

Precision, recall and mAP analysis

Camera processing with OpenCV

Bounding-box coordinate calculations

Real-time inference

Connecting model predictions to an end-user application

Understanding class imbalance and data limitations

Dataset attribution

Add the exact original dataset information before publishing:

Dataset name: ADD EXACT DATASET NAME
Creator: ADD CREATOR
Source: ADD SOURCE URL
Licence: ADD EXACT LICENCE

Any Open Images verification experiments should be credited to Open Images V7 and used according to the licensing and attribution requirements of the individual images.

Acknowledgements

Ultralytics YOLOv5 for the object-detection framework

PyTorch for model training and inference

OpenCV for camera and image processing

This project applies and integrates YOLO for an assistive-vision use case. It does not claim authorship of the YOLO architecture or framework.

Licence

Original application code is provided under the licence included in this repository. YOLOv5, pretrained weights, datasets and other third-party components remain subject to their respective licences.