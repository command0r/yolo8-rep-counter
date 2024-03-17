# Exercise Counter with YOLOv8 on NVIDIA CUDA

This is a pose estimation demo application for exercise counting with YOLOv8 using [YOLOv8-Pose](https://docs.ultralytics.com/tasks/pose) model.

You can either run it locally, deply in a Docker container or use any NVIDIA Jetson device to deploy this demo.

Currently 3 different exercise types can be counted:
- Squats
- Pushups
- Situps
More exercises can be easily added, as well as the function of <b>detecting the exercise type</b>.

## Introduction

The YOLOv8-Pose model can detect 17 key points in the human body, then select discriminative key-points based on the characteristics of the exercise. Calculate the angle between key-point lines, when the angle reaches a certain threshold, the target can be considered to have completed a certain action. By utilizing the above-mentioned mechanism, it is possible to achieve very interesting results.

## Installation

- **Step 1:** Clone the following repo

```sh
git clone https://github.com/ultralytics/ultralytics.git
```

- **Step 2:** Open pyproject.toml

```sh
cd ultralytics
nano pyproject.toml
```

- **Step 3:** Edit the following lines

```sh
# torch>=1.7.0
# torchvision>=0.8.1
```

**Note:** torch and torchvision are excluded for now because they will be installed later

- **Step 4:** Install the necessary packages

```sh
pip3 install -e .
```

- **Step 5:** If there is an error in numpy version, install the required version of numpy

```sh
pip3 install numpy==1.20.3
```

- **Step 6:** Install PyTorch and Torchvision [(Refer to here)](https://wiki.seeedstudio.com/YOLOv8-DeepStream-TRT-Jetson/#install-pytorch-and-torchvision)

- **Step 7:** Run the following command to make sure yolo is installed properly

```sh
yolo detect predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

- **Step 8:** Finally, clone this repo

## Prepare The Model File

YOLOv8-pose pre-trained pose models are PyTorch models, and you can directly use them for inferencing on any device that supports NVIDIA CUDA. However, to improve speed, you can convert the PyTorch models to TensorRT-optimized models by following the instructions below.

- **Step 1:** Download model weights in PyTorch format [(Refer to here)](https://docs.ultralytics.com/tasks/pose/#models).

- **Step 2:** Execute the following command to convert this PyTorch model into a TensorRT model:

```sh
# TensorRT FP32 export
yolo export model=yolov8s-pose.pt format=engine device=0

# TensorRT FP16 export
yolo export model=yolov8s-pose.pt format=engine half=True device=0
```

**Tip:** [Click here](https://docs.ultralytics.com/modes/export) to learn more about yolo export

- **Step 3:** Prepare a video to be tested

## Let's Run It!

To run the exercise counter, enter the following commands with the `exercise_type` as:

- sit-up
- pushup
- squat

### For video

```sh
python3 demo.py --sport <exercise_type> --model yolov8s-pose.pt --show True --input <path_to_your_video>
```

### For webcam

```sh
python3 demo.py --sport <exercise_type> --model yolov8s-pose.pt --show True --input 0
```

## Useful links & references

[https://github.com/ultralytics/](https://github.com/ultralytics/)<br />
[https://wiki.seeedstudio.com](https://wiki.seeedstudio.com/YOLOv8-DeepStream-TRT-Jetson/)
