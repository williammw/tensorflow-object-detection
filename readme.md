# Object Detection with TensorFlow

This is a simple object detection script using TensorFlow and the Object Detection API. It detects objects in real-time using your computer's webcam or a video file.

## Getting Started

### Prerequisites

- TensorFlow 2.x
- OpenCV
- Object Detection API (installation instructions [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md))

### Installation

1. Clone the repository:

2. Install the dependencies:

3. Download a pre-trained model from the [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). Extract the downloaded `.tar.gz` file to the `Frozen` directory of this repository.

4. Download the label map file for the model from the [TensorFlow Model Zoo](https://github.com/tensorflow/models/tree/master/research/object_detection/data). Save the file as `label_map.pbtxt` in the `Labels` directory of this repository.

### Usage

Run the script using the following command:

Press `q` to quit.

### Future Work

- Improve performance by using a more optimized model
- Add the ability to detect multiple objects at once
- Implement object tracking to track objects over time
- Add support for video files with different formats
