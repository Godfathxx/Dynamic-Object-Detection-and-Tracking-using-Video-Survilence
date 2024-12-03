****Dynamic Object Detection and Tracking Using Video Surveillance****

*Overview*

This repository implements dynamic object detection and tracking models for video surveillance applications. It focuses on anomaly detection in real-time scenarios, offering a comparative study of three hybrid approaches. The project includes model implementations, evaluations, and datasets for benchmarking performance.

**Models Implemented**

EfficientNetV2B0 with Histogram of Oriented Gradients (HOG):

Focus: Spatial anomaly detection in controlled environments like retail stores.

Key Features: Lightweight, accurate for edge-based anomalies.

ConvLSTM

Focus: Temporal anomaly detection by capturing sequential patterns.

Key Features: Combines convolutional layers for spatial features and LSTM layers for temporal dependencies.

HybridModel-1 (ResNet-50 + YOLO-v4)

Focus: Real-time crime detection and multi-object tracking.

Key Features: High accuracy and speed, suitable for dynamic and high-density environments.

*Dataset*

This project utilizes the DCSASS (Dynamic Crime and Security Anomaly Surveillance System) dataset and additional datasets for model training and evaluation.

Download the DCSASS Dataset [here](https://www.kaggle.com/datasets/mateohervas/dcsass-dataset).

**Dataset Details**:

-> Real-world surveillance footage capturing normal and abnormal activities.

-> Categories include theft, assault, vandalism, etc.

-> Comprehensive annotations for classification and bounding box predictions.

**Features**
-> Preprocessing Pipelines: Includes data normalization, augmentation, and frame extraction.

-> Training: Implements supervised learning with hyperparameter optimization and early stopping.

-> Evaluation: Detailed performance metrics (accuracy, precision, recall, F1-score).

-> Visualization: Results include bounding box predictions, confusion matrices, and comparative graphs.

**Research Contribution**

This repository is part of a comparative study on hybrid models for theft detection and anomaly detection in surveillance, as detailed in the attached research paper.

**Key Findings**:

-> EfficientNetV2B0 + HOG excels in spatial clarity for edge-based anomalies.

-> ConvLSTM effectively detects sequential anomalies but requires optimized hardware.

-> HybridModel-1 balances speed and accuracy, making it ideal for real-time applications.

**Setup and Installation**

1.Clone the repository:

git clone https://github.com/Godfathxx/Dynamic-Object-Detection-and-Tracking-using-Video-Survilence.git

2.Install dependencies:

pip install -r requirements.txt

3.Download the dataset and place it in the data folder.

**Usage**

1.Training a Model

python train.py --model <model_name> --dataset <path_to_dataset>

2.Testing a Model

python test.py --model <model_name> --checkpoint <path_to_checkpoint>

3.Visualizing Results

python visualize.py --results <path_to_results>

Citation

If you use this repository or the dataset in your research, please cite:

The research paper: Dynamic Objects Detection and Tracking from Videos for Surveillance Applications.
