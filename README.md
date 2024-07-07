# Bilayer Smoking Detection

This project implements a bi-layer smoking detection system using computer vision and deep learning. The system detects smoking activities from CCTV footage by analyzing skeletal poses and using a pre-trained VGG16 model for cigarette detection.

## Project Structure
```
/SmokingDetection/
├── CCTV_smoking.mp4
├── CCTV_high_res_smoking.mp4
├── main.py
├── models/
│ └── smoking_detection_vgg16_model.keras
├── utils/
│ └── logging.py
├── output_videos/
└── smoking_log.csv
```

## Requirements

- Python 3.8+
- TensorFlow
- OpenCV
- MediaPipe
- NumPy

Install the required packages using pip:

```bash
pip install tensorflow opencv-python mediapipe numpy
```

## Usage
Clone the Repository:
```bash
git clone https://github.com/aryawidjaja/BilayerSmokingDetection.git
cd BilayerSmokingDetection
```

Place the Model and Video Files:
Ensure you have the pre-trained VGG16 model and CCTV videos in the appropriate directories as shown in the project structure.

Run the Detection Script:
Execute the main script to process the videos and detect smoking activities:
```
python main.py
```
This will process the videos and save the output in the output_videos folder. Detected smoking activities will be logged in smoking_log.csv.

## Project Description
*main.py*
The main script that processes the video files, detects skeletal poses using MediaPipe, and identifies smoking activities using a pre-trained VGG16 model.

*utils/logging.py*
A utility script for logging detected smoking activities.

*Model*
The pre-trained VGG16 model (smoking_detection_vgg16_model.keras) is used to detect the presence of a cigarette in the Region of Interest (ROI) identified by the skeletal pose analysis.

## Output
Annotated Videos: The processed videos with annotated skeletal poses and detected smoking activities are saved in the output_videos folder.
Logs: Detected smoking activities are logged in smoking_log.csv with timestamps and video file names.

## License
This project is licensed under the MIT License.
