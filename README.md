# fall-detection-cnn

This project is a deep learning-based indoor human pose recognition and fall detection system, utilizing the MobileNetV2 architecture. It supports both PyTorch and ONNX formats, and provides scripts for real-time video inference and batch evaluation.

---

## Dataset Source

This project uses the [Fall Detection Dataset](https://falldataset.com/):
- Captured by a Kinect sensor, containing 5 pose categories (standing, sitting, lying, bending, crawling) and an empty class.
- Each folder contains sequential actions, with a CSV annotation file for each.
- For more details and citation, see [falldataset.com](https://falldataset.com/).

---

## Project Structure

```
fall-detection-cnn/
├── data/           # Data and annotations
│   └── labels.csv
├── models/         # Trained models
│   ├── pose_classifier.pt
│   └── pose_classifier.onnx
├── results/        # Training and evaluation charts
│   ├── acc_curve.png
│   └── confusion_matrix.png
├── scripts/        # Main Python scripts
│   ├── train.py
│   ├── evaluate_onnx.py
│   ├── demo_onnx_video.py
│   └── prepare_data.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

1. Install Python 3.8+
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Data Preparation

Place the original dataset according to the [falldataset.com](https://falldataset.com/) structure, and modify the `source_root` path in `scripts/prepare_data.py` as needed.

Run:
```bash
python scripts/prepare_data.py
```
- This will consolidate all images and annotations, generating `data/images/` and `data/labels.csv`.

---

### 2. Model Training

```bash
python scripts/train.py
```
- The dataset will be automatically split into 80% training and 20% validation sets.
- Training and validation accuracy and loss will be displayed during training.
- The best model will be saved in `models/`, and training curves will be saved in `results/`.

---

### 3. Batch Evaluation (ONNX)

```bash
python scripts/evaluate_onnx.py
```
- Uses `models/pose_classifier.onnx` to perform batch inference on `data/images/`.
- Generates a confusion matrix and classification report, saved as `results/confusion_matrix.png`.

---

### 4. Real-time Video Inference Demo

```bash
python scripts/demo_onnx_video.py
```
- Modify the `video_path` variable in the script to your video file path.
- The script will display real-time pose classification results on the video.

---

## Main Dependencies

- torch, torchvision
- pandas, numpy
- onnxruntime
- opencv-python
- scikit-learn, matplotlib, seaborn, tqdm, Pillow

---

## Notes

- Please adjust script paths according to your actual data locations.
- `models/`, `data/images/`, and large files are included in `.gitignore` and are not recommended to be uploaded to GitHub.
- Dataset source: [falldataset.com](https://falldataset.com/)
- For any questions or suggestions, feel free to open an issue!
