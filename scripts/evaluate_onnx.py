import os
import pandas as pd
import numpy as np
import onnxruntime
from PIL import Image
import cv2
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

IMG_SIZE = 192
CSV_PATH = "labels.csv"
IMG_DIR = "images"
ONNX_PATH = "pose_classifier.onnx"
LABELS = ["standing", "sitting", "lying", "bending", "crawling"]
label_map = {l: i for i, l in enumerate(LABELS)}
inv_label_map = {i: l for l, i in label_map.items()}

def preprocess_image(image_path, size=IMG_SIZE):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (size, size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    return image

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["label"])
df = df[df["label"].isin(LABELS)]
y_true = []
y_pred = []

session = onnxruntime.InferenceSession(ONNX_PATH)

print("Batch inference started...")
for idx, row in df.iterrows():
    img_path = os.path.join(IMG_DIR, row["image_path"])
    label = row["label"]
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue
    input_data = preprocess_image(img_path)
    outputs = session.run(None, {"input": input_data})
    pred = np.argmax(outputs[0])
    y_true.append(label_map[label])
    y_pred.append(pred)

cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=LABELS, digits=3)

print("\n=== Confusion Matrix ===")
print(cm)
print("\n=== Classification Report ===")
print(report)

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
print(f"Macro Precision: {precision:.3f}")
print(f"Macro Recall:    {recall:.3f}")
print(f"Macro F1-score:  {f1:.3f}")

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title(f"Confusion Matrix\nMacro Precision: {precision:.3f}  Macro Recall: {recall:.3f}  Macro F1: {f1:.3f}")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\nconfusion_matrix.png saved") 