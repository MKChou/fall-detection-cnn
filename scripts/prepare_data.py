import os
import pandas as pd
from shutil import copyfile


source_root = "D:/D/Falldataset"
output_image_dir = os.path.join(source_root, "images")
output_csv_path = os.path.join(source_root, "labels.csv")

os.makedirs(output_image_dir, exist_ok=True)


label_map = {
    0: "empty",
    1: "standing",
    2: "sitting",
    3: "lying",
    4: "bending",
    5: "crawling"
}


video_folders = [
    "1219", "1260", "1301", "1378", "1790", "1843", "1954",
    "489", "569", "581", "722", "731", "758", "807"
]

all_data = []

for folder in video_folders:
    subfolder = os.path.join(source_root, folder, folder) 
    label_csv = os.path.join(subfolder, "labels.csv")
    rgb_folder = os.path.join(subfolder, "rgb")

    if not os.path.isfile(label_csv) or not os.path.isdir(rgb_folder):
        print(f"Skip {folder}: labels.csv or rgb/ not found")
        continue

    df = pd.read_csv(label_csv)
    for _, row in df.iterrows():
        idx = int(row["index"])
        cls = int(row["class"])

        if cls == 0:
            continue  

        label = label_map.get(cls)
        filename = f"rgb_{idx:04}.png"
        src = os.path.join(rgb_folder, filename)
        dst_name = f"{folder}_{filename}"
        dst = os.path.join(output_image_dir, dst_name)

        if os.path.exists(src):
            copyfile(src, dst)
            all_data.append([dst_name, label])
        else:
            print(f"Image not found: {src}")


df_out = pd.DataFrame(all_data, columns=["image_path", "label"])
df_out.to_csv(output_csv_path, index=False)

print(f"\nDone. Processed {len(df_out)} images.")
print("Class distribution:")
print(df_out["label"].value_counts())
