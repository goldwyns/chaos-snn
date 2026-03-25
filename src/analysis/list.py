import os

base = "D:/Research_Experiments/stag-snn/chaos-snn/analysis/"

print("All files including subfolders (relative to /analysis/):")
for root, dirs, files in os.walk(base):
    for file in files:
        # Get relative path from base
        rel_path = os.path.relpath(os.path.join(root, file), base)
        print("/analysis/" + rel_path.replace("\\", "/"))