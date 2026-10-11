import onnxruntime as ort
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser(description="Score generated images with attribute classifiers")
parser.add_argument("--classifiers_dir", type=str, default=os.path.join("models", "binary_classifiers"),
                    help="Directory containing the ONNX attribute classifiers")
parser.add_argument("--output_dir", type=str, default=os.path.join("outputs", "standard", "1"),
                    help="Directory containing generated images")
args = parser.parse_args()

# Load ONNX models
sess_smile = ort.InferenceSession(os.path.join(args.classifiers_dir, "resnet18_smiling_model.onnx"))
sess_glasses = ort.InferenceSession(os.path.join(args.classifiers_dir, "resnet18_eyeglasses_model.onnx"))
sess_gender = ort.InferenceSession(os.path.join(args.classifiers_dir, "resnet18_male_model.onnx"))

def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    
    # Convert BGR to RGB (OpenCV loads as BGR)
    img = img[:, :, ::-1]
    
    # CHW format for model input
    img = img.transpose(2, 0, 1)
    
    # Apply ImageNet normalization with explicit float32 type
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((3, 1, 1))
    img = (img - mean) / std
    
    # Ensure final output is float32
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def predict(img, session):
    inputs = {session.get_inputs()[0].name: img}
    out = session.run(None, inputs)[0]
    return np.argmax(out)

# Target labels
prompt_targets = {
    0: {"smile": 0, "glasses": 0, "gender": 1},
    1: {"smile": 0, "glasses": 0, "gender": 0},
    # 2: {"smile": 1, "glasses": 1, "gender": 1},
    # 3: {"smile": 0, "glasses": 0, "gender": 0},
    # 4: {"smile": 1, "glasses": 0, "gender": 0},
    # 5: {"smile": 1, "glasses": 1, "gender": 0},
}

# Accuracy trackers
smile_correct = [0] * 6
glasses_correct = [0] * 6
gender_correct = [0] * 6
all_correct = [0] * 6  # Track when all attributes are correct
total = [0] * 6

# Output directory
output_dir = args.output_dir
print(f"Looking for images in: {output_dir}")

for i in range(0, 2):  # Range from 0 to 5 to match prompt_targets keys
    # Use glob to get all matching files
    pattern = os.path.join(output_dir, f"prompt_{i}_variant_*.png")
    files = glob(pattern)
    
    print(f"Found {len(files)} files for prompt {i}: {pattern}")
    
    target = prompt_targets[i]

    for file in tqdm(files, desc=f"Prompt {i}"):
        try:
            img = preprocess(file)
            pred_smile = predict(img, sess_smile)
            pred_glasses = predict(img, sess_glasses)
            pred_gender = predict(img, sess_gender)

            is_smile_correct = pred_smile == target["smile"]
            is_glasses_correct = pred_glasses == target["glasses"]
            is_gender_correct = pred_gender == target["gender"]
            
            if is_smile_correct:
                smile_correct[i] += 1
            if is_glasses_correct:
                glasses_correct[i] += 1
            if is_gender_correct:
                gender_correct[i] += 1
            
            # Check if all attributes are correct
            if is_smile_correct and is_glasses_correct and is_gender_correct:
                all_correct[i] += 1

            total[i] += 1
        except Exception as e:
            print(f"Error processing {file}: {e}")

# Display per-component accuracy
print("\n📊 Component-wise Accuracy per Prompt:")
for i in range(0, 2):
    if total[i] > 0:
        smile_acc = 100 * smile_correct[i] / total[i]
        glasses_acc = 100 * glasses_correct[i] / total[i]
        gender_acc = 100 * gender_correct[i] / total[i]
        all_acc = 100 * all_correct[i] / total[i]
        print(f"Prompt {i}:")
        print(f"  - Smile Accuracy   : {smile_acc:.2f}%")
        print(f"  - Glasses Accuracy : {glasses_acc:.2f}%")
        print(f"  - Gender Accuracy  : {gender_acc:.2f}%")
        print(f"  - All Attributes   : {all_acc:.2f}%")
    else:
        print(f"Prompt {i}: No valid images processed")

# Calculate overall accuracy across all prompts
overall_total = sum(total)
if overall_total > 0:
    overall_smile_acc = 100 * sum(smile_correct) / overall_total
    overall_glasses_acc = 100 * sum(glasses_correct) / overall_total
    overall_gender_acc = 100 * sum(gender_correct) / overall_total
    overall_all_acc = 100 * sum(all_correct) / overall_total
    
    print("\n📊 Overall Accuracy Across All Prompts:")
    print(f"  - Smile Accuracy   : {overall_smile_acc:.2f}%")
    print(f"  - Glasses Accuracy : {overall_glasses_acc:.2f}%")
    print(f"  - Gender Accuracy  : {overall_gender_acc:.2f}%")
    print(f"  - All Attributes   : {overall_all_acc:.2f}%")


# # Add these imports at the top
# from pytorch_fid import fid_score
# import tempfile
# import shutil
# import torch
# from multiprocessing import freeze_support

# # After your accuracy calculation code, add FID calculation:

# # Calculate FID score
# def main():
#     print("\n📊 FID Scores (Image Quality):")
#     reference_dir = "path/to/img_align_celeba"  # Directory with real face images

#     for i in range(6):
#         pattern = os.path.join(output_dir, f"prompt_{i}_variant_*.png")
#         files = glob(pattern)
        
#         if len(files) > 0:
#             # Create temporary directory for the generated images
#             with tempfile.TemporaryDirectory() as tmp_dir:
#                 # Copy generated images to temp directory
#                 for idx, file in enumerate(files):
#                     shutil.copy2(file, os.path.join(tmp_dir, f"img_{idx}.png"))
                
#                 # Calculate FID
#                 try:
#                     fid = fid_score.calculate_fid_given_paths(
#                         [reference_dir, tmp_dir],
#                         batch_size=50,
#                         device=torch.device('mps'),
#                         dims=2048
#                     )
#                     print(f"Prompt {i}: FID = {fid:.2f}")
#                 except Exception as e:
#                     print(f"Error calculating FID for prompt {i}: {e}")
#         else:
#             print(f"Prompt {i}: No images to calculate FID")

# if __name__ == "__main__":
#     freeze_support()  # Add this line to support multiprocessing
#     main()
