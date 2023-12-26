import os
import shutil

"""
원본 데이터와 전처리된 데이터를 하나의 폴더로 합치는 코드 파일입니다.

dataset_train_dir = ../input/dataset/train : 원본 훈련 데이터의 이미지 경로
rembg_dataset_train_dir = ../input/rembg_dataset/train : rembg로 변환된 훈련 데이터의 이미지 경로
all_dataset_train_dir = ../input/all_dataset/train : 원본 + 전처리 이미지의 train 데이터가 복사되어 합쳐진 폴더 경로
"""

# 디렉토리 경로 설정
input_dir = '../input'
dataset_train_dir = os.path.join(input_dir, 'dataset', 'train')
rembg_dataset_train_dir = os.path.join(input_dir, 'rembg_dataset', 'train')
all_dataset_train_dir = os.path.join(input_dir, 'all_dataset', 'train')

# 새로 저장할 디렉토리 생성
if not os.path.exists(all_dataset_train_dir):
    os.makedirs(all_dataset_train_dir)

# dataset/train에서 파일 복사
for folder_name in os.listdir(dataset_train_dir):
    source_folder = os.path.join(dataset_train_dir, folder_name)
    target_folder = os.path.join(all_dataset_train_dir, folder_name)
    if not os.path.exists(target_folder):
        shutil.copytree(source_folder, target_folder)
    else:
        pass

# rembg_dataset/train에서 파일 복사
for folder_name in os.listdir(rembg_dataset_train_dir):
    new_folder_name = 'r' + folder_name
    source_folder = os.path.join(rembg_dataset_train_dir, folder_name)
    target_folder = os.path.join(all_dataset_train_dir, new_folder_name)
    age = int(new_folder_name.split('_')[-1])

    if not os.path.exists(target_folder) and not(57 <= age <= 59 or 28 <= age <= 30): # age.drop 안하실거면 조건문 빼면 됩니다.
        
        shutil.copytree(source_folder, target_folder)
    else:
        pass
