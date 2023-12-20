import os
import shutil

# 디렉토리 경로 설정
input_dir = '../input'
dataset_train_dir = os.path.join(input_dir, 'dataset', 'train')
rembg_dataset_train_dir = os.path.join(input_dir, 'rembg_dataset_4', 'train')
yolo_dataset_train_dir = os.path.join(input_dir, 'yolo_dataset_4', 'train')
all_dataset_train_dir = os.path.join(input_dir, 'all_age_4_dataset', 'train')

# all_dataset/train 디렉토리가 없으면 생성
if not os.path.exists(all_dataset_train_dir):
    os.makedirs(all_dataset_train_dir)


# if self.age_drop and (57 <= int(age) <= 59) or (28 <= int(age) <= 30): 


# dataset/train에서 파일 복사
for folder_name in os.listdir(dataset_train_dir):
    source_folder = os.path.join(dataset_train_dir, folder_name)
    target_folder = os.path.join(all_dataset_train_dir, folder_name)
    if not os.path.exists(target_folder):
        shutil.copytree(source_folder, target_folder)
    else:
        # 중복된 폴더 처리 로직
        pass

# rembg_dataset/train에서 파일 복사 및 이름 변경
for folder_name in os.listdir(rembg_dataset_train_dir):
    new_folder_name = 'r' + folder_name
    source_folder = os.path.join(rembg_dataset_train_dir, folder_name)
    target_folder = os.path.join(all_dataset_train_dir, new_folder_name)
    age = int(new_folder_name.split('_')[-1])

    if not os.path.exists(target_folder) and not(57 <= age <= 59 or 28 <= age <= 30):
        
        shutil.copytree(source_folder, target_folder)
    else:
        # 중복된 폴더 처리 로직
        pass

# yolo_dataset/train에서 파일 복사 및 이름 변경
for folder_name in os.listdir(yolo_dataset_train_dir):
    new_folder_name = 'y' + folder_name
    source_folder = os.path.join(yolo_dataset_train_dir, folder_name)
    target_folder = os.path.join(all_dataset_train_dir, new_folder_name)
    age = int(new_folder_name.split('_')[-1])

    if not os.path.exists(target_folder) and not(57 <= age <= 59 or 28 <= age <= 30):
        shutil.copytree(source_folder, target_folder)
    else:
        # 중복된 폴더 처리 로직
        pass
