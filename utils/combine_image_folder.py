import os
import shutil

"""
-> 소윤님이 작성하신 train_va_split.py로 분할한 뒤에 사용하시면 됩니다.

dataset_train_dir = ../input/dataset/train : 원본 훈련 데이터 폴더에 있는 이미지들과
rembg_dataset_train_dir = ../input/rembg_dataset/train : rembg로 변환된 훈련 데이터의 이미지들을 합치는 코드 파일입니다.


all_dataset_train_dir은 이 합치는 파일이 저장되는 경로 입니다. -> 알아서 파일명 기입하시면 됩니다


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
        # 중복된 폴더 처리 로직
        pass

# rembg_dataset/train에서 파일 복사, 이름 중복에 의해 변경
for folder_name in os.listdir(rembg_dataset_train_dir):
    new_folder_name = 'r' + folder_name
    source_folder = os.path.join(rembg_dataset_train_dir, folder_name)
    target_folder = os.path.join(all_dataset_train_dir, new_folder_name)
    age = int(new_folder_name.split('_')[-1])

    if not os.path.exists(target_folder) and not(57 <= age <= 59 or 28 <= age <= 30): # age.drop 안하실거면 조건문 빼면 됩니다.
        
        shutil.copytree(source_folder, target_folder)
    else:
        # 중복된 폴더 처리 로직
        pass
