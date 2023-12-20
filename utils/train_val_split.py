import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import shutil

# train, validation split
data_dir = "../input/train/yolo_images/" # 각자 이미지 경로 넣어주세요 
profiles = os.listdir(data_dir) 
profiles = [profile for profile in profiles if not profile.startswith(".")]

id_list = []
new_label_list = []
val_ratio = 0.6
for index, profile in enumerate(profiles):
    id, gender, race, age = profile.split("_")
    if gender == 'male':
        gender_label = 0
    else:
        gender_label = 1
    age_label = int(age) // 2
        
    id_list.append(id)
    new_label_list.append(gender_label * 30 + age_label)

id_list = np.array(id_list)
new_label_list = np.array(new_label_list)

x_train, x_val, y_train, y_val = train_test_split(id_list, new_label_list, test_size = val_ratio, random_state =777, stratify = new_label_list)

# 폴더 복사 후 저장 
train_dir = '../input/yolo_dataset_4/train/' # 맘대로 조정하시면 됩니다!
val_dir = '../input/yolo_dataset_4/val/' # 맘대로 조정하시면 됩니다!

for dir in [train_dir, val_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"{dir} 폴더가 생성되었습니다.")

for profile in profiles:
    id, gender, race, age = profile.split("_")
    if id in x_train:
        shutil.copytree(data_dir+profile, train_dir+profile)
    if id in x_val:
        shutil.copytree(data_dir+profile, val_dir+profile)