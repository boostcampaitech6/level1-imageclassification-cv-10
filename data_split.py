import os
import numpy as np
from data.split import MultilabelStratifiedKFold
import shutil

DATA_PATH = '../train/train/images'
DST_PATH = '../split_dataset/'

paths = []
labels = []
for path in os.listdir(DATA_PATH):
    if '._' in path:
        continue
    paths.append(os.path.join(DATA_PATH, path))
    label = []
    id, gender, race, age = path.split('_')
    
    label.append(0) if gender == 'male' else label.append(1)
    label.append(int(age))
    
    labels.append(label)
    
paths = np.array(paths)
labels = np.array(labels)

print("Data num: ", len(paths))

kfold = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for idx, (train_index, test_index) in enumerate(kfold.split(paths, labels)):
    split_folder = os.path.join(DST_PATH, 'split' + str(idx))
    save_train_folder = os.path.join(split_folder, 'train')
    save_test_folder = os.path.join(split_folder, 'test')
    os.mkdir(split_folder)
    os.mkdir(save_train_folder)
    os.mkdir(save_test_folder)
    
    train_paths, test_paths = paths[train_index], paths[test_index]
    for train_path in train_paths:
        shutil.copytree(str(train_path), os.path.join(save_train_folder, train_path.split('/')[-1]))
    
    for test_path in test_paths:
        shutil.copytree(str(test_path), os.path.join(save_test_folder, test_path.split('/')[-1]))