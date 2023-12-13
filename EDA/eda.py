import os
from glob import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
from utils.util import increment_path, create_directory
import matplotlib.pyplot as plt
import seaborn as sns

def get_img_stats(img_dir, img_ids):
    img_info = dict(heights=[], widths=[], means=[], stds=[])
    for img_id in img_ids:
        for path in glob(os.path.join(img_dir, img_id, '*')):
            img = np.array(Image.open(path))
            h, w, _ = img.shape
            img_info['heights'].append(h)
            img_info['widths'].append(w)
            img_info['means'].append(img.mean(axis=(0,1)))
            img_info['stds'].append(img.std(axis=(0,1)))
    return img_info

class cfg:
    data_dir = "/usr/src/app/BoostCampl_Lv1/train/train"
    img_dir = f'{data_dir}/images'
    df_path = f'{data_dir}/train.csv'
    save_dir = "/usr/src/app/BoostCampl_Lv1/project/result/eda"


save_path = increment_path(cfg.save_dir)
create_directory(save_path)

num2class = ['incorrect_mask', 'mask1', 'mask2', 'mask3',
             'mask4', 'mask5', 'normal']
class2num = {k: v for v, k in enumerate(num2class)}


df = pd.read_csv(cfg.df_path)

img_info = get_img_stats(cfg.img_dir, df.path.values)
log_file = open(os.path.join(save_path, 'eda.txt'), 'w')

log_file.write("[Image Information]\n")
log_file.write(f'Total number of people is {len(df)}'+'\n')
log_file.write(f'Total number of images is {len(df) * 7}'+'\n')
log_file.write(f'Minimum height for dataset is {np.min(img_info["heights"])}'+'\n')
log_file.write(f'Maximum height for dataset is {np.max(img_info["heights"])}'+'\n')
log_file.write(f'Average height for dataset is {int(np.mean(img_info["heights"]))}'+'\n')
log_file.write(f'Minimum width for dataset is {np.min(img_info["widths"])}'+'\n')
log_file.write(f'Maximum width for dataset is {np.max(img_info["widths"])}'+'\n')
log_file.write(f'Average width for dataset is {int(np.mean(img_info["widths"]))}'+'\n')
log_file.write(f'RGB Mean: {np.mean(img_info["means"], axis=0) / 255.}'+'\n')
log_file.write(f'RGB Standard Deviation: {np.mean(img_info["stds"], axis=0) / 255.}'+'\n')
log_file.write('\n')

plt.figure(figsize=(6, 4.5)) 
ax = sns.countplot(data = df, x='gender')

plt.xticks(np.arange(2), ['female', 'male'] )
plt.title('Sex Ratio',fontsize= 14)
plt.xlabel('')
plt.ylabel('Number of images')

plt.savefig(os.path.join(save_path, 'gender_ratio.jpg'))

plt.clf()
plt.figure(figsize=(12, 9)) 
ax = sns.countplot(data = df, x='age')

plt.title('Age Ratio', fontsize= 14)
plt.xlabel('')
plt.ylabel('Number of images')
plt.savefig(os.path.join(save_path, 'Age_ratio.jpg'))

bins = [0, 30, 60, np.inf]
labels = ['age < 30', '30 <= age < 60', 'age <= 60']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

group_age_counts = df['age_group'].value_counts().sort_index()

plt.clf()
plt.figure(figsize=(6, 4.5)) 
ax = sns.countplot(data = df, x='age_group')

plt.title('Group age Ratio', fontsize= 14)
plt.xlabel('')
plt.ylabel('Number of images')
plt.savefig(os.path.join(save_path, 'group_age_ratio.jpg'))

log_file.write("[Gender count]\n")
gender_counts = df['gender'].value_counts()
for gender in gender_counts.keys():
    log_file.write(str(gender) + ": " + str(gender_counts[gender]) + "\n")

log_file.write("\n[Age count]\n")
age_counts = df['age'].value_counts()
for age in age_counts.keys():
    log_file.write("\'" + str(age) + "\'" + ": " + str(age_counts[age]) + "\n")

log_file.write("\n[Group age count]\n")
group_age_counts = df['age_group'].value_counts()
for group_age in group_age_counts.keys():
    log_file.write("\'" + str(group_age) + "\'" + ": " + str(group_age_counts[group_age]) + "\n")
    
log_file.close()
