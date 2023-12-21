import numpy as np
import os
from PIL import Image
import PIL

import shutil
import random

# 일정한 데이터 생성을 확인
random.seed(42222)     

# 이미지 파일들 저장된 경로 설정
data_directory = './input/dataset/train/'

def split_profile_by_gender_60s(profiles:list) ->list :
    """성별별로 데이터를 나누는 함수

    Args:
        profiles (list): list of profiles ex) 000001_female_Asian_45

    Returns:
        list: [male,female]
    """
    male,female = [], []        
    for profile in profiles:
        # 이미 생성한 데이터를 제외한 데이터들만 return
        if "Fake" in profile: 
            continue
        id, gender,species,age = profile.split("_")
        if age == "60":
            if gender == "male":
                male.append(os.path.join(data_directory,profile))
            else:
                female.append(os.path.join(data_directory,profile))
    return [male,female]

def make_folder(save_dir:str) -> None:
    """Make directory if save_dir is invalid

    Args:
        save_dir (str): Directory Path saved images
    """
    if os.path.isdir(save_dir):
        return
    else:
        os.mkdir(save_dir)
        
def not_random_make_fakes_by_gender(gender:str,profiles:list,save_dir=data_directory)->None:
    """Mixup not Randomly but Sequentially

    Args:
        gender (str): gender for split profiles
        profiles (list): profiles splitted by gender
        save_dir (str, optional): directory path for saving. Defaults==data_directory.
    """
    length = len(profiles)
    id = 0
    ids = sorted(os.listdir(save_dir))
    if ids:
        id = int(sorted(os.listdir(save_dir))[-1].split("_")[0])
    for i in range(length//2):
        id += 1
        save_profile_dir = f"{id:0>6}_{gender}_Fake_60"
        os.mkdir(os.path.join(save_dir,save_profile_dir))
        j = i+length//2

        make_images(i,j,save_dir,save_profile_dir,profiles)
        
def random_make_fakes_by_gender(gender:str,profiles:list,save_dir=data_directory)->None:       
    """Mixup Randomly

    Args:
        gender (str): gender for split profiles
        profiles (list): profiles splitted by gender
        save_dir (str, optional): directory path for saving. Defaults==data_directory.
    """
    limit = len(profiles)//2
    splited_A = set(random.sample([i for i in range(len(profiles))],limit))
    splited_B = list(set([i for i in range(len(profiles))]) - splited_A)
    splited_A = list(splited_A)
    
    id = 0
    ids = sorted(os.listdir(save_dir))
    if ids:
        id = int(sorted(os.listdir(save_dir))[-1].split("_")[0])
    for i in range(len(splited_A)):
        id += 1
        save_profile_dir = f"{id:0>6}_{gender}_Fake_60"
        os.mkdir(os.path.join(save_dir,save_profile_dir))

        make_images(splited_A[i],splited_B[i],save_dir,save_profile_dir,profiles)

def make_images(profile_i:str,profile_j:str,save_dir:str,save_profile_dir:str,profiles:list)->None:    
    """Making and Saving Mixup images.

    Args:
        profile_i (str): profile for mixup.
        profile_j (str): profile for mixup.
        save_dir (str): directory saved and saving images.
        save_profile_dir (str): profile directory saving made images.
    """
    images = ["incorrect_mask","mask1","mask2","mask3","mask4","mask5","normal"]
    ext = ".jpg"
    # all
    for image in images:
        image_A = np.array(Image.open(os.path.join(profiles[profile_i], image+ext)))//2
        image_B = np.array(Image.open(os.path.join(profiles[profile_j], image+ext)))//2
        new_image = (image_A+image_B)
        img = PIL.Image.fromarray(new_image)
        img.save(os.path.join(save_dir,save_profile_dir,image+ext))    
    

def not_random_make_fake_pics(save_dir:str=data_directory):
    """Main function for not random mixup

    Args:
        save_dir (str, optional): Saving and Saved Image Directory. Defaults to data_directory.
    """
    make_folder(save_dir)
    not_random_make_fakes_by_gender("male",male,save_dir)
    not_random_make_fakes_by_gender("female",female,save_dir)
    print("Make Done.")

def random_make_fake_pics(save_dir:str=data_directory):
    """Main function for not random mixup

    Args:
        save_dir (str, optional): Saving and Saved Image Directory Path. Defaults to data_directory.
    """
    make_folder(save_dir)
    random_make_fakes_by_gender("male",male,save_dir)
    random_make_fakes_by_gender("female",female,save_dir)
    print("Randomly Make Done.")

def rm_fake_pics(save_dir:str=data_directory):                  # 만든 fake 디렉토리 전체 제거
    """Remove every made pictures

    Args:
        save_dir (str, optional): Delete Image Directory Path. Defaults to data_directory.
    """
    if not os.path.isdir(save_dir):
        print("No folder")
        return
    for fake in [i for i in os.listdir(save_dir) if "Fake" in i]:
        fake_dir = os.path.join(save_dir,fake)
        shutil.rmtree(fake_dir)
    print("Remove Done.")

# data_directory의 os path 설정
profiles = os.listdir(data_directory)

# profiles except invalid datas
profiles = [profile for profile in profiles if not profile.startswith(".")]
male,female = split_profile_by_gender_60s(profiles)

#rm_fake_pics()
random_make_fake_pics()