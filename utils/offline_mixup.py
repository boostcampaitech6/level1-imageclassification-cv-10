import numpy as np
import os
from PIL import Image
import PIL

import shutil
import random

def split_profile_by_gender_60s(profiles:list) ->list :
    """
    프로필 목록을 성별로 나누는 함수.

    Args:
        profiles (list): 프로필 목록. 예) 000001_female_Asian_45

    Returns:
        list: [남성 프로필 목록, 여성 프로필 목록]
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
    """
    지정된 경로에 디렉토리를 생성한다. 이미 디렉토리가 존재하는 경우 생성하지 않는다.

    Args:
        save_dir (str): 생성할 디렉토리 경로.
    """
    if os.path.isdir(save_dir):
        return
    else:
        os.mkdir(save_dir)
        
def not_random_make_fakes_by_gender(gender:str,profiles:list,save_dir)->None:
    """
    성별에 따라 프로필 목록을 순차적으로 믹스업하여 가짜 이미지를 생성한다.

    Args:
        gender (str): 분류할 성별.
        profiles (list): 성별로 분류된 프로필 목록.
        save_dir (str): 생성된 이미지를 저장할 디렉토리 경로.
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
        
def random_make_fakes_by_gender(gender:str,profiles:list,save_dir)->None:       
    """
    성별에 따라 프로필 목록을 무작위로 믹스업하여 가짜 이미지를 생성한다.

    Args:
        gender (str): 분류할 성별.
        profiles (list): 성별로 분류된 프로필 목록.
        save_dir (str): 생성된 이미지를 저장할 디렉토리 경로.
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
        save_profile_dir = f"{id:0>6}_{gender}_Fake2_60"
        os.mkdir(os.path.join(save_dir,save_profile_dir))

        make_images(splited_A[i],splited_B[i],save_dir,save_profile_dir,profiles)

def make_images(profile_i:str,profile_j:str,save_dir:str,save_profile_dir:str,profiles:list)->None:    
    """
    두 프로필의 이미지를 혼합하여 새로운 이미지를 생성하고 저장한다.

    Args:
        profile_i (str): 혼합할 첫 번째 프로필.
        profile_j (str): 혼합할 두 번째 프로필.
        save_dir (str): 이미지가 저장될 디렉토리.
        save_profile_dir (str): 생성된 이미지가 저장될 프로필 디렉토리.
        profiles (list): 사용할 프로필 목록.
    """
    images = ["incorrect_mask","mask1","mask2","mask3","mask4","mask5","normal"]
    ext = ".jpg"
    # all
    weight_A = 0.5  # 70% weight for the first image
    weight_B = 0.5  # 30% weight for the second image
    for image in images:
        image_A = np.array(Image.open(os.path.join(profiles[profile_i], image+ext)))
        image_B = np.array(Image.open(os.path.join(profiles[profile_j], image+ext)))
        new_image = np.clip((image_A * weight_A + image_B * weight_B), 0, 255).astype(np.uint8)
        img = PIL.Image.fromarray(new_image)
        img.save(os.path.join(save_dir,save_profile_dir,image+ext))    
    

def not_random_make_fake_pics(save_dir):
    """
    프로필 목록을 순차적으로 믹스업하여 가짜 이미지를 생성하는 메인 함수.

    Args:
        save_dir (str): 생성된 이미지를 저장할 디렉토리 경로.
    """
    make_folder(save_dir)
    not_random_make_fakes_by_gender("male",male,save_dir)
    not_random_make_fakes_by_gender("female",female,save_dir)
    print("Make Done.")

def random_make_fake_pics(save_dir):
    """
    프로필 목록을 무작위로 믹스업하여 가짜 이미지를 생성하는 메인 함수.

    Args:
        save_dir (str): 생성된 이미지를 저장할 디렉토리 경로.
    """
    make_folder(save_dir)
    random_make_fakes_by_gender("male",male,save_dir)
    random_make_fakes_by_gender("female",female,save_dir)
    print("Randomly Make Done.")

def rm_fake_pics(save_dir):                  # 만든 fake 디렉토리 전체 제거
    """
    생성된 모든 가짜 이미지를 삭제하는 함수.

    Args:
        save_dir (str): 삭제할 이미지가 있는 디렉토리 경로.
    """
    if not os.path.isdir(save_dir):
        print("No folder")
        return
    for fake in [i for i in os.listdir(save_dir) if "Fake" in i]:
        fake_dir = os.path.join(save_dir,fake)
        shutil.rmtree(fake_dir)
    print("Remove Done.")

if __name__ == '__main__':
    # 일정한 데이터 생성을 확인
    random.seed(42)     

    # 이미지 파일들 저장된 경로 설정
    data_directory = '/data/ephemeral/home/input/dataset/train'

    # data_directory의 os path 설정
    profiles = os.listdir(data_directory)

    # profiles except invalid datas
    profiles = [profile for profile in profiles if not profile.startswith(".")]
    male,female = split_profile_by_gender_60s(profiles)

    # random 하게 데이터 생성
    random_make_fake_pics()

    # 생성된 데이터 전체 삭제
    # rm_fake_pics()