# ⭐ CV 10팀 소개 

![영-차 영-차 (1)](https://github.com/boostcampaitech6/level1-imageclassification-cv-10/assets/67350632/098b274b-42d1-4759-aa26-8f130c029330)
<br/>

> ### 🏃 멤버
<table>
    <tr height="160px">
        <td align="center" width="150px">
            <a href="https://github.com/FinalCold"><img height="120px" width="120px" src="https://github.com/boostcampaitech6/level1-imageclassification-cv-10/assets/76814748/4c47db91-cf83-473c-888f-75d3a5573ac8"></a>
            <br/>
            <a href="https://github.com/FinalCold"><strong>박찬종</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/woohee-yang"><img height="120px" width="120px" src="https://github.com/boostcampaitech6/level1-imageclassification-cv-10/assets/76814748/0548b5bf-d3f5-4c9e-a7e5-c2733bc48355"/></a>
            <br/>
            <a href="https://github.com/woohee-yang"><strong>양우희</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/lukehanjun"><img height="120px" width="120px" src="https://github.com/boostcampaitech6/level1-imageclassification-cv-10/assets/76814748/bfb37916-ed9e-405c-a981-70c0fdaa53be"/></a>
            <br/>
            <a href="https://github.com/lukehanjun"><strong>유한준</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/jinida"><img height="120px" width="120px" src=https://github.com/boostcampaitech6/level1-imageclassification-cv-10/assets/76814748/d3891701-98db-4382-abc0-3a0ab64e976e"/></a>
            <br />
            <a href="https://github.com/jinida"><strong>이영진</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/soyoonjeong"><img height="120px" width="120px" src="https://github.com/boostcampaitech6/level1-imageclassification-cv-10/assets/76814748/912e24ed-cc51-49ab-90fd-24ef0df7ce0b"/></a>
            <br />
            <a href="https://github.com/soyoonjeong"><strong>정소윤</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/classaen7"><img height="120px" width="120px" src="https://github.com/boostcampaitech6/level1-imageclassification-cv-10/assets/76814748/d5fc34c5-dfd2-400e-86d5-b32976aeb928"/></a>
              <br />
              <a href="https://github.com/classaen7"><strong>최시현</strong></a>
              <br />
          </td>
    </tr>
</table>  
<br/><br/><br/>

# 😷 마스크 착용 상태 분류 대회
> ### 🏆 대회개요
- `Multi Label Image Classification`
- `마스크를 쓰거나 쓰지 않은 아시아인들의 얼굴 사진을 보고 마스크 착용 상태와 성별, 연령대를 추론하는 문제`

<br/>
> ### 👩‍💻 팀 역할
|이름|역할|
|------|---|
|박찬종|하이퍼 파라미터 실험, 데이터 증강 실험, 싱글 및 멀티 헤드 실험, 코드 문서화|
|양우희|싱글 및 멀티헤드, 앙상블 모델 실험, github 관리, 코드 문서화|
|유한준|데이터 증강 실험, 싱글 모델 실험|
|이영진|데이터 증강, 클래스 별 모델 학습, 손실함수 실험, 코드 리팩토링|
|정소윤|데이터 전처리, 싱글 및 멀티 헤드 모델 실험|
|최시현|데이터 전처리(배경 제거, 얼굴 탐지) 실험|


<br/>
> ### ⏰ WBS
<img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-10/assets/76814748/7dc7c21d-d41c-4b28-907e-4e9a88543c44">
<a href="https://docs.google.com/spreadsheets/d/14qhqnSzOfvZsKYnmQyikYhVdpoUNx1-tdBY_Zkixy9c/edit#gid=0"> 📁 WBS</a>
<br/><br/><br/>

> ### 💻 개발 환경
```bash
- Language : Python
- Environment
  - CPU : Intel(R) Xeon(R) Gold 5120
  - GPU : Tesla V100-SXM2 32GB × 1
- Framework : PyTorch
- Collaborative Tool : Git, Wandb, Notion
```
<br/><br/><br/>
> ### 🔥 필수 라이브러리 설치
---
``` bash
pip install -r requirements.txt
```
<br/>

> ### 💽 Dataset
- 총 사진 개수 :  31500장 
- 원본 이미지 크기 : (512, 384)
- Label : 마스크 착용 상태, 성별, 연령대
- 마스크 착용 상태 (3 classes): 정상 착용, 비정상 착용, 미착용
- 성별 (2 classes): 남성, 여성
- 연령대 (3 classes): 30대 이하, 30대 이상 60대 미만, 60대 이상

<br/>

> ### 📊 EDA
<p align = "center">
<img height="300px" width="500px" src = "https://github.com/boostcampaitech6/level1-imageclassification-cv-10/assets/76814748/a5319723-0254-4b4a-9b68-74035e965f9a">
<p/>

- mask와 gender에 대한 class imbalance도 있었지만, 두 레이블에 대해 각각 single model을  실험한 결과가 좋았기 때문에 가장 문제가 된 레이블은 age 였음
- 총 3가지 클래스에 대해 분석한 결과, 60대 이상의 데이터가 가장 적어 augmentation이 꼭 필요한 것으로 보였음
- 또한, 50대 후반 데이터가 60대 데이터에 비해 너무 많아 서로를 구별하기 어려울 수 있기 때문에 57-59세 데이터는 임의로 제거하기로 함. 동일하게 20대 후반 데이터도 30대 초반과 혼동할 가능성이 있어 28-29세를 제거하였음
- 20대와 30대를 결정지을 수 있는 30대 초반의 남성 데이터가 극도로 적기 때문에 이 부분에 대해서도 augmentation을 진행하기로 함

<br/><br/>
> ### 🚀 Model
```bash
최종 모델 : Multi Task Model + Hard voting ensemble
```
1) **`Mask Model`** : Autoaugmentation + EfficientnetV2m + CrossEntropy
 - 다양한 Augmentation 기법을 실험했지만, 예상과는 다르게 분할된 데이터셋에서 성능의 차이가 크게 나타났음. 그에 따라 상대적으로 가장 강인한 효과를 보인 Autoaugmentation을 채택하게 되었음
 - 손실 함수도 상기 이유로 Cross Entropy를 채택하여 학습하였음.
2) **`Gender Model`** : Autoaugmentation + EfficientnetV2m + 입력 사이즈 [384, 288] +  Focal loss + AdamW + Cosine Scheduler
- 성별 분류 모델은 성능이 가장 좋았던 EfficientNetV2m Backbone에 AutoAugmentation을 적용한 모델을 최종 선택함
- 다양한 입력 사이즈로 실험한 결과 ½원본 크기에서 가장 높은 성능을 보여주어 채택함
- 이외에 augmentation 기법들 중 Autoaugmentation 외에는 오히려 낮은 성능을 보여주었음
3) **`Age Model`** : Custom data augmentation + Custom Mixup to 30s male, 60s + EfficientnetV2m + Hard-voting ensemble
 - 다른 분류 문제보다 클래스 불균형이 심하여 이를 해결하기위해 여러가지 기법을 사용함.
 - Augmentation 및 Loss는 e)의 실험에서 가장 좋은 효과가 나온 AutoAugmentation, FocalWithSmooting등을 사용하였음.
 그 외 기법으로 30~59 / 60~ 두 클래스의 구분을 뚜렷하게 하기위해 drop age을 적용함.
 - 또한 데이터가 적은 30대의 남자 및 60대 이상에 대해서 mixup을 사용하여 offline augmentation을 진행함.
 - 마지막으로 데이터 별 성능 편차가 있으므로 8:2로 나눈 데이터 셋들에 대해 각각의 모델을 학습시키고 최종으로 나온 모델들에 대해서 Hard-Voting을 진행함.

<br/><br/>
> ### 🐋 Training
config 폴더 안 <a href = "https://github.com/boostcampaitech6/level1-imageclassification-cv-10/blob/main/config/base.yml">yaml 파일</a>에서 training 환경 조정 가능합니다. 
 - Mask Model
```bash
python single_train.py --exp-name <이름> --dataset OnlyMaskDataset --model EfficientNetV2m --criterion cross_entropy --augmentation AutoAugmentation
```
- Gender Model
```bash
python single_train.py --exp-name <이름> --dataset OnlyGenderDataset --model EfficientNetV2m --criterion focal --augmentation AutoAugmentation --optimizer AdamW --schedular cosine
```
- Age Model 
train 하기 이전에 데이터가 적은 30대 남자 및 60대 이상에 대해서 offline mixup augmentation을 진행
```bash
python single_train.py --exp-name <이름> --dataset OnlyMaskDataset --model EfficientNetV2m --criterion focal --age-drop True  
```
<br/><br/>

> ### 🔎 Inference
각 label에 대한 Model을 Hard Voting ensemble 진행 
inference_3m 파일의 mask_model, gender_model, age_model 변수를 수정 후 
```bash
python single_inference.py 
```
<br/><br/>
> ### 📂 File Tree
```bash
  📦level1-imageclassification-cv-10
 ┣ 📂EDA
 ┃ ┗ 📜eda.ipynb
 ┣ 📂config
 ┃ ┗ 📜base.yml
 ┣ 📂data
 ┃ ┣ 📜augmentation.py
 ┃ ┣ 📜dataloader.py
 ┃ ┣ 📜datasets.py
 ┣ 📂model
 ┃ ┗ 📜model.py
 ┣ 📂utils
 ┃ ┣ 📜argparsers.py
 ┃ ┣ 📜combine_image_folder.py
 ┃ ┣ 📜evaluation.py
 ┃ ┣ 📜hpo.py
 ┃ ┣ 📜img_preprocess.py
 ┃ ┣ 📜logger.py
 ┃ ┣ 📜loss.py
 ┃ ┣ 📜lr_scheduler.py
 ┃ ┣ 📜metric.py
 ┃ ┣ 📜plot.py
 ┃ ┣ 📜train_val_split.py
 ┃ ┣ 📜data_split.py  
 ┃ ┣ 📜util.py
 ┃ ┗ 📜yolov8n-face.pt
 ┣ 📜single_inference.py
 ┣ 📜multi_inference.py
 ┣ 📜single_train.py
 ┣ 📜multi_train.py
 ┣ 📜voting.py
 ┣ 📜requirements.txt
 ┗ 📜README.md

``` 


