# ⭐ CV 10팀 소개 

**`영-차 영-차! 마지막까지 한 걸음씩 나아가자`**  
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
<br/><br/><br/>
> ### 👩‍💻 팀 역할
|이름|역할|
|------|---|
|박찬종|하이퍼 파라미터 실험, 데이터 증강 실험, 싱글 및 멀티 헤드 실험, 코드 문서화|
|양우희|싱글 및 멀티헤드, 앙상블 모델 실험, github 관리, 코드 문서화|
|유한준|데이터 증강 실험, 싱글 모델 실험|
|이영진|데이터 증강, 클래스 별 모델 학습, 손실함수 실험, 코드 리팩토링|
|정소윤|데이터 전처리, 싱글 및 멀티 헤드 모델 실험|
|최시현|데이터 전처리(배경 제거, 얼굴 탐지) 실험|


<br/><br/><br/>
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
<br/><br/><br/>

> ### 🐋 Training
config 폴더 안 <a href = "https://github.com/boostcampaitech6/level1-imageclassification-cv-10/blob/main/config/base.yml">yaml 파일</a>에서 training 환경 조정 가능합니다. 
 - Mask Model
```bash
python train.py --exp-name <이름> --dataset OnlyMaskDataset --model EfficientNetV2m --criterion cross_entropy --augmentation AutoAugmentation
```
- Gender Model
```bash
python train.py --exp-name <이름> --dataset OnlyGenderDataset --model EfficientNetV2m --criterion focal --augmentation AutoAugmentation --optimizer AdamW --schedular cosine
```
- Age Model
  <br/>
train 하기 이전에 데이터가 적은 30대 남자 및 60대 이상에 대해서 offline mixup augmentation을 진행
```bash
python train.py --exp-name <이름> --dataset OnlyMaskDataset --model EfficientNetV2m --criterion focal --age-drop True  
```
<br/><br/><br/>

> ### 🔎 Inference
각 label에 대한 Model을 Hard Voting ensemble 진행 
<br/>
inference_3m 파일의 mask_model, gender_model, age_model 변수를 수정 후 아래 코드 실행 
```bash
python inference_3m.py 
```
<br/><br/><br/>
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
 ┣ 📜inference.py
 ┣ 📜inference_3m.py
 ┣ 📜multi_inference.py
 ┣ 📜multi_train.py
 ┣ 📜train.py
 ┣ 📜train_regression.py
 ┣ 📜voting.py
 ┣ 📜requirements.txt
 ┗ 📜README.md

``` 


