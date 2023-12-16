import os
import sys
from PIL import Image

from ultralytics import YOLO
from rembg import remove

from tqdm import tqdm

def yolo_detection(img_path, model, limit_size = 0):
    image = Image.open(img_path)

    result = model(img_path,verbose=False)
    box = result[0].boxes.xyxy.tolist()
    box.sort(key=lambda x:(x[2]-x[0])*(x[3]-x[1]), reverse=True)

    if len(box) == 0:
        return image
    
    top_left_x = int(box[0][0])
    top_left_y = int(box[0][1])
    bottom_right_x = int(box[0][2])
    bottom_right_y = int(box[0][3])

    img_size = (bottom_right_x-top_left_x+bottom_right_y-top_left_y)//2

    if img_size > limit_size:
        return image.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
    return image



if __name__ == '__main__':
    task = ['all', 'yolo', 'rembg']
    assert len(sys.argv) == 2 and sys.argv[1] in task, "전처리하고 싶은 Detection : [ all, yolo, rembg ] 중 하나만 입력만 해야 합니다"

    yolo_task, rembg_task = False, False

    if sys.argv[1] == 'all':
        yolo_task, rembg_task = True, True
    elif sys.argv[1] == 'yolo':
        yolo_task = True
    else:
        rembg_task = True

    # yolo preprocessing
    if yolo_task:  
        train_image_path = '../input/train/images'

        yolo_img_save_path = '../input/train/yolo_images'
        yolo_model = YOLO('../data/preprocess/yolov8n-face.pt')

        # yolo 경로가 없을 때 진행
        print("Image preprocessing by Yolov8n : face detection")

        for image_folder in tqdm(os.listdir(train_image_path)):
            if image_folder.startswith('.'):
                continue
            

            yolo_img_folder = os.path.join(yolo_img_save_path,image_folder)
            os.makedirs(yolo_img_folder, exist_ok=True)

            if len(os.listdir(yolo_img_folder)) != 7:
                for img_name in os.listdir(os.path.join(train_image_path,image_folder)):
                    if img_name.startswith('.'):
                        continue
                    
                    image_path = os.path.join(train_image_path,image_folder,img_name)

                    yolo_img = yolo_detection(image_path, yolo_model)

                    yolo_img.save(yolo_img_folder+'/'+img_name)


    # rembg preprocessing
    
    if rembg_task:
        rembg_img_save_path = '../input/train/rembg_images'
        
        
        print("Image preprocessing by Rembg : remove background")
        os.makedirs(rembg_img_save_path, exist_ok=True)

        for image_folder in tqdm(os.listdir(train_image_path)):
            if image_folder.startswith('.'):
                continue

            rembg_img_folder = os.path.join(rembg_img_save_path,image_folder)


            os.makedirs(rembg_img_folder, exist_ok=True)
            # images_list = os.listdir(os.path.join(train_image_path,image_folder)

            if len(os.listdir(rembg_img_folder)) != 7:
                for img_name in os.listdir(os.path.join(train_image_path,image_folder)):
                    if img_name.startswith('.'):
                        continue
                                
                    image_path = os.path.join(train_image_path,image_folder,img_name)
                    
                    img = Image.open(image_path) 
                    rembg_img = remove(img).convert('RGB')
                    rembg_img.save(rembg_img_folder+'/'+img_name)


