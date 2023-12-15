from PIL import Image

import cv2
import time

# from rembg import remove


def yolo(img_path, model):

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

    if img_size < 100:
        return image

    return image.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y))


# def rembg(img_path,model):
"""
        수정 중 
"""
#     s = time.time()
#     image = Image.open(img_path)
#     print("here")
    
#     # rembg_img = model(image).convert('RGB')
#     rembg_img = remove(image).convert('RGB')
#     print(time.time() - s)

#     return rembg_img
