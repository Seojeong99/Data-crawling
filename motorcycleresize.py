import os
import glob
from PIL import Image
import cv2
import numpy as np

path = "D:/testImage/2021-11-02/17/2021-11-02-17-07-14"  # 이미지 경로
modified_path = "D:/testImage/2021-11-02/17/resize"  # resize된 이미지가 저장될 경로

img_resize=resize(path,(28,28))
img = cv2.imread("path")
res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
res.save(1,modified_path)

#files = glob.glob("D:/testImage/2021-11-02/17/*.jpg")

#size = 28, 28  # 바꾸고 싶은 사이즈

'''

for f in files:
    #name=str(f)
    img = Image.open(f)
    img_resize = img.resize(img,(28,28))
    title, ext = os.path.splitext(f)
    #os.chdir(modified_path)
    #img.save(name, "PNG")
    title, ext = os.path.splitext(f)
    img_resize.save(title + '_half' + ext)
   img_resize.save("D:/testImage/2021-11-02/17/resize")
   
'''