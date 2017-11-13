import os
import cv2
import glob
from tqdm import tqdm
import numpy as np
from PIL import Image

base_path = '/atlas/home/zwpeng/datadreams/data/'
mask_path = 'mask/'
new_mask_path = 'mask_new/'

mask_set = glob.glob(os.path.join(base_path,mask_path,'*.png'))

for i,value in tqdm(enumerate(mask_set)):
    gray = np.array(Image.open(mask_set[i]))
    _, contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros(gray.shape,np.uint8)
    cv2.drawContours(drawing, contours,-1,(255,255,255),-1)
    cv2.imwrite(os.path.join(base_path,new_mask_path,mask_set[i].split('/')[-1],'.jpg'),drawing)

