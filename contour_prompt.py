# -*- coding: utf-8 -*-
import cv2 
import numpy as np
import openslide
import imageio
import os
from PIL import Image
from tqdm import tqdm
Image.MAX_IMAGE_PIXELS = 1000000000

#本程式碼不限定輸入格式為svs檔案，mrxs檔案也可以輸入
def getprompt(origin_image):
    slide = openslide.open_slide(origin_image)
    #print('origin:',slide.dimensions)
    #print(slide.level_dimensions)
    multiplaying = 5 #縮小係數：2的n次方
    a,b=slide.dimensions
    a_1=a//(2**multiplaying)
    b_1=b//(2**multiplaying)
    slide_thumbnail = slide.get_thumbnail((a_1, b_1))
    tile = np.array(slide_thumbnail)
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 225, cv2.THRESH_BINARY_INV)
    contours0, hierarchy0 = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = tile.copy()
    prompt_list=[]
    for cidx,cnt in enumerate(contours0):
        (x0, y0, w0, h0) = cv2.boundingRect(cnt)
        shift=5
        if w0*h0>=3000:
            '''
            centerx=int(x0+0.5*w0)
            centery=int(y0+0.5*h0)
            return_x=centerx*32
            return_y=centery*32
            
            print(centerx,centery)
            print(x0,y0,w0,h0)
            cv2.circle(img,(centerx,centery),radius=1,color=(255,0,0),thickness=2)
            cv2.circle(img,(centerx-shift,centery),radius=1,color=(0,255,0),thickness=2)
            cv2.circle(img,(centerx+shift,centery+shift),radius=1,color=(0,255,0),thickness=2)
            cv2.circle(img,(centerx+shift,centery-shift),radius=1,color=(0,255,0),thickness=2)
            '''
            #cv2.rectangle(img, (x0, y0), (x0+w0, y0+h0), (0, 255, 0), 2)
            '''
            prompt_list.append(return_x)
            prompt_list.append(return_y)
            prompt_list.append((centerx-shift)*32)
            prompt_list.append(return_y)
            prompt_list.append((centerx+shift)*32)
            prompt_list.append((centery+shift)*32)
            prompt_list.append((centerx+shift)*32)
            prompt_list.append((centery-shift)*32)
            '''
            prompt_list.append([x0,y0,x0+w0,y0+h0])
        else:
            continue
    imageio.imwrite(output_dir+"CSY-A-8a_TRI.png", img)
    print('Convert completed')
    return prompt_list

input_img='../images/CSY-A-8a_TRI.svs'
output_dir='./out/'
#getprompt(input_img,output_dir)
getprompt(input_img)






