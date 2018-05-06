import cv2
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from __future__ import unicode_literals
import time
from gtts import gTTS
from googletrans import Translator
translator = Translator()

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'

hsv_lower=[22, 30, 30] 
hsv_upper=[45, 255, 255]
image = cv2.imread('sample2.jpg')
imageRGB=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(imageRGB)
plt.show()

hsv_img = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2HSV)
HSV_lower = np.array(hsv_lower, np.uint8)  # Lower HSV value
HSV_upper = np.array(hsv_upper, np.uint8)  # Upper HSV value
#Threshold
frame_threshed = cv2.inRange(hsv_img, HSV_lower, HSV_upper)
    
# find connected components
_, contours, hierarchy, = cv2.findContours(frame_threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
threshold_area = 400
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    if area > threshold_area:
        #hili=cv2.polylines(image,contours[i],True,255,2,cv2.LINE_AA)
        hili=cv2.drawContours(image, contours, i, (0,0,255), 2)
RGB = cv2.cvtColor(hili, cv2.COLOR_RGB2BGR)
plt.imshow(RGB)
plt.show()        
mask = np.zeros_like(RGB)
for i in range(len(contours)):
 # The index of the contour that surrounds your object
 # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask, contours, i, 255, -1) # Draw filled contour in mask
    out = np.zeros_like(RGB) # Extract out the object and place into output image
    out[mask == 255] = RGB[mask == 255]

out_BGR = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
plt.imshow(out)
plt.show()
output=cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

kernel = np.ones((2, 2), np.uint8)
img = cv2.dilate(output, kernel, iterations=1)
img = cv2.erode(output, kernel, iterations=1)

cv2.imwrite('rmv_noise_image.png', img)

(thresh, im_bw) = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
#im_bw = cv2.adaptiveThreshold(output, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
#im_bw = cv2.threshold(output, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imwrite('bw_image.png', im_bw)

imgx=cv2.imread('bw_image.png',1)
plt.imshow(imgx)
plt.show()
#img = cv2.morphologyEx(imgx, cv2.MORPH_CLOSE, kernel)
text = pytesseract.image_to_string(Image.open('bw_image.png'))
print (text)
voice = gTTS(text=text, lang='en', slow=False)
voice.save("line.mp3")
ttext=translator.translate("Instead of a series of steps specifyin how the computer must work to solve a problem",src='en', dest='ar')
s = unicode(ttext)
print (s)
voice2=gTTS(text=ttext.text, lang='ar', slow=False)
voice2.save("arabline.mp3")