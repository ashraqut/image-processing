import kivy
import random
#from __future__ import unicode_literals
from kivy.app import App
from kivy.properties import StringProperty, ObjectProperty
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.event import EventDispatcher
from kivy.core.audio import SoundLoader
from pygame import mixer
import cv2
from PIL import Image
from tempfile import TemporaryFile
#from __future__ import unicode_literals
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import time
from gtts import gTTS
from googletrans import Translator
os.path.dirname(sys.argv[0])
translator = Translator()
mixer.init()

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
import os

class MyApp(App):	
	def build(self):
		layout = BoxLayout(padding=10, orientation='vertical')
		btn1 = Button(text="enter image name")
		btn1.bind(on_press=self.buttonClicked)
		layout.add_widget(btn1)
		self.lbl1 = Label(text="")
		self.s1=Label(text="")
		self.s2=Label(text="")
		layout.add_widget(self.lbl1)
		self.txt1 = TextInput(text='', multiline=False)
		layout.add_widget(self.txt1)
		btn2 = Button(text="Read")
		btn2.bind(on_press=self.button2Clicked)
		layout.add_widget(btn2)
		btn3 = Button(text="Translate")
		btn3.bind(on_press=self.button3Clicked)
		layout.add_widget(btn3)
		return layout

	def process(self,index):
		hsv_lower=[22, 30, 30]
		hsv_upper=[45, 255, 255]
		original = cv2.imread(index)
		imageRGB=cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
		hsv_img = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2HSV)
		HSV_lower = np.array(hsv_lower, np.uint8)  # Lower HSV value
		HSV_upper = np.array(hsv_upper, np.uint8)  # Upper HSV value
		frame_threshed = cv2.inRange(hsv_img, HSV_lower, HSV_upper)

    # find connected components
		_, contours, hierarchy, = cv2.findContours(frame_threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		threshold_area = 400
		for i in range(len(contours)):
			area = cv2.contourArea(contours[i])
			if area > threshold_area:
                #hili=cv2.polylines(image,contours[i],True,255,2,cv2.LINE_AA)
				hili=cv2.drawContours(original, contours, i, (0,0,255), 2)
		RGB = cv2.cvtColor(hili, cv2.COLOR_RGB2BGR)
		mask = np.zeros_like(RGB)
		for i in range(len(contours)):
     # The index of the contour that surrounds your object
     # Create mask where white is what we want, black otherwise
			cv2.drawContours(mask, contours, i, 255, -1) # Draw filled contour in mask
			out = np.zeros_like(RGB) # Extract out the object and place into output image
			out[mask == 255] = RGB[mask == 255]
		out_BGR = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
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

    #OCR to turn the text detected inside the image into a string
		text = pytesseract.image_to_string(Image.open('bw_image.png'))
		ttext = translator.translate(text,src='en', dest='ar')
		#s = unicode(ttext)
		return text

	def voice(self,text,trans):
		#voice = gTTS(text=text, lang='en', slow=False)
		#mixer.music.load("C:/Users/Eman ElAgamy/line.mp3")
		#mixer.music.play()
		#f = TemporaryFile()
		#voice.write_to_fp(f)
		#Play (f)
		#f.close()
		voice2=gTTS(text=trans, lang='ar', slow=False)
		mixer.music.load("C:/Users/Eman ElAgamy/arabline.mp3")
		mixer.music.play()
		#voice.write_to_fp(f)
		#Play (f)
		#f.close()
	

	def buttonClicked(self,btn):
		index=self.txt1.text
		str1=self.process(index)
		self.s1.text=str1
	
    # button click function
	def button2Clicked(self,btn2):
		self.lbl1.text = self.s1.text

	def button3Clicked(self,btn3):
		self.voice(self.s1.text,self.s1.text)

    # run app
if __name__ == "__main__":
    MyApp().run()
