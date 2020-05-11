import numpy as np
from scipy.spatial import distance
#from playsound import playsound
import imutils
from imutils import face_utils
#from imutils.video import VideoStream
import dlib
import cv2
from threading import Thread
import os
import pygame
#pygame.init()
#pygame.mixer.music.load('C:\\Users\\msi1\\Desktop\\MajorProject\\1-SleepinessDetection\\v02-Final\\alarm.wav')
#pygame.mixer.music.play()
#time.sleep(2)
#pygame.mixer.music.stop()
	#playsound.playsound('C:\\Users\\msi1\\Desktop\\qMajorProject\\1-SleepinessDetection\\v02-Final\\audio\\Alarm01.wav')
	#alarm = Thread(target=playsound, args=('C:\\Users\\msi1\\Desktop\\MajorProject\\1-SleepinessDetection\\v02-Final\\audio\\Alarm01.wav',))
	#alarm.start()
	
def eye_aspect_ratio(eye):
	#Euclidean dist b/w the set of two set of vertical eyes landmark
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	#b/w horizontal
	C = distance.euclidean(eye[0], eye[3])
	#Compute eye aspact ratio
	ear = (A + B) / (2.0 * C)
	# return aspact ratio
	return ear

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--alarm", type=str, default="",
	help="path alarm .WAV file")
args = vars(ap.parse_args())
#ALARM_ON = False

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("C:/Users/msi1/Desktop/MajorProject/1-SleepinessDetection/v02-Final/DataPrepration/shape_predictor_68_face_landmarks.dat") #Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
cap=cv2.VideoCapture(0)
flag=0
while True:
	# grab screen, resix
	ret, frame=cap.read()
	#frame = imutils.resize(frame, width=600)
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)#converting to NumPy Array
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		if ear < thresh:
			flag += 1
			print (flag)
			#import subprocess
			if flag >= frame_check:
				os.system('python drive.py model.h5')
				#if not ALARM_ON:
				#	ALARM_ON = True
				#	if args["alarm"] != "":
				#		t = Thread(target=sound_alarm, args=(args["alarm"],))
				#		t.deamon = True
				#		t.start()
				cv2.putText(frame, "*********  WAKE UP SIR  *************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "*********  ALERT!  *************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				
				#print ("Drowsy")
				#subprocess.call(['python', 'C:\\Users\\msi1\\Desktop\\MajorProject\\1-SleepinessDetection\\v02-Final\\drive.py'])
				#subprocess.call(['python', 'C:\\Users\\msi1\\Desktop\\MajorProject\\1-SleepinessDetection\\v02-Final\\model.h5'])
		else:
			flag = 0
	cv2.imshow("Sleepiness Detection System - Online", frame)
	# 0xFF is used to mask off the last 8-bit of Sequence and the ord() of any keyboard character will not be greater than 255
	key = cv2.waitKey(1) & 0xFF
	# ord() return ASCII value of q (DEC, HEX and OCT - 113,71 and 161 respectively)
	if key == ord("q"):
		break
cv2.destroyAllWindows()
cap.stop()
