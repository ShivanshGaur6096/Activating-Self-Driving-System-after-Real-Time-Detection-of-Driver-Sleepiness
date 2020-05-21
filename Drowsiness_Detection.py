import numpy as np
from scipy.spatial import distance

import imutils
from imutils import face_utils
import dlib
import cv2

from playsound import playsound 

from subprocess import Popen

def play_alarm(path):
	playsound(path)
		
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

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("C:/Users/msi1/Desktop/MajorProject/1-SleepinessDetection/v02-Final/DataPrepration/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
cap=cv2.VideoCapture(0)

flag=0
#buzzerCount = 0

while True:
	ret, frame=cap.read()
	
	frame = imutils.resize(frame, width=550)
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		
		cv2.putText(frame, "Visuals: {:.2f}".format(ear), (0,30),cv2.FONT_HERSHEY_SIMPLEX,0.7, (255,0,0),2)
		
		if ear < thresh:
			flag += 1
			#print (flag)
			cv2.putText(frame, "B L I N K",(430,20),cv2.FONT_HERSHEY_SIMPLEX,0.7, (100,155,0),2)
			
			if flag >= frame_check:
				cv2.putText(frame, "********* WAKE UP SIR ********", (10, 70),
					cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
					# image, "TEXT", (x,y), fontFamily-eg. SIMPLEX / PLAIN, frontScale, (B,G,R), thickness
				cv2.putText(frame, "*** SLEEPINESS DETECTED ***", (10,355),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				
				play_alarm('C:\\Users\\msi1\\Desktop\\MajorProject\\1-SleepinessDetection\\v02-Final\\alarm.wav')
				buzzerCount = buzzerCount + 1
				print(buzzerCount)
				#while(flag >= 2):
				while(buzzerCount >= 2):
					Popen('python drive.py model.h5')
					break

					
		else:
			flag = 0
			buzzerCount = 0
	cv2.imshow("Sleepiness Detection System - ACTIVE", frame)
	# 0xFF is used to mask off the last 8-bit of Sequence and the ord() of any keyboard character will not be greater than 255
	key = cv2.waitKey(1) & 0xFF
	# ord() return ASCII value of q (DEC, HEX and OCT - 113,71 and 161 respectively)
	if key == ord("q"):
		break
cv2.destroyAllWindows()
cap.stop()
