# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 10:09:12 2025

@author: kasun
"""

import cv2
import mediapipe as mp
import math 

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

im = cv2.imread("hand1.jpg");
RGB_im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
results = hands.process(RGB_im)
# print(results) 
# print(results.multi_hand_landmarks)

if (results.multi_hand_landmarks):
    # print(len(results.multi_hand_landmarks)) 
    # print(results.multi_hand_landmarks[0]) 

    for handLms in results.multi_hand_landmarks:
        all_lm_x = []
        all_lm_y = []

        for ilm,lm in enumerate(handLms.landmark):
            h,w,c = RGB_im.shape
            lm_x,lm_y = int(lm.x*w),int(lm.y*h)
            # print(ilm,lm_x,lm_y) 

            all_lm_x.append(lm_x)   # Store the x and y coordinates
            all_lm_y.append(lm_y)

            if(ilm == 4 or ilm == 20):
                cv2.circle(im,(lm_x,lm_y),10,(255,0,255),cv2.FILLED)
               
        mpDraw.draw_landmarks(im, handLms, mpHands.HAND_CONNECTIONS)
        x1, y1 = all_lm_x[4], all_lm_y[4] 
        x2, y2 = all_lm_x[20], all_lm_y[20] 
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) # Calculateing Euclidean distance
        print(f"Distance between landmark 5 and landmark 21: {distance:.2f} pixels")

        cv2.line(im, (x1, y1), (x2, y2), (200, 200, 0), 2) 



cv2.namedWindow("ColorImage",cv2.WINDOW_NORMAL)
cv2.imshow("ColorImage",im )

cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()