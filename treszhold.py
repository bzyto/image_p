import cv2
import numpy as np
from matplotlib import pyplot as plt
import csv
frame = cv2.imread('klatki/frame_30.jpg')
frame = frame[15:-15, :]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur = cv2.fastNlMeansDenoising(gray, None, 9, 7, 21)
blur = cv2.GaussianBlur(blur, (5, 5), 0)
frame_for_treshold = blur[:, 400:]
ret2,th2 = cv2.threshold(frame_for_treshold,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
full_ret, full_tresh = cv2.threshold(blur, ret2, 255, 0)
full_tresh= cv2.bitwise_not(full_tresh)
kernel = np.array([[0, 0, 1, 0 ,0,], [0, 1, 1, 1, 0],[0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0],[0, 0, 1, 0, 0]], np.uint8)
erosion = cv2.dilate(full_tresh,kernel,iterations = 2)
erosion = cv2.erode(erosion,kernel,iterations = 2)
edges = cv2.Canny(erosion, 100, 200)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
j=0
for i, cnt in enumerate(contours):
    #CREATE AN IF LOOP FOR TIME TEXT and SCALE TEXT
    #finding the top and bottom points of the fractures/contours
    top = cnt[cnt[:, :, 1].argmin()][0]
    bottom = cnt[cnt[:, :, 1].argmax()][0]
    distance = np.linalg.norm(np.array(top) - np.array(bottom))
    #print(distance)
    if distance >15:
        cv2.drawContours(frame, contours, i, (0, 255, 0), 3)
        j+=1
print(j)

cv2.imshow('frame2', th2)
cv2.imshow('frame', frame)
cv2.waitKey(0)