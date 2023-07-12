import cv2
import numpy as np
import csv
import os
glob_lengths = []    
def read_image_and_write_to_csv(image):
    #reads images and creates an array of lengths of fractures
    #measured in pixels, from the furthest top point to the furthest bottom point
    frame = cv2.imread(image)
    frame = frame[15:-15, :]
    #sometimes there are some issues with the top and bottom of the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaus = cv2.GaussianBlur(gray, (5, 5), 0)
    #here we use Gaussian blur to reduce the noise, I am still trying to figure out
    #how to do it better
    edges = cv2.Canny(gaus, 50, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #here we find the contours of the fractures, unfortunately sometimes it doesn't
    #find all contours, this should be fixed by better/smarter bluring
    #also, sometimes it measures the same fracture twice, however this can be dealt with
    # in data analysis, I think
    lengths = []
    for cnt in contours:
        #CREATE AN IF LOOP FOR TIME TEXT and SCALE TEXT
        #finding the top and bottom points of the fractures/contours
        top = cnt[cnt[:, :, 1].argmin()][0]
        bottom = cnt[cnt[:, :, 1].argmax()][0]
        distance = np.linalg.norm(np.array(top) - np.array(bottom))
        lengths.append(distance)
    lengths = np.array(lengths)
    lengths.sort()
    #we return the sorted array of lengths
    return lengths
for filename in os.scandir('klatki2'):
    #iterating through frames obtained from the video
    if filename.is_file():
        glob_lengths.append(read_image_and_write_to_csv(filename.path))
f = open('lengths.csv', 'w')
writer = csv.writer(f)
for l in glob_lengths:
    writer.writerow(l)
f.close()
#the code outputs a csv file with a row for each frame in the video
