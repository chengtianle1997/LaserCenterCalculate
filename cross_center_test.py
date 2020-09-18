import os
from cv2 import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import ndimage

# Read Single Image from a folder
# src: folder path, num: the n_th image of the folder
# return: the image
def ReadImg(src, num):
    imgs = os.listdir(src)
    img = cv2.imread(src + '/' + imgs[num])
    img = RgbToGrey(img)
    return img

# Convert image from BGR to GRAY
# img: input image
# return: the converted image
def RgbToGrey(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert image from GRAY to BGR
# img: input image
# return: the converted image
def GreyToRgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Save Images
# img: input image, foldername: the folder to save images, filename: the name for the image file
# Attention: All of the images are saved in the sub-folder of thefolder named "Output"
def SaveImg(imglist, foldername, filename):
    save_dir = "Output/" + foldername
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_name = save_dir + "/" + filename + ".jpg"
    cv2.imwrite(img_name, img)

minBright = 100

# Calculate the weight center of the laser line
def WeightCal(img):
    # Gauss Blur
    
    #img = SelfAdapterdRanger(img)
    #img = cv2.GaussianBlur(img, (9, 9), 0)
    '''
    cv2.namedWindow("Gaussian Blur", 0)
    cv2.imshow("Gaussian Blur", img)
    cv2.waitKey(0)
    '''
    width, height = img.shape[1], img.shape[0]
    eva_center = []
    max_pixel = []
    # Initial Params
    xRange = 10
    minError = 0.6
    # Find the Evaluated Center of the Laser Line
    for i in range(height):
        max_index = 0
        max_val = 0
        for j in range(width):
            if img[i][j] > max_val:
                max_val = img[i][j]
                max_index = j
        eva_center.append(max_index)
        max_pixel.append(max_val)
    centers = []
    for i in range(height):
        gpoint = []
        # Choose Guass Points
        for j in range(int(eva_center[i] - xRange), min(int(eva_center[i] + xRange), width - 1)):
            if img[i][j] > max_pixel[i] * minError:
                point = [j, img[i][j]]
                gpoint.append(point)
        if len(gpoint) < 1:
            centers.append(0)
        else:
            sum_num = 0
            sum_pos = 0
            for [m, w] in gpoint:
                sum_num += m * w
                sum_pos += w
            center = sum_num / sum_pos
            if img[i][int(center)] > minBright:
                centers.append(center)
            else:
                centers.append(0)
    return centers

def WeightCalVert(img):
    width, height = img.shape[1], img.shape[0]
    eva_center = []
    max_pixel = []
    # Initial Params
    yRange = 10
    minError = 0.6
    # Find the Evaluated Center of the Laser Line
    for i in range(width):
        max_index = 0
        max_val = 0
        for j in range(height):
            if img[j][i] > max_val:
                max_val = img[j][i]
                max_index = j
        eva_center.append(max_index)
        max_pixel.append(max_val)
    centers = []
    for i in range(width):
        gpoint = []
        # Choose Guass Points
        for j in range(int(eva_center[i] - yRange), min(int(eva_center[i] + yRange), height - 1)):
            if img[j][i] > max_pixel[i] * minError:
                point = [j, img[j][i]]
                gpoint.append(point)
        if len(gpoint) < 1:
            centers.append(0)
        else:
            sum_num = 0
            sum_pos = 0
            for [m, w] in gpoint:
                sum_num += m * w
                sum_pos += w
            center = sum_num / sum_pos
            if img[int(center)][i] > minBright:
                centers.append(center)
            else:
                centers.append(0)
    return centers

# Draw Centers for testing
# img: input image, centers: centers output by calculation function
# return: a visual result image
def DrawCenter(img, centers, ver_centers, displayed = False):
    n = len(centers)
    img = GreyToRgb(img)
    for i in range(n):
        # Draw points on the image in red (BGR param)
        cv2.circle(img, (int(centers[i]), i), 1, (0, 0, 255))
    m = len(ver_centers)
    for i in range(m):
        cv2.circle(img, (i, int(ver_centers[i])), 1, (255, 0, 0))
    if displayed:
        cv2.namedWindow("Centers", 0)
        cv2.imshow("Centers", img)
        cv2.waitKey(0)
    return img

def FindWindow(img, centers, ver_centers):
    n, m = len(centers), len(ver_centers)
    min_h, max_h, min_w, max_w = 0, 0, 0, 0
    for i in range(n):
        if not centers[i] < 20:
            if centers[i] > max_w:
                max_w = centers[i]
            elif centers[i] < min_w:
                min_w = centers[i]
    for j in range(m):
        if not ver_centers[j] < 200:
            if ver_centers[j] > max_h:
                max_h = centers[j]
            elif ver_centers[j] < min_h:
                min_h = centers[j]
    cv2.rectangle(img, (int(min_h), int(min_w)), (int(max_h), int(max_w)), (0, 255, 0), 2)
    cv2.namedWindow("Rect", 0)
    cv2.imshow("Rect", img)
    cv2.waitKey(0)
    return min_h, max_h, min_w, max_w

# counter by line (rows)
def hori_counter(img):
    width, height = img.shape[1], img.shape[0]
    hori_stat = [sum(img[i,0:width]) for i in range(height)]
    return hori_stat

def vert_counter(img):
    width, height = img.shape[1], img.shape[0]
    vert_stat = [sum(img[0:height, i]) for i in range(width)]
    return vert_stat
'''
img = ReadImg("cross_center", 1)
hori_stat = hori_counter(img)
vert_stat = vert_counter(img)

plt.plot(vert_stat, 'r')
plt.plot(hori_stat, 'b')
plt.show()
'''
'''
# Test Demo 
img = ReadImg("cross_center", 0)
centers = WeightCal(img)
ver_centers = WeightCalVert(img)
img = DrawCenter(img, centers, ver_centers)
FindWindow(img, centers, ver_centers)
'''

def FindRectBox(img, debug=True):
    x_range, y_range = 300, 600
    width, height = img.shape[1], img.shape[0]
    ret, img = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
    hori_stat = [sum(img[i,0:width]) for i in range(height)]
    vert_stat = [sum(img[0:height, i]) for i in range(width)]
    hori_center = hori_stat.index(max(hori_stat))
    vert_center = vert_stat.index(max(vert_stat))
    if debug:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(rgb_img, (vert_center - y_range, hori_center - x_range), (vert_center + y_range, hori_center + x_range), (0, 0, 255), 2)
        cv2.namedWindow("Rect", 0)
        cv2.imshow("Rect", rgb_img)
        plt.plot(vert_stat, 'r')
        plt.plot(hori_stat, 'b')
        plt.show()
        cv2.waitKey(0)
    return img[max(hori_center - x_range, 0) : min(hori_center + x_range, width), max(vert_center - y_range, 0): min(vert_center + y_range, height) ]

img = ReadImg("cross_center", 1)
img = FindRectBox(img, debug=True)
cv2.namedWindow("Rect", 0)
cv2.imshow("Rect", img)
cv2.waitKey(0)