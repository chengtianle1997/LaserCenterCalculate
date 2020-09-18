# -*- coding: UTF-8 -*-

# Filename :    main.py
# author by :   Jiang Lei
# email:        jiang.lei@tongji.edu.cn

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import getopt
import sys
from shapely.geometry import Polygon
import time
import shutil


debug_mode = False
crop_img = {}


def get_orthogonal_lines(lines):
    # find orthogonal line pairs
    newlines = []
    newlines.append(lines.pop())
    for line in lines:
        flag = 0
        for newline in newlines:
            delta_theta = abs(line[1]-newline[1])
            delta_theta = abs(delta_theta - np.pi / 2)
            if(delta_theta < 0.2):
                flag = 1
        if(flag == 1):
            newlines.append(line)
    return newlines


def cleanlines(lines, maxLineGap):
    # clean lines by erase nearby theta lines
    for lineindex, line in enumerate(lines):
        if line[0] < 0:
            lines[lineindex][0] = -line[0]
            lines[lineindex][1] = line[1]-np.pi
    newlines = []
    newlines.append(lines.pop())
    for line in lines:
        flag = 0
        for newline in newlines:
            if(abs(line[0]-newline[0]) < maxLineGap) and (abs(line[1]-newline[1]) < 0.15):
                flag = 1
        if(flag == 0):
            newlines.append(line)
    return newlines


def Cline2Pline(line):
    if line[0] != 0:
        k = line[1]/line[0]
        b = line[3] - k * line[2]

        theta = np.arctan(k)
        # if theta < 0:
        #     theta += np.pi
        # rho = b * np.cos(np.arctan(k))
        rho = np.abs(b) * np.cos(theta)
    else:
        theta = np.pi / 2
        if line[2] > 0:
            rho = line[2]
        else:
            rho = 0 - line[2]
            theta = theta + np.pi

    return rho[0], theta[0]


def beam_center(line, threshold):
    sum = 0.0
    samples = 0
    polygon_pts = []
    for pt_idx, pt_lumi in enumerate(line):
        if pt_lumi > threshold and abs(pt_idx - len(line)/2) < 60:
            polygon_pts.append((pt_idx, pt_lumi))

    for idx, pt in enumerate(polygon_pts):
        if idx == 0 and idx < len(polygon_pts)-1:
            if polygon_pts[idx+1][0] - pt[0] > 1:
                polygon_pts.pop(idx)
        elif idx == len(polygon_pts)-1:
            if pt[0] - polygon_pts[idx-1][0] > 1:
                polygon_pts.pop(idx)
        elif polygon_pts[idx+1][0] - pt[0] > 1 or pt[0] - polygon_pts[idx-1][0] > 1:
            polygon_pts.pop(idx)

    if len(polygon_pts) > 4:
        ref_polygon = Polygon(polygon_pts)
        center_x = ref_polygon.centroid.x
    else:
        center_x = -1
    return center_x


def PreciseIntersectionPoint(img_gray, threshold):
    global crop_img
    global debug_mode

    height = img_gray.shape[0]
    width = img_gray.shape[1]

    hline_centers = []
    vline_centers = []

    for row in range(0, height):
        section = img_gray[row, :]
        x = beam_center(section, threshold)
        y = row
        if x < width and x >= 0:
            vline_centers.append([x, y])

        # cv2.circle(crop_img,(int(x+0.5), int(y+0.5)), 1, (255, 0, 0), 1)

    for col in range(0, width):
        section = img_gray[:, col]
        y = beam_center(section, threshold)
        x = col
        if y < height and y >= 0:
            hline_centers.append([x, y])
        # cv2.circle(crop_img,(int(x), int(y)), 1, (0, 0, 255), 1)

    # draw dots on vline
    # cv2.namedWindow('crop_img', cv2.WINDOW_NORMAL)
    # cv2.imshow('crop_img',crop_img)
    # cv2.waitKey(20000)

    # linear regression

    # convert to np array
    hline_centers = np.array(hline_centers)
    h_line = cv2.fitLine(hline_centers, cv2.DIST_L2, 0, 0.01, 0.01)

    if h_line[0] != 0:
        k = h_line[1] / h_line[0]
        x1 = h_line[2] + 300
        x2 = h_line[2] - 300
        y1 = h_line[3] + 300*k
        y2 = h_line[3] - 300*k
    else:
        x1 = x2 = h_line[2]
        y1 = h_line[3] + 300
        y2 = h_line[3] - 300
    horizon_line = [x1, y1, x2, y2]

    if debug_mode:
        cv2.line(crop_img, (int(x1+0.5), int(y1+0.5)),
                 (int(x2+0.5), int(y2+0.5)), (0, 0, 255), 1)
        cv2.namedWindow('h_pline', cv2.WINDOW_NORMAL)
        cv2.imshow('h_pline', crop_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    vline_centers = np.array(vline_centers)
    v_line = cv2.fitLine(vline_centers, cv2.DIST_L2, 0, 0.01, 0.01)

    if v_line[0] != 0:
        k = v_line[1] / v_line[0]
        x1 = v_line[2] + 300
        x2 = v_line[2] - 300
        y1 = v_line[3] + 300*k
        y2 = v_line[3] - 300*k
    else:
        x1 = x2 = v_line[2]
        y1 = v_line[3] + 300
        y2 = v_line[3] - 300

    vertical_line = [x1, y1, x2, y2]

    if debug_mode:
        cv2.line(crop_img, (int(x1+0.5), int(y1+0.5)),
                 (int(x2+0.5), int(y2+0.5)), (0, 0, 255), 1)
        cv2.namedWindow('v_pline', cv2.WINDOW_NORMAL)
        cv2.imshow('v_pline', crop_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    pt_center = cross_point(horizon_line, vertical_line)
    return pt_center


def cross_point(line1, line2):  # calculate cross point

    point_is_exist = False
    x = 0
    y = 0
    x1 = line1[0]
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    if (x2 - x1) == 0:
        k1 = None
    else:
        k1 = (y2 - y1) / (x2 - x1)  # calc k1,convert to float first
        b1 = y1 - x1 * k1

    if (x4 - x3) == 0:  # k of L2 does not exist
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # k of L2 exists
        b2 = y3 * 1.0 - x3 * k2 * 1.0

    if k1 is None:
        if not k2 is None:
            x = x1
            y = k2 * x1 + b2
            point_is_exist = True
    elif k2 is None:
        x = x3
        y = k1*x3+b1
    elif not k2 == k1:
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
        point_is_exist = True
    return point_is_exist, [x, y]


def IntersectionPoints(lines):

    points = []
    if len(lines) == 2 or len(lines) == 4:
        horLine = []
        verLine = []
        for line in lines:
            if line[1] > (0-0.785) and line[1] < (0+0.785):
                # if line[1] > 0.78 and line[1] < 2.35:
                horLine.append(line)
            else:
                verLine.append(line)
        # print(horLine)
        for l1 in horLine:
            for l2 in verLine:
                a = np.array([
                    [np.cos(l1[1]), np.sin(l1[1])],
                    [np.cos(l2[1]), np.sin(l2[1])]
                ])
                b = np.array([l1[0], l2[0]])
                points.append(np.linalg.solve(a, b))
        return points
    else:
        print("the number of lines error")


def FindRectBox(img, debug=False):
    x_range, y_range = 300, 600
    img_c = img
    width, height = img.shape[1], img.shape[0]
    ret, img = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
    hori_stat = [sum(img[i, 0:width]) for i in range(height)]
    vert_stat = [sum(img[0:height, i]) for i in range(width)]
    hori_center = hori_stat.index(max(hori_stat))
    vert_center = vert_stat.index(max(vert_stat))
    if debug:
        cv2.rectangle(img, (vert_center - y_range, hori_center - x_range),
                      (vert_center + y_range, hori_center + x_range), (0, 0, 255), 2)
        cv2.namedWindow("Rect", 0)
        cv2.imshow("Rect", img)
        cv2.waitKey(0)
    return max(hori_center - x_range, 0), max(vert_center - y_range, 0), img_c[max(hori_center - x_range, 0): min(hori_center + x_range, width), max(vert_center - y_range, 0): min(vert_center + y_range, height)]


use_rect_finding = True
save_ori_and_res = False


def GetCrossCenter(img_file):
    global crop_img
    center = [0, 0]
    img = cv2.imread(img_file)
    file_num = 0

    # Save Test images
    if save_ori_and_res:
        file_list = os.listdir("origin")
        file_num = len(file_list) + 1
        file_name = "origin/image{}.bmp".format(file_num)
        shutil.copy(img_file, file_name)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, im2 = cv2.threshold(
    #     img_gray, 50, 255, cv2.THRESH_BINARY)

    # showimg(im2, "THRESH_BINARY")
    if use_rect_finding:
        c_x, c_y, img_gray = FindRectBox(img_gray)
        img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    gimg = cv2.GaussianBlur(img_gray, (3, 3), 0)
    showimg(gimg, "GaussianBlur")

    gret, gim2 = cv2.threshold(
        gimg, 60, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # gret, gim2 = cv2.threshold(
    #     gimg, 135, 255, cv2.THRESH_BINARY)

    showimg(gim2, "THRESH_BINARY")

    edges = cv2.Canny(gim2, 30, 150)

    showimg(edges, "Canny")

    minLineLength = 120
    maxLineGap = 50
    lines = cv2.HoughLines(edges, 1, np.pi/180, minLineLength)
    # lines = cv2.HoughLines(gim2, 1, np.pi/180, minLineLength)

    lines = [line[0] for line in lines.tolist()]

    lines = cleanlines(lines, maxLineGap)
    lines = get_orthogonal_lines(lines)

    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # lines visualize
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(edges, (x1, y1), (x2, y2), (0, 0, 255), 5)
    showimg(edges, "Houghlines")

    point = IntersectionPoints(lines)

    pt_x = int(point[0][0])
    pt_y = int(point[0][1])

    img_heigh = img.shape[0]
    img_width = img.shape[1]

    if pt_y > 200 and pt_y < (img_heigh-200):
        crop_img = img[pt_y-200:pt_y+200, pt_x-200:pt_x+200]
        img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        showimg(img_gray, "Cropped")
        crop_center_pt = PreciseIntersectionPoint(img_gray, 70)

        center[0] = point[0][0] + crop_center_pt[1][0][0] - 200
        center[1] = point[0][1] + crop_center_pt[1][1][0] - 200

    elif pt_y >= (img_heigh-200):
        enlarged_img = cv2.copyMakeBorder(
            img, 0, 200, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        crop_img = enlarged_img[pt_y-200:pt_y+200, pt_x-200:pt_x+200]
        img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        showimg(img_gray, "Cropped DN extend")
        crop_center_pt = PreciseIntersectionPoint(img_gray, 70)

        center[0] = point[0][0] + crop_center_pt[1][0][0] - 200
        center[1] = point[0][1] + crop_center_pt[1][1][0] - 200

    else:
        enlarged_img = cv2.copyMakeBorder(
            img, 200, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        crop_img = enlarged_img[pt_y:pt_y+400, pt_x-200:pt_x+200]

        img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        showimg(img_gray, "Cropped UP extend")
        crop_center_pt = PreciseIntersectionPoint(img_gray, 70)

        center[0] = point[0][0] + crop_center_pt[1][0][0] - 200
        center[1] = point[0][1] + crop_center_pt[1][1][0] - 200

    if use_rect_finding:
        center[0] += c_y
        center[1] += c_x

    # Save Result Images
    if save_ori_and_res:
        img = cv2.imread(img_file)
        cpt = (int(center[0]), int(center[1]))
        cv2.circle(img, cpt, 3, (0, 255, 0), 1)
        res_file = "result/image{}.bmp".format(file_num)
        cv2.imwrite(res_file, img)

    return center


def showimg(img, title):
    if debug_mode:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main(argv):
    global crop_img
    global debug_mode

    en_dispimg = False
    img_fname = ''
    help_str = '''
        'get_cross_center.py -h -f <filepath>'
    '''
    try:
        opts, args = getopt.getopt(argv, "hdf:", ["file=", "debug"])
    except getopt.GetoptError:
        print(help_str)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help_str)
            sys.exit(0)

        elif opt in ("-f", "--file"):
            img_fname = arg

        elif opt in ("-d", "--debug"):
            debug_mode = True

    img_file = os.getcwd() + '\\' + img_fname
    if not os.path.isfile(img_file):
        print("invalid image file")
        sys.exit(1)

    center_pt = GetCrossCenter(img_file)

    if debug_mode:
        print('center = {0},{1}'.format(center_pt[0], center_pt[1]))
    else:
        print('CENTER=%.3f,%.3f' % (center_pt[0], center_pt[1]))

    if debug_mode:
        img = cv2.imread(img_file)

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #cx, cy, img = FindRectBox(img)
        # for line in lines:
        #     rho, theta = line
        #     #print(rho, theta)
        #     a = np.cos(theta)
        #     b = np.sin(theta)
        #     x0 = a*rho
        #     y0 = b*rho
        #     x1 = int(x0 + 2000*(-b))
        #     y1 = int(y0 + 2000*(a))
        #     x2 = int(x0 - 2000*(-b))
        #     y2 = int(y0 - 2000*(a))
        #     cv2.line(img,(x1, y1), (x2, y2), (0, 255, 0), 1)
        cpt = (int(center_pt[0]), int(center_pt[1]))
        cv2.circle(
            img, cpt, 3, (255, 255, 0), 1)
        result_img = img
        result_img = img[cpt[1]-100:cpt[1]+100, cpt[0]-100:cpt[0]+100]
        cv2.namedWindow('result_img', cv2.WINDOW_NORMAL)
        cv2.imshow('result_img', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


test_all = False
timer_on = False

# execute only if run as a script
if __name__ == "__main__":
    if test_all:
        dirs = os.listdir("test_images")
        dirs.sort()
        args = []
        for src in dirs:
            print("image{}".format(src))
            start_time = time.time()
            args.append("--file=test_images/{}".format(src))
            # args.append("--debug")
            main(args)
            stop_time = time.time()
    else:
        main(sys.argv[1:])
