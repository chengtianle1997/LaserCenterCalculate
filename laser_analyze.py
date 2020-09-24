import os
#from cv2 import cv2
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import ndimage
# import cross_center

# Read Single Image from a folder
# src: folder path, num: the n_th image of the folder
# return: the image
def ReadImg(src, num=-1):
    if num == -1:
        img = cv2.imread(src)
        img = RgbToGrey(img)
        return img
    else:
        imgs = os.listdir(src)
        imgs.sort(key=lambda x: int(x[:-4]))
        img = cv2.imread(src + '/' + imgs[num])
        img = RgbToGrey(img)
        return img

# Read all Images from a folder
# src: folder path
# return: the images
def ReadImgs(src):
    imgs = os.listdir(src)
    imgs.sort(key=lambda x: int(x[:-4]))
    imglist = []
    for img in imgs:
        img = cv2.imread(src + '/' + img)
        img = RgbToGrey(img)
        imglist.append(img)
    return imglist

# Read a video
# src: video path
# return: the images
def ReadVideo(src, start_line=0, end_line=0):
    ROI_on = True
    if start_line == 0 and end_line == 0:
        ROI_on = False
    cap = cv2.VideoCapture(src)
    frame_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    counter = 0
    imglist = []
    while cap.isOpened():
        ret, frame = cap.read()
        #frame = RgbToGrey(frame)
        if not ret:
            break
        if ROI_on:
            try:
                frame = frame[start_line:end_line, :]
            except:
                print("ROI Error: Check the input param")
                ROI_on = False
        imglist.append(frame)
        counter += 1
        if counter % 5 == 0:
                print("\r-->Decoding...{}/{}".format(counter, frame_all), end="")
    cap.release()
    return imglist

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
def SaveImg(img, foldername, filename):
    save_dir = "Output/" + foldername
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_name = save_dir + "/" + filename + ".jpg"
    cv2.imwrite(img_name, img)

# Draw Centers for testing
# img: input image, centers: centers output by calculation function
# return: a visual result image
def DrawCenter(img, centers, displayed = False):
    n = len(centers)
    img = GreyToRgb(img)
    for i in range(n):
        # Draw points on the image in red (BGR param)
        cv2.circle(img, (int(centers[i]), i), 1, (0, 0, 255))
    if displayed:
        cv2.namedWindow("Centers", 0)
        cv2.imshow("Centers", img)
        cv2.waitKey(0)
    return img

# Calculate the Gauss Center of every single line
# img: input image (Grey)
# return: centers of every single line
def GaussCal(img):
    width, height = img.shape[1], img.shape[0]
    eva_center = []
    max_pixel = []
    # Initial Params
    xRange = 15
    minError = 0.13
    maxError = 0.13
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
    # Analyze the Gauss Center of every line
    for i in range(height):
        gpoint = []
        # Choose Guass Points
        for j in range(int(eva_center[i] - xRange), min(int(eva_center[i] + xRange), width - 1)):
            if img[i][j] > max_pixel[i] * minError and img[i][j] < max_pixel[i] * (1 - maxError):
                point = [j, img[i][j]]
                gpoint.append(point)
        if len(gpoint) < 3:
            centers.append(0)
        else:
            n = len(gpoint)
            X = np.zeros([n, 3])
            Z = np.zeros([n, 1])
            for k in range(n):
                for m in range(3):
                    X[k][m] = pow(gpoint[k][0], m)
                    Z[k][0] = math.log(gpoint[k][1])
            SA = np.matmul(X.T, X)
            SAN = np.linalg.inv(SA)
            SC = np.matmul(SAN, X.T)
            B = np.matmul(SC, Z)
            if B[2][0] == 0:
                centers.append(0)
            else:
                centers.append(-B[1][0] * 1.0 / (2 * B[2][0]))
    return centers

# Calculate the weight center of the laser line
def WeightCal(img):
    # Gauss Blur
    
    #img = SelfAdapterdRanger(img)
    img = cv2.GaussianBlur(img, (9, 9), 0)
    
    # cv2.namedWindow("Gaussian Blur", 0)
    # cv2.imshow("Gaussian Blur", img)
    # cv2.waitKey(0)
    
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
            centers.append(center)
    return centers

# Self-adapted Ranger
def SelfAdapterdRanger(img):
    phi = -0.25
    m = 0.00952
    kernel_adp = np.array([[phi*m, phi*m, phi*m, phi, phi, phi, phi, phi, phi, phi*m, phi*m, phi*m],
                        [phi*m, phi*m, phi*m, phi, phi, phi, phi, phi, phi, phi*m, phi*m, phi*m],
                        [phi*m, phi*m, phi*m, phi, phi, phi, phi, phi, phi, phi*m, phi*m, phi*m],
                        [phi*m, phi*m, phi*m, phi, phi, phi, phi, phi, phi, phi*m, phi*m, phi*m],
                        [phi*m, phi*m, phi*m, phi, phi, phi, phi, phi, phi, phi*m, phi*m, phi*m],
                        [phi*m, phi*m, phi*m, phi, phi, phi, phi, phi, phi, phi*m, phi*m, phi*m],
                        [phi*m, phi*m, phi*m, phi, phi, phi, phi, phi, phi, phi*m, phi*m, phi*m],
                        ]) 
    kernel_te = np.array([[-1,-1,-1,-1],
                            [-1,5,5,-1],
                            [-1,-1,-1,-1]])                   
    #img = ndimage.convolve(img, kernel_adp)
    img = cv2.Canny(img, 30, 150)
    #cv2.threshold(img, 50, 255, cv2.THRESH_OTSU, img)
    return img

# Show the Trend of the centers
def ShowCenter_1D(centers):
    # multi-line average
    n = len(centers)
    counter = 0
    centers_avg = []
    sum_center = 0
    for center in centers:
        sum_center += center
        counter += 1
        if counter > 9:
            centers_avg.append(sum_center/counter)
            counter = 0
            sum_center = 0
    centers = centers_avg
    plt.plot(centers)
    plt.show()

# Show the Trend of Brightness of a single line
def ShowBrightness(img, line):
    x = []
    width, height = img.shape[1], img.shape[0]
    for i in range(width):
        x.append(img[line, i])
    plt.plot(x)
    plt.show()

def ErrorVisualize(img, centers):
    width, height = img.shape[1], img.shape[0]
    pic_row = 6
    pic_col = 8
    x_range = 20
    pic_num = pic_row * pic_col
    n = len(centers)
    pic_int = int(n / pic_num)
    plt.figure()
    rows = []
    for i in range(pic_row):
        for j in range(pic_col):
            pic_n = i * pic_col + j
            row = pic_n * pic_int
            rows.append(row)
            plt.subplot(pic_row, pic_col, pic_n + 1)
            Brightness = img[row][max(int(centers[row]) - x_range, 0):min(int(centers[row]) + x_range, width)]
            x = [x for x in range(max(int(centers[row]) - x_range, 0), min(int(centers[row]) + x_range, width))]
            plt.plot(x, Brightness, "b-")
            plt.axvline(centers[row], color='r')
            plt.title("line" + str(row))
    img = DrawCenter(img, centers)
    for i in range(len(rows)):
        cv2.line(img, (max(int(centers[rows[i]]) - x_range * 2, 0), rows[i]), (min(int(centers[rows[i]]) + x_range * 2, width), rows[i]), (118, 244, 246), 1)
        cv2.putText(img, "line" + str(rows[i]), (min(int(centers[rows[i]]) + x_range + 50 , width - 220), rows[i]), cv2.FONT_HERSHEY_SIMPLEX, 1, (118, 244, 246))
    cv2.namedWindow("image", 0)
    cv2.imshow("image", img)
    plt.savefig("error_partition")
    #plt.tight_layout()
    plt.show()
    cv2.waitKey(0)
    return img

# Show the repeatability of the algorithm
def TimeErrorVisualize(src, image_num):
    imgs = os.listdir(src)
    imgs.sort()
    samp_line = 8
    imglist = []
    centers = []
    x_range = 10
    y_range = 10
    brightness_displayed = True
    for num in range(len(image_num)):
        img = cv2.imread(src + '/' + imgs[image_num[num]])
        img = RgbToGrey(img)
        imglist.append(img)
        #center = GaussCal(img)
        center = WeightCal(img)
        centers.append(center)
    # Several plots of brightness and one center changing plot
    col = len(imglist) + 1
    # With one line plot and one line image
    row = samp_line * 2
    plot_counter = 1
    # Get the height and width of the image
    width, height = imglist[0].shape[1], imglist[0].shape[0]
    row_int = int(height / samp_line)
    for i in range(samp_line):
        row_num = row_int * i
        y_center = []
        # the First line
        # First len(imglist) columns
        for j in range(col - 1):
            plt.subplot(row, col, plot_counter)
            Brightness = imglist[j][row_num][max(int(centers[j][row_num]) - x_range, 0):min(int(centers[j][row_num]) + x_range, width)]
            x = [x for x in range(max(int(centers[j][row_num]) - x_range, 0), min(int(centers[j][row_num]) + x_range, width))]
            plt.plot(x, Brightness, "b-")
            plt.axvline(centers[j][row_num], color='r')
            y_center.append(centers[j][row_num])
            plot_counter += 1
        # Last column
        plt.subplot(row, col, plot_counter)
        plt.plot(y_center)
        plt.title("line" + str(row_num))
        plot_counter += 1
        # the Second line
        for j in range(col - 1):
            plt.subplot(row, col, plot_counter)
            img = imglist[j][max(row_num - y_range, 0) : min(row_num + y_range, height), max(int(centers[j][row_num]) - x_range, 0) : min(int(centers[j][row_num]) + x_range, width)]
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
            plot_counter += 1
        plot_counter += 1
    plt.show()

# 10-line average
mul_line_avg_on = False
# 5 frame average
mul_frame_avg_on = False
# Show the repeatability of the algorithm
def TimeErrorVisualize_lineplot(src, image_num):
    imgs = os.listdir(src)
    imgs.sort(key=lambda x: int(x[:-4]))
    samp_line = 10
    imglist = []
    centers = []
    x_range = 10
    y_range = 10
    brightness_displayed = True
    print("\nProcessing images...")
    for num in range(len(image_num)):
        img = cv2.imread(src + '/' + imgs[image_num[num]])
        img = RgbToGrey(img)
        imglist.append(img)
        #center = GaussCal(img)
        center = WeightCal(img)
        centers.append(center)
        print("\r-->Processing......{}/{}".format(num, len(image_num)), end = "")
    # Several plots of brightness and one center changing plot
    col = 1
    # With one line plot and one line image
    row = samp_line
    plot_counter = 1
    # Get the height and width of the image
    width, height = imglist[0].shape[1], imglist[0].shape[0]
    row_int = int(height / samp_line)
    print("\nProcessing line plot...")
    for i in range(samp_line):
        row_num = row_int * i
        y_center = []
        for j in range(len(imglist)):
            sum_center = 0
            sum_counter = 0
            if mul_line_avg_on:
                for x in range(max(row_num - 5, 0), min(row_num + 5, height)):
                    sum_center += centers[j][x]
                    sum_counter += 1
                y_center.append(sum_center / sum_counter)
            else:
                y_center.append(centers[j][row_num])
            print("\r-->Processing......{}/{}".format(j + i*len(imglist),samp_line*len(imglist)), end = "")
            
        # Last column
        plt.subplot(row, col, plot_counter)
        plt.plot(y_center)
        plt.title("line" + str(row_num))
        plot_counter += 1      
    print("\nPlot Finished!")
    plt.show()


def resolution_test(test_seq, image_num):
    seq_num = len(test_seq)
    sample_num = 1
    line_num = 10
    center_x = []
    center_xc = []
    center_y = []
    center_pos = []
    for seq in test_seq:
        img_list = os.listdir("Output/" + seq + "-L")
        img_l_list = os.listdir("Output/" + seq)
        img_list.sort()
        img_l_list.sort()
        center = [0, 0]
        for i in range(sample_num):
            center_s = cross_center.GetCrossCenter("Output/" + seq + "-L/" + img_list[image_num + i])
            center[0] += center_s[0]
            center[1] += center_s[1]
        center[0] = center[0]/sample_num
        center[1] = center[1]/sample_num
        center_xc.append(center[0])
        center_y.append(center[1])
        centers_x = 0
        centers_x_ml = 0
        ml_counter = 0
        for i in range(sample_num):
            img = cv2.imread("Output/" + seq + "/" + img_l_list[image_num + i])
            img = RgbToGrey(img)
            centers = WeightCal(img)
            centers_x += centers[int(round(center[1]))]
            for l in range(int(round(center[1])) - int(line_num/2), int(round(center[1])) + int(line_num/2)):
                centers_x_ml += centers[l]
                ml_counter += 1
        centers_x_ml = centers_x_ml / ml_counter
        centers_x = centers_x_ml
        #centers_x = centers_x/sample_num
        center_x.append(centers_x)
        center_pos.append(int(seq))
        print("{} : x={}, y={}, xc={}".format(seq, centers_x, center[1], center[0]))
    plt.plot(center_pos, center_x)
    plt.show()

# resolution_test(["2750", "2751", "2752", "2753", "2754", "2755", "2756", "2757", "2758", "2759", "2760"], 5)

# Time Error Visualize
# Instant laser line analyze

# TimeErrorVisualize("Output/ReflectDark", [0,1,2,3,4,5,6,7,8,9,10,11])

# Uniform Sampling
'''
sample_seq = 1
sample_max = 100
src_dir = "Output/t=2000_g=0"
image_list = os.listdir(src_dir)
# image_list.sort(key=lambda x: int(x[:-4]))
image_sum = len(image_list)
image_num_input = []
for x in range(image_sum):
    if len(image_num_input) > sample_max:
        break
    if x % sample_seq == 0:
        image_num_input.append(x)
TimeErrorVisualize_lineplot(src_dir, image_num_input)
'''
'''
# Find Proper Range
img = ReadImg("Images", 18)
ShowBrightness(img, 600)
'''

# Test Demo
'''
img = ReadImg("Output/ReflectDark", 0)
centers = GaussCal(img)
ErrorVisualize(img, centers)
'''
'''
img = DrawCenter(img, centers)
SaveImg(img, "Test", "Single GaussCal")
'''
'''
img = ReadImg("Output/Reflection", 6)
centers = WeightCal(img)
#img = DrawCenter(img, centers )
img = ErrorVisualize(img, centers)
SaveImg(img, "Test", "Reflection")
'''
'''
# Test Numbers of Images
imgs = ReadImgs("Images") 
counter = 0
num = len(imgs)
for img in imgs:
    centers = GaussCal(img)
    img = DrawCenter(img, centers)
    counter = counter + 1
    if counter % 10 == 0:
        print("Solving image {}/{}".format(counter, num))
    SaveImg(img, "Single GaussCal", str(counter))
'''
'''
imgs = ReadImgs("Images")
counter = 0
num = len(imgs)
for img in imgs:
    centers = WeightCal(img)
    img = DrawCenter(img, centers)
    counter = counter + 1
    if counter % 10 == 0:
        print("Solving image {}/{}".format(counter, num))
    SaveImg(img, "Single WeightCal", str(counter))
'''

# Read Video
'''
seq = "t=2000_g=0"
start_line = 800
end_line = 1500
imglist = ReadVideo("Videos/" + seq + ".avi", start_line, end_line)
counter = 0
for img in imglist:
    SaveImg(img, seq, str(counter))
    counter += 1 
'''
'''
# Noise Visualize
img = ReadImg("Output/2750", 0)
# img = ReadImg("Output/t=2000_g=0", 0)
centers = WeightCal(img)
#centers = GaussCal(img)
ShowCenter_1D(centers)
#img = DrawCenter(img, centers )
'''
# Solve Pic-Stream
'''
src = "fix_cable_pic"
src_d = "Output/" + src
start_line = 800
end_line = 1500
imglist = ReadImgs(src_d)
# imglist = imglist.sort(key=lambda x: int(x[:-4]))
counter = 0
for img in imglist:
    counter += 1
    img = img[start_line:end_line, :]
    SaveImg(img, src + "_roi", str(counter))
'''
def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        pass
    return False

def is_video(s):
    if not os.path.isfile(s):
        return False
    kind = filetype.guess(s)
    if not kind:
        return False
    file_type = kind.MIME.split("/")[0]
    if file_type == "video":
        return True
    return False

def is_image(s):
    if not os.path.isfile(s):
        return False
    kind = filetype.guess(s)
    if not kind:
        return False
    file_type = kind.MIME.split("/")[0]
    if file_type == "image":
        return True
    return False

def get_file_name(s):
    if not os.path.isfile(s):
        return False
    file_path, tempfile_name = os.path.split(s)
    file_name, extension = os.path.splitext(tempfile_name)
    return file_name

def get_folder_name(s):
    if not os.path.exists(s):
        return False
    file_path = os.path.split(s)[-1]
    return file_path

import sys, getopt
import filetype
def main(argv):
    # Input params
    function = 1
    roi = 0
    start_line = 0
    end_line = 0
    input_file = ''
    calc = 1
    save = False
    save_path = ''
    try:
        opts, args = getopt.getopt(argv, "hf:r:s:e:i:c:a:", ["help", "func=", "roi=", "start=", "end=", "input=", "calc=", "save="])
    except getopt.GetoptError:
        print("\
            -h <help> help option\n\
            -f <func> function you choose \n\
                1:Analyze video (input a video)\n\
                2:Analyze pictures (input a folder fullfilled with images)\n\
                3:Analyze one frame (input an image) \n\
                4:Decode video (input a video)\n\
            -r <roi> ROI enable and you have to set start and end line 0:disable 1:enable\n\
            -s <start> start line for ROI\n\
            -e <end> end line for ROI\n\
            -i <input> input file path\n\
            -c <calc> center calculation mathod you choose 1:Weight center 2:Gauss center\n\
            -a <save> save the plot result to file in root folder, or input the save path\n\
            All default option is always the first one (if has), e.g: the default function is 1")
        sys.exit()
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("\
            -h <help> help option\n\
            -f <func> function you choose \n\
                1:Analyze video (input a video)\n\
                2:Analyze pictures (input a folder fullfilled with images)\n\
                3:Analyze one frame (input an image) \n\
                4:Decode video (input a video)\n\
            -r <roi> ROI enable and you have to set start and end line 0:disable 1:enable\n\
            -s <start> start line for ROI\n\
            -e <end> end line for ROI\n\
            -i <input> input file path\n\
            -c <calc> center calculation mathod you choose 1:Weight center 2:Gauss center\n\
            -a <save> save the plot result to file in root folder, or input the save path\n\
            All default option is always the first one (if has), e.g: the default function is 1")
        elif opt in ("-f", "--func"):
            if not is_number(arg):
                print("No function {} found, please choose from function 1 to 4".format(arg))
                sys.exit()
            function = int(arg)
            if function < 1 or function > 4:
                print("No function {} found, please choose from function 1 to 4".format(arg))
                sys.exit()
        elif opt in ("-r", "--roi"):
            if not is_number(arg):
                print("--roi (-r) can only be 0 or 1, please try again.")
                sys.exit()
            roi = int(arg)
            if roi < 0 or roi > 1:
                print("--roi (-r) can only be 0 or 1, please try again.")
                sys.exit()
        elif opt in ("-s", "--start"):
            if not is_number(arg):
                print("start_line incorrect, please check and try again")
                sys.exit()
            start_line = int(arg)
        elif opt in ("-e", "--end"):
            if not is_number(arg):
                print("end_line incorrect, please check and try again")
                sys.exit()
            end_line = int(arg)
        elif opt in ("-i", "--input"):
            if not arg:
                print("input file cannot be empty, please input a correct file path")
                sys.exit()
            if not os.path.exists(arg):
                print("input file path not exist, please input a correct file path")
            input_file = arg
        elif opt in ("-c", "--calc"):
            if not is_number(arg):
                print("calculation method (-calc / -c) can only be 1 or 2")
                sys.exit()
            calc = int(arg)
            if calc < 1 or calc > 2:
                print("calculation method (-calc / -c) can only be 1 or 2")
                sys.exit()
        elif opt in ("-a", "--save"):
            save = True
            if arg:
                save_path = arg
    # Function
    # Function 1: Analyze Video
    if function == 1:
        # Check the input file
        if not is_video(input_file):
            print("Analyze video error: input file is not a video")
            sys.exit()
        # Decode Video
        print("Decode Video")
        seq = get_file_name(input_file)
        if roi == 0:
            start_line, end_line = 0, 0
        elif roi == 1:
            seq += "_roi_{}_{}".format(start_line, end_line)
        imglist = ReadVideo(input_file, start_line, end_line)
        counter = 0
        n = len(imglist)
        sample_seq = 1
        sample_max = 100
        if n > sample_max:
            imglist = imglist[0:sample_max]
            n = sample_max
        print("\nSave Images")
        for img in imglist:
            SaveImg(img, seq, str(counter))
            counter += 1 
            print("\r-->Saving... {}/{}".format(counter, n), end="")
        # Analyze Images
        src_dir = "Output/" + seq
        image_list = os.listdir(src_dir)
        # image_list.sort(key=lambda x: int(x[:-4]))
        image_sum = len(image_list)
        image_num_input = []
        for x in range(image_sum):
            if len(image_num_input) > sample_max:
                break
            if x % sample_seq == 0:
                image_num_input.append(x)
        TimeErrorVisualize_lineplot(src_dir, image_num_input)
    
    # Function 2: Analyze images
    elif function == 2:
        # Check the input path

        # ROI
        if roi == 1:
            src = get_folder_name(input_file) + "_roi_{}_{}".format(start_line, end_line)
            imglist = ReadImgs(input_file)
            # imglist = imglist.sort(key=lambda x: int(x[:-4]))
            counter = 0
            n = len(imglist)
            print("Save ROI images to {}".format("Output/" + src))
            for img in imglist:
                counter += 1
                img = img[start_line:end_line, :]
                SaveImg(img, src, str(counter))
                print("\r-->Saving... {}/{}".format(counter, n), end="")
            input_file = "Output/" + src
        # Analyze Images
        sample_seq = 1
        sample_max = 100
        src_dir = input_file
        image_list = os.listdir(src_dir)
        # image_list.sort(key=lambda x: int(x[:-4]))
        image_sum = len(image_list)
        image_num_input = []
        for x in range(image_sum):
            if len(image_num_input) > sample_max:
                break
            if x % sample_seq == 0:
                image_num_input.append(x)
        TimeErrorVisualize_lineplot(src_dir, image_num_input)
    
    # Analyze a single frame
    elif function == 3:
        # Check the input file
        if not is_image(input_file):
            print("Analyze frame error: input file is not an image")
            sys.exit()
        # Show 1D plot
        img = ReadImg(input_file)
        centers = WeightCal(img)
        #centers = GaussCal(img)
        ShowCenter_1D(centers)
    
    # Decode video
    elif function == 4:
        # Check the input file
        if not is_video(input_file):
            print("Analyze video error: input file is not a video")
            sys.exit()
        # Decode Video
        print("Decode Video")
        seq = get_file_name(input_file)
        if roi == 0:
            start_line, end_line = 0, 0
        elif roi == 1:
            seq += "_roi_{}_{}".format(start_line, end_line)
        imglist = ReadVideo(input_file, start_line, end_line)
        counter = 0
        n = len(imglist)
        print("\nSave Images")
        for img in imglist:
            SaveImg(img, seq, str(counter))
            counter += 1 
            print("\r-->Saving... {}/{}".format(counter, n), end="")

if __name__ == "__main__":
    main(sys.argv[1:])


