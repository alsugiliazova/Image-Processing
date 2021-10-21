from sys import argv
import cv2 as cv
# import numpy as np
# from PIL import Image
# from numpy import asarray
# import skimage.io
# from skimage import io
# import skimage.filters
# import sys
# import skimage
# from skimage import img_as_ubyte
# from skimage.metrics import structural_similarity as ssim
# import argparse
# import imutils
# from math import *


def median_filter(data, filter_size):
    temp = []
    return_data = []
    indexer = filter_size // 2
    for i in range(len(data)):
        for j in range(len(data[0])):
            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            return_data[i][j] = temp[len(temp) // 2]
            temp = []
    return return_data

command = argv[1]
if command == 'median':
    rad = int(argv[2])
    input_file = argv[3]
    output_file = argv[4]
    img = cv.imread(input_file)
    out = median_filter(img, 2*rad+1)
    cv.imwrite(output_file, out)

if command == 'gauss':
    sigma = float(argv[2])
    input_file = argv[3]
    output_file = argv[4]
    image = io.imread(fname=input_file)
    blurred = skimage.filters.gaussian(image, sigma=(sigma, sigma), truncate=3, multichannel=True)
    blurred = img_as_ubyte(blurred)
    io.imsave(output_file, blurred)
   
if command == 'bilateral':
    sigma_d = float(argv[2])
    sigma_r = float(argv[3])
    input_file = argv[4]
    output_file = argv[5]
    img = cv.imread(input_file, 0)
    bilateral = cv.bilateralFilter(img, int(3*sigma_d), sigma_r, sigma_d)
    cv.imwrite(output_file, bilateral)


def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err

if command == 'mse':
    input_file_1 = argv[2]
    input_file_2 = argv[3]
    img1 = cv.imread(input_file_1)
    img2 = cv.imread(input_file_2)
    print(mse(img1, img2))


if command == 'ssim':
    input_file_1 = argv[2]
    input_file_2 = argv[3]
    imageA = cv.imread(input_file_1)
    imageB = cv.imread(input_file_2)
    grayA = cv.cvtColor(imageA, cv.COLOR_BGR2GRAY)
    grayB = cv.cvtColor(imageB, cv.COLOR_BGR2GRAY)
    (score, diff) = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    print(format(score))

if command == 'psnr':
    input_file_1 = argv[2]
    input_file_2 = argv[3]
    img1 = np.array(Image.open(input_file_1))
    img2 = np.array(Image.open(input_file_2))
    mse = mse(img1, img2)
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    print(psnr)