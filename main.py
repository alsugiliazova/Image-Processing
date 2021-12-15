from sys import argv
import cv2 as cv
import numpy as np
from math import log10, sqrt, pi

def median_filter(data, filter_size):
    temp = []
    result = np.zeros(data.shape)
    for i in range(len(data)):
        for j in range(len(data[0])):
            for z in range(-filter_size, filter_size+1):
                for k in range(-filter_size, filter_size+1):
                    res_i = 0 if i + z < 0 else (len(data) - 1 if i + z > len(data) - 1 else i + z)
                    res_j = 0 if j + k < 0 else (len(data[0]) - 1 if j + k > len(data[0]) - 1 else j + k)
                    temp.append(data[res_i][res_j])
            temp.sort()
            result[i][j] = temp[len(temp) // 2]
            temp = []
    return result

def gaussian_filter(img, rad, sigma):
    result = np.zeros(img.shape)
    kernel = np.zeros((2 * rad + 1, 2 * rad + 1))
    for i in range(-rad, rad + 1):
        for j in range(-rad, rad + 1):
            kernel[i + rad][j + rad] = 1 / (2 * pi * (sigma**2)) * np.exp(-(i**2 + j**2) / (2 * (sigma**2)))
    img_with_borders = cv.copyMakeBorder(img, rad, rad, rad, rad, cv.BORDER_REPLICATE)
    for i in range(rad, rad + len(img)):
        for j in range(rad, rad + len(img[0])):
            g = img_with_borders[i - rad: i + rad + 1, j - rad: j + rad + 1]
            result[i - rad][j - rad] = np.sum(kernel * g)
    return result

def bilateral_filter(img, rad, sigma_d, sigma_r):
    img = img.astype('int')
    result = np.zeros(img.shape)
    img_with_borders = cv.copyMakeBorder(img, rad, rad, rad, rad, cv.BORDER_REPLICATE)
    kernel = np.zeros((2 * rad + 1, 2 * rad + 1))
    for i in range(-rad, rad + 1):
        for j in range(-rad, rad + 1):
            kernel[i + rad][j + rad] = 1 / (2 * pi * (sigma_d ** 2)) * np.exp(-(i**2 + j**2) / (2 * (sigma_d**2)))
    gauss = kernel * (2 * pi * (sigma_d**2))
    for i in range(rad, rad + len(img)):
        for j in range(rad, rad + len(img[0])):
            g = img_with_borders[i - rad: i + rad + 1, j - rad: j + rad + 1]
            bilateral = np.exp(-((g - img_with_borders[i, j])**2 / (2 * (sigma_r**2))))
            kernel = bilateral * gauss
            result[i - rad][j - rad] = np.sum(g * kernel) / np.sum(kernel)
    return result

command = argv[1]
if command == 'median':
    rad = int(argv[2])
    input_file = argv[3]
    output_file = argv[4]
    img = cv.imread(input_file, cv.IMREAD_GRAYSCALE)
    output = median_filter(img, rad)
    cv.imwrite(output_file, output)

if command == 'gauss':
    sigma = float(argv[2])
    input_file = argv[3]
    output_file = argv[4]
    rad = int(3 * sigma)
    img = cv.imread(input_file, cv.IMREAD_GRAYSCALE)
    output = gaussian_filter(img, rad, sigma)
    cv.imwrite(output_file, output)
   
if command == 'bilateral':
    sigma_d = float(argv[2])
    sigma_r = float(argv[3])
    input_file = argv[4]
    output_file = argv[5]
    img = cv.imread(input_file, cv.IMREAD_GRAYSCALE)
    output = bilateral_filter(img, int(3*sigma_d), sigma_d, sigma_r)
    cv.imwrite(output_file, output)


def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err

def psnr(im1, im2):
    return 20 * log10(255 / sqrt(mse(im1, im2)))

def ssim(im1, im2):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    im1 = im1.astype('float')
    im2 = im2.astype('float')
    mean1 = np.mean(im1)
    mean2 = np.mean(im2)
    contrast1 = np.var(im1)
    contrast2 = np.var(im2)
    covariance = np.mean((im1 - mean1) * (im2 - mean2))
    result = ((2 * mean1 * mean2 + c1) * (2 * covariance + c2)) / ((mean1 ** 2 + mean2 ** 2 + c1) * (contrast1 + contrast2 + c2))
    return result

if command == 'mse':
    input_file_1 = argv[2]
    input_file_2 = argv[3]
    imageA = cv.imread(input_file_1, cv.IMREAD_GRAYSCALE)
    imageB = cv.imread(input_file_2, cv.IMREAD_GRAYSCALE)
    print(mse(imageA, imageB))

if command == 'ssim':
    input_file_1 = argv[2]
    input_file_2 = argv[3]
    imageA = cv.imread(input_file_1, cv.IMREAD_GRAYSCALE)
    imageB = cv.imread(input_file_2, cv.IMREAD_GRAYSCALE)
    print(ssim(imageA, imageB))

if command == 'psnr':
    input_file_1 = argv[2]
    input_file_2 = argv[3]
    imageA = cv.imread(input_file_1, cv.IMREAD_GRAYSCALE)
    imageB = cv.imread(input_file_2, cv.IMREAD_GRAYSCALE)
    print(psnr(imageA, imageB))