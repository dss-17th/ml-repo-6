import cv2
import matplotlib.pyplot as plt
import os
import sys
import numpy as np


base_dir = '/Users/riversong/Desktop/my_project/binary_classification/archive/'
train_painting_img_dir = '/Users/riversong/Desktop/my_project/binary_classification/archive/train/painting/'
train_photos_img_dir = '/Users/riversong/Desktop/my_project/binary_classification/archive/train/photos/'
valid_painting_img_dir = '/Users/riversong/Desktop/my_project/binary_classification/archive/valid/painting/'
valid_photos_img_dir = '/Users/riversong/Desktop/my_project/binary_classification/archive/valid/photos/'
train_paintings = os.listdir('archive/train/painting')
train_photos = os.listdir('archive/train/photos')
valid_paintings = os.listdir('archive/valid/painting')
valid_photos = os.listdir('archive/valid/photos')

for train_painting in train_paintings:
    # print(train_painting)
    src = cv2.imread(train_painting_img_dir + train_painting)
    ## 미묘한 차이밖에 없어서 커널의 사이즈를 각각 더 크게 차이나도록 할 필요가 있을듯.
    dst_1 = cv2.GaussianBlur(src, (5, 5), 3)
    dst_2 = cv2.GaussianBlur(src, (21, 21), 6)
    dst_3 = cv2.GaussianBlur(src, (43, 43), 9)
    dst_4 = cv2.GaussianBlur(src, (93, 93), 21)
    cv2.imwrite('/Users/riversong/Desktop/my_project/binary_classification/gaussian_1/train/paintings/{}'.format(train_painting),dst_1)
    cv2.imwrite('/Users/riversong/Desktop/my_project/binary_classification/gaussian_2/train/paintings/{}'.format(train_painting),dst_2)
    cv2.imwrite('/Users/riversong/Desktop/my_project/binary_classification/gaussian_3/train/paintings/{}'.format(train_painting),dst_3)
    cv2.imwrite('/Users/riversong/Desktop/my_project/binary_classification/gaussian_4/train/paintings/{}'.format(train_painting),dst_4)

for train_photo in train_photos:
    src = cv2.imread(train_photos_img_dir + train_photo)
    dst_1 = cv2.GaussianBlur(src, (5, 5), 3)
    dst_2 = cv2.GaussianBlur(src, (21, 21), 6)
    dst_3 = cv2.GaussianBlur(src, (43, 43), 9)
    dst_4 = cv2.GaussianBlur(src, (93, 93), 21)
    cv2.imwrite('/Users/riversong/Desktop/my_project/binary_classification/gaussian_2/train/photos/{}'.format(train_photo),dst_2)
    cv2.imwrite('/Users/riversong/Desktop/my_project/binary_classification/gaussian_1/train/photos/{}'.format(train_photo),dst_1)
    cv2.imwrite('/Users/riversong/Desktop/my_project/binary_classification/gaussian_3/train/photos/{}'.format(train_photo),dst_3)
    cv2.imwrite('/Users/riversong/Desktop/my_project/binary_classification/gaussian_4/train/photos/{}'.format(train_photo),dst_4)

for valid_painting in valid_paintings:
    src = cv2.imread(valid_painting_img_dir + valid_painting)
    dst_1 = cv2.GaussianBlur(src, (5, 5), 3)
    dst_2 = cv2.GaussianBlur(src, (21, 21), 6)
    dst_3 = cv2.GaussianBlur(src, (43, 43), 9)
    dst_4 = cv2.GaussianBlur(src, (93, 93), 21)
    cv2.imwrite('/Users/riversong/Desktop/my_project/binary_classification/gaussian_1/valid/paintings/{}'.format(valid_painting),dst_1)
    cv2.imwrite('/Users/riversong/Desktop/my_project/binary_classification/gaussian_2/valid/paintings/{}'.format(valid_painting),dst_2)
    cv2.imwrite('/Users/riversong/Desktop/my_project/binary_classification/gaussian_3/valid/paintings/{}'.format(valid_painting),dst_3)
    cv2.imwrite('/Users/riversong/Desktop/my_project/binary_classification/gaussian_4/valid/paintings/{}'.format(valid_painting),dst_4)

for valid_photo in valid_photos:
    src = cv2.imread(valid_photos_img_dir + valid_photo)
    dst_1 = cv2.GaussianBlur(src, (5, 5), 3)
    dst_2 = cv2.GaussianBlur(src, (21, 21), 6)
    dst_3 = cv2.GaussianBlur(src, (43, 43), 9)
    dst_4 = cv2.GaussianBlur(src, (93, 93), 21)
    cv2.imwrite('/Users/riversong/Desktop/my_project/binary_classification/gaussian_2/valid/photos/{}'.format(valid_photo),dst_2)
    cv2.imwrite('/Users/riversong/Desktop/my_project/binary_classification/gaussian_1/valid/photos/{}'.format(valid_photo),dst_1)
    cv2.imwrite('/Users/riversong/Desktop/my_project/binary_classification/gaussian_3/valid/photos/{}'.format(valid_photo),dst_3)
    cv2.imwrite('/Users/riversong/Desktop/my_project/binary_classification/gaussian_4/valid/photos/{}'.format(valid_photo),dst_4)

# 가우시안 필터 적용

# for train_painting in train_paintings[:5]:
#     plt.imshow(train_painting_img_dir + train_painting)
#     plt.show()
# src = cv2.imread(train_painting_img_dir + train_paintings[0])
# dst = cv2.GaussianBlur(src, (3, 3), 3)
# dst2 = cv2.GaussianBlur(src, (7, 7), 3)
# cv2.imshow('src', src)
# cv2.imshow('dst', dst)
# cv2.imshow('dst2',dst2)
# cv2.waitKey()
# cv2.destroyAllWindows()