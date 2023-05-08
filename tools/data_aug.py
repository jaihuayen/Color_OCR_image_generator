#!/usr/env/bin python3

import random
import cv2
import numpy as np
import albumentations as A


def prob(percent):
    assert 0 <= percent <= 1
    if random.uniform(0, 1) <= percent:
        return True
    return False


def apply_blur_on_output(img):
    if prob(0.5):
        return apply_gauss_blur(img, [3, 5])
    else:
        return apply_norm_blur(img)


def apply_gauss_blur(img, ks=None):
    if ks is None:
        ks = [7, 9, 11, 13]
    ksize = random.choice(ks)

    sigmas = [0, 1, 2, 3, 4, 5, 6, 7]
    sigma = 0
    if ksize <= 3:
        sigma = random.choice(sigmas)
    img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return img


def apply_norm_blur(img, ks=None):
    # kernel == 1, the output image will be the same
    if ks is None:
        ks = [2, 3]
    kernel = random.choice(ks)
    img = cv2.blur(img, (kernel, kernel))
    return img


def apply_prydown(img):
    scale = random.uniform(1, 1.5)
    height = img.shape[0]
    width = img.shape[1]

    out = cv2.resize(img, (int(width / scale), int(height / scale)),
                     interpolation=cv2.INTER_AREA)
    return cv2.resize(out, (width, height), interpolation=cv2.INTER_AREA)


def apply_lr_motion(image):
    kernel_size = 5
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel_motion_blur = kernel_motion_blur / kernel_size
    image = cv2.filter2D(image, -1, kernel_motion_blur)
    return image


def apply_up_motion(image):
    kernel_size = 9
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    kernel_motion_blur[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    kernel_motion_blur = kernel_motion_blur / kernel_size
    image = cv2.filter2D(image, -1, kernel_motion_blur)
    return image


def random_augment_img(img):
    # gets PIL image and returns augmented PIL image
    # only augment 3/4th the images
    if random.randint(1, 4) > 3:
        return img

    img = np.asarray(img)  # convert to numpy for opencv

    # morphological alterations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if random.randint(1, 5) == 1:
        # dilation because the image is not inverted
        img = cv2.erode(img, kernel, iterations=random.randint(1, 2))
    if random.randint(1, 6) == 1:
        # erosion because the image is not inverted
        img = cv2.dilate(img, kernel, iterations=random.randint(1, 1))

    transform = A.Compose([

        A.OneOf([
            # add black pixels noise
            A.OneOf([
                A.RandomRain(brightness_coefficient=1.0, drop_length=2, drop_width=2, drop_color=(
                    0, 0, 0), blur_value=1, rain_type='drizzle', p=0.05),
                A.RandomShadow(p=1),
                A.PixelDropout(p=1),
            ], p=0.9),

            # add white pixels noise
            A.OneOf([
                A.PixelDropout(dropout_prob=0.5, drop_value=255, p=1),
                A.RandomRain(brightness_coefficient=1.0, drop_length=2, drop_width=2, drop_color=(
                    255, 255, 255), blur_value=1, rain_type=None, p=1),
            ], p=0.9),
        ], p=1),

        # transformations
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0.25, rotate_limit=2,
                               border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=8,
                               border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.15, rotate_limit=11,
                               border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=1),
            A.Affine(shear=random.randint(-5, 5),
                     mode=cv2.BORDER_CONSTANT, cval=(255, 255, 255), p=1)
        ], p=0.5),
        A.Blur(blur_limit=5, p=0.25),
    ])
    img = transform(image=img)['image']

    return img
