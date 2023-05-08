# -*- coding: utf-8 -*-

import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm
import argparse

from tools.config import load_config
from tools.noiser import Noiser
from tools.utils import *
from tools.data_aug import *


def generate_text_picture(bg_imnames, chars, fonts_list, font_unsupport_chars, args):
    retry = 0

    # random add spacing between characters with 0.5 probability
    rd = random.random()
    if rd < 0.5:

        while True:
            bg_imname = random.choice(bg_imnames)
            img_path = os.path.join(bg_img_root_path, bg_imname)
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            w, h = img.size

            width = 0
            height = 0
            chars_size = []

            font_path = random.choice(fonts_list)
            font_size = random.randint(args.font_min_size, args.font_max_size)

            font = ImageFont.truetype(font_path, font_size)
            unsupport_chars = font_unsupport_chars[font_path]

            not_in_fonts = word_in_font(chars, unsupport_chars, font_path)

            if not_in_fonts:
                print('font not support. font pick retry.')
                continue

            for c in chars:
                size = font.getsize(c)
                chars_size.append(size)
                width += size[0]

                # set max char height as word height
                if size[1] > height:
                    height = size[1]

            char_space_width = int(height * np.random.uniform(-0.1, 0.3))
            width += (char_space_width * (len(chars) - 1))

            f_w, f_h = width, height
            if f_w < w:
                x1 = random.randint(0, w - f_w)
                y1 = random.randint(0, h - f_h)
                x2 = x1 + f_w
                y2 = y1 + f_h

                if args.random_offset:
                    rd = random.random()
                    if rd < 0.3:
                        crop_y1 = y1 - random.random() / 5 * f_h
                        crop_x1 = x1 - random.random() / 2 * f_h
                        crop_y2 = y2 + random.random() / 5 * f_h
                        crop_x2 = x2 + random.random() / 2 * f_h
                        crop_y1 = int(max(0, crop_y1))
                        crop_x1 = int(max(0, crop_x1))
                        crop_y2 = int(min(h, crop_y2))
                        crop_x2 = int(min(w, crop_x2))
                    else:
                        crop_y1 = y1
                        crop_x1 = x1
                        crop_y2 = y2
                        crop_x2 = x2
                else:
                    crop_y1 = y1
                    crop_x1 = x1
                    crop_y2 = y2
                    crop_x2 = x2

                crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                crop_lab = cv2.cvtColor(
                    np.asarray(crop_img), cv2.COLOR_RGB2Lab)

                # background color filter
                if (np.linalg.norm(np.reshape(np.asarray(crop_lab), (-1, 3)).std(axis=0)) > 55) and retry < 30:
                    retry += 1
                    print('crop background color too abundant retry.')
                    continue
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                best_color = (r, g, b)

                break
            else:
                print("background image too small retry. Image name:{}".format(img_path))
                pass

        draw = ImageDraw.Draw(img)

        for i, c in enumerate(chars):
            draw.text((x1, y1), c, best_color, font=font)
            x1 += (chars_size[i][0] + char_space_width)

        crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    else:
        while True:
            bg_imname = random.choice(bg_imnames)
            img_path = os.path.join(bg_img_root_path, bg_imname)
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            w, h = img.size

            font_path = random.choice(fonts_list)
            font_size = random.randint(args.font_min_size, args.font_max_size)

            font = ImageFont.truetype(font_path, font_size)
            unsupport_chars = font_unsupport_chars[font_path]
            not_in_fonts = word_in_font(chars, unsupport_chars, font_path)

            if not_in_fonts:
                print('font not support. font pick retry.')
                continue

            f_w, f_h = font.getsize(chars)
            if f_w < w:
                x1 = random.randint(0, w - f_w)
                y1 = random.randint(0, h - f_h)
                x2 = x1 + f_w
                y2 = y1 + f_h

                if args.random_offset:
                    rd = random.random()
                    if rd < 0.3:
                        crop_y1 = y1 - random.random() / 10 * f_h
                        crop_x1 = x1 - random.random() / 8 * f_h
                        crop_y2 = y2 + random.random() / 10 * f_h
                        crop_x2 = x2 + random.random() / 8 * f_h
                        crop_y1 = int(max(0, crop_y1))
                        crop_x1 = int(max(0, crop_x1))
                        crop_y2 = int(min(h, crop_y2))
                        crop_x2 = int(min(w, crop_x2))
                    else:
                        crop_y1 = y1
                        crop_x1 = x1
                        crop_y2 = y2
                        crop_x2 = x2
                else:
                    crop_y1 = y1
                    crop_x1 = x1
                    crop_y2 = y2
                    crop_x2 = x2

                crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                crop_lab = cv2.cvtColor(
                    np.asarray(crop_img), cv2.COLOR_RGB2Lab)

                # background color filter
                if (np.linalg.norm(np.reshape(np.asarray(crop_lab), (-1, 3)).std(axis=0)) > 55) and retry < 30:
                    retry += 1
                    print('crop background color too abundant retry')
                    continue

                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                best_color = (r, g, b)

                break
            else:
                print("background image too small retry. Image name:{}".format(img_path))
                pass

        draw = ImageDraw.Draw(img)
        draw.text((x1, y1), chars, best_color, font=font)
        crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    return crop_img, chars


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_img', type=int, default=30,
                        help="Number of images per text to generate")

    parser.add_argument('--font_min_size', type=int, default=30,
                        help="Can help adjust the size of the generated text and the size of the picture")

    parser.add_argument('--font_max_size', type=int, default=50,
                        help="Can help adjust the size of the generated text and the size of the picture")

    parser.add_argument('--bg_path', type=str, default='./background',
                        help='The generated text pictures will use the pictures of this folder as the background')

    parser.add_argument('--fonts_path', type=str, default='./fonts/greek',
                        help='The font used to generate the picture')

    parser.add_argument('--corpus_path', type=str, default='./corpus',
                        help='The corpus used to generate the text picture')

    parser.add_argument('--chars_file',  type=str, default='./dictionary/greek_eng_dict_v2.txt',
                        help='Chars allowed to be appear in generated images')

    parser.add_argument('--blur', action='store_true', default=False,
                        help="Apply gauss blur to the generated image")

    parser.add_argument('--prydown', action='store_true', default=False,
                        help="Blurred image, simulating the effect of enlargement of small pictures")

    parser.add_argument('--lr_motion', action='store_true', default=False,
                        help="Apply left and right motion blur")

    parser.add_argument('--ud_motion', action='store_true', default=False,
                        help="Apply up and down motion blur")

    parser.add_argument('--random_offset', action='store_true', default=True,
                        help="Randomly add offset")

    # enable noise.enable as true in noise.yaml to randomly add noice in generating images
    parser.add_argument('--config_file', type=str, default='./tools/noise.yaml',
                        help='Set the parameters when rendering images')

    parser.add_argument('--random_augmentation', action='store_true', default=False,
                        help='Set random augmentations')

    parser.add_argument('--output_dir', type=str,
                        default='./output/', help='Images save dir')

    parser.add_argument('--label_file', type=str,
                        default='labels.txt', help='Label txt save path')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    flag = load_config(args.config_file)

    # noiser parameters
    noiser = Noiser(flag)

    # load font path
    fonts_path = args.fonts_path
    fonts_list = get_fonts(fonts_path)

    # Read Corpus file
    txt_root_path = args.corpus_path
    char_lines = get_char_lines(txt_root_path=txt_root_path)

    bg_img_root_path = args.bg_path
    bg_imnames = os.listdir(bg_img_root_path)

    labels_path = args.label_file

    # dictionary path
    chars_file = args.chars_file
    font_unsupport_chars = get_unsupported_chars(fonts_list, chars_file)

    f = open(labels_path, 'a', encoding='utf-8')
    print('start generating images')
    for j in tqdm(range(len(char_lines))):
        img_n = 0
        for i in range(args.num_img):
            img_n += 1
            gen_img, chars = generate_text_picture(
                bg_imnames, char_lines[j], fonts_list, font_unsupport_chars, args)
            save_img_name = 'img_' + \
                str(j).zfill(3) + '_' + str(i).zfill(3) + '.jpg'

            if gen_img.mode != 'RGB':
                gen_img = gen_img.convert('RGB')

            # gauss blur
            if args.blur:
                image_arr = np.array(gen_img)
                gen_img = apply_blur_on_output(image_arr)
                gen_img = Image.fromarray(np.uint8(gen_img))

            # Blurred image
            if args.prydown:
                image_arr = np.array(gen_img)
                gen_img = apply_prydown(image_arr)
                gen_img = Image.fromarray(np.uint8(gen_img))

            # left and right motion blur
            if args.lr_motion:
                image_arr = np.array(gen_img)
                gen_img = apply_lr_motion(image_arr)
                gen_img = Image.fromarray(np.uint8(gen_img))

            # up and down motion blur
            if args.ud_motion:
                image_arr = np.array(gen_img)
                gen_img = apply_up_motion(image_arr)
                gen_img = Image.fromarray(np.uint8(gen_img))

            if apply(flag.noise):
                gen_img = np.clip(gen_img, 0., 255.)
                gen_img = noiser.apply(gen_img)
                gen_img = Image.fromarray(np.uint8(gen_img))

            # random data augmentation
            if args.random_augmentation:
                image_arr = np.array(gen_img)
                gen_img = random_augment_img(image_arr)
                image = Image.fromarray(np.uint8(gen_img))

            gen_img.save(args.output_dir+save_img_name)
            f.write(save_img_name+'\t'+chars+'\n')
    f.close()
    print('end generating images')
