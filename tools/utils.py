#!/usr/env/bin python3

import os
import random
import cv2
import numpy as np
import hashlib
import sys
import pickle
from fontTools.ttLib import TTCollection, TTFont


def prob(percent):
    """
    percent: 0 ~ 1, e.g: 如果 percent=0.1, 有 10% 的可能性
    """
    assert 0 <= percent <= 1
    if random.uniform(0, 1) <= percent:
        return True
    return False


def draw_box(img, pnts, color):
    """
    :param img: gray image, will be convert to BGR image
    :param pnts: left-top, right-top, right-bottom, left-bottom
    :param color:
    :return:
    """
    if isinstance(pnts, np.ndarray):
        pnts = pnts.astype(np.int32)

    if len(img.shape) > 2:
        dst = img
    else:
        dst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    thickness = 1
    linetype = cv2.LINE_AA
    cv2.line(dst, (pnts[0][0], pnts[0][1]), (pnts[1][0], pnts[1][1]), color=color, thickness=thickness,
             lineType=linetype)
    cv2.line(dst, (pnts[1][0], pnts[1][1]), (pnts[2][0], pnts[2][1]), color=color, thickness=thickness,
             lineType=linetype)
    cv2.line(dst, (pnts[2][0], pnts[2][1]), (pnts[3][0], pnts[3][1]), color=color, thickness=thickness,
             lineType=linetype)
    cv2.line(dst, (pnts[3][0], pnts[3][1]), (pnts[0][0], pnts[0][1]), color=color, thickness=thickness,
             lineType=linetype)
    return dst


def draw_bbox(img, bbox, color):
    pnts = [
        [bbox[0], bbox[1]],
        [bbox[0] + bbox[2], bbox[1]],
        [bbox[0] + bbox[2], bbox[1] + bbox[3]],
        [bbox[0], bbox[1] + bbox[3]]
    ]
    return draw_box(img, pnts, color)


def load_bgs(bg_dir):
    dst = []

    for root, sub_folder, file_list in os.walk(bg_dir):
        for file_name in file_list:
            image_path = os.path.join(root, file_name)

            # For load non-ascii image_path on Windows
            bg = cv2.imdecode(np.fromfile(
                image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

            dst.append(bg)

    print("Background num: %d" % len(dst))
    return dst


def load_chars(filepath):
    if not os.path.exists(filepath):
        print("Chars file not exists.")
        exit(1)

    ret = ''
    with open(filepath, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            ret += line[0]
    return ret


def md5(string):
    m = hashlib.md5()
    m.update(string.encode('utf-8'))
    return m.hexdigest()


def apply(cfg_item):
    """
    :param cfg_item: a sub cfg item in default.yml, it should contain enable and fraction. such as
                prydown:
                    enable: true
                    fraction: 0.03
    :return: True/False
    """

    if cfg_item.enable and prob(cfg_item.fraction):
        return True

    return False


def get_platform():
    platforms = {
        'linux1': 'Linux',
        'linux2': 'Linux',
        'darwin': 'OS X',
        'win32': 'Windows'
    }
    if sys.platform not in platforms:
        return sys.platform

    return platforms[sys.platform]


def get_fonts(fonts_path):
    font_files = os.listdir(fonts_path)
    fonts_list = []
    for font_file in font_files:
        font_path = os.path.join(fonts_path, font_file)
        fonts_list.append(font_path)
    return fonts_list


def get_unsupported_chars(fonts, chars_file):
    """
    Get fonts unsupported chars by loads/saves font supported chars from cache file
    :param fonts:
    :param chars_file:
    :return: dict
        key -> font_path
        value -> font unsupported chars
    """
    charset = load_chars(chars_file)
    charset = ''.join(charset)
    fonts_chars = get_fonts_chars(fonts, chars_file)
    fonts_unsupported_chars = {}
    for font_path, chars in fonts_chars.items():
        unsupported_chars = list(filter(lambda x: x not in chars, charset))
        fonts_unsupported_chars[font_path] = unsupported_chars
    return fonts_unsupported_chars


def load_chars(filepath):
    if not os.path.exists(filepath):
        print("Chars file not exists.")
        exit(1)

    ret = ''
    with open(filepath, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            ret += line[0]
    return ret


def get_fonts_chars(fonts, chars_file):
    """
    loads/saves font supported chars from cache file
    :param fonts: list of font path. e.g ['./data/fonts/msyh.ttc']
    :param chars_file: arg from parse_args
    :return: dict
        key -> font_path
        value -> font supported chars
    """
    out = {}

    cache_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../', '.caches'))
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    chars = load_chars(chars_file)
    chars = ''.join(chars)

    for font_path in fonts:
        string = ''.join([font_path, chars])
        file_md5 = md5(string)

        cache_file_path = os.path.join(cache_dir, file_md5)

        if not os.path.exists(cache_file_path):
            ttf = load_font(font_path)
            _, supported_chars = check_font_chars(ttf, chars)
            print('Save font(%s) supported chars(%d) to cache' %
                  (font_path, len(supported_chars)))

            with open(cache_file_path, 'wb') as f:
                pickle.dump(supported_chars, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(cache_file_path, 'rb') as f:
                supported_chars = pickle.load(f)
            print('Load font(%s) supported chars(%d) from cache' %
                  (font_path, len(supported_chars)))

        out[font_path] = supported_chars

    return out


def load_font(font_path):
    """
    Read ttc, ttf, otf font file, return a TTFont object
    """

    # ttc is collection of ttf
    if font_path.endswith('ttc'):
        ttc = TTCollection(font_path)
        # assume all ttfs in ttc file have same supported chars
        return ttc.fonts[0]

    if font_path.endswith('ttf') or font_path.endswith('TTF') or font_path.endswith('otf'):
        ttf = TTFont(font_path, 0, allowVID=0,
                     ignoreDecompileErrors=True, fontNumber=-1)
        return ttf


def md5(string):
    m = hashlib.md5()
    m.update(string.encode('utf-8'))
    return m.hexdigest()


def check_font_chars(ttf, charset):
    """
    Get font supported chars and unsupported chars
    :param ttf: TTFont ojbect
    :param charset: chars
    :return: unsupported_chars, supported_chars
    """
    chars_int = set()
    for table in ttf['cmap'].tables:
        for k, v in table.cmap.items():
            chars_int.add(k)

    unsupported_chars = []
    supported_chars = []
    for c in charset:
        if ord(c) not in chars_int:
            unsupported_chars.append(c)
        else:
            supported_chars.append(c)

    ttf.close()
    return unsupported_chars, supported_chars


def get_char_lines(txt_root_path):
    txt_files = os.listdir(txt_root_path)
    char_lines = []
    for txt in txt_files:
        f = open(os.path.join(txt_root_path, txt), mode='r', encoding='utf-8')
        lines = f.readlines()
        f.close()
        for line in lines:
            char_lines.append(line.strip().replace(
                '\xef\xbb\xbf', '').replace('\ufeff', ''))
    return char_lines


def word_in_font(word, unsupport_chars, font_path):
    for c in word:
        if c in unsupport_chars:
            # print('Retry pick_font(), \'%s\' contains chars \'%s\' not supported by font %s' % (word, c, font_path))
            return True
        else:
            continue


if __name__ == '__main__':
    print(md5('test123'))
