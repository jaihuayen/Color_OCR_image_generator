import os
import shutil

for i in range(1,80001):
    file_name = 'img_' + str(i).zfill(3) + '.jpg'
    src = './training_set/' + file_name
    dst = './new_training_set/' + file_name
    shutil.copyfile(src, dst)