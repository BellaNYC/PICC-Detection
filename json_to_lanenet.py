import numpy as np
import argparse
import random
import shutil
import json
import glob
import cv2
import os


def create_mask(gt, exp_folder, json_folder, png_folder):
    dst = os.path.join(exp_folder, gt)
    if not os.path.exists(dst):
        os.makedirs(dst)
    png_list = glob.glob(os.path.join(png_folder, '*.png'))

    for file in png_list:
        filename = (os.path.basename(file).split('.')[0])  # .split('_')[0]
        jsonname = filename + '.json'
        full_path = os.path.join(json_folder, jsonname)
        try:
            with open(full_path, 'r') as f:
                jsonfile = json.load(f)

        except:
            continue
        print(full_path)

        img = cv2.imread(file)
        #
        h, w, _ = img.shape
        # canvas = np.zeros([h, w, 3])
        color_map = {'Region': 255, 'non': 255}
        # color_map = {'Region':(255,0,0), 'non':(0,255,0)}
        # labeldict = {'Region':'picc' 'non':'nonpicc'}

        if 'ROIPoints' in jsonfile:
            jsonfile = [jsonfile]
        for ind, label in enumerate(jsonfile):
            # print(label)
            canvas = np.zeros([h, w])
            arr = [eval(x) for x in label['ROIPoints']]
            arr = np.array([arr], dtype=np.int32)
            if label['DataSummary']['Name'] == 'Region':
                shutil.copy2(file, os.path.join(exp_folder, 'image'))
                color = color_map[label['DataSummary']['Name']]

                cv2.fillPoly(canvas, arr, color)
                labelname = filename+'.png'
                filename = os.path.join(dst, labelname)
                cv2.imwrite(filename, canvas)

def create_labelfile(name, label_list, exp_name):
    current_path = os.path.dirname(os.path.abspath(__file__))

    exp_folder = os.path.join(current_path, exp_name)
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    fpath = os.path.join(exp_folder, name)

    with open(fpath, 'w') as f:
        for l in label_list:
            an = l.split(os.path.sep)[-1].split('.')[0]
            img = 'image/%s.png'%an
            binary = 'gt_image_binary/%s.png'%an
            instance = 'gt_image_instance/%s.png'%an
            f_img = os.path.join(os.path.join(current_path, exp_name),img)
            f_binary = os.path.join(os.path.join(current_path, exp_name), binary)
            f_instance = os.path.join(os.path.join(current_path, exp_name), instance)

            if not os.path.exists(f_img):
                continue

            string = (f_img + ' ' + f_binary + ' ' + f_instance + '\n')
            if l == label_list[-1]:
                string = string.strip()
            f.write(string)
    return exp_folder

def create_split(folder, exp_name):
    json_labels = glob.glob(os.path.join(folder, '*png'))
    random.shuffle(json_labels)

    val = json_labels[:20]
    train = json_labels[20:]
    create_labelfile('val.txt', val, exp_name)
    return create_labelfile('train.txt', train, exp_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('convert Osirix json output to lanenet dataset format')
    parser.add_argument('--json_folder', type=str, default='H:/Study/lanenet/data/picc_json', help='path to json folder')
    parser.add_argument('--png_input', type=str, default='H:/Study/lanenet/data/picc_png', help=' path to png input')
    parser.add_argument('--exp_name', type=str, default='H:/Study/lanenet/lanenet-lane-detection/data/train_data', help=' path to experiment folder')
    args = parser.parse_args()
    exp_folder = create_split(args.png_input, args.exp_name)
    create_mask('gt_image_binary', exp_folder, args.json_folder, args.png_input)
    create_mask('gt_image_instance', exp_folder, args.json_folder, args.png_input)
