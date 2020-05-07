#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
test LaneNet model on single image
"""
import argparse
import os.path as ops
import time

import cv2
import glog as log
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import global_config
from lanenet_model import lanenet
from sklearn.metrics import auc, roc_curve

CFG = global_config.cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def _test_lanenet(image_path, weights_path):
    """

    :param image_path:
    :param weights_path:
    :return:
    """
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)

    log.info('Start reading image and preprocessing')
    t_start = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_vis = image
    image = cv2.resize(image, (CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
    image = image / 127.5 - 1.0
    log.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model', return_score=True)

    saver = tf.train.Saver()

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():
        model_file = tf.train.latest_checkpoint(weights_path)
        saver.restore(sess=sess, save_path=model_file)

        t_start = time.time()
        binary_seg_image, instance_seg_image = sess.run(
            [binary_seg_ret, instance_seg_ret],
            feed_dict={input_tensor: [image]}
        )
        t_cost = time.time() - t_start
        log.info('Single imgae inference cost time: {:.5f}s'.format(t_cost))
        plt.figure('src_image')
        plt.imshow(image_vis[:, :, (2, 1, 0)])
        plt.axis('off')
        plt.figure('binary_image')
        plt.imshow(binary_seg_image[0,:,:,1] * 255, cmap='gray')
        plt.axis('off')
        plt.show()

    sess.close()

    return

def plot_auc(txt_file, weights_path):

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH, 3],
                                  name='input_tensor')

    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model', return_score=True)

    saver = tf.train.Saver()

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():
        model_file = tf.train.latest_checkpoint(weights_path)
        saver.restore(sess=sess, save_path=model_file)
        gts, preds = [], []
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                image_path, gt_path, _ = line.strip().split()
                log.info('Processing {}.'.format(image_path))
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
                image = image / 127.5 - 1.0
                seg_image_gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                seg_image_gt = cv2.resize(seg_image_gt, (CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
                seg_image_score = sess.run([binary_seg_ret], feed_dict={input_tensor: [image]})
                seg_image_score = seg_image_score[0][0, :, :, 1]
                gts.append((seg_image_gt>0))
                preds.append(seg_image_score)
    sess.close()

    gt = np.array(gts)
    pred = np.array(preds)
    fpr, tpr, _ = roc_curve(gt.flatten(), pred.flatten())
    plt.title('ROC')
    plt.plot(fpr, tpr)
    plt.xlabel('fp')
    plt.ylabel('tp')
    plt.savefig('./tboard/roc_{}.png'.format(txt_file.split('/')[-1][:-4]))

    return

if __name__ == '__main__':

    weights_path = 'G:/Study/lanenet/lanenet-lane-detection/model/tusimple_lanenet_vgg'
    image_path = './data/train_data/image/5492674.png'
    _test_lanenet(image_path, weights_path)

    # txt_file = './data/train_data/val.txt'
    # plot_auc(txt_file, weights_path)