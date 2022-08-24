# Copyright (c) Gorilla Lab, SCUT.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Training/testing routines of DualPoseNet for category-level pose estimation on CAMERA25 or REAL275.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import pprint
pp = pprint.PrettyPrinter()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR


import tensorflow as tf
import utils.configs as configs
from model.dualposenet import DualPoseNet
from utils.evaluation_utils import evaluate


def run():
    FLAGS = configs.parse()
    assert FLAGS.dataset=='REAL275' or FLAGS.dataset=='CAMERA25' or FLAGS.dataset=='realsense_data' \
           or FLAGS.dataset == 'bsyn' or FLAGS.dataset == 'bsyn_segmented',\
        'Error dataset of {}, which should be chosen from [REAL275, CAMERA25, realsense_data, bsyn]'.format(FLAGS.dataset)
    assert FLAGS.phase in ['train', 'test', 'test_refine_encoder', 'test_refine_feature', 'transfer_train'],\
        'Error dataset of {}, which should be chosen from [train, test, test_refine_encoder, test_refine_feature, transfer_train]'.format(FLAGS.phase)

    FLAGS.log_dir = os.path.join('log', FLAGS.model)
    if not os.path.exists('log'):
        os.makedirs('log')
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    if FLAGS.phase !='train':
        FLAGS.test_log_dir = os.path.join(FLAGS.log_dir, FLAGS.phase + '_epoch' + str(FLAGS.test_epoch))
        if not os.path.exists(FLAGS.test_log_dir):
            os.makedirs(FLAGS.test_log_dir)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        model = DualPoseNet(FLAGS, sess)

        if FLAGS.phase == 'train':
            model.train()
        elif FLAGS.phase == 'transfer_train':
            model.transfer_train()
        else:
            if FLAGS.phase == 'test':
                model.test()
                # pass
            elif FLAGS.phase == 'test_refine_encoder':
                model.test_refine_encoder()
            elif FLAGS.phase == 'test_refine_feature':
                model.test_refine_feature()

            print('\n*********** Evaluate the results on {} ***********'.format(FLAGS.dataset))
            evaluate(os.path.join(FLAGS.test_log_dir, FLAGS.dataset), FLAGS.n_classes)


def main(unused_argv):
    run()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()



