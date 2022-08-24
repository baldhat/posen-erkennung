import os
import numpy as np
import queue
import threading
import glob
import _pickle as cPickle


class Fetcher(threading.Thread):
    def __init__(self, opts):
        super(Fetcher, self).__init__()
        self.queue = queue.Queue(50)
        self.stopped = False
        self.opts = opts

        self.data_paths = glob.glob('D:/code/python/DualPoseNet/data/training_instance/CAMERA25_*.pkl')
        self.data_paths.append('D:/code/python/DualPoseNet/data/training_instance/REAL275.pkl')
        # data_paths.append('E:/data/training_instance/baldhatsyn.pkl')


        num_items = 0
        for data_path in self.data_paths:
            print(data_path)
            with open(data_path, 'rb') as f:
                data = cPickle.load(f)
            num_items += data['observed_pc'].shape[0]

        self.batch_size = self.opts.batch_size
        self.sample_cnt = num_items
        self.num_batches = self.sample_cnt//self.batch_size
        print ("NUM_INSTANCE is %s"%(self.sample_cnt))
        print ("NUM_BATCH is %s"%(self.num_batches))

    def run(self):
        pkl_index = 0
        while not self.stopped:
            with open(self.data_paths[pkl_index], 'rb') as f:
                data = cPickle.load(f)

            idx = np.arange(self.sample_cnt)
            np.random.shuffle(idx)
            self.observed_pc = self.observed_pc[idx, ...]
            self.input_dis = self.input_dis[idx, ...]
            self.input_rgb = self.input_rgb[idx, ...]
            self.rotation = self.rotation[idx, ...]
            self.translation = self.translation[idx, ...]
            self.scale = self.scale[idx, ...]

            for batch_idx in range(self.num_batches):
                if self.stopped:
                    return None
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size

                batch_input_dis = self.input_dis[start_idx:end_idx, :, :, :].copy()
                batch_input_rgb = self.input_rgb[start_idx:end_idx, :, :, :].copy()
                batch_observed_pc = self.observed_pc[start_idx:end_idx, :, :].copy()
                batch_rotation = self.rotation[start_idx:end_idx, :].copy()
                batch_translation = self.translation[start_idx:end_idx, :].copy()
                batch_scale = self.scale[start_idx:end_idx, :].copy()
                self.queue.put((batch_input_dis, batch_input_rgb, batch_observed_pc, batch_rotation, batch_translation, batch_scale))
        return None

    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        print ("Shutdown .....")
        while not self.queue.empty():
            self.queue.get()
        print ("Remove all queue data")

