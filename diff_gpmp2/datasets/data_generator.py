import os
import numpy as np
from imageio import imread
import random
from scipy.ndimage import distance_transform_edt as dist_trans


class DataGenerator(object):
    def __init__(self, data_dir=None, im_rows=256, im_cols=256, im_chan=1,
                 batch_size=32, cell_size=0.01, epsilon_sdf=0.1, shuffle=True):
        self.im_rows = im_rows
        self.im_cols = im_cols
        self.im_chan = im_chan
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.cell_size = cell_size
        self.epsilon_sdf = epsilon_sdf

    def __len__(self):
        indices = self._get_exploration_order()
        return int(len(indices) / self.batch_size)

    def generate(self):
        while True:
            folders = self._get_exploration_order()
            for i in range(0,len(folders)-self.batch_size,self.batch_size):
                list_IDs_tmp = folders[i:i+self.batch_size]
                I_obs, I_cost = self._data_generation(list_IDs_tmp)
                yield I_obs, I_cost

    def _get_exploration_order(self):
        folders = os.listdir(self.data_dir)
        if self.shuffle == True:
            random.shuffle(folders)
        return folders

    def get_costmap(self,im):
        # Signed distance field
        map_im = im > 0.75
        inv_map_im = 1 - map_im
        map_dist = dist_trans(map_im)
        inv_map_dist = dist_trans(inv_map_im)
        field = map_dist - inv_map_dist
        field = field * self.cell_size

        # Hinge Loss
        loss = -1.0 * field + self.epsilon_sdf
        hinge = field <= self.epsilon_sdf
        cost_map = hinge * loss
        return cost_map

    def _data_generation(self, list_IDs):
        I_obs = np.empty((self.batch_size, self.im_rows, self.im_cols, self.im_chan))
        I_cost = np.empty((self.batch_size, self.im_rows, self.im_cols, self.im_chan))

        for i, ID in enumerate(list_IDs):
            im_obs = imread('{}/{}/obstacles.png'.format(self.data_dir,ID)).reshape(
                         (self.im_rows, self.im_cols, self.im_chan))
            im_obs = im_obs/255
            I_obs[i,:,:,:] = im_obs
            I_cost[i,:,:,:] = self.get_costmap(im_obs)

        return I_obs, I_cost