import os
import numpy as np
import torch.utils.data as torch_data
import lib.utils.calibration as calibration
import lib.utils.kitti_utils as kitti_utils
from PIL import Image
import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
import lib.datasets.ground_segmentation as gs
from pyntcloud import PyntCloud
from lib.config import cfg
import random

import glob
import os

from argoverse.data_loading.simple_track_dataloader import SimpleArgoverseTrackingDataLoader

class KittiDataset(torch_data.Dataset):
    def __init__(self, root_dir, split='train'):
        self.split = split
        is_test = (self.split == 'test')
        self.lidar_pathlist = []
        self.label_pathlist = []
        
        #self.lidar_dir = os.path.join(root_dir)
        self._lidar_list = {}

        self.lidar_pathlist = sorted(glob.glob(os.path.join(root_dir, "lidar", "*.npy")))        

        self.label_pathlist = sorted(glob.glob(os.path.join(root_dir, cfg.label_name, "*.npy"))) 
               
        print('--------------------------------------------------------------len 1st data loader', len(self.lidar_pathlist) )

        self.lidar_filename = [x.split('.')[0].rsplit('/',1)[1] for x in self.lidar_pathlist]
        
        assert len(self.lidar_pathlist) == len(self.label_pathlist)

        self.num_sample = len(self.lidar_pathlist)
        self.lidar_idx_list = ['%06d'%l for l in range(self.num_sample)]
        
        self.lidar_idx_table = dict(zip(self.lidar_idx_list, self.lidar_filename))
        
        self.kitti_to_kitti = np.array([[1, 0, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 1, 0, 0 ],
                                       [0, 0, 0, 1]])
        
        self.ground_removal = False
        
        self.lidar_dir = os.path.join('/data/')        
        self.label_dir = os.path.join('/data/')
        
    def get_lidar(self,idx):
        lidar_file = self.lidar_pathlist[idx]
        assert os.path.exists(lidar_file)
        
        data = np.load(lidar_file)
        lidar_time = data[:,0:5]
        lidar_time = np.delete(lidar_time,3,1)
        s_lidar_time = lidar_time[-(cfg.past_frame) <= lidar_time[:,3]]
        pts_lidar = s_lidar_time[cfg.future_frame >= s_lidar_time[:,3]]

        # transform the data to kitti format        
        pts_lidar = np.dot(self.kitti_to_kitti,pts_lidar.T).T
        
        if self.ground_removal: 
            pts_lidar = gs.ground_segmentation(pts_lidar)
        
        return pts_lidar
        
        
    def get_label(self,idx):
        
        label_file = self.label_pathlist[idx]
        assert os.path.exists(label_file)
        
        return kitti_utils.get_objects_from_label(label_file)
    

