from __future__ import division, print_function, absolute_import
import glob
import copy
import os.path as osp
from scipy.io import loadmat

from ..utils.tools import read_json, write_json

from ..utils.data import BaseImageDataset


class CUHKSYSU(BaseImageDataset):
    """CUHKSYSU.
    This dataset can only be used for model training.
    Reference:
        Xiao et al. End-to-end deep learning for person search.
    URL: `<http://www.ee.cuhk.edu.hk/~xgwang/PS/dataset.html>`_

    Dataset statistics:
        - identities: 11,934
        - images: 34,574
    """
    _train_only = True
    dataset_dir = 'cuhksysu'
    dataset_name = 'cuhksysu'

    def __init__(self, root='', verbose=True, combineall=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = self.root
        self.data_dir = osp.join(self.dataset_dir, 'cropped_images')

        # image name format: p11422_s16929_1.jpg
        train = self.process_dir(self.data_dir)

        query = [copy.deepcopy(train[0])]
        gallery = [copy.deepcopy(train[0])]

        self.train = train
        self.query = query
        self.gallery = gallery

        if combineall:
            self.train = self.combine_all(train, query, gallery)

        if verbose:
            print("=> CUHK-SYSU loaded")
            self.print_dataset_statistics(self.train, query, gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def process_dir(self, dirname):
            img_paths = glob.glob(osp.join(dirname, '*.jpg'))
            # num_imgs = len(img_paths)

            # get all identities:
            pid_container = set()
            for img_path in img_paths:
                img_name = osp.basename(img_path)
                pid = img_name.split('_')[0]
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

            # num_pids = len(pid_container)

            # extract data
            data = []
            for img_path in img_paths:
                img_name = osp.basename(img_path)
                pid = img_name.split('_')[0]
                label = pid2label[pid]
                data.append((img_path, label, 0))  # dummy camera id

            return data