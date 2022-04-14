from __future__ import print_function, absolute_import
import os.path as osp
import tarfile

import glob
import re
import urllib
import zipfile

from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json
from ..utils.data import BaseImageDataset


class Dataset_MSMT(BaseImageDataset):
    dataset_name = 'msmt17'
    def __init__(self, root):
        super(Dataset_MSMT, self).__init__()
        self.root = root
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0

    @property
    def images_dir(self):
        return osp.join(self.root, 'MSMT17_V1')

    def load(self, verbose=True, combineall=False):
        exdir = osp.join(self.root, 'MSMT17_V1')
        self.train = self._pluck_msmt(osp.join(exdir, 'list_train.txt'), 'train', relabel=True)
        self.val = self._pluck_msmt(osp.join(exdir, 'list_val.txt'), 'train', relabel=True)
        self.train = self.train + self.val
        self.query = self._pluck_msmt(osp.join(exdir, 'list_query.txt'), 'test')
        self.gallery = self._pluck_msmt(osp.join(exdir, 'list_gallery.txt'), 'test')

        if combineall:
            self.train = self.combine_all(self.train, self.query, self.gallery)

        if verbose:
            print("=> MSMT17 loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _pluck_msmt(self, list_file, subdir, pattern=re.compile(r'([-\d]+)_([-\d]+)_([-\d]+)'), relabel=False):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        ret = []
        for line in lines:
            line = line.strip()
            fname = line.split(' ')[0]
            pid, _, camid = map(int, pattern.search(osp.basename(fname)).groups())
            if relabel:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            ret.append((osp.join(self.images_dir, subdir, fname), pid, camid))
        return ret


class MSMT17(Dataset_MSMT):

    def __init__(self, root, split_id=0, download=True, verbose=True, combineall=False):
        super(MSMT17, self).__init__(root)

        if download:
            self.download()

        self.load(verbose=verbose, combineall=combineall)


    def download(self):

        import re
        import hashlib
        import shutil
        from glob import glob
        from zipfile import ZipFile

        raw_dir = osp.join(self.root)
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        fpath = osp.join(raw_dir, 'MSMT17_V1')
        if osp.isdir(fpath):
            pass
            # print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please download the dataset manually to {}".format(fpath))
