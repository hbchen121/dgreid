# encoding: utf-8
import numpy as np

from torch.utils.data import Dataset
from .base_dataset import BaseImageDataset


class SplitDataset(BaseImageDataset):
    """Image Person ReID Dataset, Split one dataset"""
    def __init__(self, img_items, verbose=True, num_sets=3):
        super(SplitDataset, self).__init__()
        self.img_items = img_items
        self.num_sets = num_sets

        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))

        def chunks(l, n):
            if len(l) % n == 0:
                n = len(l) // n
            else:
                n = len(l) // n + 1
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]

        self.cams_list = [cam for cam in chunks(self.cams, num_sets)]

        self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
        self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

        def in_subset(l, cid):
            for i, subset in enumerate(l):
                if cid in subsgitet:
                    return i
            assert "Invaild camera id"
            return -1

        self.train = [[] for _ in range(num_sets)]

        for img, pid, camid in img_items:
            set_idx = in_subset(self.cams_list, camid)
            self.train[set_idx].append((img, self.pid_dict[pid], self.cam_dict[camid]))

        # train = [(img, self.pid_dict[pid], self.cam_dict[camid]) for img, pid, camid in img_items]
        # self.train = train
        self.query = []
        self.gallery = []

        # self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        if verbose:
            print("=> Split Dataset loaded")
            for i in range(num_sets):
                print("=> Sub-Dataset {}".format(str(i)))
                self.print_dataset_statistics(self.train[i], self.query, self.gallery)

    def align_train(self, pid_dict, cam_dict):
        self.pid_dict = pid_dict
        self.cam_dict = cam_dict
        train = [(img, self.pid_dict[pid], self.cam_dict[camid]) for img, pid, camid in self.img_items]
        self.train = train
        self.query = []
        self.gallery = []
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)


class CommDataset(BaseImageDataset):
    """Image Person ReID Dataset, combine all datasets"""

    def __init__(self, img_items, verbose=True):
        super(CommDataset, self).__init__()
        self.img_items = img_items

        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))

        self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
        self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

        train = [(img, self.pid_dict[pid], self.cam_dict[camid]) for img, pid, camid in img_items]
        self.train = train
        self.query = []
        self.gallery = []
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        if verbose:
            print("=> Combine-all-reID loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

    def align_train(self, pid_dict, cam_dict):
        self.pid_dict = pid_dict
        self.cam_dict = cam_dict
        train = [(img, self.pid_dict[pid], self.cam_dict[camid]) for img, pid, camid in self.img_items]
        self.train = train
        self.query = []
        self.gallery = []
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)


def align_label(datasets:list):
    """ datasets: [ CommDataset1, CommDataset2, ... ]
    map the label of all commDataset into unified space """
    pids_set = set()
    cams_set = set()
    for dataset in datasets:
        for pid in dataset.pids:
            pids_set.add(pid)
        for camid in dataset.cams:
            cams_set.add(camid)
    pid_dict = dict([(p, i) for i, p in enumerate(list(pids_set))])
    cam_dict = dict([(p, i) for i, p in enumerate(list(cams_set))])
    for dataset in datasets:
        dataset.align_train(pid_dict, cam_dict)
    return datasets
