# coding: utf-8

from __future__ import division

import os
import shutil
import tarfile
import itertools
import cv2
import numpy as np
import git
import glob


# default dataset directory
DEFAULT_DATADIR = 'data/muct'


class MuctDataset(object):
    # landmark pair connections
    PAIRS = (
        # jaw
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
        (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14),
        # right eyebrow
        (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 15),
        # left eyebrow
        (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 21),
        # left eye
        (27, 68), (68, 28), (28, 69), (69, 29),
        (29, 70), (70, 30), (30, 71), (71, 27),
        # right eye
        (32, 72), (72, 33), (33, 73), (73, 34),
        (34, 74), (74, 35), (35, 75), (75, 32),
        # nose
        (37, 38), (38, 39), (39, 40), (40, 41),
        (41, 42), (42, 43), (43, 44), (44, 45),
        # nose tip
        (41, 46), (46, 67), (67, 47), (47, 41),
        # upper lip
        (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),
        (48, 65), (65, 64), (64, 63), (63, 54),
        # lower lip
        (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),
        (48, 60), (60, 61), (61, 62), (62, 54),
    )

    # landmark flipping correspondences
    SYMMETRY = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 21,
                22, 23, 24, 25, 26, 15, 16, 17, 18, 19, 20, 32, 33, 34,
                35, 36, 27, 28, 29, 30, 31, 45, 44, 43, 42, 41, 40, 39,
                38, 37, 47, 46, 54, 53, 52, 51, 50, 49, 48, 59, 58, 57,
                56, 55, 62, 61, 60, 65, 64, 63, 66, 67, 72, 73, 74, 75,
                68, 69, 70, 71]

    # dataset urls
    URL = "https://github.com/StephenMilborrow/muct.git"

    def __init__(self, datadir=DEFAULT_DATADIR):
        self._datadir = os.path.join(os.path.dirname(os.path.realpath(__file__)), datadir)
        self._img_fname = os.path.join(os.path.dirname(os.path.realpath(__file__)), datadir, 'jpg/%s.jpg')

    def download(self):
        """downloads and unpacks the muct dataset"""
        # delete datadir if it already exists
        if os.path.exists(self._datadir):
            shutil.rmtree(self._datadir)
        # create datadir
        os.makedirs(self._datadir)
        # clone muct datasets
        git.Git(self._datadir.split('/')[0]).clone(self.URL)
        # change directory to datadir but don't forget where you came from
        cwd = os.getcwd()
        os.chdir(self._datadir)
        # unpack file if needed
        for filename in glob.glob('*.tar.gz'):
            with tarfile.open(filename) as tar:
                tar.extractall()
        # return to original directory
        os.chdir(cwd)

    def load(self, clean=False):
        # read landmarks file
        fname = os.path.join(os.path.dirname(os.path.realpath(__file__)), self._datadir,\
                              'muct-landmarks/muct76-opencv.csv')
        data = np.loadtxt(fname, delimiter=',', skiprows=1, dtype=str)
        # separate data
        names = np.char.array(data[:,0])
        tags = data[:,1]
        landmarks = data[:,2:].astype(float)
        # find flipped data
        flipped = names.startswith('ir')
        # keep data in self
        self.names = names[~flipped]
        self.tags = tags[~flipped]
        self.landmarks = landmarks[~flipped]
        self.landmarks_flip = landmarks[flipped]
        if clean:
            self.clean()

    def clean(self):
        """remove landmarks with unavailable points"""
        # unavailable points are marked with (0,0)
        is_complete = lambda x: all(x[::2] + x[1::2] != 0)
        keep = list(map(is_complete, self.landmarks))
        self.names = self.names[keep]
        self.tags = self.tags[keep]
        self.landmarks = self.landmarks[keep]
        self.landmarks_flip = self.landmarks_flip[keep]

    def ignore(self, name):
        keep = self.names != name
        self.names = self.names[keep]
        self.tags = self.tags[keep]
        self.landmarks = self.landmarks[keep]
        self.landmarks_flip = self.landmarks_flip[keep]

    def image(self, name, flip=False):
        img = cv2.imread(self._img_fname % name)
        return cv2.flip(img, 1) if flip else img

    def iterimages(self, mirror=False):
        # iterate over all images
        for n in self.names:
            yield self.image(n)
        # iterate over all mirror images if required
        if mirror:
            for n in self.names:
                yield self.image(n, flip=True)

    def iterdata(self):
        return zip(self.names, self.tags, self.landmarks, self.landmarks_flip)

    def all_lmks(self):
        return np.concatenate((self.landmarks, self.landmarks_flip))


# download dataset with command:
#   $ python -m pyaam.muct
if __name__ == '__main__':
    muct = MuctDataset()
    muct.download()
