# --------------------------------------------------------
# Copyright (c) 2016
# Author: Lyne P. Tchapmi
# --------------------------------------------------------

""" Initialize Paths to project main modules
"""
import sys
import os
import os.path as osp

def addpath(path):
    sys.path.insert(0, path)

# Add Project folder and subfolders
project_folder =os.getcwd()
addpath(project_folder)
addpath(osp.join(project_folder, 'utils'))
addpath(osp.join(project_folder, 'models'))
addpath(osp.join(project_folder, '../shared/'))
addpath(osp.join(project_folder, '../shared/datasets'))
addpath(osp.join(project_folder, 'utils/emd'))
addpath(osp.join(project_folder, 'utils/chamfer'))
