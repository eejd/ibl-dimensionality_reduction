#
# Simple test of dPCA on IBL data
#

# system imports
import sys
from pathlib import Path

sys.path.append('/Users/dep/Workspaces/dPCA/python')

# core imports
import matplotlib.pyplot as plt
import numpy as np

# ibl imports
import alf.io
from brainbox.processing import bincount2D
import ibllib.plots as iblplt

# external packages
from dPCA import dPCA


#code_camp_data = Path('~/Workspaces/Rodent/IBL/code_camp/data/')
code_camp_data = Path('/Users/dep/Workspaces/Rodent/IBL/code_camp/data/')
sessions = {
    'A': code_camp_data.joinpath(Path('ZM_1735/2019-08-01/001')),
    'B': code_camp_data.joinpath(Path('ibl_witten_04/2019-08-04/002')),
    'C': code_camp_data.joinpath(Path('ZM_1736/2019-08-09/004')),
    'D': code_camp_data.joinpath(Path('ibl_witten_04/2018-08-11/001')),
    'E': code_camp_data.joinpath(Path('KS005/2019-08-29/001')),
    'F': code_camp_data.joinpath(Path('KS005/2019-08-30/001')),
}

# select a session from the bunch
session_id = 'B'
session_path = Path(sessions[session_id])

# read in the alf objects
alf_path = session_path.joinpath('alf')
spikes = alf.io.load_object(alf_path, 'spikes')
clusters = alf.io.load_object(alf_path, 'clusters')
channels = alf.io.load_object(alf_path, 'channels')
trials = alf.io.load_object(alf_path, '_ibl_trials')
wheel = alf.io.load_object(alf_path, '_ibl_wheel')

T_BIN = 0.1

# compute raster map as a function of cluster number
R, times, clusters = bincount2D(spikes['times']/30000, spikes['clusters'], T_BIN)

