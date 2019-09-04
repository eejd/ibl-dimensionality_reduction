#
# tests for dimensionality reduction
# Authors: dim_reduction hackathon workgroup
# IBL Code Camp 2019
#
#

# imports from system
from pathlib import Path

# imports from ibllib conda environment
import numpy.ma as ma
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA

# imports from ibllib
import alf.io  
import ibllib.plots as iblplt

# imports from dim_reduction work
import dim_reduce

main_path = Path('/home/mic/drive_codecamp/')
SES = {
    # RSC --> CA1 --> midbrain, good behavior, bad recroding
    'A': main_path.joinpath(Path('ZM_1735/2019-08-01/001')),
    # visual cortex, good behavior, noisy recording
    'B': main_path.joinpath(Path('ibl_witten_04/2018-08-11/001')),
    # left probe, bad behavior, good recording
    'C': main_path.joinpath(Path('ZM_1736/2019-08-09/004')),
    # motor cortex, bad beahvior, good recording
    'D': main_path.joinpath(Path('ibl_witten_04/2019-08-04/001')),
    # activity in in red nucleaus, bad recording (serious lick artifacts and some units saturated) 
    'E': main_path.joinpath(Path('KS005/2019-08-29/001')),
    # too large, didnt download for now
    'F': main_path.joinpath(Path('KS005/2019-08-30/001')),
}
# select a session from the bunch
sid = 'B'
ses_path = Path(SES[sid])
# read in the alf objects
alf_path = ses_path / 'alf'
spikes = alf.io.load_object(alf_path, 'spikes')  # spikes['time'] or spikes.time
clusters = alf.io.load_object(alf_path, 'clusters')
channels = alf.io.load_object(alf_path, 'channels')
trials = alf.io.load_object(alf_path, '_ibl_trials')
wheel = alf.io.load_object(alf_path, '_ibl_wheel')

# bin data on all trial variables
binned_data = dim_reduce.bin_types(spikes, trials, wheel)

# produce dimensionality reduced data sets
isomap_transform = manifold.Isomap(n_components=3).fit_transform(X)

# plot transforms
dim_reduce.color_attractor(, , )
