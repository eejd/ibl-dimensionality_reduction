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
import dim_reduce.bin_types
import dim_reduce.find_nearest
import dim_reduce.color_attractor

main_path = Path('/home/mic/drive_codecamp/')
SES = {
    'A': main_path.joinpath(Path('ZM_1735/2019-08-01/001')), # RSC --> CA1 --> midbrain, good behavior, bad recroding
    'B': main_path.joinpath(Path('ibl_witten_04/2018-08-11/001')), # visual cortex, good behavior, noisy recording
    'C': main_path.joinpath(Path('ZM_1736/2019-08-09/004')),  # left probe, bad behavior, good recording
    'D': main_path.joinpath(Path('ibl_witten_04/2019-08-04/001')), # motor cortex, bad beahvior, good recording
    'E': main_path.joinpath(Path('KS005/2019-08-29/001')), # activity in in red nucleaus, bad recording (serious lick artifacts and some units saturated) 
#    'F': main_path.joinpath(Path('KS005/2019-08-30/001')), # too large, didnt download for now
}
# select a session from the bunch
sid = 'B'
ses_path = Path(SES[sid])
# read in the alf objects
alf_path = ses_path / 'alf'
spikes = alf.io.load_object(alf_path, 'spikes')  #can be addressed as spikes['time'] or spikes.time
clusters = alf.io.load_object(alf_path, 'clusters')
channels = alf.io.load_object(alf_path, 'channels')
trials = alf.io.load_object(alf_path, '_ibl_trials')
wheel = alf.io.load_object(alf_path, '_ibl_wheel')

color_attractor(, manifold.Isomap(n_components=3).fit_transform(X), )
