#
# Eric's test
#

# imports from system
import sys
from pathlib import Path

sys.path.append('/Users/dep/Workspaces/Rodent/IBL/ibllib')
sys.path.append('/Users/dep/Workspaces/Rodent/IBL/code_camp/ibl-dimensionality_reduction/')
sys.path.append('/Users/dep/Workspaces/dPCA/python')

# imports from ibllib conda environment
import numpy.ma as ma
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap as ISOMap
from sklearn.decomposition import PCA

# imports from ibllib
import alf.io  
import ibllib.plots as iblplt

# imports from dim_reduction work
import dim_reduce

main_path = Path('/Users/dep/Workspaces/Rodent/IBL/code_camp/data/')

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
sid = 'A'
ses_path = Path(SES[sid])

# read in the alf objects
alf_path = ses_path / 'alf'
spikes = alf.io.load_object(alf_path, 'spikes')
clusters = alf.io.load_object(alf_path, 'clusters')
channels = alf.io.load_object(alf_path, 'channels')
trials = alf.io.load_object(alf_path, '_ibl_trials')
wheel = alf.io.load_object(alf_path, '_ibl_wheel')

# bin data on all trial variables
binned = dim_reduce.bin_types(spikes, trials, wheel, 0.2)

# produce dimensionality reduced data sets
# isomap_transform = ISOMap(n_components=3).fit_transform(neural_data)

# get a trial
# trial_neural_data, trial_variable_data = dim_reduce.get_trial()

# plot transforms
# p = dim_reduce.color_attractor(neural_data, isomap_transform, 100)

# plt.title("Guido's motor cortex --> thalamus recording vs wheel speed")
# plt.scatter(Y.T[0,:],Y.T[1,:],s=1,alpha=0.9,c=D_trial['wheel_velocity'][trial])
# plt.show()
