from pathlib import Path

import numpy as np
#from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
import numpy.ma as ma

import alf.io  
from brainbox.processing import bincount2D
import matplotlib.pyplot as plt
import ibllib.plots as iblplt

# define the path to the sessions we downloaded 
main_path = Path(r'C:\Leenoy\Postdoc 1st year\IBL\Code_camp_September_2019\data_code_camp')
SES = {
    'A': main_path.joinpath(Path('ZM_1735/2019-08-01/001')), # RSC --> CA1 --> midbrain, good behavior, bad recroding
    'B': main_path.joinpath(Path('ibl_witten_04_002/2019-08-04/002')), # visual cortex, good behavior, noisy recording
    'C': main_path.joinpath(Path('ZM_1736/2019-08-09/004')),  # left probe, bad behavior, good recording
    'D': main_path.joinpath(Path('ibl_witten_04_001/2018-08-11/001')), # motor cortex, bad beahvior, good recording
    'E': main_path.joinpath(Path('KS005/2019-08-29/001')), # activity in in red nucleaus, bad recording (serious lick artifacts and some units saturated) 
#    'F': main_path.joinpath(Path('KS005/2019-08-30/001')), # too large, didnt download for now
}

# select a session from the bunch
sid = 'D'
ses_path = Path(SES[sid])

# read in the alf objects
alf_path = ses_path / 'alf'
spikes = alf.io.load_object(alf_path, 'spikes')  #can be addressed as spikes['time'] or spikes.time
clusters = alf.io.load_object(alf_path, 'clusters')
channels = alf.io.load_object(alf_path, 'channels')
trials = alf.io.load_object(alf_path, '_ibl_trials')
wheel = alf.io.load_object(alf_path, '_ibl_wheel')


T_BIN = 0.1

# compute raster map as a function of cluster number
R, times, clusters = bincount2D(spikes['times']/30000, spikes['clusters'], T_BIN)

# plot raster map
plt.imshow(R, aspect='auto', cmap='binary', vmax=T_BIN / 0.001 / 4,
           extent=np.r_[times[[0, -1]], clusters[[0, -1]]], origin='lower')
# plot trial start and reward time
reward = trials['feedback_times'][trials['feedbackType'] == 1]
iblplt.vertical_lines(trials['intervals'][:, 0], ymin=0, ymax=clusters[-1],
                      color='k', linewidth=0.5, label='trial starts')
iblplt.vertical_lines(reward, ymin=0, ymax=clusters[-1], color='m', linewidth=0.5,
                      label='valve openings')
plt.xlabel('Time (s)')
plt.ylabel('Cluster #')
plt.legend()

### Playing with Isomap, PCA, TSNE, UMAP 
plt.ion()
T_BIN = 0.01

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def bin_types(spikes, trials, wheel):
    T_BIN = 0.2  # [sec]
    # TO GET MEAN: bincount2D(..., weight=positions) / bincount2D(..., weight=None)
    reward_times = trials['feedback_times'][trials['feedbackType'] == 1]
    trial_start_times = trials['intervals'][:, 0]
    # trial_end_times = trials['intervals'][:, 1] #not working as there are
    # nans
    # compute raster map as a function of cluster number
    
    R1, times1, _ = bincount2D(spikes['times'], spikes['clusters'], T_BIN, weights=spikes['amps'])
    R2, times2, _ = bincount2D(reward_times, np.array([0] * len(reward_times)), T_BIN)
    R3, times3, _ = bincount2D(trial_start_times, np.array([0] * len(trial_start_times)), T_BIN)
    R4, times4, _ = bincount2D(wheel['times'], np.array([0] * len(wheel['times'])), T_BIN, weights=wheel['position'])
    R5, times5, _ = bincount2D(wheel['times'], np.array([0] * len(wheel['times'])), T_BIN, weights=wheel['velocity'])
    #R6, times6, _ = bincount2D(trial_end_times, np.array([0]*len(trial_end_times)), T_BIN)
    start = max([x for x in [times1[0], times2[0], times3[0], times4[0], times5[0]]])
    stop = min([x for x in [times1[-1], times2[-1], times3[-1], times4[-1], times5[-1]]])
    time_points = np.linspace(start, stop, int((stop - start) / T_BIN))
    binned_data = {}
    binned_data['wheel_position'] = np.interp(time_points, wheel['times'], wheel['position'])
    binned_data['wheel_velocity'] = np.interp(time_points, wheel['times'], wheel['velocity'])
    binned_data['summed_spike_amps'] = R1[:, find_nearest(times1, start):find_nearest(times1, stop)]
    binned_data['reward_event'] = R2[0, find_nearest(times2, start):find_nearest(times2, stop)]
    binned_data['trial_start_event'] = R3[0, find_nearest(times3, start):find_nearest(times3, stop)]
    # binned_data['trial_end_event']=R6[0,find_nearest(times6,start):find_nearest(times6,stop)]
    # np.vstack([R1,R2,R3,R4])
    return binned_data


def Isomap_colored(binned_data, bounds):
    # obs_limit=1000 #else it's too slow
    low, high = bounds
    X = binned_data['summed_spike_amps'][:, low:high].T
    Y = Isomap(n_components=3).fit_transform(X)
    x, y, z = np.split(Y, 3, axis=1)
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter(x, y, z, s=20, alpha=0.25, c=abs(binned_data['wheel_velocity'][low:high]), cmap='ocean')
    fig.colorbar(p)
    plt.title("Isomap Alex's motor cortex recording vs wheel speed")
    # plt.scatter(Y.T[0,:],Y.T[1,:],s=1,alpha=0.9,c=D_trial['wheel_velocity'][trial])
    plt.show()

def PCA_colored(binned_data, bounds):
    # obs_limit=1000 #else it's too slow
    low, high = bounds
    X = binned_data['summed_spike_amps'][:, low:high].T
    Y = PCA(n_components=3, svd_solver = 'full').fit_transform(X)
    x, y, z = np.split(Y, 3, axis=1)
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter(x, y, z, s=20, alpha=0.25, c=abs(binned_data['wheel_velocity'][low:high]), cmap='ocean')
    fig.colorbar(p)
    plt.title("PCA Alex's motor cortex recording vs wheel speed")
    # plt.scatter(Y.T[0,:],Y.T[1,:],s=1,alpha=0.9,c=D_trial['wheel_velocity'][trial])
    plt.show()

def TSNE_colored(binned_data, bounds):
    # obs_limit=1000 #else it's too slow
    low, high = bounds
    X = binned_data['summed_spike_amps'][:, low:high].T
    Y = TSNE(n_components=3).fit_transform(X)
    x, y, z = np.split(Y,3, axis=1)
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter(x, y, z, s=20, alpha=0.25, c=abs(binned_data['wheel_velocity'][low:high]), cmap='ocean')
    fig.colorbar(p)
    plt.title("TSNE Alex's motor cortex recording vs wheel speed")
    # plt.scatter(Y.T[0,:],Y.T[1,:],s=1,alpha=0.9,c=D_trial['wheel_velocity'][trial])
    plt.show()

# lets run and plot 
binned_data = bin_types(spikes, trials, wheel)
# binned_data.keys()
Isomap_colored(binned_data, (0,1000))
PCA_colored(binned_data, (0,1000))
TSNE_colored(binned_data, (0,1000)) # takes a long time