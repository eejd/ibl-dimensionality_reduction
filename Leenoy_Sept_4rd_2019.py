# we need to cross validate all the methods

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

from sklearn.manifold import Isomap, MDS, TSNE, LocallyLinearEmbedding
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import sys
sys.path.append(r'C:\Leenoy\Postdoc 1st year\IBL\Code_camp_September_2019\data_code_camp\dim_red_WG\umap-master')
import umap
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns

import alf.io  
from brainbox.processing import bincount2D
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
sid = 'A'
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

## plot raster map
#plt.imshow(R, aspect='auto', cmap='binary', vmax=T_BIN / 0.001 / 4,
#           extent=np.r_[times[[0, -1]], clusters[[0, -1]]], origin='lower')
## plot trial start and reward time
#reward = trials['feedback_times'][trials['feedbackType'] == 1]
#iblplt.vertical_lines(trials['intervals'][:, 0], ymin=0, ymax=clusters[-1],
#                      color='k', linewidth=0.5, label='trial starts')
#iblplt.vertical_lines(reward, ymin=0, ymax=clusters[-1], color='m', linewidth=0.5,
#                      label='valve openings')
#plt.xlabel('Time (s)')
#plt.ylabel('Cluster #')
#plt.legend()

### Playing with Isomap, PCA, TSNE, UMAP 
plt.ion()
T_BIN = 0.01

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def filliti(v):
    """
    Fill values in between trials with common variable of trial. 
    E.g All the bins within a trials hould have the same choice value
    v, param:  one-dimensional vector with trial variable only in the first bin
    of each trial
    """
    for x in range(len(v[0])):
        if v[0,x]==0:
            v[0,x] = v[0,x-1]
    return v 


def bin_types(spikes, trials, wheel, t_bin):
    T_BIN = t_bin  # [sec]

    # TO GET MEAN: bincount2D(..., weight=positions) / bincount2D(..., weight=None)
    reward_times = trials['feedback_times']
    trial_start_times = trials['intervals'][:, 0]
    # trial_end_times = trials['intervals'][:, 1] #not working as there are
    # nans
    # compute raster map as a function of cluster number
    # Load in different things
    R1, times1, _ = bincount2D(
        spikes['times'], spikes['clusters'], T_BIN, weights=spikes['amps'])
    R2, times2, _ = bincount2D(
        reward_times, np.array(
            [0] * len(reward_times)), T_BIN)
    R3, times3, _ = bincount2D(
        trial_start_times, np.array(
            [0] * len(trial_start_times)), T_BIN)
    R4, times4, _ = bincount2D(wheel['times'], np.array(
        [0] * len(wheel['times'])), T_BIN, weights=wheel['position'])
    R5, times5, _ = bincount2D(wheel['times'], np.array(
        [0] * len(wheel['times'])), T_BIN, weights=wheel['velocity'])

    #Add choice
    R6, times6, _ = bincount2D(trials['goCue_times'],trials['choice'], T_BIN)
    # Flatten choice -1 Left, 1  Right
    R6 = np.sum(R6*np.array([[-1], [1]]),axis=0)
    R6 = np.expand_dims(R6, axis=0)
    # Fill 0 between trials with choice outcome of trial
    R6 =  filliti(R6)
    R6[R6==-1]=0
    #Add reward
    R7, times7, _ = bincount2D(trials['goCue_times'],trials['feedbackType'], T_BIN)
    # Flatten reward -1 error, 1  reward
    R7 = np.sum(R7*np.array([[-1], [1]]),axis=0)
    R7 = np.expand_dims(R7, axis=0)
    # Fill 0 between trials with reward outcome of trial
    R7 =  filliti(R7)
    R7[R7==-1]=0

    start = max([x for x in [times1[0], times3[0], times4[0], times5[0],times6[0], times7[0]]])
    stop = min([x for x in [times1[-1], times2[-1],
                            times3[-1], times4[-1], times5[-1], times6[-1], times7[-1]]])
    time_points = np.linspace(start, stop, int((stop - start) / T_BIN))
    binned_data = {}
    binned_data['wheel_position'] = np.interp(time_points, wheel['times'], wheel['position'])
    binned_data['wheel_velocity'] = np.interp(time_points, wheel['times'], wheel['velocity'])
    binned_data['summed_spike_amps'] = R1[:, find_nearest(times1, start):find_nearest(times1, stop)]
    binned_data['reward_event'] = R2[0, find_nearest(times2, start):find_nearest(times2, stop)]
    binned_data['trial_start_event'] = R3[0, find_nearest(times3, start):find_nearest(times3, stop)]
    binned_data['choice'] = R6[0, find_nearest(times6, start):find_nearest(times6, stop)]
    binned_data['reward'] = R7[0, find_nearest(
        times7, start):find_nearest(times7, stop)]
    binned_data['trial_number'] = np.digitize(time_points,trials['goCue_times'])
    print('Range of trials: ',[binned_data['trial_number'][0],binned_data['trial_number'][-1]])
    return binned_data

def Isomap_colored(binned_data, trial_range, n_comps, n_neigh, behav_var):
    # trial_range is an array of two numbers defining the range of concatenated trials
    # Find first and last bin index for given trial   
    a = list(binned_data['trial_number']) 
    #first = a.index(trial_number)
    #last  = len(a) - 1 - a[::-1].index(trial_number)
    first = a.index(trial_range[0])
    last  = len(a) - 1 - a[::-1].index(trial_range[1])

    # load neural data and reduce dimensions
    X = binned_data['summed_spike_amps'][:, first:last].T
    Y = Isomap(n_components=n_comps, n_neighbors=n_neigh).fit_transform(X)

    # color it with some other experimental parameter
    x, y, z = np.split(Y, 3, axis=1)
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter(x, y, z, s=40, alpha=0.25, c=abs(
        binned_data[behav_var][first:last]), cmap='bwr')
    fig.colorbar(p)
    #plt.title("Isomap Alex's visual cortex --> hippocampus recording vs wheel speed")
    #plt.title("Isomap Alex's motor cortex --> thalamus recording vs %s" %behav_var)
    plt.title("Isomap Guido's RSC --> CA1 --> midbrain recording vs %s" %behav_var, fontsize=40)
    # plt.scatter(Y.T[0,:],Y.T[1,:],s=1,alpha=0.9,c=D_trial['wheel_velocity'][trial])
    plt.show()

def PCA_colored(binned_data, trial_range, n_comps, behav_var):
    # trial_range is an array of two numbers defining the range of concatenated trials
    # Find first and last bin index for given trial   
    a = list(binned_data['trial_number']) 
    #first = a.index(trial_number)
    #last  = len(a) - 1 - a[::-1].index(trial_number)
    first = a.index(trial_range[0])
    last  = len(a) - 1 - a[::-1].index(trial_range[1])

    # load neural data and reduce dimensions
    X = binned_data['summed_spike_amps'][:, first:last].T
    Y = PCA(n_components=n_comps, svd_solver='full').fit_transform(X)
    # color it with some other experimental parameter
    x, y, z = np.split(Y, 3, axis=1)
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter(x, y, z, s=40, alpha=0.25, c=abs(
        binned_data[behav_var][first:last]), cmap='ocean')
    fig.colorbar(p)
    #plt.title("PCA Alex's visual cortex --> hippocampus recording vs wheel speed")
    #plt.title("PCA Alex's motor cortex --> thalamus recording vs %s" %behav_var)
    plt.title("PCA Guido's RSC --> CA1 --> midbrain recording vs %s" %behav_var, fontsize=40)
    # plt.scatter(Y.T[0,:],Y.T[1,:],s=1,alpha=0.9,c=D_trial['wheel_velocity'][trial])
    plt.show()


def TSNE_colored(binned_data, trial_range, n_comps, perp, behav_var):
    # trial_range is an array of two numbers defining the range of concatenated trials
    # Find first and last bin index for given trial   
    a = list(binned_data['trial_number']) 
    #first = a.index(trial_number)
    #last  = len(a) - 1 - a[::-1].index(trial_number)
    first = a.index(trial_range[0])
    last  = len(a) - 1 - a[::-1].index(trial_range[1])

    # load neural data and reduce dimensions
    X = binned_data['summed_spike_amps'][:, first:last].T
    Y = TSNE(n_components=n_comps, perplexity=perp).fit_transform(X)
    # color it with some other experimental parameter
    x, y, z = np.split(Y, 3, axis=1)
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter(x, y, z, s=40, alpha=0.25, c=abs(
        binned_data[behav_var][first:last]), cmap='jet')
    fig.colorbar(p)
    #plt.title("TSNE Alex's visual cortex --> hippocampus recording vs wheel speed")
    #plt.title("TSNE Alex's motor cortex --> thalamus recording vs %s" %behav_var)
    plt.title("TSNE Guido's RSC --> CA1 --> midbrain recording vs %s" %behav_var, fontsize=40)
    # plt.scatter(Y.T[0,:],Y.T[1,:],s=1,alpha=0.9,c=D_trial['wheel_velocity'][trial])
    plt.show()


def FA_colored(binned_data, trial_range, n_comps, behav_var):
    # trial_range is an array of two numbers defining the range of concatenated trials
    # Find first and last bin index for given trial   
    a = list(binned_data['trial_number']) 
    #first = a.index(trial_number)
    #last  = len(a) - 1 - a[::-1].index(trial_number)
    first = a.index(trial_range[0])
    last  = len(a) - 1 - a[::-1].index(trial_range[1])

    # load neural data and reduce dimensions
    X = binned_data['summed_spike_amps'][:, first:last].T
    Y = FactorAnalysis(n_components=n_comps).fit_transform(X)
    # color it with some other experimental parameter
    x, y, z = np.split(Y, 3, axis=1)
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter(x, y, z, s=40, alpha=0.25, c=abs(
        binned_data[behav_var][first:last]), cmap='hsv')
    fig.colorbar(p)
    #plt.title("FA Alex's visual cortex --> hippocampus recording vs wheel speed")
    #plt.title("FA Alex's motor cortex --> thalamus recording vs %s" %behav_var)
    plt.title("Factor Analysis Guido's RSC --> CA1 --> midbrain recording vs %s" %behav_var, fontsize=40)
    # plt.scatter(Y.T[0,:],Y.T[1,:],s=1,alpha=0.9,c=D_trial['wheel_velocity'][trial])
    plt.show()


def LLE_colored(binned_data, trial_range, n_comps, n_neigh, behav_var):
    # trial_range is an array of two numbers defining the range of concatenated trials
    # Find first and last bin index for given trial   
    a = list(binned_data['trial_number']) 
    #first = a.index(trial_number)
    #last  = len(a) - 1 - a[::-1].index(trial_number)
    first = a.index(trial_range[0])
    last  = len(a) - 1 - a[::-1].index(trial_range[1])

    # load neural data and reduce dimensions
    X = binned_data['summed_spike_amps'][:, first:last].T
    Y = LocallyLinearEmbedding(n_components=n_comps, n_neighbors=n_neigh).fit_transform(X)
    # color it with some other experimental parameter
    x, y, z = np.split(Y, 3, axis=1)
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter(x, y, z, s=40, alpha=0.25, c=abs(
        binned_data[behav_var][first:last]), cmap='jet')
    fig.colorbar(p)
    #plt.title("LLE Alex's visual cortex --> hippocampus recording vs wheel speed")
    #plt.title("LLE Alex's motor cortex --> thalamus recording vs %s" %behav_var)
    plt.title("LLE Guido's RSC --> CA1 --> midbrain recording vs %s" %behav_var, fontsize=40)
    # plt.scatter(Y.T[0,:],Y.T[1,:],s=1,alpha=0.9,c=D_trial['wheel_velocity'][trial])
    plt.show()

def MDS_colored(binned_data, trial_range, n_comps, behav_var):
    # trial_range is an array of two numbers defining the range of concatenated trials
    # Find first and last bin index for given trial   
    a = list(binned_data['trial_number']) 
    #first = a.index(trial_number)
    #last  = len(a) - 1 - a[::-1].index(trial_number)
    first = a.index(trial_range[0])
    last  = len(a) - 1 - a[::-1].index(trial_range[1])

    # load neural data and reduce dimensions
    X = binned_data['summed_spike_amps'][:, first:last].T
    Y = MDS(n_components=n_comps).fit_transform(X)
    # color it with some other experimental parameter
    x, y, z = np.split(Y, 3, axis=1)
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter(x, y, z, s=40, alpha=0.25, c=abs(
        binned_data[behav_var][first:last]), cmap='jet')
    fig.colorbar(p)
    #plt.title("MDS Alex's visual cortex --> hippocampus recording vs wheel speed")
    #plt.title("MDS Alex's motor cortex --> thalamus recording vs %s" %behav_var)
    plt.title("MultiDimensional Scaling Guido's RSC --> CA1 --> midbrain recording vs %s" %behav_var, fontsize=40)
    # plt.scatter(Y.T[0,:],Y.T[1,:],s=1,alpha=0.9,c=D_trial['wheel_velocity'][trial])
    plt.show()


def UMAP_colored(binned_data, trial_range, n_comps, n_neigh, min_distance, behav_var):

    # trial_range is an array of two numbers defining the range of concatenated trials
    # Find first and last bin index for given trial   
    a = list(binned_data['trial_number']) 
    #first = a.index(trial_number)
    #last  = len(a) - 1 - a[::-1].index(trial_number)
    first = a.index(trial_range[0])
    last  = len(a) - 1 - a[::-1].index(trial_range[1])

    # load neural data and reduce dimensions
    X = binned_data['summed_spike_amps'][:, first:last].T
    Y = umap.UMAP(n_components=n_comps, n_neighbors=n_neigh, min_dist=min_distance).fit_transform(X)
    # color it with some other experimental parameter
    x, y, z = np.split(Y, 3, axis=1)
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter(x, y, z, s=40, alpha=0.25, c=abs(
        binned_data[behav_var][first:last]), cmap='ocean')
    fig.colorbar(p)
    #plt.title("UMAP Alex's visual cortex --> hippocampus recording vs wheel speed")
    #plt.title("UMAP Alex's motor cortex --> thalamus recording vs %s" %behav_var)
    plt.title("UMAP Guido's RSC --> CA1 --> midbrain recording vs %s" %behav_var, fontsize=40)
    # plt.scatter(Y.T[0,:],Y.T[1,:],s=1,alpha=0.9,c=D_trial['wheel_velocity'][trial])
    plt.show()

def LDA_colored(binned_data, n_comps, bounds):
    # obs_limit=1000 #else it's too slow
    low, high = bounds
    X = binned_data['summed_spike_amps'][:, low:high].T
    Y = LinearDiscriminantAnalysis(binned_data, 3).fit_transform(X)
    x, y, z = np.split(Y,3, axis=1)
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter(x, y, z, s=20, alpha=0.25, c=abs(binned_data['wheel_velocity'][low:high]), cmap='ocean')
    fig.colorbar(p)
    #plt.title("LDA Alex's visual cortex--> hippocamus recording vs wheel speed")
    plt.title("LDA Alex's motor cortex --> thalamus recording vs wheel speed")
    # plt.scatter(Y.T[0,:],Y.T[1,:],s=1,alpha=0.9,c=D_trial['wheel_velocity'][trial])
    plt.show()



# lets run and plot 
binned_data = bin_types(spikes, trials, wheel, 0.2)
# binned_data.keys()

# behav_var takes the following options: 'choice'/ 'reward'/ 'wheel_velocity'/ 'wheel_position'
Isomap_colored(binned_data, [40, 90], 3, 5, 'choice') # second variable is range of trials, then number of components, then number of neighbors, last is behavioral variable
PCA_colored(binned_data, [40, 90], 3, 'wheel_velocity') # this is a PPCA implementation of PCA
TSNE_colored(binned_data, [40, 90], 3, 30.0, 'reward') # takes a long time # last variable here is perplexity. One should always run TSNE multiple times with perplexity ranging betoween 5 to 50 in order to decide on the best one
FA_colored(binned_data, [40, 90], 3, 'wheel_position')
LLE_colored(binned_data, [40, 90], 3, 30, 'wheel_velocity')
MDS_colored(binned_data, [40, 90], 3, 'choice')
#UMAP_colored(binned_data, [40, 90], 3, 30.0, 0.3, 'choice') # variable one before last is number of neighbors, last is minimum distance
#LDA_colored(binned_data, 3, (1,1000))

