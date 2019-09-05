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
sid = 'A'
ses_path = Path(SES[sid])
# read in the alf objects
alf_path = ses_path / 'alf'
spikes = alf.io.load_object(alf_path, 'spikes')  #can be addressed as spikes['time'] or spikes.time
clusters = alf.io.load_object(alf_path, 'clusters')
channels = alf.io.load_object(alf_path, 'channels')
trials = alf.io.load_object(alf_path, '_ibl_trials')
wheel = alf.io.load_object(alf_path, '_ibl_wheel')


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# get the data from flatiron and the current folder
#one = ONE()
#eid = one.search(subject='ZM_1736', date='2019-08-09', number=4)
#D = one.load(eid[0], clobber=False, download_only=True)
#session_path = Path(D.local_path[0]).parent
#session_path = Path(
#    '/home/mic/Downloads/FlatIron/mainenlab/ZM_1735_2019-08-01_001/mnt/s0/Data/Subjects/ZM_1735/2019-08-01/001/alf')

## load objects
#spikes = ioalf.load_object(session_path, 'spikes')
#trials = ioalf.load_object(session_path, '_ibl_trials')
#wheel = ioalf.load_object(session_path, '_ibl_wheel')


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


def bin_types(spikes, trials, wheel):
    T_BIN = 0.01  # [sec]

    # TO GET MEAN: bincount2D(..., weight=positions) / bincount2D(..., weight=None)

    reward_times = trials['feedback_times']
    trial_start_times = trials['intervals'][:, 0]
    # trial_end_times = trials['intervals'][:, 1] #not working as there are
    # nans
    # compute raster map as a function of cluster number
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

    #Add chouce
    R6, times6, _ = bincount2D(trials['goCue_times'],trials['choice'], T_BIN)
    # Flatten choice -1 Left, 1  Right
    R6 = np.sum(R6*np.array([[-1], [1]]),axis=0)
    R6 = np.expand_dims(R6, axis=0)
    # Fill 0 between trials with choice outcome of trial
    R6 =  filliti(R6)
    #Add reward
    R7, times7, _ = bincount2D(trials['goCue_times'],trials['feedbackType'], T_BIN)
    # Flatten reward -1 error, 1  reward
    R7 = np.sum(R7*np.array([[-1], [1]]),axis=0)
    R7 = np.expand_dims(R7, axis=0)
    # Fill 0 between trials with reward outcome of trial
    R7 =  filliti(R7)

    start = max([x for x in [times1[0], times3[0], times4[0], times5[0],times6[0], times7[0]]])
    stop = min([x for x in [times1[-1], times2[-1],
                            times3[-1], times4[-1], times5[-1], times6[-1], times7[-1]]])
    time_points = np.linspace(start, stop, int((stop - start) / T_BIN))
    binned_data = {}
    binned_data['wheel_position'] = np.interp(
        time_points, wheel['times'], wheel['position'])
    binned_data['wheel_velocity'] = np.interp(
        time_points, wheel['times'], wheel['velocity'])
    binned_data['summed_spike_amps'] = R1[:, find_nearest(
        times1, start):find_nearest(times1, stop)]
    binned_data['reward_event'] = R2[0, find_nearest(
        times2, start):find_nearest(times2, stop)]
    binned_data['trial_start_event'] = R3[0, find_nearest(
        times3, start):find_nearest(times3, stop)]
    binned_data['choice'] = R6[0, find_nearest(
        times6, start):find_nearest(times6, stop)]
    binned_data['reward'] = R7[0, find_nearest(
        times7, start):find_nearest(times7, stop)]

    # get trial number for each time bin   
    binned_data['trial_number'] = np.digitize(time_points,trials['goCue_times'])
    print('Range of trials: ',[binned_data['trial_number'][0],binned_data['trial_number'][-1]])

    return binned_data 


def color_attractor_bounds(binned_data, bounds):

    first, last = bounds

    # load neural dataand reduce dimensions
    X = binned_data['summed_spike_amps'][:, first:last].T
    Y = Isomap(n_components=3).fit_transform(X)

    # color it with some other experimental parameter
    x, y, z = np.split(Y, 3, axis=1)
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter(x, y, z, s=20, alpha=0.25, c=
        binned_data['choice'][first:last], cmap='bwr')
    fig.colorbar(p)
    plt.title("Guido's motor cortex --> thalamus recording vs wheel speed")
    # plt.scatter(Y.T[0,:],Y.T[1,:],s=1,alpha=0.9,c=D_trial['wheel_velocity'][trial])
    plt.show()



def color_attractor(binned_data, trial_number):

    # Find first and last bin index for given trial   
    a = list(binned_data['trial_number']) 
    first = a.index(trial_number)
    last  = len(a) - 1 - a[::-1].index(trial_number)
    print(last-first)


    # load neural dataand reduce dimensions
    X = binned_data['summed_spike_amps'][:, first:last].T
    Y = Isomap(n_components=3).fit_transform(X)

    # color it with some other experimental parameter
    x, y, z = np.split(Y, 3, axis=1)
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter(x, y, z, s=20, alpha=0.25, c=
        binned_data['choice'][first:last], cmap='binary')
    fig.colorbar(p)
    plt.title("Guido's motor cortex --> thalamus recording vs wheel speed")
    # plt.scatter(Y.T[0,:],Y.T[1,:],s=1,alpha=0.9,c=D_trial['wheel_velocity'][trial])
    plt.show()




# for ZM_1735 use left probe only

