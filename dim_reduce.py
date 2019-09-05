#
# functions for dimensionality reduction
# Authors: dim_reduction hackathon workgroup
# IBL Code Camp 2019
#

# imports from core ibllib conda environment
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# imports from ibllib
from brainbox.processing import bincount2D

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


def get_trial(binned_data, variable, trial_number):
    # Find first and last bin index for given trial
    a = list(binned_data['trial_number'])
    first = a.index(trial_number)
    last  = len(a) - 1 - a[::-1].index(1)
    neural_data = binned_data['summed_spike_amps'][:, first:last].T
    variable_data = binned_data[variable][first:last]


def color_3D_projection(projection, variable):
    # color it with some other experimental parameter
    x, y, z = np.split(projection, 3, axis=1)
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter(x, y, z, s=20, alpha=0.25, c=abs(variable), cmap='binary')
    fig.colorbar(p)
    return p
