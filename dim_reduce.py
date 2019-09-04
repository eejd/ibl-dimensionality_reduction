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

def bin_types(spikes, trials, wheel):

    T_BIN = 0.01  # [sec]

    # TO GET MEAN: bincount2D(..., weight=positions) / bincount2D(..., weight=None)

    reward_times = trials['feedback_times'][trials['feedbackType'] == 1]
    trial_start_times = trials['intervals'][:, 0]
    # trial_end_times = trials['intervals'][:, 1] #not working as there are
    # nans

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

    # R6, times6, _ = bincount2D(trial_end_times, np.array([0]*len(trial_end_times)), T_BIN)
    start = max([x for x in [times1[0], times2[0], times3[0], times4[0], times5[0]]])
    stop = min([x for x in [times1[-1], times2[-1],
                            times3[-1], times4[-1], times5[-1]]])

    time_points = np.linspace(start, stop, int((stop - start) / T_BIN))
    
    # either find nearest or interpolate in order to bin 
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

    # get trial number for each time bin   
    binned_data['trial_number'] = np.digitize(time_points,trials['goCue_times'])
    print('Range of trials: ',[binned_data['trial_number'][0],binned_data['trial_number'][-1]])
    return binned_data

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def color_attractor(binned_data, projection, trial_number):

    # Find first and last bin index for given trial   
    a = list(binned_data['trial_number']) 
    first = a.index(trial_number)
    last  = len(a) - 1 - a[::-1].index(1)

    # load neural dataand reduce dimensions
    X = binned_data['summed_spike_amps'][:, first:last].T
    Y = projection

    # color it with some other experimental parameter
    x, y, z = np.split(Y, 3, axis=1)
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter(x, y, z, s=20, alpha=0.25, c=abs(
        binned_data['wheel_velocity'][first:last]), cmap='binary')
    fig.colorbar(p)
    plt.title("Guido's motor cortex --> thalamus recording vs wheel speed")
    # plt.scatter(Y.T[0,:],Y.T[1,:],s=1,alpha=0.9,c=D_trial['wheel_velocity'][trial])
    plt.show()
