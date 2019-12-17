'''
json_processing.py
Written by JINSEOK OH

[Objective]
json_processing.py will read all the json data files in a folder
and merge them in preparation of future analyses

[Data structure]
Each .json file would be a nested dictionary
Keys of a dictionary would be the subject ID's (e.g. SUBJ_001)
The value to a key of the dictionary would also be a dictionary
with three keys
    - 'subj_info'
    - 'subj_anth'
    - 'subj_trial_info'

Each key would again have a dictionary as a corresponding value
    1) subj_info
        - age
        - gender: M or F
        - right_used: True or False
        - Staircase used: Psi-Marginal or Adaptive-Staircase
    2) subj_anth
        - flen: finger length
        - fwid: finger width
        - init_step: Initial step size; N/A for Psi-Marginal Staircase
        - MPJR: Metacarpal Joint Radius
    3) subj_trial_info
        - TRIAL_XX (XX ranges from 0 to 45)
            - trial_num
            - Psi_obj: A or B
            - Psi-stimulus(deg): A deviation from the 'false' reference;
                                 This could be understood as a stimulus value
            - Visual_stimulus(deg): The acute angle between the slant line
                                    and the bottom line of the color screen
            - correct_ans: right or left
            - response
            - response_correct: 0 (wrong) or 1 (correct)
'''

import os
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Run this script in the folder where the json files are located
all_files = os.listdir('./')

# filter out the json files
json_files = list(filter(lambda x: x[-5:] == '.json', all_files))

# A complete dictionary is prepared
c_dict = dict()

# Iterating over the list of json files
# and merge them to 'c_dict'
for a_file in json_files:
    with open(a_file, 'r') as f:
        temp = json.load(f)
        c_dict.update(temp)
    f.close()

# Variables that may be useful

# Subject ID's
subj_ids = [*c_dict]

# Ages of the subjects
age_dist = [c_dict[key]['subj_info']['age'] for key in c_dict.keys()]

# Gender
gender_dist = [c_dict[key]['subj_info']['gender'] for key in c_dict.keys()]

# Handedness
hand_dist = [c_dict[key]['subj_info']['right_used'] for key in c_dict.keys()]

subj_infos = {key:c_dict[key]['subj_info'] for key in c_dict.keys()}

'''
A set of three variables
    - Psi-stimulus for each trial(psi_stim)
    - Psi-object(psi_obj)
    - The correctness of the response for that stimulus(if_corr)

Data structure for the two variables: dictionary
    1) psi_stim: key = SUBJ_XX, value = the list of psi_stimuli for the total of 50 trials
    2) psi_obj: key = SUBJ_XX, value = the list of psi_objs used for the total of 50 trials
    3) if_corr: key = SUBJ_XX, value = the list of right or wrong for the total of 50 trials

There are now 3 catch trials: 12, 28, 44 for short(Total 53 trials)
'''
catch_trials = [12,28,44]
psi_trials = [i for i in range(53) if i not in catch_trials]

psi_stim = dict()
psi_obj = dict()
if_corr = dict()
for subj in c_dict.keys():
    psi_stim[subj] = [c_dict[subj]['subj_trial_info']['_'.join(['TRIAL', str(i)])]['Psi_stimulus(deg)'] for i in psi_trials]
    psi_obj[subj] = [c_dict[subj]['subj_trial_info']['_'.join(['TRIAL', str(i)])]['Psi_obj'] for i in psi_trials]
    if_corr[subj] = [c_dict[subj]['subj_trial_info']['_'.join(['TRIAL', str(i)])]['response_correct'] for i in range(53)]


'''
plot_sub_performance is a function to draw to plots
    - Performance per trial
    - Performance per Psi-stimulus(deg)

The input parameter should be the c_dict, or a nested dictionay
The output of this function would be the plots of each subject saved as png format in the
current working directory
'''
def plot_sub_performance(subj_ids, psi_stim, psi_obj, if_corr, catch_trials, psi_trials):

    for subj_id in subj_ids:

        e_scheme = ['r' if x == 'A' else 'b' for x in psi_obj[subj_id]]
        c_scheme = ['r' if (x, y) == ('A', 1) else 'b' if (x, y) == ('B', 1) else 'none' for x, y in zip(psi_obj[subj_id], np.delete(if_corr[subj_id], catch_trials))]

        legend_elements = [Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'r', markeredgecolor = 'r', label = 'A, Correct'), Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'none', markeredgecolor = 'r', label = 'A, Wrong'), Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'b', markeredgecolor = 'b', label = 'B, Correct'), Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'none', markeredgecolor = 'b', label = 'B, Wrong')]
        fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize = (12, 6))

        ax1.scatter(psi_trials, psi_stim[subj_id], c = c_scheme, edgecolors = e_scheme, label = ['A:1', 'A:0', 'B:1', 'B:0'])
        ax1.axhline(y=5)
        linesty = ['solid' if if_corr[subj_id][i] is 1 else 'dashed' for i in catch_trials]
        ax1.vlines(catch_trials, ymin=0, ymax=35, colors='g', linestyles=linesty)

        ax1.legend(handles = legend_elements, loc = 0, ncol = 4)

        ax1.set_xlabel('Trials')
        ax1.set_ylabel('Psi-stimulus(Deg)')
        ax1.set_title(':'.join([subj_id, 'Performance per trial / catch_hit rate', str(np.mean([if_corr[subj_id][k] for k in catch_trials]))]))


        # Plotting for Psychophysical analysis
        ax2.scatter(psi_stim[subj_id], np.delete(if_corr[subj_id],catch_trials), c = c_scheme, edgecolors = e_scheme)
        ax2.set_xlabel('Psi-stimulus(Deg)')
        ax2.set_ylabel('Response')
        ax2.set_title(':'.join([subj_id, 'Performance per stimulus']))
        ax2.legend(handles = legend_elements, loc = 0, ncol = 4)

        fig.tight_layout()

        plt.rc('legend', fontsize = 'small', handlelength = 2)

        plt.savefig('.'.join([subj_id,'png']), dpi = 300, format = 'png')

        plt.close() 

plot_sub_performance(subj_ids, psi_stim, psi_obj, if_corr, catch_trials, psi_trials)

def plot_sub_performance2(c_dict, labelsize = 20, markersize = 15):
    subj_ids = [*c_dict]

    for subj_id in subj_ids:

        e_scheme = ['r' if x == 'A' else 'b' for x in psi_obj[subj_id]]
        c_scheme = ['r' if (x, y) == ('A', 1) else 'b' if (x, y) == ('B', 1) else 'none' for x, y in zip(psi_obj[subj_id], if_corr[subj_id])]

        legend_elements = [Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'r', markeredgecolor = 'r', label = 'A, Correct'), Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'none', markeredgecolor = 'r', label = 'A, Wrong'), Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'b', markeredgecolor = 'b', label = 'B, Correct'), Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'none', markeredgecolor = 'b', label = 'B, Wrong')]

        signed_stim = [(p - 5)*q for p, q in zip(psi_stim[subj_id], [1 if x == 'A' else -1 for x in psi_obj[subj_id]])]
        fig, ax1 = plt.subplots(figsize = (12, 6))

        ax1.scatter(range(1, 47), signed_stim, c = c_scheme, edgecolors = e_scheme, label = ['A:1', 'A:0', 'B:1', 'B:0'], s = markersize)

        ax1.legend(handles = legend_elements, loc = 0, ncol = 4)

        ax1.set_xlabel('Trials', fontsize = labelsize)
        ax1.tick_params(axis="x", labelsize = 20)
        ax1.set_ylabel('Psi-stimulus(Deg)', fontsize = labelsize) 
        ax1.tick_params(axis="y", labelsize = 20)

        plt.rc('legend', fontsize = 15, handlelength = 2)

        fig.tight_layout()

        plt.savefig(''.join([subj_id,'_signed.png']), dpi = 300, format = 'png')

        plt.close() 
'''
This is just for testing purpose

plot_sub_performance2(c_dict)

from collections import defaultdict

probs = defaultdict(list)
subjs = ['_'.join(['SUBJ', str(x)]) for x in [121, 125, 144, 141, 147, 149, 161]]
new_dict = defaultdict(list)

for sbj in subjs:
    new_dict[sbj].append(c_dict[sbj])

plot_sub_performance2(new_dict, 25, 100)

for subj in c_dict.keys():
    for i in range(len(psi_stim[subj])):
        if (psi_obj[subj][i] == 'A'):
            probs[-psi_stim[subj][i]].append(if_corr[subj][i])
        else:
            probs[psi_stim[subj][i]].append(if_corr[subj][i])

real_probs = defaultdict(list)
for key in probs.keys():
    real_probs[key] = sum(probs[key])/len(probs[key])
    
for key, val in real_probs.items():
    if ((val >= 0.5) and (key < 15.1)):
        plt.scatter(key, val)
'''
