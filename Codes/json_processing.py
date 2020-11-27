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

import numpy as np
import sys
import os
import json
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit, fmin, minimize
from collections import defaultdict
from collections import OrderedDict

from scipy.optimize import curve_fit
from scipy import pi
from scipy.special import erf

arg = sys.argv[1:]

# If an argument is not given, then just search all the files in the working directory 
if arg is None: 
    all_files = os.listdir('./')

    # and filter out the json files among those files
    json_files = list(filter(lambda x: x[-5:] == '.json', all_files))

# Otherwise, simply process those given json files
else:
    json_files = arg[::-1] 

# A complete dictionary is prepared
c_dict = dict()

# Iterating over the list of json files
# and merge them to 'c_dict'
for a_file in json_files:
    with open(a_file, 'r') as f:
        temp = json.load(f)
        c_dict.update(temp)
    f.close()
print(json_files)
# Variables that may be useful

# Ages of the subjects
#age_dist = [c_dict[key]['subj_info']['age'] for key in c_dict.keys()]

# Gender
#gender_dist = [c_dict[key]['subj_info']['gender'] for key in c_dict.keys()]

# Handedness
#hand_dist = [c_dict[key]['subj_info']['right_used'] for key in c_dict.keys()]

#subj_infos = {key:c_dict[key]['subj_info'] for key in c_dict.keys()}

# SUBJ_HS17, SUBJ_hqy, SUBJ_HS25: These are all the things in the past
#del c_dict['SUBJ_HS16']
#del c_dict['SUBJ_HS15']
#del c_dict['SUBJ_HS13']
#del c_dict['SUBJ_HS14']
#del c_dict['SUBJ_HS17']
#del c_dict['SUBJ_hqy']
#del c_dict['SUBJ_HS25']

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
In total there are 53 trials. The catch trial array would be different for Psi_Marginal_7
and Psi_Marginal_10. We skip the catch trials with their types
'''
#psi_trials = [i for i in range(53) if i not in catch_trials]

psi_stim = dict()
psi_obj = dict()
if_corr = dict()
if_right = dict()
vis_stim = dict()
#for subj in c_dict.keys():
#for subj in ['SUBJ_HS821', 'SUBJ_HS82', 'SUBJ_HS822']:
for subj in c_dict.keys():
    n_total = len(c_dict[subj]['subj_trial_info'].keys())
    trial_info = c_dict[subj]['subj_trial_info']
    psi_stim[subj] = [trial_info['_'.join(['TRIAL', str(i)])]['Psi_stimulus(deg)'] for i in range(n_total) if 'Psi_stimulus(deg)' in trial_info['_'.join(['TRIAL', str(i)])]]
    vis_stim[subj] = [trial_info['_'.join(['TRIAL', str(i)])]['Visual_stimulus(deg)'] for i in range(n_total)]
    psi_obj[subj] = [trial_info['_'.join(['TRIAL', str(i)])]['Psi_obj'] if 'Psi_obj' in trial_info['_'.join(['TRIAL', str(i)])] else 'C' for i in range(n_total)]
    if_corr[subj] = [trial_info['_'.join(['TRIAL', str(i)])]['response_correct'] for i in range(n_total)]
    if_right[subj] = [trial_info['_'.join(['TRIAL', str(i)])]['response'] for i in range(n_total)]

# Right = 1, Left = 0
for subj in c_dict.keys():
    if_right[subj] = [1 if if_right[subj][i] == 'right' else 0 for i in range(len(if_right[subj]))]

## Now make a function that would iterate over subjects
## The input should be the subject number + binsize

def bin_and_plot(subj_id, binsize, dp_on = False, prob=False, bias_precision=False):

    # make a dictionary again
    per_subj = defaultdict(list)
    for i, k in enumerate(vis_stim[subj_id]):
        per_subj[k - 50].append(if_right[subj_id][i])

    per_subj_od = OrderedDict(sorted(per_subj.items()))

    # Remove the two very obvious data points
    del per_subj_od[30.0]
    del per_subj_od[-30.0]

    # binned data / bin width should vary... how are you going to address this?  
    output = dict() 
    count = range(int(np.ceil(len(per_subj_od)/binsize)))
    for i in count: 
        temp_key = np.mean(list(per_subj_od.keys())[3*i:3*i+3])
        temp_val = np.mean(np.concatenate(list(per_subj_od.values())[3*i:3*i+3]))
        output[temp_key] = temp_val
    
    # try sigmoid function
    def sigmoid(x, x0, k):
        y = 1/(1+np.exp(-k*(x-x0))) 
        return(y)

    popt, pcov = curve_fit(sigmoid, list(output.keys()), list(output.values()), p0=[5.,0.3])
    
    x_data_new = np.linspace(-30, 30, 200)
    y_data_new = sigmoid(x_data_new, *popt)
    # threshold = stimulus that is the closest to 0.5 probability
    threshold = x_data_new[np.argmin(abs(y_data_new - 0.5))]

    # precision = the difference between 0.25 and 0.75 prob stimulus
    point75 = x_data_new[np.argmin(abs(y_data_new - 0.75))]
    point25 = x_data_new[np.argmin(abs(y_data_new - 0.25))]
    precision = abs(point75 - point25) 

    if bias_precision:
        return((threshold, precision))
        
    else: 
        fig, ax = plt.subplots()
        ax.set_xlim([-10,15])
        ax.set_ylim([-0.02,1.02])
        if dp_on: 
            ax.scatter(list(output.keys()), list(output.values()))
        ax.plot(x_data_new, y_data_new, 'r-', linewidth = 2)

        # bunch of lines...
        ax.vlines(x = threshold, ymin= -0.1, ymax = 0.5, linestyles='dashed', color = 'r') # Threshold position
        ax.vlines(x = 0.0, ymin = -0.1, ymax = 0.5, linestyles='dashed', color = 'gray') # This is the non-biased position
        ax.vlines(x = point75, ymin = -0.1, ymax = 0.75, linestyles = 'dotted', color = 'gray')
        ax.vlines(x = point25, ymin= -0.1, ymax = 0.25, linestyles = 'dotted', color = 'gray')
        ax.hlines(xmin = -30, xmax = threshold, y = 0.5, linestyles = 'dotted', color = 'gray') # threshold line
        ax.hlines(xmin = -30, xmax = point25, y = 0.25, linestyles = 'dotted', color = 'gray') # 0.25 line
        ax.hlines(xmin = -30, xmax = point75, y = 0.75, linestyles = 'dotted', color = 'gray') # 0.75 line

        # arrows, (A) and (B)
        ax.annotate('', xy = (-0.15, 0.4), xytext = (threshold+0.1, 0.4), arrowprops=dict(arrowstyle="<->"))
        ax.annotate('', xy = (2, 0.405), xytext = (9, 0.515), arrowprops=dict(arrowstyle="->"))
        ax.text(11.2, 0.5, 'Threshold', ha = 'center', fontsize = 13)
        ax.annotate('', xy = (point75-0.1,0.10), xytext = (point25+0.1,0.10), arrowprops = dict(arrowstyle="<->"))
        ax.annotate('', xy = (threshold, 0.1), xytext = (9, 0.211), arrowprops=dict(arrowstyle="->"))
        ax.text(11.2, 0.2, 'Precision', ha = 'center', fontsize = 13)

        # x and y label names
        if prob: 
            ylab = 'Probability of rightward responses (%)'
        else:
            ylab = 'Proportion of rightward responses'

        ax.set_ylabel(ylab, fontsize = 13, fontweight='bold') 
        ax.set_xlabel("Stimulus (deg)", fontsize = 13, fontweight='bold')

        # Remove the outer lines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(axis='both', which='major', labelsize = 13, length = 0)
        plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
        if prob:
            ax.set_yticklabels(['0', '25', '50', '75', '100'])
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams["font.family"] = 'Times New Roman'
        plt.savefig(subj_id + '.png', dpi = 300, format = 'png')
        plt.close()

bias_precision_dict = dict()
#for s in list(c_dict.keys()):
#for s in list(['SUBJ_HS821', 'SUBJ_HS82', 'SUBJ_HS822']):
#    out = bin_and_plot(s, 3, bias_precision = True)
#    if out[0] > -2.5:
#        bias_precision_dict[out[0]] = out[1]

#fig, ax = plt.subplots()
##ax.set_ylim([-0.02,4.02])
#ax.scatter(list(bias_precision_dict.keys()), list(bias_precision_dict.values()), s = 150) 
#ax.set_ylabel('Precision (deg)', fontsize = 15, fontweight = 'bold') 
#ax.set_xlabel("Threshold (deg)", fontsize = 15, fontweight = 'bold')
#ax.tick_params(axis = 'both', which='major', labelsize = 13, length = 0)
#ax.spines["top"].set_visible(False)
#ax.spines["right"].set_visible(False)
#plt.rcParams['font.weight'] = 'bold'
#plt.rcParams["font.family"] = 'Times New Roman'
#plt.savefig('bias_precision.pdf', format = 'pdf')
#plt.show()


# plot the individuals
#for s in list(c_dict.keys()):
#    bin_and_plot(s, 3, prob=True)

        
'''
plot_sub_performance is a function to draw to plots
    - Performance per trial
    - Performance per Psi-stimulus(deg)

The input parameter should be the c_dict, or a nested dictionay
The output of this function would be the plots of each subject saved as png format in the
current working directory
'''
def plot_sub_performance(subj_ids, psi_obj, if_corr, psi_stim):

    #subj_ids = [*c_dict]

    for subj_id in subj_ids:

        catch_trials = np.where(np.array(psi_obj[subj_id]) == 'C')[0]
        # psi_obj_ce is the list of psi_objects, catch excluded
        psi_obj_ce = [y for x, y in enumerate(psi_obj[subj_id]) if x not in catch_trials]
        if_corr_ce = [y for x, y in enumerate(if_corr[subj_id]) if x not in catch_trials]
        
        e_scheme = ['r' if x == 'A' else 'b' for x in psi_obj_ce]
        c_scheme = ['r' if (x, y) == ('A', 1) else 'b' if (x, y) == ('B', 1) else 'none' for x, y in zip(psi_obj_ce, if_corr_ce)]


        legend_elements = [Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'r', markeredgecolor = 'r', label = 'A, Correct'), Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'none', markeredgecolor = 'r', label = 'A, Wrong'), Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'b', markeredgecolor = 'b', label = 'B, Correct'), Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'none', markeredgecolor = 'b', label = 'B, Wrong')]
        fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize = (12, 6))

        ax1.scatter(range(1, 51), psi_stim[subj_id], c = c_scheme, edgecolors = e_scheme, label = ['A:1', 'A:0', 'B:1', 'B:0'])

        ax1.legend(handles = legend_elements, loc = 0, ncol = 4)

        ax1.set_xlabel('Trials')
        ax1.set_ylabel('Psi-stimulus(Deg)')
        ax1.set_title(':'.join([subj_id, 'Performance per trial']))


        # Plotting for Psychophysical analysis
        ax2.scatter(psi_stim[subj_id], if_corr_ce, c = c_scheme, edgecolors = e_scheme)
        ax2.set_xlabel('Psi-stimulus(Deg)')
        ax2.set_ylabel('Response')
        ax2.set_title(':'.join([subj_id, 'Performance per stimulus']))
        ax2.legend(handles = legend_elements, loc = 0, ncol = 4)

        fig.tight_layout()

        plt.rc('legend', fontsize = 'small', handlelength = 2)

        plt.savefig('.'.join([subj_id,'png']), dpi = 300, format = 'png')

        plt.close() 

def plot_sub_performance2(subj_ids, psi_obj, if_corr, psi_stim, labelsize = 20, markersize = 150): 

    for subj_id in subj_ids:

        catch_trials = np.where(np.array(psi_obj[subj_id]) == 'C')[0]
        # psi_obj_ce is the list of psi_objects, catch excluded
        psi_obj_ce = [y for x, y in enumerate(psi_obj[subj_id]) if x not in catch_trials]
        if_corr_ce = [y for x, y in enumerate(if_corr[subj_id]) if x not in catch_trials]

        e_scheme = ['limegreen' if x == 'A' else 'b' for x in psi_obj_ce]
        c_scheme = ['limegreen' if (x, y) == ('A', 1) else 'b' if (x, y) == ('B', 1) else 'none' for x, y in zip(psi_obj_ce, if_corr_ce)]

        legend_elements = [Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'limegreen', markeredgecolor = 'limegreen', label = 'A, Correct'), Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'none', markeredgecolor = 'limegreen', label = 'A, Wrong'), Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'b', markeredgecolor = 'b', label = 'B, Correct'), Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'none', markeredgecolor = 'b', label = 'B, Wrong')]

        signed_stim = [(p - 5)*q for p, q in zip(psi_stim[subj_id], [-1 if x == 'A' else 1 for x in psi_obj_ce])]
        fig, ax1 = plt.subplots(figsize = (12, 6))

        ax1.scatter(range(1,51), signed_stim, c = c_scheme, edgecolors = e_scheme, label = ['A:1', 'A:0', 'B:1', 'B:0'], s = markersize)

        #ax1.legend(handles = legend_elements, loc = 0, ncol = 4, markerscale = 2.0)

        ax1.set_xlabel('Trials', fontsize = labelsize, fontweight='bold')
        ax1.tick_params(axis="x", labelsize = 20)
        ax1.set_ylabel('Stimulus(Deg)', fontsize = labelsize, fontweight='bold') 
        ax1.tick_params(axis="y", labelsize = 20)

        plt.rc('legend', fontsize = 15, handlelength = 2)
        plt.rcParams["font.family"] = 'Times New Roman'

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

plot_sub_performance([*c_dict], psi_obj, if_corr, psi_stim)
plot_sub_performance2([*c_dict], psi_obj, if_corr, psi_stim)

subj_id = 'SUBJ_130'
catch_trials = np.where(np.array(psi_obj[subj_id]) == 'C')[0]
psi_obj_ce = [y for x, y in enumerate(psi_obj[subj_id]) if x not in catch_trials]
vis_stim_ce = [y for x, y in enumerate(vis_stim[subj_id]) if x not in catch_trials]
if_corr_ce = [y for _, y in enumerate(if_corr[subj_id]) if x not in catch_trials]
if_corr_ce_inverted = [int(not(y)) for x, y in enumerate(if_corr[subj_id]) if x not in catch_trials]

test = sorted([(x,y) for x, y in zip(vis_stim_ce, if_corr_ce_inverted)])

fig, ax = plt.subplots(1,1, dpi=300)
ax.scatter(*zip(*sorted([(x,y) for x, y in zip(vis_stim_ce, if_corr_ce)])), s=6)
ax.set_title("%s, RAW DATA" % subj_id)
ax.set_xlabel('Visual stimulus (deg)')
ax.set_ylabel('Correctness')
plt.show()


fig, ax = plt.subplots(1,1, dpi=300)
ax.scatter(*zip(*test), s=6)
ax.set_title("%s, RAW DATA" % subj_id)
ax.set_xlabel('Visual stimulus (deg)')
ax.set_ylabel('Correctness')
plt.show()

# Done nothing
stim_fi = defaultdict(list)
for x,y in zip(vis_stim_ce,if_corr_ce_inverted):
    stim_fi[x].append(y)

stim_fi2 = {x:np.mean(y) for x, y in stim_fi.items()}

fig, ax = plt.subplots(1,1, dpi=300)
ax.scatter(*zip(*stim_fi2.items()), s=8)
ax.set_title("%s, CLEANED DATA" % subj_id)
ax.set_xlabel('Visual stimulus (deg)')
ax.set_ylabel('Relative Frequency (f_i)')
plt.show()

denom = sum(stim_fi2.values())
stim_pi = {x:y/denom for x, y in stim_fi2.items()}

μ = sum([x*y for x, y in stim_pi.items()])
μ 
σ = np.sqrt(sum([y*(x-μ)**2 for x, y in stim_pi.items()]))
σ

fig, ax = plt.subplots(1,1,dpi=300)
ax.plot(*zip(*sorted(stim_pi.items())), markersize=3, marker = 'o', mfc='b', mec='b', c='r')
ax.axvline(μ, alpha=0.4)
ax.set_ylabel('density', fontsize=10)
ax.set_xlabel('Visual stimulus (deg)', fontsize = 10)
ax.set_title('%s RAW DATA, non-parametric (μ = %.1f, σ = %.1f)' % (subj_id, μ, σ))
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.tight_layout()
plt.show()

calc_density(stim_pi)
raw_skewed = skewfit(stim_pi)
raw_skewed_by_density = skewfit(stim_pi, byDensity=True)


def calc_skewed_params(a_dict): 

    π = np.pi
    α = a_dict['popt'][2]
    print(α)
    σ = a_dict['popt'][1]
    ξ = a_dict['popt'][0]
    δ = α / (np.sqrt(1+α**2))
    μz = np.sqrt(2/π)*δ
    σz = np.sqrt(1 - μz**2)
    γ1 = (4 - π)/2 * ((δ*np.sqrt(2/π))**3)/(1-2*δ**2/π)**(3/2)
    m0a = μz - γ1*σz/2 - np.sign(α)/2*np.exp(-2*π/abs(α))
    mode = ξ + σ*m0a
    mean = ξ + σ*δ*np.sqrt(2/π)

    return mode, mean

mode0, mean0 = calc_skewed_params(raw_skewed)
mode, mean = calc_skewed_params(raw_skewed_by_density)

newx = np.linspace(20, 80, 200)
markersize = 10
fig, ax = plt.subplots(2,1, dpi=300, sharex=True, sharey=True)
ax[0].plot(*zip(*sorted(stim_pi.items())), markersize=3, marker = 'o', mfc='b', mec='b', c='r')
ax[0].axvline(μ, alpha=0.4)
ax[0].set_title('%s RAW DATA, non-parametric (μ = %.1f, σ = %.1f)' % (subj_id, μ, σ), fontsize=8)
ax[1].scatter(raw_skewed['xdata'], raw_skewed['ydata'], color='blue', s = markersize)
ax[1].plot(newx, skew(newx, *raw_skewed['popt']), color='red')
#ax[1].text(35, 0.2, subj_id)
ax[1].axvline(mode0, alpha= 0.5)
ax[1].axvline(mean0, alpha=0.5, c='g')
ax[1].set_title('%s RAW DATA, skewnorm (mode = %.1f, mean = %.1f)' %  (subj_id, mode0, mean0), fontsize=8)
ax[1].set_ylabel('density')
plt.rc('xtick', labelsize=5)
plt.xlabel('Visual stimulus (deg)', fontsize=6)
plt.ylabel('density', fontsize=6)
plt.tight_layout()
plt.show()


# Forward moving average
n = 3
out = defaultdict(list) 
for i in range(len(test)-n+1):
    temp = test[i:i+n]
    returnval = np.mean([x for x, _ in temp])
    returnfreq = np.mean([y for _, y in temp])
    out[returnval].append(returnfreq)
out_cleaned = {x:np.mean(y) for x, y in out.items()}

# Forward-backward moving average
out2 = defaultdict(list)
floor = int(np.floor(n/2)); ceil = int(np.ceil(n/2))
for i in range(floor, len(test) - ceil):
    temp = test[i-floor:i+floor]
    returnval = np.mean([x for x, _ in temp])
    returnfreq = np.mean([y for _, y in temp])
    out2[returnval].append(returnfreq)
out2_cleaned = {x:np.mean(y) for x, y in out2.items()}

# Try binning
binout = defaultdict(list)
limit = int(np.ceil((len(test) - n )/ n))
for i in range(limit):
    temp = test[(n*i):n+(n*i)]
    returnval = np.mean([x for x, _ in temp])
    returnfreq = np.mean([y for _, y in temp])
    binout[returnval].append(returnfreq)
binout_cleaned = {x:np.mean(y) for x, y in binout.items()}

plt.scatter(*zip(*sorted(binout.items())))

fig, ax = plt.subplots(3,1, sharex=True, sharey=True, dpi=300)
ax[0].scatter(*zip(*sorted(out_cleaned.items())), s=10)
ax[0].set_title('n = %d, MW-forward only' % n)
ax[1].scatter(*zip(*sorted(out2_cleaned.items())), s=10)
ax[1].set_title('n = %d, MW-back and forth' % n)
ax[2].scatter(*zip(*sorted(binout.items())), s=10)
ax[2].set_title('n = %d, binned data' % n)
plt.tight_layout()
plt.show()

def calc_density(dict):
    denom = sum(dict.values())
    pi = {x:y/denom for x, y in dict.items()}

    μ = sum([x*y for x, y in pi.items()])
    σ = np.sqrt(sum([y*(x-μ)**2 for x, y in pi.items()]))

    return {'μ':μ, 'σ':σ, 'pi_dict':pi}

output1 = calc_density(out_cleaned)
output2 = calc_density(out2_cleaned)
output3 = calc_density({x:np.mean(y) for x, y in binout.items()})

def pdf(x):
    return 1/np.sqrt(2*np.pi)*np.exp(-x**2/2)

def cdf(x):
    return (1 + erf(x/np.sqrt(2)))/2

def skew(x, e=0, w=1, a=0):
    t = (x-e)/w
    return (2/w)*pdf(t)*cdf(a*t)

def skewfit(orgdict, byDensity = False):
    outdict = calc_density(orgdict)
    xdata, ydata = zip(*outdict['pi_dict'].items())
    xdata = list(xdata); ydata = list(ydata)
    if byDensity:
        popt, pcov = curve_fit(skew, xdata, ydata, p0 = [outdict['μ'], outdict['σ'], 1])
    else:
        popt, pcov = curve_fit(skew, xdata, list(orgdict.values()), p0 = [outdict['μ'], outdict['σ'], 1])
    return {'popt':popt, 'pcov':pcov, 'xdata':xdata, 'ydata':ydata}

skew1 = skewfit(out_cleaned)
skew2 = skewfit(out2_cleaned)

mode2, mean2 = calc_skewed_params(skew1)
mode3, mean3 = calc_skewed_params(skew2)

# Let's compare different 'binning' methods
newx = np.linspace(20, 80, 200)
markersize = 10
fig, ax = plt.subplots(3,1, sharex=True, sharey=True, dpi=300)
ax[0].scatter(skew1['xdata'], skew1['ydata'], color='blue', s = markersize)
ax[0].plot(newx, skew(newx, *skew1['popt']), color='red')
ax[0].text(35, 0.2, subj_id)
ax[0].axvline(mode2, alpha= 0.5)
ax[0].set_title('Moving Avg. (n = %d, F) - skewnorm (mode = %.1f)' % (n, mode2))

ax[1].scatter(skew2['xdata'], skew2['ydata'], color='blue', s = markersize)
ax[1].plot(newx, skew(newx, *skew2['popt']), color='red')
ax[1].axvline(mode3, alpha = 0.5)
ax[1].set_title('Moving Avg. (n = %d, BF) - skewnorm (mode = %.1f)' % (n, mode3))

ax[2].plot(*zip(*output3['pi_dict'].items()), marker='o', color='red', mec='blue', mfc='blue', markersize = 3)
ax[2].axvline(output3['μ'], alpha=0.5)
ax[2].set_title('n = %d, nonparametric (μ = %.1f, σ = %.1f)' % (n, output3['μ'], output3['σ']))

plt.tight_layout()
plt.show()

# Let's compare parametric vs non-parametric: Moving Avg, Forward only
fig, ax = plt.subplots(3,1, sharex=True, sharey=True, dpi=300)
ax[0].plot(*zip(*sorted(stim_pi.items())), markersize=3, marker = 'o', mfc='b', mec='b', c='r')
ax[0].axvline(μ, alpha=0.4)
ax[0].set_ylabel('density')
ax[0].set_title('%s RAW DATA, non-parametric (μ = %.1f, σ = %.1f)' % (subj_id, μ, σ))

ax[1].scatter(skew1['xdata'], skew1['ydata'], color='blue', s=markersize)
ax[1].plot(newx, skew(newx, *skew1['popt']), color='red')
ax[1].axvline(mode2, alpha= 0.5)
ax[1].axvline(mean2, alpha=0.5, c='g')
ax[1].set_title('Moving Avg. (n = %d, F) - skewnorm (mode = %.1f, mean = %.1f)' % (n, mode2, mean2))

ax[2].plot(skew1['xdata'], skew1['ydata'], color='red', markersize = 3, marker = 'o', mfc='blue', mec='blue')
ax[2].axvline(output1['μ'], alpha=0.5)
ax[2].set_title('Moving Avg. (n = %d, F) - nonparametric (μ = %.1f, σ = %.1f)' % (n, output1['μ'], output1['σ']))

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, dpi=300)
ax[0].plot(*zip(*sorted(stim_pi.items())), markersize=3, marker = 'o', mfc='b', mec='b', c='r')
ax[0].axvline(μ, alpha=0.4)
ax[0].set_ylabel('density')
ax[0].set_title('%s RAW DATA, non-parametric (μ = %.1f, σ = %.1f)' % (subj_id, μ, σ))

ax[1].scatter(skew2['xdata'], skew2['ydata'], color='blue', s=markersize)
ax[1].plot(newx, skew(newx, *skew2['popt']), color='red')
ax[1].axvline(mode3, alpha= 0.5)
ax[1].axvline(mean3, alpha= 0.5, c='g')
ax[1].set_title('Moving Avg. (n = %d, FB) - skewnorm (mode = %.1f, mean = %.1f)' % (n, mode3, mean3))

ax[2].plot(skew2['xdata'], skew2['ydata'], color='red', markersize = 3, marker = 'o', mfc='blue', mec='blue')
ax[2].axvline(output2['μ'], alpha=0.5)
ax[2].set_title('Moving Avg. (n = %d, FB) - nonparametric (μ = %.1f, σ = %.1f)' % (n, output2['μ'], output2['σ']))

plt.tight_layout()
plt.show()

output['μ']
output['σ']

for x in xdata:
    print(skew(x, 51.3, 1.75, 1))


stim_fi = defaultdict(list)
for x,y in zip(vis_stim_ce,if_corr_ce):
    stim_fi[x].append(y)

angleorder = defaultdict(list)
for x, y in zip(vis_stim_ce, psi_obj_ce):
    angleorder[x].append(y)

stim_fi2 = {x:np.mean(y) for x, y in stim_fi.items()}

denom = sum(stim_fi2.values())
stim_pi = {x:y/denom for x, y in stim_fi2.items()}

μ = sum([x*y for x, y in stim_pi.items()])
μ 
σ = np.sqrt(sum([y*(x-μ)**2 for x, y in stim_pi.items()]))
σ

c_scheme = ['limegreen' if (x, y) == ('A', 1) else 'b' if (x, y) == ('B', 1) else 'none' for x, y in zip(psi_obj_ce, if_corr_ce)]
fig, ax = plt.subplots(1,1)
ax.scatter(*zip(*sorted(pi.items())), marker='o')
ax.axvline(μ, color='red')

# Optimization algorithm

# Gumbel.py has two functions
#
#   1) unpackparams(*params): Given the parameter inputs, returns α, β, γ, λ
#   2) Gumbel(*params, x, Type = None):
#       - params: the parameters (at least α,β; possibly γ,λ)
#       - x: a value whose probability density is seeked
#       - Type: decides what probability density is calculated; 
#               'Inverse', 'Derivative', 'None'(default) are available
#       'Gumbel' returns the Gumbel probability density of the given input, x
from Gumbel import *
from scipy.stats import skewnorm

# loss function
def loss_func(params, data_stimuli, data_response, Type=None):
    # Negative Log-likelihood function for directional tuning curves??

    return -np.log(np.prod(skewnorm.pdf(data_stimuli, a=params[0], loc=params[1], scale=params[2])))


    # log likelihood of poisson: logλ*(ΣX_i) - nλ - Σlog(X_i!)
    # I assume predictedF == λ
    #logP = sum(data_response) * np.log(predictedF) - len(data_response)*predictedF - np.log(math.factorial(data_response))

    #nLL = -sum([x*y for x,y in zip(data_response,np.log(predictedF))]) - sum([(1-x)*y for x,y in zip(data_response, np.log(1-predictedF))])

    #return nLL

out = minimize(loss_func, x0 = [1, 5, 5], args = (psi_stim['SUBJ_HS891'], if_corr['SUBJ_HS891']))

def skewed_norm(data, loc, scale, alpha):
    return [2*norm.pdf(x, loc=loc, scale=scale)*norm.cdf(alpha*x, loc=loc, scale=scale) for x in data]
    
data_stimuli = sorted(vis_stim['SUBJ_HS891'])
data_response = if_corr['SUBJ_HS891']

skewed = skewnorm.pdf(data_stimuli, a=1, loc=50, scale=5)
plt.plot(data_stimuli, skewed)
plt.show()

popt, pcov = curve_fit(skewed_norm, xdata=data_stimuli, ydata=[1-x for x in data_response])

plt.scatter(data_stimuli, [1-x for x in data_response])
plt.plot(np.arange(30,80,0.1), skewnorm.pdf(np.arange(30,80,0.1),1, 1,1))
plt.show()

params = [50,5,1]
test = [2*norm.pdf(x, loc=params[0], scale=params[1])*norm.cdf(params[2]*x, loc=params[0], scale=params[1]) for x in data_stimuli]
