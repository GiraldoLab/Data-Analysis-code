from turtle import delay
import matplotlib.pyplot as plt
import csv
import pandas as pd
from dateutil import parser
import numpy as np
import scipy as sp
import os
import glob
from circstats import difference, wrapdiff
 

# Calculates mean heading angle of fly
def full_arctan(x,y):
    angle = np.arctan(y/x)+np.pi if x<0 else np.arctan(y/x)
    return angle if angle <= np.pi else angle -2*np.pi  

def angle_avg(data):
    return full_arctan(np.cos(data*np.pi/180).sum(),np.sin(data*np.pi/180).sum())*180/np.pi  


#Calculates variance 
def angle_var(data):                                #Returns same value as 'circvar' in flight_arena_circular_analysis.py
    return 1-np.sqrt(np.sin(data*np.pi/180).sum()**2 + np.cos(data*np.pi/180).sum()**2)/len(data)

def angle_std(variance):
    return np.sqrt(-2*np.log(1-variance))
def yamartino_angle_std(variance):
    e = np.sqrt(1-(1-variance)**2)
    return np.arcsin(e)*(1+(2/np.sqrt(3)-1)*e**3)

def circdiff(alpha, beta):
    D = np.arctan2(np.sin(alpha*np.pi/180-beta*np.pi/180),np.cos(alpha*np.pi/180-beta*np.pi/180))
    return D

def sun_angle(data):
    if data == 19:
        return 135
    elif data == 57:
        return 45
    elif data == 93:
        return -45
    elif data == 129:
        return -135
    else:
        return 0

allFliesMeanAnglesListA = list()
allFliesRotatedMeanAnglesListA = list()
allFliesAngleVarsListA = list()
allFliesSunListA = list()

allFliesMeanAnglesListB = list()
allFliesRotatedMeanAnglesListB = list()
allFliesAngleVarsListB = list()
allFliesSunListB = list()


path = '/home/giraldolab/catkin_ws/src/magno-test/nodes/data/For Data Analysis/HCS-5min relax in the arena'
#Sorts all the csv files in order in 'path' and sorts it to A trials and B trials
experiment_names = glob.glob(os.path.join(path, "*.csv"))
experiment_names.sort(reverse=False)

experiment_namesA = [f for f in experiment_names if f[-5:]=='A.csv']
experiment_namesA.sort(reverse=False)
#Sorts all fly A trials in numerical order - file names needs to start from fly01,fly02,.. when naming file
experiment_namesB = [f for f in experiment_names if f[-5:]=='B.csv']
experiment_namesB.sort(reverse=False)
#print(experiment_names)



experiments =[]
output= {}
time_delay = 0

for experiment_name in experiment_names:
    experiment = pd.read_csv(experiment_name)
    output[str(experiment_name)] = {}
    experiment['Image Time'] = experiment['Image Time'].apply(parser.parse)
    experiment_data = experiment.values
    sun_change_indexes = [0]+[i for i in range(1,len(experiment_data)) if experiment_data[i,3]!=experiment_data[i-1,3]]
    sun_periods = [experiment[sun_change_indexes[-1]:-1]]
    for sun_period in sun_periods:
        delayed_sun_period = sun_period.loc[[(frame_time - sun_period['Image Time'].iloc[0]).seconds>time_delay for frame_time in sun_period['Image Time']]]
        sun_position = sun_angle(delayed_sun_period['Sun Position'].iloc[0]) #to show where sun stimulus was during the experiment
        #print('sun_position:', sun_position)
        heading_angle = angle_avg(delayed_sun_period['Heading Angle']) #original: pos_angles(delayed_sun_period['Heading Angle'])

        heading_angle_var = angle_var(delayed_sun_period['Heading Angle'])#original: pos_angles(delayed_sun_period['Heading Angle'])

        heading_angle_std = angle_std(heading_angle_var)

        heading_angle_yamartino_std = yamartino_angle_std(heading_angle_var)

        rotated_sun_position = sun_position - sun_position

        output[str(experiment_name)][str(sun_period["Sun Position"].iloc[0])]= {
            "heading angle": heading_angle,
            "heading angle var": heading_angle_var,
            "heading angle std": heading_angle_std,
            "heading angle yamartino_std": heading_angle_yamartino_std,
        } 
        if experiment_name in experiment_namesA:     
            allFliesMeanAnglesListA.append(heading_angle)
            allFliesAngleVarsListA.append(heading_angle_var)
            allFliesSunListA.append(sun_position)

        elif experiment_name in experiment_namesB:
            allFliesMeanAnglesListB.append(heading_angle)
            allFliesAngleVarsListB.append(heading_angle_var)
            allFliesSunListB.append(sun_position)

allFliesMeanAnglesA = np.array(allFliesMeanAnglesListA)
allFliesAngleVarsA = np.array(allFliesAngleVarsListA)         
allFliesSunA = np.array(allFliesSunListA)


allFliesMeanAnglesB = np.array(allFliesMeanAnglesListB)
allFliesAngleVarsB = np.array(allFliesAngleVarsListB)
allFliesSunB = np.array(allFliesSunListB)

##apply cutt off filter of vecstrength<0.2 ###
allFliesAngleVarsACutOff = np.zeros (1)
allFliesAngleVarsBCutOff = np.zeros (1)
allFliesMeanAnglesACutOff = np.zeros (1)
allFliesMeanAnglesBCutOff = np.zeros (1)
allFliesSunACutOff = np.zeros (1)
allFliesSunBCutOff = np.zeros (1)  ##added 10/20 HP JL

for x in range (allFliesAngleVarsA.size):
    if allFliesAngleVarsA[x] > 0.8 or allFliesAngleVarsB[x] > 0.8:                                      ####MAX_VAR=0.8
        pass
    else:
        allFliesAngleVarsACutOff = np.append(allFliesAngleVarsACutOff, allFliesAngleVarsA[x])
        allFliesMeanAnglesACutOff = np.append(allFliesMeanAnglesACutOff, allFliesMeanAnglesA[x])
        allFliesSunACutOff = np.append(allFliesSunACutOff, allFliesSunA[x])

        allFliesAngleVarsBCutOff = np.append(allFliesAngleVarsBCutOff, allFliesAngleVarsB[x])
        allFliesMeanAnglesBCutOff = np.append(allFliesMeanAnglesBCutOff, allFliesMeanAnglesB[x])
        allFliesSunBCutOff = np.append(allFliesSunBCutOff, allFliesSunB[x])

allFliesAngleVarsACutOff = allFliesAngleVarsACutOff[1:]
allFliesMeanAnglesACutOff = allFliesMeanAnglesACutOff[1:]
print('allFliesMeanAnglesACutOff:', allFliesMeanAnglesACutOff)
allFliesSunACutOff = allFliesSunACutOff[1:]

allFliesAngleVarsBCutOff = allFliesAngleVarsBCutOff[1:]
allFliesMeanAnglesBCutOff = allFliesMeanAnglesBCutOff[1:]
print('allFliesMeanAnglesBCutOff:', allFliesMeanAnglesBCutOff)
allFliesSunBCutOff = allFliesSunBCutOff[1:]


print('Atrialvecstrengthsavg:',1-sum(allFliesAngleVarsACutOff)/len(allFliesAngleVarsACutOff))
print('Btrialvecstrengthsavg:',1-sum(allFliesAngleVarsBCutOff)/len(allFliesAngleVarsBCutOff))
print('allFliesAngleVarsListA:',allFliesAngleVarsListA)
print('A_Vars_cutoff', allFliesAngleVarsACutOff)
print('allFliesAngleVarsListB:',allFliesAngleVarsListB)
print('B_Vars_cutoff', allFliesAngleVarsBCutOff)


rotated_heading_anglesA = difference(allFliesMeanAnglesACutOff, allFliesSunACutOff, deg= True)

rotated_heading_anglesB = difference(allFliesMeanAnglesBCutOff, allFliesSunBCutOff, deg= True)

heading_difference= difference(rotated_heading_anglesA ,rotated_heading_anglesB, deg=True)


############## For polar plot ###############
fig, (ax1,ax2,ax3) = plt.subplots(1,3, subplot_kw = {'projection': 'polar'}, sharex = True, sharey=True)
r=1

for i in range(len(rotated_heading_anglesA)):
    ax1.plot((0,rotated_heading_anglesA[i]*np.pi/180.0),(0,1.0-allFliesAngleVarsACutOff[i]),'k', linewidth=2)
    ax1.set_title('A Trials Heading')
    ax2.plot((0,rotated_heading_anglesB[i]*np.pi/180.0),(0,1.0-allFliesAngleVarsBCutOff[i]),'k', linewidth=2)
    ax2.set_title('B Trials Heading')
    ax3.plot((0,heading_difference[i]*np.pi/180.0),(0,1.0),'k', linewidth=2)
    ax3.set_title('Heading Change')

for ax in (ax1,ax2,ax3):
    ax.set_theta_zero_location ("N")
    ax.set_theta_direction(-1)
    ax.set_rlim((0, 1.0))
    ax.spines['polar'].set_visible(True)
    ax.grid(True)
    circle = plt.Circle((0.0, 0.0), 1., transform=ax.transData._b, edgecolor='k', linewidth=2, facecolor= 'w', zorder=0)
    ax.add_artist(circle)

    #ax.axis('off')




############## For linear plot ###############
fig = plt.figure(figsize=(5,10))
fig.set_facecolor('w')
ax = fig.add_subplot(1,1,1)
for axis in ['left','bottom']:
            ax.spines[axis].set_linewidth(2.0)
ax.xaxis.set_tick_params(width=2.0, length=8.0, direction = 'in')
ax.yaxis.set_tick_params(width=2.0, length=8.0, direction = 'in') 
ax.scatter(rotated_heading_anglesA,rotated_heading_anglesB, s=35, color='k', zorder=10)
ax.scatter(rotated_heading_anglesA,rotated_heading_anglesB +360.0, s=35, color='k', zorder=10)
ax.errorbar(x=rotated_heading_anglesA, y=rotated_heading_anglesB, xerr=allFliesAngleVarsACutOff*36, yerr= allFliesAngleVarsBCutOff*36, fmt='none', linewidth =3, ecolor=[.6,.6,.6], capsize = 0, zorder=5)
ax.errorbar(x=rotated_heading_anglesA, y=rotated_heading_anglesB +360.0, xerr=allFliesAngleVarsACutOff*36 , yerr= allFliesAngleVarsBCutOff*36, fmt='none', linewidth =3, ecolor=[.6,.6,.6], capsize = 0, zorder=5)
ax.plot([-180,180], [-180,180], color=[.7, .7, .7], zorder=1, linewidth=3, linestyle='--')
ax.plot([-180,180], [180,540], color=[.7, .7, .7], zorder=1, linewidth=3, linestyle='--')


ax.set_title('A-B Heading change', fontsize =18)
ax.set_xticks((-180,-90, 0, 90,180))
ax.set_yticks((-180,-90, 0, 90,180, 270, 360, 450, 540))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_bounds(-180,540)
ax.spines['bottom'].set_bounds(-180,180)
ax.set_yticklabels(('-180','-90', '0', '90','180', '270', '360', '450', '540'), color='k', fontsize=12)
ax.set_xticklabels(('-180','-90', '0', '90','180'), color='k', fontsize=12)
ax.set_ylabel('Second sun headings', fontsize=18)
ax.set_xlabel('First sun headings', fontsize=18)
ax.axis('equal')
#plt.show()  ###uncomment if not bootstrapping



############# For Bootstrap ###############

#observedDiffs = circdiff(allFliesMeanAnglesB, allFliesMeanAnglesA) # this uses all the data
observedDiffs = circdiff(allFliesMeanAnglesACutOff, allFliesMeanAnglesBCutOff)# previously had with allFliesMeanAnglesA and allFliesMeanAngles B which doesn't make sense because we need to use the 0.2 cutoff with this
observedDiffMean = np.mean(np.abs(observedDiffs))


#%%run with the same randomization

NUM_RESAMPLES = 10000
resampledDiffMeans = np.zeros(NUM_RESAMPLES, dtype='float')
for resampleInd in range(NUM_RESAMPLES):
    resampledB = np.random.permutation(allFliesMeanAnglesBCutOff)
    resampledDiffs = circdiff(resampledB, allFliesMeanAnglesACutOff)
    resampledDiffMean = np.mean(np.abs(resampledDiffs))
    resampledDiffMeans[resampleInd] = resampledDiffMean

pval = np.sum(resampledDiffMeans <= observedDiffMean)/float(NUM_RESAMPLES)
pval = np.around(pval, decimals=3)

observed_diff=observedDiffMean*180./np.pi
observed_diff=np.around(observed_diff, decimals=3)

bootstrap_mean=np.mean(resampledDiffMeans)*180./np.pi
bootstrap_mean=np.around(bootstrap_mean, decimals=3)

#%% plotting using Peter's circ diff
print 'pval= ', pval
print 'observed diff= ', observed_diff
print 'bootstrap mean= ', bootstrap_mean
fig = plt.figure(figsize=(4,4))
fig.set_facecolor('w')
ax = fig.add_subplot(1, 1, 1)
for axis in ['left','bottom']:
            ax.spines[axis].set_linewidth(2.0)
ax.xaxis.set_tick_params(width=2.0, length = 6.0, direction = 'in')
ax.yaxis.set_tick_params(width=2.0, length = 6.0, direction = 'in')
ax.hist(resampledDiffMeans*180./np.pi, bins=15, histtype='stepfilled', color = [0.7, 0.7, 0.7])
ax.axvline(observedDiffMean*180./np.pi, ymin=0.078, ymax=0.88, color='r', linewidth=2)
ax.text((observedDiffMean+0.6)*180./np.pi, NUM_RESAMPLES/30., 'pval = '+str(pval), fontsize=14)
ax.text((observedDiffMean+0.6)*180./np.pi, NUM_RESAMPLES/15., 'ob diff='+str(observed_diff), fontsize=14)
ax.text((observedDiffMean+0.6)*180./np.pi, NUM_RESAMPLES/20., 'bs mean= '+str(bootstrap_mean), fontsize=14)
## following commands to make figure "pretty"
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
ax.set_xlabel('Mean angle difference',  fontsize=14)
ax.set_ylabel('Counts', fontsize=14)
ax.set_xlim((-15, 200))
ax.set_ylim((-200, 2500))
ax.spines['bottom'].set_bounds(0, 180)
ax.spines['left'].set_bounds(0, 2500)
ax.set_xticks((0, 90, 180))
ax.set_yticks((0, 2500))
ax.set_yticklabels((0, 2500), fontsize=14)
ax.set_xticklabels((0, 90, 180), fontsize=14)
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.show()
#fig.savefig('2hoursunA_vs_stripe_bootstrap_PW_method_2500y.pdf', transparent=True)
