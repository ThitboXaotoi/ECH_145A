#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 07:29:58 2020

@author: tfinney
"""

import numpy
import matplotlib.pyplot as pyplot
import scipy.optimize
import glob
import scipy.stats
import csv
import scipy.signal


import pint

u = pint.UnitRegistry()


#calibration curve
weight_file = "Lab2.1_data/weights.csv" #raw weights data

def open_file(file_name):
    with open(file_name,'r') as f:
        raw_data = list(csv.reader(f))
        f.close()
    return raw_data
        

raw_weight_data = open_file(weight_file) #read in weight data
header = raw_weight_data[0] #extract header 
weight_data = []
for i in range(1,len(raw_weight_data)):
    weight_data.append(raw_weight_data[i][1:]) #extract out data from csv excluding labels
    
weight_data = numpy.array(weight_data,dtype=float) #cast as array
mean_weight_data = numpy.mean(weight_data,axis=1) * u.g #weight data in g 


#in this data set, we did no weight, then 1, 1+2, 1+2+3 etc
#cum sum nicely does this for us.
weight_cfgs = numpy.cumsum(mean_weight_data) 

#define some constants plunger measurements are necessary so we can make "pressure"
g = (1*u.gravity).to(u.m/u.s**2)
plunger_diameter = 0.0325 * u.m 
plunger_area = numpy.pi * plunger_diameter**2/4 

#multiply weight times g to get force  (F = ma) then divide by area
pressure_data = (weight_cfgs * g)/plunger_area 
pressure_data = pressure_data.to(u.Pa)
print(pressure_data)




"""
Now do the calibration curve, reading in the data!


factory calibration

"""
calibration = 0.2584 * u.mV/u.psi/u.V

#%%
"""

Nonlinear CURVE FITTING sho' nuff

"""
#For reference if we want to offset to absolute pressure
sl_P = 1020.4 * u.millibar

#read in one file
osc_data_file = "/media/infidel2/ECH145/git/ECH145/ECH145A/Lab_2/Lab2.1_data/osc_nw_1.csv"

#extract data from file
raw_osc_data = open_file(osc_data_file)
header = raw_osc_data[:7]
osc_data = numpy.array(raw_osc_data[7:],dtype=float)

voltage = osc_data[:,2] #mV
time = osc_data[:,1] #s

#correct voltage data by dividing by the input 9V
corrected_voltage = (voltage * u.mV) / (9*u.V)

#convert to pressure using our calibration
pressure = corrected_voltage / calibration

#convert to psi because....
pressure = pressure.to(u.psi)

#start plotting
pyplot.figure(1)
start_at = numpy.argmax(voltage) #find where the max voltage is -- likely this is where we start analysing

#figure out start time
time_start = time[start_at]

#stop time is arbitrary, I picked 500 data points beyond, but whatever is cool.
end_at = start_at + 500

#take a subset of the pressure data that we want to actually analyze
analysis_P = pressure[start_at:end_at].magnitude #pressure 
analysis_t = time[start_at:end_at] - time[start_at]#s  #correct time so start of oscilliations is zero.

#plot the subset
pyplot.plot(analysis_t,analysis_P,'o',label='Data')

#Guess parameters -- this isn't really necessary, but interested to do. 
P_0 = analysis_P[-1] # pressure 
C = max(analysis_P - P_0) #guess for C -- really not bad

#you can compute the max and min of each oscillation using scipy signal
rel_maxes = scipy.signal.argrelmax(analysis_P)
rel_mins = scipy.signal.argrelmin(analysis_P)
rel_maxes = numpy.append([0],rel_maxes) #add the first step as one
rel_mins = rel_mins[0] #some weird formatting in scipy that we are correcting here

rel_maxes = rel_maxes[:] #if we want to exclude data points you can do so here
rel_mins = rel_mins[:10] 

#to guess k -- I ddid this quite regression
def curve_exp(t,k):
    f =  numpy.exp(-k/2 * t )
    return f
    
k_times = analysis_t[rel_maxes]
k_P = analysis_P[rel_maxes] - P_0 #it worked pretty well!

exp_params = scipy.optimize.curve_fit(curve_exp,k_times,k_P)
kk = exp_params[0]

#find omega using our relative mins -- once again this isn't necessary but cool
t_mins = analysis_t[rel_mins]
t_mins_diff = numpy.diff(t_mins)
omega = 2*numpy.pi / (numpy.mean(t_mins_diff))


#Do the curve fitting -- fucntion we want to fit is defined here
def osc_func(t,P_0,C,k,omega,phi):
    """

    """
    P = P_0 + C * numpy.exp(-k * t / 2 ) * numpy.cos( omega * t + phi)
    return P
    
    
#actually do the fitting
fitting_params = scipy.optimize.curve_fit(osc_func,analysis_t,analysis_P)

#generate some filler data to make a smooth line
filler = numpy.linspace(analysis_t[0],analysis_t[-1],1000)

#plot the data -- *unpacks variables into the arguments (neat)
pyplot.plot(filler,osc_func(filler,*fitting_params[0]),label='Best Fit')

pyplot.xlabel('Time (s)')
pyplot.ylabel('Pressure (psi)')
pyplot.legend()




"""

Compute kappa -- for one trial

obivously you want to analyze all the data above first

"""

def kappa(omega, k, m, V, P_abs, A):
    """
    
    Omega, k are fitting parameters from above
    
    m is the total oscillating mass
    
    V is the volume of the air in the spring
    
    P_abs is the absolute pressure
    
    A is the cross sectional area of the plunger
    
    """

    kappa = (omega**2 + k**2/4 ) * (m*V / (P_abs * A**2))

    return kappa

















