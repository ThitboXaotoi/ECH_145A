import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import glob
import scipy.stats
import csv
import scipy.signal

import pint

u = pint.UnitRegistry()

#calibration curve
weight_file = "/Users/Nguyen/Documents/ECH_145/Lab-2_1/Sample_data/Calibration_1.csv" #raw weights data

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
    
weight_data = np.array(weight_data,dtype=float) #cast as array
mean_weight_data = np.mean(weight_data,axis=1) * u.g #weight data in g 

weight_data
