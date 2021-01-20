#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import files

uploaded = files.upload()


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from glob import glob 
from scipy import stats


# In[6]:


temp_cold_1 = "TC1.csv"
temp_cold_2 = "TC2.csv"
temp_cold_3 = "TC3.csv"
temp_hot_1 = "TH1.csv"
temp_hot_2 = "TH2.csv"
temp_hot_3 = "TH3.csv"
temp_room_1 = "TR1.csv"
temp_room_2 = "TR2.csv"
temp_room_3 = "TR3.csv"
volt_cold_1 = "VC1.csv"
volt_cold_2 = "VC2.csv"
volt_cold_3 = "VC3.csv"
volt_hot_1 = "VH1.csv"
volt_hot_2 = "VH2.csv"
volt_hot_3 = "VH3.csv"
volt_room_1 = "VR1.csv"
volt_room_2 = "VR2.csv"
volt_room_3 = "VR3.csv"


def read_raw_data(file_name):
    with open(file_name, 'r') as f:
        raw_data = list(csv.reader(f)) #use csv reader function
        f.close()
    return raw_data

raw_data = read_raw_data(temp_cold_1)

vc1 = pd.read_csv(volt_cold_1,header=6)
th1 = pd.read_csv(temp_hot_1, header=6)
#extract header from rest of data 
header = th1[:7]
data = np.array(th1[7:],dtype=float) #convert data into a 4 x n numpy array so we can do math

sample_no = data[:,0] 
time = data[:,1]
channel_0_T = data[:,2] #commercial thermocouple
channel_1_T = data[:,3] #homemade thermocouple


# In[7]:


vc1


# In[8]:


plt.figure(0, figsize=(12, 10))
plt.plot(time,channel_0_T,label='Commercial Thermocouple') #plot x then y
plt.plot(time,channel_1_T,label='Handmade Thermocouple')
plt.legend() #add legend
 
plt.xlabel('Time (s)')
plt.ylabel('Temperature ($^\circ$C)')
plt.show()


# In[9]:


mean_T = np.mean(channel_0_T)
std_T = np.std(channel_0_T)

print('Temperature mean,std')
print(mean_T,std_T)


# In[10]:


#get all the cold temp thermocouple files
list_of_tc = glob('TC*.csv')
list_of_vc = glob('VC*.csv')

#get all the hot temp thermocouple files
list_of_th = glob('TH*.csv')
list_of_vh = glob('VH*.csv')

#get all the room temp thermocouple files
list_of_tr = glob('TR*.csv')
list_of_vr = glob('VR*.csv')


# In[11]:


def read_cooked_data(list_of_data_files):
    comm_data = []
    hand_data = []
    for i in range(len(list_of_data_files)):
        # tc_raw_data.append(read_raw_data(list_of_data_files[i]))
        raw_data = read_raw_data(list_of_data_files[i])
        
        data = np.array(raw_data[7:],dtype=float) #convert data into a 4 x n numpy array so we can do math

        sample_no = data[:,0] 
        time = data[:,1]
        C_H_0 = data[:,2] #commercial thermocouple
        C_H_1 = data[:,3] #homemade thermocouple

        comm_data.append(C_H_0) #append commercial data to a list
        hand_data.append(C_H_1)

    return sample_no, time, comm_data, hand_data


# In[44]:


# Reading all sample data
tc_sample_no, tc_time, tc_comm_data, tc_hand_data = read_cooked_data(list_of_tc)
th_sample_no, th_time, th_comm_data, th_hand_data = read_cooked_data(list_of_th)
tr_sample_no, tr_time, tr_comm_data, tr_hand_data = read_cooked_data(list_of_tr)
vc_sample_no, vc_time, vc_comm_data, vc_hand_data = read_cooked_data(list_of_vc)
vh_sample_no, vh_time, vh_comm_data, vh_hand_data = read_cooked_data(list_of_vh)
vr_sample_no, vr_time, vr_comm_data, vr_hand_data = read_cooked_data(list_of_vr)

print(len(vr_comm_data[1]))


# In[69]:


""" Here is the super annoying part:  The glob.glob randomly merge the file so everytime I reupload the file, the order is messed up"""
xerr = 0.05
yerr = 0.000005

# cold bath
plt.figure(0, figsize=(15, 10))
plt.errorbar(tc_comm_data[0][:300],vc_comm_data[2][:300], xerr=xerr, yerr=yerr, marker='o', fmt='.', label='Commercial Thermocouple (trial 1)') # First 300 samples
plt.errorbar(tc_hand_data[0][:300],vc_hand_data[2][:300], xerr=xerr, yerr=yerr, marker='o', fmt='.', label='Handmade Thermocouple (trial 1)')
plt.errorbar(tc_comm_data[1][:359],vc_comm_data[1][:359], xerr=xerr, yerr=yerr, marker='o', fmt='.', label='Commercial Thermocouple (trial 2)') # First 359 samples
plt.errorbar(tc_hand_data[1][:359],vc_hand_data[1][:359], xerr=xerr, yerr=yerr, marker='o', fmt='.', label='Handmade Thermocouple (trial 2)')
plt.errorbar(tc_comm_data[2][:298],vc_comm_data[0][:298], xerr=xerr, yerr=yerr, marker='o', fmt='.', label='Commercial Thermocouple (trial 3)') # First 298 samples
plt.errorbar(tc_hand_data[2][:298],vc_hand_data[0][:298], xerr=xerr, yerr=yerr, marker='o', fmt='.', label='Handmade Thermocouple (trial 3)')
plt.title('Scatter plot between the measured volume vs temperature in the cold bath (all trials)', fontsize=15 )

plt.legend()
 
plt.xlabel('Temperature ($^\circ$C)')
plt.ylabel('Measured Voltage (mV)')

# hot bath

plt.figure(1, figsize=(15, 10))
plt.errorbar(th_comm_data[2][:281],vh_comm_data[0][:281], xerr=xerr, yerr=yerr, marker='o', fmt='.', label='Commercial Thermocouple (trial 1)') # First 281 samples
plt.errorbar(th_hand_data[2][:281],vh_hand_data[0][:281], xerr=xerr, yerr=yerr, marker='o', fmt='.', label='Handmade Thermocouple (trial 1)')
plt.errorbar(th_comm_data[1][:282],vh_comm_data[1][:282], xerr=xerr, yerr=yerr, marker='o', fmt='.', label='Commercial Thermocouple (trial 2)') # First 282 samples
plt.errorbar(th_hand_data[1][:282],vh_hand_data[1][:282], xerr=xerr, yerr=yerr, marker='o', fmt='.', label='Handmade Thermocouple (trial 2)')
plt.errorbar(th_comm_data[0][:267],vh_comm_data[2][:267], xerr=xerr, yerr=yerr, marker='o', fmt='.', label='Commercial Thermocouple (trial 3)') # First 267 samples
plt.errorbar(th_hand_data[0][:267],vh_hand_data[2][:267], xerr=xerr, yerr=yerr, marker='o', fmt='.', label='Handmade Thermocouple (trial 3)')
plt.title('Scatter plot between the measured volume vs temperature in the hot bath (all trials)', fontsize=15 )

plt.legend()
 
plt.xlabel('Temperature ($^\circ$C)')
plt.ylabel('Measured Voltage (mV)')

# room bath
plt.figure(2, figsize=(15, 10))
plt.errorbar(tr_comm_data[2][:369],vr_comm_data[0][:369], xerr=xerr, yerr=yerr, marker='o', fmt='.', label='Commercial Thermocouple (trial 1)') # First 368 samples
plt.errorbar(tr_hand_data[2][:369],vr_hand_data[0][:369], xerr=xerr, yerr=yerr, marker='o', fmt='.', label='Handmade Thermocouple (trial 1)')
plt.errorbar(tr_comm_data[0][:367],vr_comm_data[2][:367], xerr=xerr, yerr=yerr, marker='o', fmt='.', label='Commercial Thermocouple (trial 2)') # First 367 samples
plt.errorbar(tr_hand_data[0][:367],vr_hand_data[2][:367], xerr=xerr, yerr=yerr, marker='o', fmt='.', label='Handmade Thermocouple (trial 2)')
plt.errorbar(tr_comm_data[1][:354],vr_comm_data[1][:354], xerr=xerr, yerr=yerr, marker='o', fmt='.', label='Commercial Thermocouple (trial 3)') # First 354 samples
plt.errorbar(tr_hand_data[1][:354],vr_hand_data[1][:354], xerr=xerr, yerr=yerr, marker='o', fmt='.', label='Handmade Thermocouple (trial 3)')
plt.title('Scatter plot between the measured volume vs temperature in the room temperature bath (all trials)', fontsize=15 )

plt.legend()
 
plt.xlabel('Temperature ($^\circ$C)')
plt.ylabel('Measured Voltage (mV)')

plt.show()


# In[74]:


def data_mean(data, sample_count):
    mean = np.zeros((3, sample_count))
    for i in range(3):
        mean[i] = data[i][:sample_count]
    mean = np.mean(mean, axis=0)
    return mean

def lin_reg_plot(T, V, color):
    lin_reg = stats.linregress(T, V)
    print(lin_reg)
    slope = lin_reg.slope
    intercept = lin_reg.intercept
    r_value = lin_reg.rvalue
    xerr = 0.03
    yerr = 0.0000005
    label = 'y = {} x + {}'.format(round(slope, 5), round(intercept, 5))
    print(label)
    plt.errorbar(T, slope * T + intercept, color = color, label = label)

tc_comm_mean = data_mean(tc_comm_data, 298)
vc_comm_mean = data_mean(vc_comm_data, 298)
tc_hand_mean = data_mean(tc_hand_data, 298)
vc_hand_mean = data_mean(vc_hand_data, 298)

th_comm_mean = data_mean(th_comm_data, 267)
vh_comm_mean = data_mean(vh_comm_data, 267)
th_hand_mean = data_mean(th_hand_data, 267)
vh_hand_mean = data_mean(vh_hand_data, 267)

tr_comm_mean = data_mean(tr_comm_data, 354)
vr_comm_mean = data_mean(vr_comm_data, 354)
tr_hand_mean = data_mean(tr_hand_data, 354)
vr_hand_mean = data_mean(vr_hand_data, 354)


plt.figure(0, figsize=(18, 6))
plt.plot(tc_comm_mean, vc_comm_mean , 'o', label='Commercial Thermocouple')
lin_reg_plot(tc_comm_mean, vc_comm_mean, 'b')
plt.plot(tc_hand_mean, vc_hand_mean , 'o', label='Handmade Thermocouple')
lin_reg_plot(tc_hand_mean, vc_hand_mean, 'orange')
plt.title('Mean value of three trials between temperature and voltage in cold bath with linear regression', fontsize=15 )
plt.xlabel('Temperature ($^\circ$C)')
plt.ylabel('Measured Voltage (mV)')
plt.legend()

plt.figure(1, figsize=(18, 6))
plt.plot(th_comm_mean, vh_comm_mean , 'go', label='Commercial Thermocouple')
lin_reg_plot(th_comm_mean, vh_comm_mean, 'green')
plt.plot(th_hand_mean, vh_hand_mean , 'ro', label='Handmade Thermocouple')
lin_reg_plot(th_hand_mean, vh_hand_mean, 'red')
plt.title('Mean value of three trials between temperature and voltage in hot bath with linear regression', fontsize=15 )
plt.xlabel('Temperature ($^\circ$C)')
plt.ylabel('Measured Voltage (mV)')
plt.legend()

plt.figure(2, figsize=(18, 6))
plt.plot(tr_comm_mean, vr_comm_mean , 'o', color = 'magenta', label='Commercial Thermocouple')
lin_reg_plot(tr_comm_mean, vr_comm_mean, 'magenta')
plt.plot(tr_hand_mean, vr_hand_mean , 'o', color = 'brown', label='Handmade Thermocouple')
lin_reg_plot(tr_hand_mean, vr_hand_mean, 'brown')
plt.title('Mean value of three trials between temperature and voltage in room temperature bath with linear regression', fontsize=15 )
plt.xlabel('Temperature ($^\circ$C)')
plt.ylabel('Measured Voltage (mV)')
plt.legend()

plt.show()


# In[122]:





# In[ ]:




