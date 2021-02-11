import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
import pandas as pd
from scipy.stats import linregress
from scipy import constants
from scipy import signal
from uncertainties import unumpy
from uncertainties import *


weights = np.array([ufloat(35, 0.5), ufloat(110.685, 0.5), ufloat(118.428, 0.5), ufloat(259.595, 0.5)])

Pl_diameter = ufloat(35.56, 0.01)/1000
area = (Pl_diameter/2)**2*np.pi
pressure = []

for i in range(len(weights)):
    force = weights[i]/1000*constants.g
    pressure.append(force/area*0.00014503)  # newton

pressure = np.array(pressure)
p_err = unumpy.std_devs(pressure)
pressure = unumpy.nominal_values(pressure)
rel_vol = np.zeros(4)

for i in range(0, 4):
    voltage = np.zeros(3)
    for j in range(0, 3):
        file = pd.read_csv(
            f'/Users/binhco/Documents/ECH_145A/Lab-2_2/sample/Calibration_{i+1}_T{j+1}.csv', header=6)
        voltage[j] = np.mean(file['AI4 (mV)'].to_numpy())

    rel_vol[i] = np.mean(voltage/9)

lin_reg = linregress(pressure, rel_vol)
slope = np.round_(lin_reg[0], 5)
intercept = np.round_(lin_reg[1], 5)
stderr = 2*np.round_(lin_reg[4], 5)
calibration = slope

plt.figure(0, figsize=(10, 8))
plt.errorbar(pressure, rel_vol, xerr=p_err, yerr=stderr,
             marker='o', ls='None', label='raw data')

print(f'Calibration: ', calibration)
print(f'Error: ', stderr)
x = np.linspace(0, 0.5, 6969)
y = intercept + slope*x
plt.plot(x, y, label=f'lin-reg: y ={slope}*x +{intercept}')
plt.ylabel('Relative Voltage (mV/V)')
plt.xlabel('Pressure (psi)')
plt.legend()
plt.savefig('/Users/binhco/Documents/ECH_145A/Lab-2_2/Calibration.png')

"""
Calculating the parameter
"""


def osc_func(t, P_0, C, k, omega, phi):
    P = P_0+C*np.exp(-k*t/2)*np.cos(omega*t+phi)
    return P


def best_fit(analysis_P, analysis_t, pressure):
    """
    Guessing every parameter for initial guess
    """
    P0_range = analysis_P[-30:-1]
    P_0 = np.mean(P0_range)
    P0_st = 2*np.std(P0_range)

    C = max(analysis_P - P_0)

    rel_maxes = signal.argrelmax(analysis_P)
    rel_mins = signal.argrelmin(analysis_P)
    rel_maxes = np.append([0], rel_maxes)
    rel_mins = rel_mins[0]

    rel_maxes = rel_maxes[:]
    rel_mins = rel_mins[:6]

    t_mins = analysis_t[rel_mins]
    t_mins_diff = np.diff(t_mins)
    omega = 2*np.pi/np.mean(t_mins_diff)

    # Finding k manually
    k_t_y = C/2
    possible_k_t_y = np.isclose(pressure, k_t_y, 0.1)
    possible_k_t_y = possible_k_t_y * pressure
    k_t_index = possible_k_t_y.nonzero()[-1]
    k_t = time[k_t_index[-1]] - start_at

    k = -2*np.log(0.5)/k_t

    manual_params = np.array([P_0, C, k, omega, 0.00])

    """End here""" 
    """Using Scipy curvefit using manual parameter as initial guess""" 

    return so.curve_fit(osc_func, analysis_t, analysis_P, manual_params)

gas = ['Ar', 'CO2']
mass = ['NoWeight', 'Mass1', 'Mass2']
trial = ['T1', 'T2', 'T3', 'T4', 'T5']

for i in range (3):
    osc_file = pd.read_csv(f'/Users/binhco/Documents/ECH_145A/Lab-2_2/sample/Osc_Ar_{mass[i]}_T1.csv',
                           header=6)

    voltage = osc_file['AI4 (mV)'].to_numpy()
    time = osc_file['Time (s)'].to_numpy()
    corrected_vol = voltage/9
    pressure = corrected_vol/calibration

    start_at = np.argmax(voltage) + 7
    end_at = start_at + 400

    analysis_P = (pressure[start_at:end_at])
    analysis_t = (time[start_at:end_at]) - time[start_at]

    plt.figure(i+1, figsize=(10, 8))
    plt.plot(analysis_t, analysis_P, '.', label='Raw Data')

    filler = np.linspace(analysis_t[0], analysis_t[-1], 6969)

    fitting_params = best_fit(analysis_P, analysis_t, pressure)

    plt.plot(filler, osc_func(filler, *fitting_params[0]), label='Best fit')

    plt.ylabel('Pressure (psi)')
    plt.xlabel('Time (s)')
    plt.title(f'Argon ({mass[i]})')
    plt.legend()
    plt.savefig(f'/Users/binhco/Documents/ECH_145A/Lab-2_2/Argon ({mass[i]}).png')

plt.show()

def kappa(P_abs, k, omega, m, V, A):
    kappa = (omega**2 + k**2/4) * (m*V/(P_abs * A**2))
    return kappa

# all SI units
Patm = 101325 #Pa
Ar_height = np.array([ufloat(68, 1), ufloat(56, 8), ufloat(56, 3)])/1000 # m
CO2_height = np.array([ufloat(68, 1), ufloat(60, 3), ufloat(57, 2)])/1000
tube_length = ufloat(280, 1)/1000
tube_diameter = ufloat(3.59, 0.02)/1000

V_Ar = area*Ar_height + tube_length*(tube_diameter/2)**2*np.pi
V_CO2 = area*CO2_height + tube_length*(tube_diameter/2)**2*np.pi

# Argon
Argon = {
    'P_abs': [],
    'C': [],
    'k': [],
    'omega': [],
    'phi': [], 
    'Kappa': []
}
for i in range(0, 3):
    for j in [0, 2, 3]:
        osc_file = pd.read_csv(f'/Users/binhco/Documents/ECH_145A/Lab-2_2/sample/Osc_Ar_{mass[i]}_{trial[j]}.csv',
                               header=6)

        voltage = osc_file['AI4 (mV)'].to_numpy()
        time = osc_file['Time (s)'].to_numpy()
        corrected_vol = voltage/9
        pressure = corrected_vol/calibration
        #change to Pa
        pressure = pressure*6894.7572

        start_at = np.argmax(voltage) + 10
        end_at = start_at + 400

        analysis_P = (pressure[start_at:end_at])
        analysis_t = (time[start_at:end_at]) - time[start_at]
        
        fitting_params, pcov = best_fit(analysis_P, analysis_t, pressure)
        fitting_err = 2*np.sqrt(np.diag(pcov))

        P_abs = np.round_(Patm + fitting_params[0], 4)
        P_abs_err = np.round_(fitting_err[0], 4) 
        a1 = ufloat(P_abs, P_abs_err)
        Argon['P_abs'].append(a1)
        
        c = np.round_(fitting_params[1], 4)
        c_err = np.round_(fitting_err[1], 4)
        a2 = ufloat(c, c_err)
        Argon['C'].append(a2)

        k = np.round_(fitting_params[2], 4)
        k_err = np.round_(fitting_err[2], 4)
        a3 = ufloat(k, k_err)
        Argon['k'].append(a3)

        omega = np.round_(fitting_params[3], 4)
        omega_err = np.round_(fitting_err[3], 4)
        a4 = ufloat(omega, omega_err)
        Argon['omega'].append(a4)

        phi = np.round_(fitting_params[4], 4)
        phi_err = np.round_(fitting_err[4], 4)
        a5 = ufloat(phi, phi_err)
        Argon['phi'].append(a5)
        
        Kappa = kappa(a1, a3, a4, weights[i]/1000, V_Ar[i], area)
        Argon['Kappa'].append(Kappa)

#CO2
CO2 = {
    'P_abs': [],
    'C': [],
    'k': [],
    'omega': [],
    'phi': [],
    'Kappa': []
}

for i in range(0, 3):
    for j in range(0, 3):
        osc_file = pd.read_csv(f'/Users/binhco/Documents/ECH_145A/Lab-2_2/sample/Osc_CO2_{mass[i]}_{trial[j]}.csv',
                               header=6)

        voltage = osc_file['AI4 (mV)'].to_numpy()
        time = osc_file['Time (s)'].to_numpy()
        corrected_vol = voltage/9
        pressure = corrected_vol/calibration
        pressure = pressure*6894.7572

        start_at = np.argmax(voltage) + 7
        end_at = start_at + 400

        analysis_P = (pressure[start_at:end_at])
        analysis_t = (time[start_at:end_at]) - time[start_at]
        
        fitting_params, pcov = best_fit(analysis_P, analysis_t, pressure)
        fitting_err = 2*np.sqrt(np.diag(pcov))

        P_abs = np.round_(Patm + fitting_params[0], 4)
        P_abs_err = np.round_(fitting_err[0], 4) 
        a1 = ufloat(P_abs, P_abs_err)
        CO2['P_abs'].append(a1)
        
        c = np.round_(fitting_params[1], 4)
        c_err = np.round_(fitting_err[1], 4)
        a2 = ufloat(c, c_err)
        CO2['C'].append(a2)

        k = np.round_(fitting_params[2], 4)
        k_err = np.round_(fitting_err[2], 4)
        a3 = ufloat(k, k_err)
        CO2['k'].append(a3)

        omega = np.round_(fitting_params[3], 4)
        omega_err = np.round_(fitting_err[3], 4)
        a4 = ufloat(omega, omega_err)
        CO2['omega'].append(a4)

        phi = np.round_(fitting_params[4], 4)
        phi_err = np.round_(fitting_err[4], 4)
        a5 = ufloat(phi, phi_err)
        CO2['phi'].append(a5)
        
        Kappa = kappa(a1, a3, a4, weights[i]/1000, V_CO2[i], area)
        CO2['Kappa'].append(Kappa)
        

a = pd.DataFrame(Argon)
b = pd.DataFrame(CO2)
print(np.mean(CO2['Kappa']))
