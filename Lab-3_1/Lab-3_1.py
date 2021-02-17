import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
import pandas as pd
from scipy.stats import linregress
from scipy import constants
from scipy import signal
from uncertainties import unumpy
from uncertainties import *
from true_proof import *
import datetime
start_time = datetime.datetime.now()
# insert code snippet here


def fluid_density(rho_w, M_e, M_w, M_f):
    return rho_w * (M_f - M_e)/(M_w - M_e)


data_raw = pd.read_excel(
    '/Users/binhco/Documents/ECH_145A/Lab-3_1/Sample data/ECH145A_LAB3.1_data.xlsx', header=2)

data_raw = data_raw.fillna(0)
data_raw = data_raw.drop(data_raw.index[[10, 11]])

data = np.array(data_raw, dtype=float)

density = np.linspace(0.8133, 0.9976, 8)*1000
temperature_C = np.array([15.56, 20, 25])
e_mole_frac = np.zeros((3, len(density)))
true_pr = np.zeros((3, len(density)))

for i in range(len(temperature_C)):
    for j in range(len(density)):
        try:
            e_mole_frac[i][j] = true_proof(temperature_C[i], density[j])[
                'ethanol_mole_frac']
            true_pr[i][j] = true_proof(temperature_C[i], density[j])[
                'true_proof']
        except:
            continue

# print(true_proof(30, 950))

plt.figure(0, figsize=(10, 8))
plt.plot(density, e_mole_frac[0], label='60˚F')
plt.plot(density, e_mole_frac[1], label='20˚C')
plt.plot(density, e_mole_frac[2], label='25˚C')
plt.xlabel('Density (kg/m3)')
plt.ylabel('Mole Fraction')
plt.legend()
# plt.savefig('/Users/binhco/Documents/ECH_145A/Lab-3_1/figure1.png')
plt.show()

plt.figure(1, figsize=(10, 8))
plt.plot(density, true_pr[0], label='60˚F')
plt.plot(density, true_pr[1], label='20˚C')
plt.plot(density, true_pr[2], label='25˚C')
plt.xlabel('Density (kg/m3)')
plt.ylabel('True Proof (˚P)')
plt.legend()
# plt.savefig('/Users/binhco/Documents/ECH_145A/Lab-3_1/figure2.png')
plt.show()

rho_d = data[:, 10]*1000
temp_d = data[:, 9]
true_pr_d = np.zeros(10)

for i in range(10):
    try:
        true_pr_d[i] = true_proof(temp_d[i], rho_d[i])['true_proof']
        print(f"Approximate True Proof with density {rho_d[i]} (kg/m3) at temperature {temp_d[i]} (˚C): ",
              true_pr_d[i], '(˚P)')
    except:
        continue

M_e = ufloat(35.069, 0.001)/1000
M_w = ufloat(60.638, 0.001)/1000
rho_w = ufloat(0.9976, 0.0005)*1000
M_f = np.array([ufloat(60.222, 0.001),
                ufloat(60.018, 0.001),
                ufloat(59.696, 0.001),
                ufloat(59.112, 0.001),
                ufloat(58.2, 0.001),
                ufloat(57.559, 0.001),
                ufloat(56.985, 0.001),
                ufloat(56.541, 0.001),
                ufloat(55.851, 0.001)])


M_f = M_f/1000

rho_f = fluid_density(rho_w, M_e, M_w, M_f)

end_time = datetime.datetime.now()
print(end_time - start_time)
