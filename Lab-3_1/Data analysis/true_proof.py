#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:37:14 2020

@author: tfinney
"""
import numpy
import scipy.interpolate
import pint
import pandas
import bisect
import copy

u = pint.UnitRegistry()

def true_proof(temperature_C, density):
    """
    does interpolation
    
    takes temperature in deg C and density in kg / m**3
    """
    Q_ = u.Quantity
    ref_temp = Q_(60,u.degF)
    #reference data
    ref_etOH_density = 793.13 * u.kg/u.m**3 #At 60 F
    ref_water_density = 999.04 * u.kg/u.m**3 #At 60 F
    
    mw_etOH = 46.06844 * u.gram/u.mol
    mw_water = 18.01528 * u.gram/u.mol
        
    alpha_C = 25e-6 #linear coefficient of thmernal expansion for glass  (1/C)
    alpha_F = 5/9 * alpha_C #Q_(alpha_C * 5/9,1/u.degF) #in 1/F
        
    input_temp = Q_(temperature_C,u.degC) #add units for quick conversion maybe?
    input_density = density * u.kg/u.m**3
    
    """
    read in TTB_Table_6
    """
    #Proof, Alcohol, Water, SG in Air, SG in Vacpython find two closest value in list interpolation
    ttb6_raw = pandas.read_excel('/Users/Nguyen/Documents/ECH_145/Lab-3_1/Data analysis/TTB_Table_6_digitized.xlsx',header=5, engine='openpyxl')
    
    ttb6_raw = ttb6_raw.dropna(axis=1) #get rid of nans
    #convert to array beacuse dataframs are annoying
    ttb6 = numpy.array(ttb6_raw)
    
    
    
    #convert input density and temp C to specific gravity and farenheit for some reason
    temp_F = (input_temp.to(u.degF)).magnitude #F
    # print(temp_F)
    
    #compute density 
    #first convert from whatever te  return trueProofOut, x_etOH_Out, et_OH_massFraction_Out
    #mp we are at to Reference -- which is 60 F
    density_at_ref_T = input_density * (1 + (alpha_F * (temp_F - ref_temp.magnitude)))
    
    user_sg = (density_at_ref_T / ref_water_density).magnitude
    # print(user_sg)
    #find the nearest specific gravity (vacuum ) to the one we provide
        
    sg_data = ttb6[:,4] #specific gravity in vacuum (pycnometer)
    proof_data = ttb6[:,0] #proof list
    booze_data = ttb6[:,1] #alcohol content
    water_data = ttb6[:,2] #water content
       
    #interpolate specific gravity to find proof at our provided SG
    interp_sg_proof = scipy.interpolate.interp1d(sg_data,proof_data)
    calculated_proof = interp_sg_proof(user_sg)
    
    """
    now we take our calculcated proof and find the TRUE PROOF 
    
    YOU CAN'T HANDLE THE TRUTH 
                    -- G. Miller on Separation of Variables Winter 2019
    """
    ttb1_raw = pandas.read_excel('/Users/Nguyen/Documents/ECH_145/Lab-3_1/Data analysis/TTB_Table_1_digitized.xlsx',header=5, engine='openpyxl')
    ttb1_raw = ttb1_raw.fillna(0) #nans mess with the interpolation so lets bump em.
    temperature_header = numpy.arange(1,101) #fuck the real header, this is the shit.
    ttb1 = numpy.array(ttb1_raw,dtype=float) #convert to numpy array because I can.
    
    ttb1 = ttb1[:,1:] #cut out the first column as it is the hydrometer readings (don't need)
    
    ttb1_y_size,ttb1_x_size = numpy.shape(ttb1) # get the size of our array
    
    ttb1_ys = numpy.arange(0,ttb1_y_size) # proof readings not true but apparent
    ttb1_xs = numpy.arange(1,ttb1_x_size+1) # temp in deg F, starts at 1 deg F ends at 100 deg F
    
    #using miller's bs notation here 
    #see Gauging Alcohol.pdf to figure out WTF this means.
    
    rd_proof = numpy.floor(calculated_proof) # proof rounded down
    rd_T = numpy.floor(temp_F) #rounded down temp in (deg F)
    
    temp_plus_1 = rd_T + 1
    proof_plus_1 = rd_proof +1 
    
    #this is really stupid and you don't need to do this...
    temp_idx = numpy.where(ttb1_xs == rd_T)[0][0]
    proof_idx = numpy.where(ttb1_ys == rd_proof)[0][0]
    
    f = ttb1[proof_idx,temp_idx] #the proof at the rounded down value
    
    f_proof_plus_1 = ttb1[proof_idx+ 1,temp_idx ]
    f_temp_plus_1 = ttb1[proof_idx,temp_idx + 1]
    
    # print(f)
    
    """
    df    True Proof at 1 Apparent Proof Higher - True Proof at rounded down Proof
    -- = ---------------------------------------------------------------------------
    dC      +1  Apparent Proof - Rounded Down Apparent Proof
    
    
    Same for df / dT except denominator is temperature 
    """
    
    
    df_dC = (f_proof_plus_1 - f) / (proof_plus_1 - rd_proof) # take small difference between change lower and higher proof
    df_dT = (f_temp_plus_1 - f) / (temp_plus_1 - rd_T) # take small difference between higher and lower temp values
    
    # print(df_dC)
    # print(df_dT)
    
    the_true_proof = f + (calculated_proof - rd_proof) * df_dC + (temp_F - rd_T) * df_dT
    
    
    #bilinear interpolation is wrong see gauging alcohol for the real shit.
    # ttb1_interpolate = scipy.interpolate.interp2d(ttb1_xs,ttb1_ys,ttb1,kind='linear') #generate
    #bilinear interpolator using the above defined shit.
    
    # the_true_proof = ttb1_interpolate(temp_F,calculated_proof) # calculate true proof. 
    # print(calculated_proof)
    # print(the_true_proof)
    """
    now we need mole fraction etc
    """
    # interpolate to get alcohol content, arbitrary volume
    proofen_boozen_interp = scipy.interpolate.interp1d(proof_data,booze_data)
    interp_proof_sg = scipy.interpolate.interp1d(proof_data,sg_data)
    alcohol_content = proofen_boozen_interp(the_true_proof) * u.mL #volume
    water_content = 100 * u.mL - alcohol_content
    
    #compute volume fraction for fun
    etOH_v_frac = alcohol_content / (water_content + alcohol_content)
    water_v_frac = 1 - etOH_v_frac
    
    
    # print('Alcohol per 100 mL',alcohol_content)
    
    #recalculate specific gravity
    true_sg = interp_proof_sg(the_true_proof)
    
    #get real density from SG
    true_etOH_density = true_sg * ref_water_density
    
    #calculate mass frac for fun
    etOH_mass_frac = alcohol_content * true_etOH_density /(alcohol_content * true_etOH_density + water_content * ref_water_density)
    water_mass_frac = 1 - etOH_mass_frac
    
    #calculate moles
    etOH_mol = alcohol_content * (true_etOH_density/mw_etOH)
    # print(etOH_mol.to(u.mol))
    water_mol = water_content * (ref_water_density/mw_water)
    
    #finally calculate mole fraction -- almost everything above was unnecessary.
    etOH_mol_frac = etOH_mol / (etOH_mol + water_mol)
    etOH_mol_frac.ito_base_units()
    etOH_mol_frac = etOH_mol_frac.magnitude
    water_mol_frac = 1 - etOH_mol_frac
    # print(calculated_proof)
    return {'true_proof':the_true_proof,
            'ethanol_mole_frac':etOH_mol_frac,
            'ethanol_mass_frac':etOH_mass_frac.magnitude,
            'density':true_etOH_density.magnitude,
            'ethanol_moles':(etOH_mol.to(u.mol)).magnitude}


print(true_proof(31,850))






