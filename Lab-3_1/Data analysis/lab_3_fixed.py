#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:06:31 2020

@author: tfinney
"""


import scipy.interpolate
import numpy
import pandas


ttb6_raw = pandas.read_excel('TTB_Table_6_digitized.xlsx',header=5, engine='openpyxl')

ttb6_raw = ttb6_raw.dropna(axis=1)

ttb6 = numpy.array(ttb6_raw)

density = 0.85 
ref_density = 0.99904

sg = density/ref_density

sg_data = ttb6[:,4]
proof_data = ttb6[:,0]
booze_data = ttb6[:,1]
water_data = ttb6[:,2]

interp_sg_proof = scipy.interpolate.interp1d(sg_data,proof_data)

calculated_proof = interp_sg_proof(sg)

temp_F = 75

ttb1_raw = pandas.read_excel('TTB_Table_1_digitized.xlsx',header=5, engine='openpyxl')
ttb1_raw = ttb1_raw.fillna(0)

temperature_header = numpy.arange(1,101)

ttb1 = numpy.array(ttb1_raw,dtype=float)


ttb1 = ttb1[:,1:]

ttb1_y_size,ttb1_x_size = numpy.shape(ttb1)

ttb1_ys = numpy.arange(0,ttb1_y_size)
ttb1_xs = numpy.arange(0,ttb1_x_size)

ttb1_interpolate = scipy.interpolate.interp2d(ttb1_xs,ttb1_ys,ttb1,kind='cubic')

the_true_proof = ttb1_interpolate(temp_F,calculated_proof)



