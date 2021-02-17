import scipy.interpolate
import numpy
import pandas

"""
Copy Pasta from this part
"""

user_sg = 0.851145 #specific gravity
temp_F = 87.8 #degF

ttb6_raw = pandas.read_excel('TTB_Table_6_digitized.xlsx',header=5, engine='openpyxl') # Read in TTB6

ttb6_raw = ttb6_raw.dropna(axis=1) #get rid of first column

ttb6 = numpy.array(ttb6_raw,dtype=float) #Convert to Array

sg_data = ttb6[:,4] #get the specific gravity data 
proof_data = ttb6[:,0] #get the embenzalmine nitrotomine proof data

interp_sg_proof = scipy.interpolate.interp1d(sg_data,proof_data)

calculated_proof = interp_sg_proof(user_sg)

ttb1_raw = pandas.read_excel('TTB_Table_1_digitized.xlsx',header=5, engine='openpyxl') #read in ttb1

ttb1_raw = ttb1_raw.fillna(0) #nans mess with the interpolation so lets bump em.

ttb1 = numpy.array(ttb1_raw,dtype=float) #convert ttb1 to array

ttb1 = ttb1[:,1:] #exclude column 1

ttb1_y_size,ttb1_x_size = numpy.shape(ttb1)  #get the x and y shape of ttb1

ttb1_ys = numpy.arange(0,ttb1_y_size) #create an array of ints based on shape of ttb1
ttb1_xs = numpy.arange(1,ttb1_x_size+1)

proof_rd = numpy.floor(calculated_proof) #get the proof value rounded down
T_rd = numpy.floor(temp_F) # get the temp in Freedom Units rounded down

temp_idx = numpy.where(ttb1_xs == T_rd )[0][0] #find where the proof array is equal to our rounded Temp
proof_idx = numpy.where(ttb1_ys == proof_rd)[0][0] #same for proof

f = ttb1[proof_idx,temp_idx] #get this value  from ttb1



temp_plus_1 = T_rd + 1 # following along with that pdf 
proof_plus_1 = proof_rd + 1

f_proof_plus_1 = ttb1[proof_idx + 1, temp_idx] 
f_temp_plus_1 = ttb1[proof_idx,temp_idx + 1]

"""
End Here
"""

alpha = 15E-6

# df_dC = (f_proof_plus_1 - f) / (proof_plus_1 - proof_rd) 

dAP_dTP = (proof_plus_1 - proof_rd) / (f_proof_plus_1 - f)
dTP_dT = (f_temp_plus_1 - f) / (temp_plus_1 - T_rd)

# the_true_proof = f + (calculated_proof - proof_rd) * df_dC + (temp_F - T_rd) * df_dT

# drho_dAP = 0.99904 * ()

print(calculated_proof)
print(the_true_proof)
# Thants & Blants
