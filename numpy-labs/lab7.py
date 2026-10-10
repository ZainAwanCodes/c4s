import numpy as np 
m = np.array(
    [[10,20,30],
     [40,50,60],
     [70,80,90]])
print(m.shape) #Shape
print(m.ndim) #Dimensions
print(m[1][1]) #Print 50
print(m[1,:]) #Print 2nd row
print(m[:,2]) #Print 3rd Col
print(m.mean()) #Mean
print(m.mean(axis=0)) #Col means
