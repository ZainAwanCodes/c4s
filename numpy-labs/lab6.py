import numpy as np
a = np.array([5,10,15,20,25,30,35,40,45,50])
print(a[0]) #First Element
print(a[-1]) #Last Element
print(a[0:5]) #First 5 Elements
print(a[2:6]) #Elements from index 2 to 6
print(a[1::2]) #Every 2nd Value
a = np.flip(a)
print(a) #Reversed Array