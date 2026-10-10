import numpy as np
sensor = np.array([22.4, 22.7, 23.1, 30.5, 23.0, 22.8, 31.2, 22.9])
print(sensor.mean()) #Average
count=0
for i in sensor:
    if i > 30:
        print(i) #Print readings >30
        count+=1
print(count)
sensor = np.where(sensor > 30, 30, sensor)
print(sensor)
print(sensor.mean())