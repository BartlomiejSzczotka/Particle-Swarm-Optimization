# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 22:33:13 2019

@author: Smail
"""

import random
import matplotlib.pyplot as plt
import numpy as np

#Boundaries of the contour map
x_min = -1.5
x_max = 1.5
y_min = -1
y_max = 3


def rosenbrockfunction(x,y):
    return (0-x)**2+100*(y-x**2)**2



fig = plt.figure()
ax = fig.add_subplot(111)
plt.axis([x_min, x_max, y_min,y_max])
plt.title("Plot of the last 50 generations with a progressive alpha")
ux = np.linspace(x_min,x_max,100) #linearly spaced x points between the boundaries
uy = np.linspace(y_min,y_max,100) #linearly spaced y points between the boundaries

x,y = np.meshgrid(ux,uy) #blends the coordinates into a matrix

Z = rosenbrockfunction(x,y)
cs = ax.contourf(ux,uy,Z) #plotting the contour map (filled)


x_list = [] #list of x points to be filled for the plotting
y_list = [] #list of y points to be filled for the plotting


a = 0.9 #as advised in the course material
b = random.uniform(0,0.5)#b within these boundaries gave me the best results
c = random.uniform(0,1)    

#Initial random velocities need to be bound otherwise the algorithm may never converge
#Boundaries need to be close to the space boundaries    
Vmin = -1.5
Vmax = 1.5


number_generations = 100
total_population = 30
popx = [] #array of initial population x coordinates
popy = [] #array of initial population y coordinates
velocities = [] 


#function to get the global best coordinates
#it searches for the minimum value given by an array of coordinates
def get_gbest(position_array):
    x_best,y_best = position_array[0][0],position_array[0][1]
    for pos in position_array:
        if(rosenbrockfunction(pos[0],pos[1])<rosenbrockfunction(x_best,y_best)):
            x_best,y_best = pos[0],pos[1]
    return np.vstack((x_best,y_best)).T #stacks the 2 arrays into 1 array of 2 columns

#initialization of the first coordinates and velocities
for i in range(total_population):
    popx.append(random.uniform(x_min,x_max))
    popy.append(random.uniform(y_min,y_max))
    velocities.append(random.uniform(Vmin,Vmax))


position = np.vstack((popx,popy)).T
best_position = position    #setting the best_position array to the current position array
global_best = get_gbest(best_position)

print("Starting PSO ...")

for i in range(number_generations):
    for j in range(total_population):
        r = random.uniform(0,1)    
        global_best = get_gbest(best_position)
        velocities[j] = a * velocities[j] +  b * r * (best_position[j] - position[j]) + c * r * (global_best - position[j]) #velocity update
        position[j] = position[j]+velocities[j] #position update
        if(rosenbrockfunction(position[j][0],position[j][1])<rosenbrockfunction(best_position[j][0],best_position[j][1])):
            best_position[j] = position[j]
    a = a - ((0.9 - 0.4)/number_generations) #linear decrease of "a" to have a final value of 0.4 
    #plotting part
    x_list.clear()
    y_list.clear()
    if(i > number_generations-50): #we are interested into the last 50 generations
        for p in position:
            x_list.append(p[0])
            y_list.append(p[1])
        plt.plot(x_list, y_list, 'orange',marker='.' ,linestyle='None', alpha =float(i)/100)

print("\nPSO done")
print("\nPlotting the last 50 generations : 1500 points\n")
plt.show()  
     
best = get_gbest(best_position)
print("\nX and Y coordinates of best global position of last generation : " + str(best) )

print("\nValue of these coordinates : "+ str(rosenbrockfunction(best[0][0],best[0][1])) + "\n")
  
print("a: " + str(a) + " b: " + str(b) + " c: " + str(c) )

