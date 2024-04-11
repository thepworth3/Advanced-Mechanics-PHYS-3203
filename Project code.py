# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:11:13 2024

@author: thoma
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as smp
from scipy.integrate import odeint


# plt.style.use('seaborn-v0_8-poster')  


M = 100  #our gyroscope in the first year lab is probably around 4 pounds
g = 9.81
R = 10
I = 100
                                          #SI units!!!!!!
pphi = 2000        
ppsi = 1000       #like 1 makes it spikey, like 1000 makes it smooth


#defines the vector, V,  of the two coupled equations
def dVdt(t, V):
    the, pthe = V           #we verified this equation
    return [pthe/I,
           M*g*R*np.sin(the) +( (pphi**2 + ppsi**2)/I )*(np.cos(the)/np.sin(the))*(1/np.sin(the))**2 + pphi*ppsi/I*( 1/np.sin(the) -2*(1/np.sin(the))**3 )   ]
the_0 = 1.0
pthe_0 = 0         #how diff is trajectory in time based on the IC
V_0 = (the_0, pthe_0)    #define the vector of inititial conditions

t=np.linspace(0,2,1000)  #define the time range from _ to _ seconds with _ points inbetween
#record all this info in report, 1000 data points, ICs etc. etc.
#play with ICs to get theta to stay basically constant

sol1 = odeint(dVdt, y0=V_0, t=t, tfirst=True)

the_sol1 = sol1.T[0]    #picks off each variable from the solution vector sol, above
pthe_sol1 = sol1.T[1]


plt.plot(t, the_sol1, "--.", label = "theta")
# plt.legend("center right")
plt.xlabel("time (s)")
plt.ylabel(r"$\theta$ (rads)")
plt.text(1.75,1.1, r"$\theta$ = 1", fontsize = 18)
plt.text(1.75,1.05, r"p$_{\theta}$ = 0", fontsize = 18)
plt.show()

plt.plot(t, pthe_sol1, "--.", label = "ptheta")   #fix label to latex later
# plt.legend()
plt.xlabel("time (s)")
plt.ylabel(r"p$_{\theta}$ (kgm$^2$/s)")
plt.text(1.65,-360, r"$\theta$ = 1", fontsize = 18)
plt.text(1.65,-400, r"p$_{\theta}$ = 0", fontsize = 18)

plt.show()

# plt.plot(the_sol, pthe_sol, ".")     #this isn't right. How do I pick a time???

# print(max(the_sol)) #just to see things are changing


######Now copy paste this and slightly change initial conditions so I can make new arrays

the_0 = 1.2
pthe_0 = 0         #how diff is trajectory in time based on the IC
V_0 = (the_0, pthe_0)    #define the vector of inititial conditions



sol2 = odeint(dVdt, y0=V_0, t=t, tfirst=True)

the_sol2 = sol2.T[0]    #picks off each variable from the solution vector sol, above
pthe_sol2 = sol2.T[1]


plt.plot(t, the_sol2, "--.", label = "theta")
# plt.legend()
plt.xlabel("time (s)")
plt.ylabel(r"$\theta$ (rads)")
plt.text(1.85,1.25, r"$\theta$ = 1.2", fontsize = 18)
plt.text(1.85,1.2255, r"p$_{\theta}$ = 0", fontsize = 18)
plt.show()

plt.plot(t, pthe_sol2, "--.", label = "ptheta")   #fix label to latex later
# plt.legend()
plt.xlabel("time (s)")
plt.ylabel(r"p$_{\theta}$ (kgm$^2$/s)")
plt.text(1.7,-135, r"$\theta$ = 1.2", fontsize = 18)
plt.text(1.7,-150, r"p$_{\theta}$ = 0", fontsize = 18)
plt.show()
# print(max(the_sol)) #just to see things are changing
#############################


the_0 = 0.9
pthe_0 = 0         #how diff is trajectory in time based on the IC
V_0 = (the_0, pthe_0)    #define the vector of inititial conditions



sol3 = odeint(dVdt, y0=V_0, t=t, tfirst=True)

the_sol3 = sol3.T[0]    #picks off each variable from the solution vector sol, above
pthe_sol3 = sol3.T[1]
# print(the_sol3)

plt.plot(t, the_sol3, "--.", label = "theta")
# plt.legend()
plt.xlabel("time (s)")
plt.ylabel(r"$\theta$ (rads)")
plt.text(1.675,1, r"$\theta$ = 0.9", fontsize = 18)
plt.text(1.65,0.9, r"p$_{\theta}$ = 0", fontsize = 18)
plt.show()

plt.plot(t, pthe_sol3, "--.", label = "ptheta")   #fix label to latex later
# plt.legend()
plt.xlabel("time (s)")
plt.ylabel(r"p$_{\theta}$ (kgm$^2$/s)")
plt.text(1.6,-700, r"$\theta$ = 0.9", fontsize = 18)
plt.text(1.6,-600, r"p$_{\theta}$ = 0", fontsize = 18)
plt.show()
# print(max(the_sol)) #just to see things are changing
###############################

the_0 = 1.1
pthe_0 = 0         #how diff is trajectory in time based on the IC
V_0 = (the_0, pthe_0)    #define the vector of inititial conditions



sol4 = odeint(dVdt, y0=V_0, t=t, tfirst=True)

the_sol4 = sol4.T[0]    #picks off each variable from the solution vector sol, above
pthe_sol4 = sol4.T[1]


plt.plot(t, the_sol4, "--.", label = "theta")
# plt.legend()
plt.xlabel("time (s)")
plt.ylabel(r"$\theta$ (rads)")
plt.text(1.8,1.15, r"$\theta$ = 1.1", fontsize = 18)
plt.text(1.8,1.1, r"p$_{\theta}$ = 0", fontsize = 18)
plt.show()

plt.plot(t, pthe_sol4, "--.", label = "ptheta")   #fix label to latex later
# plt.legend()
plt.xlabel("time (s)")
plt.ylabel(r"p$_{\theta}$ (kgm$^2$/s)")
plt.text(1.7,-300, r"$\theta$ = 1.1", fontsize = 18)
plt.text(1.75,-400, r"p$_{\theta}$ = 0", fontsize = 18)
plt.show()
# print(max(the_sol)) #just to see things are changing




###################
#Now that we have 4 different sets of IC's, lets make our arrays
#We want each time to have the same colour
#Put the data table in my report (appendix maybe)

#notation: t0: t = 0s array, t1: t = 1s array, t1_5: t = 1.5s array
# can append each sol array (the first column and pthe second column and then plot each)

#since we have 1000 times and 2s intevral, we can get which index of the time array coresponds to each solution 

# print(sol1)
# print(t)

thet0 = []
pthet0 = []
print("time is", t[1])
thet0.append(the_sol1[1])
thet0.append(the_sol2[1])
thet0.append(the_sol3[1])
thet0.append(the_sol4[1])

pthet0.append(pthe_sol1[1])
pthet0.append(pthe_sol2[1])
pthet0.append(pthe_sol3[1])
pthet0.append(pthe_sol4[1])


# plt.plot(thet0, pthet0)
# print(thet0)
# print(pthet0)


thet1 = []  #500th point
pthet1 = [] 
print("time is", t[500])

thet1.append(the_sol1[500])
thet1.append(the_sol2[500])
thet1.append(the_sol3[500])
thet1.append(the_sol4[500])

pthet1.append(pthe_sol1[500])
pthet1.append(pthe_sol2[500])
pthet1.append(pthe_sol3[500])
pthet1.append(pthe_sol4[500])


thet1_5 = [] #750th point
pthet1_5 = []
print("time is", t[750])

thet1_5.append(the_sol1[750])
thet1_5.append(the_sol2[750])
thet1_5.append(the_sol3[750])
thet1_5.append(the_sol4[750])

pthet1_5.append(pthe_sol1[750])
pthet1_5.append(pthe_sol2[750])
pthet1_5.append(pthe_sol3[750])
pthet1_5.append(pthe_sol4[750])


thet2 = []   #1000th point
pthet2 = []
print("time is", t[999])

thet2.append(the_sol1[999])
thet2.append(the_sol2[999])
thet2.append(the_sol3[999])
thet2.append(the_sol4[999])

pthet2.append(pthe_sol1[999])
pthet2.append(pthe_sol2[999])
pthet2.append(pthe_sol3[999])
pthet2.append(pthe_sol4[999])

plt.plot(thet0,pthet0, "ro", label = "t = 0.002s")
plt.plot(thet1,pthet1, "bo", label = "t = 1.001s")
plt.plot(thet1_5,pthet1_5, "go", label = "t = 1.501s")
plt.plot(thet2,pthet2, "mo", label = "t = 2.0s")
plt.xlabel(r"$\theta$ (rads)")
plt.ylabel(r"p$_{\theta}$ (kgm$^2$/s)")
plt.legend()

#make sure to add time stamp to legend, label t0, t1 etc.






