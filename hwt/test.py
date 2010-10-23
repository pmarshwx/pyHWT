from __future__ import print_function, division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import hwt
import time

x = 101
y = 101
dx = 4.
roi = 40.

fcst1 = np.zeros([101, 101], dtype=np.float64)
obs1 = np.zeros([101, 101], dtype=np.float64)
"""
fcst1[48, 48] = 1
obs1[52, 52] = 1
"""
fcst1[40:60, 49:51] = 1
obs1[49:51, 40:60] = 1

fcst2 = hwt.hitormiss(fcst1, 1., roi, dx)
obs2 = hwt.hitormiss(obs1, 1., roi, dx)

a1, b1, c1, d1 = hwt.getContingency(fcst1, obs1)
a2, b2, c2, d2 = hwt.getContingency(fcst2, obs1)
a3, b3, c3, d3 = hwt.getContingency(fcst2, obs2)
a4, b4, c4, d4 = hwt.getRadiusContingency(fcst1, obs1, roi, dx)
#a5, b5, c5, d5 = hwt.getRadiusContingency(fcst2, obs2, roi, dx)


print('raw:raw', a1, b1, c1, d1, a1/(a1+c1), b1/(a1+b1), b1/(b1+d1), a1/(a1+b1+c1), (a1+b1)/(a1+c1))
print('40km:raw', a2, b2, c2, d2, a2/(a2+c2), b2/(a2+b2), b2/(b2+d2), a2/(a2+b2+c2), (a2+b2)/(a2+c2))
print('40km:40km', a3, b3, c3, d3, a3/(a3+c3), b3/(a3+b3), b3/(b3+d3), a3/(a3+b3+c3), (a3+b3)/(a3+c3))
print('raw:raw:roi', a4, b4, c4, d4, a4/(a4+c4), b4/(a4+b4), b4/(b4+d4), a4/(a4+b4+c4), (a4+b4)/(a4+c4))
#print('40km:40km:roi', a5, b5, c5, d5, a5/(a5+c5), b5/(a5+b5), b5/(b5+d5), #a5/(a5+b5+c5), (a5+b5)/(a5+c5))

print(101*101 - (a4+b4+c4+d4))


fig = plt.figure(figsize=(10,10))
plt.subplot(2,3,1)
plt.imshow(fcst1)
plt.subplot(2,3,2)
plt.imshow(obs1)
plt.subplot(2,3,3)
plt.imshow(fcst1+obs1)
plt.subplot(2,3,4)
plt.imshow(fcst2)
plt.subplot(2,3,5)
plt.imshow(obs2)
plt.subplot(2,3,6)
plt.imshow(fcst2+obs2)
plt.show()



"""
dt2 = time.time()
c = hwt.gauss_smooth(b, 120., 4., 5.)
print(time.time() - dt2)

print(a.mean(), b.mean(), c.mean())
fig = plt.figure(figsize=(20,8))
plt.subplot(1,3,1)
plt.imshow(a)
plt.subplot(1,3,2)
plt.imshow(b)
plt.subplot(1,3,3)
plt.imshow(c)
plt.show()
"""
