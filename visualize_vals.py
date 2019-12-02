import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import cm
#import importlib
#from mpl_toolkits.mplot3d import axes3d, Axes3D

#fig = plt.figure()
#ax = fig.gca(projection='3d')

vals = np.load('char_scores.npy')
print vals.shape
X = np.arange(0, vals.shape[1], 1)
Y = np.arange(0, vals.shape[2], 1)
Z = vals[1, :, :]

print divmod(Z.argmax(), Z.shape[1])

#print Z

# plot surface
#surf = ax.plot_surface(X,Y,Z, cmap = cm.coolwarm, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()

#plt.imshow(Z)
#plt.show()
for i in range(vals.shape[0]):
	print i
	for j in range(vals.shape[2]):
    		for k in range(vals.shape[3]):
        		plt.plot(vals[i,:,j,k])
	plt.savefig('visualize_im_'+str(i)+'.png')
	plt.clf()
#plt.show()
'''
Q = np.zeros((vals.shape[0], vals.shape[2], vals.shape[3]))
for j in range(vals.shape[2]):
    for k in range(vals.shape[3]):
        Q[0,j,k] = np.sum(vals[0,:,j,k] > 0.5)

plt.imshow(Q[0,:,:])
plt.show()
'''

