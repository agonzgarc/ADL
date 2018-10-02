import matplotlib.pyplot as plt
import numpy as np

num_videos = 398

points = np.array([1, 2, 3])*num_videos
rnd = np.array([74.83,75.39,76.55])
rndErr = np.array([0.52,0.55,0.10])
ent = np.array([74.83,76.03,76.72])
entErr = np.array([0.52,0.09,0.04])
drp = np.array([74.83,75.60,75.60])
drpErr = np.array([0.52,0.53,0.20])
tcfp3n = np.array([74.83,76.05])
tcfp3nErr = np.array([0.52,0.28])


plt.errorbar(points,rnd,rndErr,color='b',label='Random',marker='o',capsize=2)
plt.errorbar(points,ent,entErr,color=(1.0,0.5,0),label='Entropy',marker='o',capsize=2)
plt.errorbar(points,drp,drpErr,color='r',label='Dropout',marker='o',capsize=2)
plt.errorbar(points[:-1],tcfp3n,tcfp3nErr,color='c',label='TCFP-3N',marker='o',capsize=2)

plt.legend()

plt.xticks(points,['1 f/v','2 f/v', '3 f/v'])
plt.xlabel('Number of frames')
plt.ylabel('mAP')

plt.show()

## 3 points version
#points = np.array([1, 2, 3])*num_videos
#rnd = np.array([74.83,75.39,76.55])
#rndErr = np.array([0.52,0.55,0.10])
#ent = np.array([74.83,76.03,76.72])
#entErr = np.array([0.52,0.09,0.04])
#drp = np.array([74.83,75.60,75.60])
#drpErr = np.array([0.52,0.53,0.20])
#tcfp3n = np.array([74.83,76.05])
#tcfp3nErr = np.array([0.52,0.28])


#plt.errorbar(points,rnd,rndErr,color='b',label='Random',marker='o',capsize=2)
#plt.errorbar(points,ent,entErr,color=(1.0,0.5,0),label='Entropy',marker='o',capsize=2)
#plt.errorbar(points,drp,drpErr,color='r',label='Dropout',marker='o',capsize=2)
#plt.errorbar(points[:-1],tcfp3n,tcfp3nErr,color='c',label='TCFP-3N',marker='o',capsize=2)

#plt.legend()

#plt.xticks(points,['1 f/v','2 f/v', '3 f/v'])
#plt.xlabel('Number of frames')
#plt.ylabel('mAP')

#plt.show()
