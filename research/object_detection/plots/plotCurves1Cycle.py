import matplotlib.pyplot as plt
import numpy as np

data_dir = '/home/abel/Documents/graphics/ADL/curves/'
num_videos = 398
maxAP = 81.76

points = np.array([1, 2, 4])*num_videos
rnd_all = np.array([[74.83,75.25,77.13],
                   [74.7,75.98,77.89],
                    [73.88,75.96,77.71]])

ent_all = np.array([[73.70,75.78],
                    [73.84,76.00],
                    [74.28,75.96]])

drp_all = np.array([[74.17,76.35],
                    [75.30,76.46],
                    [75.31,76.52]])

tcfp_all = np.array([[74.26,76.98]])

#ent_all = np.array([[[72.90,72.16,79.43],[72.12,71.86,80.12],[70.21,70.89,80.53]],
                    #[[73.92,72.18,82.28],[73.64,72.00,82.38],[73.34,72.65,81.89]],
                    #[[74.85,72.67,82.13],[74.63,72.48,82.75],[75.20,71.58,82.99]]])

##drp_all = np.array([[[72.07,71.23,79.20],[72.67,72.63,80.61],[72.20,73.34,80.40]],
#drp_all = np.array([[[72.90,72.16,79.43],[72.12,71.86,80.12],[70.21,70.89,80.53]],
                    #[[72.93,72.55,79.99],[71.91,72.55,81.92],[72.74,73.19,82.64]],
                    #[[72.51,71.69,81.91],[72.13,71.98,83.04],[72.68,72.03,82.46]]])
#tcfp_all = np.array([[[72.90,72.16,79.43],[72.12,71.86,80.12],[70.21,70.89,80.53]],
                     #[[74.90,72.17,80.14],[75.29,71.99,81.16],[74.67,72.12,82.04]],
                     #[[76.00,72.39,83.44],[76.06,73.23,81.75],[76.25,71.93,83.37]]])


#ent_avg = np.zeros(3)
#ent_std = np.zeros(3)
#drp_avg = np.zeros(3)
#drp_std = np.zeros(3)
#tcfp_avg = np.zeros(3)
#tcfp_std = np.zeros(3)

rnd_avg = rnd_all.mean(axis=0)
rnd_std = rnd_all.std(axis=0)
ent_avg = ent_all.mean(axis=0)
ent_std = ent_all.std(axis=0)
drp_avg = drp_all.mean(axis=0)
drp_std = drp_all.std(axis=0)
tcfp_avg = tcfp_all.mean(axis=0)
tcfp_std = tcfp_all.std(axis=0)

plt.errorbar(points,rnd_avg,rnd_std,color='b',label='Random',marker='o',capsize=2)
plt.errorbar(points[:-1],ent_avg,ent_std,color=(1.0,0.5,0),label='Entropy',marker='o',capsize=2)
plt.errorbar(points[:-1],drp_avg,drp_std,color='r',label='Dropout',marker='o',capsize=2)
#plt.errorbar(points[:-1],tcfp_avg,tcfp_std,color='c',label='TCFP-3N',marker='o',capsize=2)
#plt.plot([points[0],points[-1]], [maxAP,maxAP],color='k',linestyle='dashed')

plt.legend()

plt.xticks(points,['1 f/v','2 f/v', '4 f/v'])
plt.xlabel('Number of frames')
plt.ylabel('mAP')

plt.savefig(data_dir+'baselines1-4')
#plt.savefig(data_dir+'baselinesWOurs1-4')

plt.show()


