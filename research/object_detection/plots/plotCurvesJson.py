import matplotlib.pyplot as plt
import numpy as np
import json


data_dir = '/home/abel/Documents/graphics/ADL/curves/'
num_videos = 398

json_file = '/home/abel/DATA/faster_rcnn/resnet101_coco/performances/Rndr3c10-Total.json'

with open(json_file) as f:
    data = json.load(f)

print(data['run2c6'][0])

points = np.array(range(1,11))*num_videos


rnd_avg = np.zeros(10)
rnd_std = np.zeros(10)

for cycle in range(1,11):
    ap_cycle = np.array([data['run'+str(r)+'c'+str(cycle)][0] for r in range(1,4)])
    rnd_avg[cycle-1] = ap_cycle.mean()
    rnd_std[cycle-1] = ap_cycle.std()


plt.errorbar(points,rnd_avg,rnd_std,color='b',label='Random',marker='o',capsize=2)
#plt.errorbar(points,ent_avg,ent_std,color=(1.0,0.5,0),label='Entropy',marker='o',capsize=2)
#plt.errorbar(points,drp_avg,drp_std,color='r',label='Dropout',marker='o',capsize=2)
#plt.errorbar(points,tcfp_avg,tcfp_std,color='c',label='TCFP-3N',marker='o',capsize=2)

plt.legend()

#plt.xticks(points,[str(f)+' f/v' for f in range(1,11)])
plt.xticks(points,range(1,11))
plt.xlabel('Number of frames/video')
plt.ylabel('mAP')

plt.savefig(data_dir+'10cycles-rnd')

plt.show()


