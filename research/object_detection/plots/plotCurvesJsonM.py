import matplotlib.pyplot as plt
import numpy as np
import json


data_dir = '/home/abel/Documents/graphics/ADL/curves/'
save_name = '10-rnd'

num_videos = 398
points = np.array(range(1,11))*num_videos

json_files = [
    '/home/abel/DATA/faster_rcnn/resnet101_coco/performances/cluster/RndFromScratchr3c10.json',
    '/home/abel/DATA/faster_rcnn/resnet101_coco/performances/cluster/Rnd20KR1r1c10-total.json']

labels = ['Random-40K (pre)','Random-20K (pre)']
colors = ['b','b']
linestyles = ['--','-']

num_curves = len(json_files)


for i in range(num_curves):
    with open(json_files[i]) as f:
        data = json.load(f)

    rnd_avg = np.zeros(10)
    rnd_std = np.zeros(10)

    for cycle in range(1,11):
        ap_cycle = np.array([data['run'+str(r)+'c'+str(cycle)][0] for r in range(1,4)])
        rnd_avg[cycle-1] = ap_cycle.mean()
        rnd_std[cycle-1] = ap_cycle.std()

    plt.errorbar(points,rnd_avg,rnd_std,color=colors[i],label=labels[i],linestyle=linestyles[i],marker='o',capsize=2)



plt.legend()

#plt.xticks(points,[str(f)+' f/v' for f in range(1,11)])
plt.xticks(points,range(1,11))
plt.xlabel('Number of frames/video')
plt.ylabel('mAP')

plt.savefig(data_dir+save_name)

plt.show()


