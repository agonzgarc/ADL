import matplotlib.pyplot as plt
import numpy as np
import json
import pdb

data_dir = '/home/abel/Documents/graphics/ADL/curves/'
save_name = 'Rnd-Lst'

budget = 3200

json_files = [
    '/home/abel/DATA/faster_rcnn/resnet50_coco/performances/Rnd-Total.json',
    '/home/abel/DATA/faster_rcnn/resnet50_coco/performances/Lst-Total.json']

labels = ['Random', 'Least Confidence']
colors = ['b',[1,.4,.7]]
linestyles = ['-','-']

num_curves = len(json_files)


num_cycles = [6,6]
num_runs = [5,5]


first_cycle = 0

points = np.array(range(np.max(num_cycles)+1))*budget

for i in range(num_curves):
    with open(json_files[i]) as f:
        data = json.load(f)

    rnd_avg = np.zeros(num_cycles[i]+1)
    rnd_std = np.zeros(num_cycles[i]+1)

    for cycle in range(first_cycle,num_cycles[i]+1):
        ap_cycle = np.array([data['R'+str(r)+'c'+str(cycle)][0] for r in range(1,num_runs[i]+1)])
        rnd_avg[cycle] = ap_cycle.mean()
        rnd_std[cycle] = ap_cycle.std()

    plt.errorbar(points[first_cycle:num_cycles[i]+1],rnd_avg[first_cycle:],rnd_std[first_cycle:],color=colors[i],label=labels[i],linestyle=linestyles[i],marker='o',capsize=2)

#for i in range(num_curves):
    #with open(json_files[i]) as f:
        #data = json.load(f)

    #for r in range(1,num_runs[i]+1):
        #ap_cycle = np.zeros(num_cycles[i]+1)
        #for cycle in range(num_cycles[i]+1):
            #print(cycle)
            #ap_cycle[cycle] = data['R'+str(r)+'c'+str(cycle)][0]
        #plt.plot(points[:num_cycles[i]+1],ap_cycle,label=labels[i]+str(r),linestyle=linestyles[i],marker='o')



plt.legend()

plt.xticks(points[first_cycle:],range(first_cycle,np.max(num_cycles)+1))
plt.xlabel('Cycle')
#plt.xlabel('Number of frames/video')
plt.ylabel('mAP')

plt.savefig(data_dir+save_name)

plt.show()
