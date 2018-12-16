import matplotlib.pyplot as plt
import numpy as np
import json
import pdb

data_dir = '/datatmp/Experiments/Javad/tf/data/ILSVRC/performances/'
save_name = 'FP_vs_FN'

#num_videos = 398
#points = np.array(range(1,6))*num_videos


json_files = [
    #'/home/abel/DATA/faster_rcnn/resnet101_coco/performances/cluster/RndFromScratchr3c10.json',
    #'/home/abel/DATA/faster_rcnn/resnet101_coco/performances/cluster/Rnd20KR1r1c10-total.json',
    #'/datatmp/Experiments/Javad/tf/data/ILSVRC/performances/RndClean-total.json',
    #'/datatmp/Experiments/Javad/tf/data/ILSVRC/performances/RndxVidFrom0-Total.json',
    #'/datatmp/Experiments/Javad/tf/data/ILSVRC/performances/TCFP_total.json',
    '/datatmp/Experiments/Javad/tf/data/ILSVRC/performances/FP_gtR1c20.json',
    '/datatmp/Experiments/Javad/tf/data/ILSVRC/performances/FN_gtR1c20.json',
    '/datatmp/Experiments/Javad/tf/data/ILSVRC/performances/RndR1c20.json']

#labels = ['Random','Random_with_freezing','TCFP','False Positive','False Negative']
labels = ['False Positive','False Negative','Random']

colors = ['b','c','r','g']
linestyles = ['-','-','-','-']

num_curves = len(json_files)

for i in range(num_curves):

    with open(json_files[i]) as f:
        data = json.load(f)

    if i==0:
      rnd_avg = np.zeros(8)
      rnd_std = np.zeros(8)
      for cycle in range(1,9):	
       print(i)
       ap_cycle = np.array([data['R'+str(r)+'c'+str(cycle)][0] for r in range(1,2)])
       rnd_avg[cycle-1] = ap_cycle.mean()
       rnd_std[cycle-1] = ap_cycle.std()
      points=range(1,9)
      plt.errorbar(points,rnd_avg,rnd_std,color=colors[i],label=labels[i],linestyle=linestyles[i],marker='o',capsize=2)

    if i==1:
      rnd_avg = np.zeros(10)
      rnd_std = np.zeros(10)
      for cycle in range(1,11):	
       print(i)
       ap_cycle = np.array([data['R'+str(r)+'c'+str(cycle)][0] for r in range(1,2)])
       rnd_avg[cycle-1] = ap_cycle.mean()
       rnd_std[cycle-1] = ap_cycle.std()
      points=range(1,11)
      plt.errorbar(points,rnd_avg,rnd_std,color=colors[i],label=labels[i],linestyle=linestyles[i],marker='o',capsize=2)

    elif i==2:
      rnd_avg = np.zeros(7)
      rnd_std = np.zeros(7)
      for cycle in range(1,8):	
       print(i)
       ap_cycle = np.array([data['R'+str(r)+'c'+str(cycle)][0] for r in range(1,2)])
       rnd_avg[cycle-1] = ap_cycle.mean()
       rnd_std[cycle-1] = ap_cycle.std()
      points=range(1,8)
      plt.errorbar(points,rnd_avg,rnd_std,color=colors[i],label=labels[i],linestyle=linestyles[i],marker='o',capsize=2)

    elif i==3:
      rnd_avg = np.zeros(8)
      rnd_std = np.zeros(8)
      for cycle in range(1,9):	
       print(i)
       ap_cycle = np.array([data['R'+str(r)+'c'+str(cycle)][0] for r in range(1,4)])
       rnd_avg[cycle-1] = ap_cycle.mean()
       rnd_std[cycle-1] = ap_cycle.std()
      points=range(1,9)
      plt.errorbar(points,rnd_avg,rnd_std,color=colors[i],label=labels[i],linestyle=linestyles[i],marker='o',capsize=2)

plt.legend()

#plt.xticks(points,[str(f)+' f/v' for f in range(1,11)])
plt.xticks(points,range(1,9))
plt.xlabel('Number of frames/video')
plt.ylabel('mAP')

plt.savefig(data_dir+save_name)

#plt.show()


