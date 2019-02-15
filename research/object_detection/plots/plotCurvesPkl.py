import matplotlib.pyplot as plt
import numpy as np
import json
import pdb
import pickle

data_dir = '/home/abel/Documents/graphics/ADL/curves/'

all_metrics = [['DetectionBoxes_Precision/mAP', 'DetectionBoxes_Precision/mAP@.50IOU','DetectionBoxes_Precision/mAP@.75IOU'],
             ['DetectionBoxes_Precision/mAP (small)', 'DetectionBoxes_Precision/mAP (medium)','DetectionBoxes_Precision/mAP (large)'],
             ['DetectionBoxes_Recall/AR@1','DetectionBoxes_Recall/AR@10','DetectionBoxes_Recall/AR@100']]


budget = 3200

pkl_files = [
    '/home/abel/DATA/faster_rcnn/resnet50_coco/performances/imagenet/Rnd',
    '/home/abel/DATA/faster_rcnn/resnet50_coco/performances/imagenet/Lst',
    '/home/abel/DATA/faster_rcnn/resnet50_coco/performances/imagenet/EntAvg',
    '/home/abel/DATA/faster_rcnn/resnet50_coco/performances/imagenet/TCFP']


labels = ['Random', 'Least Confidence (Avg)', 'Entropy (Avg)', 'TCFP', 'Least Confidence (Max)', 'Entropy (Avg)']

colors = ['b',[1,.4,.7],'r','c',[1,.4,.7]]
linestyles = ['-','-','-','-']

num_cycles = [6,6,6,6]
num_runs = [5,5,5,5]

#points = np.array(range(np.max(num_cycles)))*budget
points = np.array(range(np.max(num_cycles)))

#num_curves = len(pkl_files)
num_curves = 4



for metric_set in range(3):

    all_metrics = all_metrics[metric_set]
    print(all_metrics)


    save_name = 'Rnd-Lst-TCFP-{}.png'.format(metric_set)

    fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(10,3))

    row=col=0


    for metric in all_metrics:
        metric_name = metric.split('/')
        print(metric_name)
        metric_name = metric_name[1]

        for i in range(num_curves):
            values = np.zeros((np.max(num_runs),num_cycles[i]))
            for r in range(num_runs[i]):
                with open(pkl_files[i]+'R{}c6.pkl'.format(r+1),'rb') as f:
                    data = pickle.load(f,encoding='latin1' )

                for cycle in range(1,num_cycles[i]+1):
                    values[r,cycle-1] = data['R'+str(r+1)+'c'+str(cycle)][metric]

            val_avg = values.mean(axis=0)
            val_std = values.std(axis=0)

            #axes[row,col].errorbar(points[:num_cycles[i]+1],val_avg,val_std,color=colors[i],label=labels[i],linestyle=linestyles[i],marker='o',capsize=2,linewidth=0.7, markersize=0.7)
            axes[col].errorbar(points[:num_cycles[i]+1],val_avg,val_std,color=colors[i],label=labels[i],linestyle=linestyles[i],marker='o',capsize=2,linewidth=1.5, markersize=4)


        axes[col].set_xticks(points,range(1,np.max(num_cycles)+1))
        axes[col].set_xlabel('Cycle')
        axes[col].set_title(metric_name)
        #axes[col].set(adjustable='datalim',aspect=1)

        col +=1
        if col >2:
            col = 0
            row += 1


    plt.legend()

#plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.90, hspace=0.5,
                        #wspace=0.4)
    plt.subplots_adjust(top=0.80, bottom=0.15, left=0.05, right=0.95, hspace=0.2, wspace=0.3)
    plt.savefig(data_dir+save_name)

#for metric in all_metrics:
##metric = all_metrics[0]
    #metric_name = metric.split('/')
    #metric_name = metric_name[1]

    #save_name = 'Rnd-Lst-Ent-' + metric_name + '.png'


    #colors = ['b',[1,.4,.7], 'r' ,'c']
    #linestyles = ['-','-','-']

        ##num_curves = len(pkl_files)
    #num_curves = 3


    #num_cycles = [6,6,6]
    #num_runs = [5,5,5]


    #points = np.array(range(np.max(num_cycles)))*budget

    #fig,ax = plt.subplots()

    #for i in range(num_curves):
        #values = np.zeros((np.max(num_runs),num_cycles[i]))
        ##rnd_std = np.zeros((np.max(num_runs),num_cycles[i]+1))
        #for r in range(num_runs[i]):
            #with open(pkl_files[i]+'R{}c6.pkl'.format(r+1),'rb') as f:
                #data = pickle.load(f,encoding='latin1' )

            #for cycle in range(1,num_cycles[i]+1):
                #values[r,cycle-1] = data['R'+str(r+1)+'c'+str(cycle)][metric]

        #print(values)

        #val_avg = values.mean(axis=0)
        #val_std = values.std(axis=0)

        #ax.errorbar(points[:num_cycles[i]+1],val_avg,val_std,color=colors[i],label=labels[i],linestyle=linestyles[i],marker='o',capsize=2)


    #ax.legend()

    #plt.xticks(points,range(1,np.max(num_cycles)+1))
    #plt.xlabel('Cycle')
##plt.xlabel('Number of frames/video')
    #plt.ylabel(metric_name)

    #plt.savefig(data_dir+save_name)

