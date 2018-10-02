import matplotlib.pyplot as plt
import numpy as np

data_dir = '/home/abel/Documents/graphics/ADL/curves/'
num_videos = 398

num_cycles = 4

points = np.array(range(1,num_cycles+1))*num_videos

rnd_all = np.array([[[72.90,72.16,79.43],[72.12,71.86,80.12],[70.21,70.89,80.53]],
                [[73.63,72.22,82.02],[72.87,71.89,79.85],[72.88,71.98,81.15]],
                [[74.00,73.25,82.30],[73.89,71.65,80.60],[73.85,73.41,81.35]],
                [[74.81,73.81,80.88],[74.55,72.79,82.14],[75.03,73.52,82.53]]])
#ent_all = np.array([[[70.87,70.84,79.40],[69.94,72.37,79.20],[71.22,72.24,79.39]],
ent_all = np.array([[[72.90,72.16,79.43],[72.12,71.86,80.12],[70.21,70.89,80.53]],
                    [[73.92,72.18,82.28],[73.64,72.00,82.38],[73.34,72.65,81.89]],
                    [[74.38,72.50,82.23],[75.67,72.85,83.11],[75.17,72.11,82.49]],
                    [[74.84,73.28,83.21],[75.37,73.18,82.89],[74.23,72.68,82.54]]])

#drp_all = np.array([[[72.07,71.23,79.20],[72.67,72.63,80.61],[72.20,73.34,80.40]],
drp_all = np.array([[[72.90,72.16,79.43],[72.12,71.86,80.12],[70.21,70.89,80.53]],
                    [[72.93,72.55,79.99],[71.91,72.55,81.92],[72.74,73.19,82.64]],
                    [[73.13,72.26,82.25],[73.04,72.27,82.33],[73.23,71.93,83.37]],
                    [[74.49,72.76,84.34],[73.58,72.66,83.18],[73.24,72.88,83.64]]])

tcfp_all = np.array([[[72.90,72.16,79.43],[72.12,71.86,80.12],[70.21,70.89,80.53]],
                     [[74.90,72.17,80.14],[75.29,71.99,81.16],[74.67,72.12,82.04]],
                     [[76.00,72.39,83.44],[76.06,73.23,81.75],[76.25,71.93,83.37]],
                     [[77.04,72.26,83.51],[76.14,72.33,82.08],[76.86,72.54,82.99]]])

per_class = False


rnd_avg = np.zeros(num_cycles)
rnd_std = np.zeros(num_cycles)
ent_avg = np.zeros(num_cycles)
ent_std = np.zeros(num_cycles)
drp_avg = np.zeros(num_cycles)
drp_std = np.zeros(num_cycles)
tcfp_avg = np.zeros(num_cycles)
tcfp_std = np.zeros(num_cycles)


if per_class:
    class_names = ['Bike','Car','Motorbike']
    idx_class = 1

    for cycle in range(num_cycles):
        rnd_avg[cycle] = rnd_all[cycle].mean(axis=0)[idx_class]
        rnd_std[cycle] = rnd_all[cycle].std(axis=0)[idx_class]
        ent_avg[cycle] = ent_all[cycle].mean(axis=0)[idx_class]
        ent_std[cycle] = ent_all[cycle].std(axis=0)[idx_class]
        drp_avg[cycle] = drp_all[cycle].mean(axis=0)[idx_class]
        drp_std[cycle] = drp_all[cycle].std(axis=0)[idx_class]
        tcfp_avg[cycle] = tcfp_all[cycle].mean(axis=0)[idx_class]
        tcfp_std[cycle] = tcfp_all[cycle].std(axis=0)[idx_class]


    plt.errorbar(points,rnd_avg,rnd_std,color='b',label='Random',marker='o',capsize=2)
    plt.errorbar(points,ent_avg,ent_std,color=(1.0,0.5,0),label='Entropy',marker='o',capsize=2)
    plt.errorbar(points,drp_avg,drp_std,color='r',label='Dropout',marker='o',capsize=2)
    plt.errorbar(points,tcfp_avg,tcfp_std,color='c',label='TCFP-3N',marker='o',capsize=2)

    plt.legend()

    plt.xticks(points,['1 f/v','2 f/v', '3 f/v'])
    plt.xlabel('Number of frames')
    plt.ylabel('AP')
    plt.title('Class {}'.format(class_names[idx_class]))

    plt.savefig(data_dir+ str(num_cycles) + 'cycles-{}'.format(class_names[idx_class]))

    plt.show()

else:
    for cycle in range(num_cycles):

        mAP = rnd_all[cycle].mean(axis=1)
        rnd_avg[cycle] = mAP.mean()
        rnd_std[cycle] = mAP.std()
        mAP = ent_all[cycle].mean(axis=1)
        ent_avg[cycle] = mAP.mean()
        ent_std[cycle] = mAP.std()
        mAP = drp_all[cycle].mean(axis=1)
        drp_avg[cycle] = mAP.mean()
        drp_std[cycle] = mAP.std()
        mAP = tcfp_all[cycle].mean(axis=1)
        tcfp_avg[cycle] = mAP.mean()
        tcfp_std[cycle] = mAP.std()

    plt.errorbar(points,rnd_avg,rnd_std,color='b',label='Random',marker='o',capsize=2)
    plt.errorbar(points,ent_avg,ent_std,color=(1.0,0.5,0),label='Entropy',marker='o',capsize=2)
    plt.errorbar(points,drp_avg,drp_std,color='r',label='Dropout',marker='o',capsize=2)
    plt.errorbar(points,tcfp_avg,tcfp_std,color='c',label='TCFP-3N',marker='o',capsize=2)

    plt.legend()

    plt.xticks(points,[str(f) + ' f/v' for f in range(1,num_cycles+1)])
    plt.xlabel('Number of frames')
    plt.ylabel('mAP')

    plt.savefig(data_dir+ str(num_cycles) + 'cycles-all')

    plt.show()

## 3 points version
#points = np.array([1, 2, 3])*num_videos
#rnd_all = np.array([[[72.90,72.16,79.43],[72.12,71.86,80.12],[70.21,70.89,80.53]],
                #[[73.63,72.22,82.02],[72.87,71.89,79.85],[72.88,71.98,81.15]],
                #[[74.00,73.25,82.30],[74.36,72.64,82.38],[73.69,73.87,82.42]]])
##ent_all = np.array([[[70.87,70.84,79.40],[69.94,72.37,79.20],[71.22,72.24,79.39]],
#ent_all = np.array([[[72.90,72.16,79.43],[72.12,71.86,80.12],[70.21,70.89,80.53]],
                    #[[73.92,72.18,82.28],[73.64,72.00,82.38],[73.34,72.65,81.89]],
                    #[[74.38,72.50,82.23],[75.67,72.85,83.11],[75.17,72.11,82.49]]])

##drp_all = np.array([[[72.07,71.23,79.20],[72.67,72.63,80.61],[72.20,73.34,80.40]],
#drp_all = np.array([[[72.90,72.16,79.43],[72.12,71.86,80.12],[70.21,70.89,80.53]],
                    #[[72.93,72.55,79.99],[71.91,72.55,81.92],[72.74,73.19,82.64]],
                    #[[73.13,72.26,82.25],[73.04,72.27,82.33],[73.23,71.93,83.37]]])

#tcfp_all = np.array([[[72.90,72.16,79.43],[72.12,71.86,80.12],[70.21,70.89,80.53]],
                     #[[74.90,72.17,80.14],[75.29,71.99,81.16],[74.67,72.12,82.04]],
                     #[[76.00,72.39,83.44],[76.06,73.23,81.75],[76.25,71.93,83.37]]])

#per_class = False

#rnd_avg = np.zeros(3)
#rnd_std = np.zeros(3)
#ent_avg = np.zeros(3)
#ent_std = np.zeros(3)
#drp_avg = np.zeros(3)
#drp_std = np.zeros(3)
#tcfp_avg = np.zeros(3)
#tcfp_std = np.zeros(3)


#if per_class:
    #class_names = ['Bike','Car','Motorbike']
    #idx_class = 1

    #for cycle in range(3):
        #rnd_avg[cycle] = rnd_all[cycle].mean(axis=0)[idx_class]
        #rnd_std[cycle] = rnd_all[cycle].std(axis=0)[idx_class]
        #ent_avg[cycle] = ent_all[cycle].mean(axis=0)[idx_class]
        #ent_std[cycle] = ent_all[cycle].std(axis=0)[idx_class]
        #drp_avg[cycle] = drp_all[cycle].mean(axis=0)[idx_class]
        #drp_std[cycle] = drp_all[cycle].std(axis=0)[idx_class]
        #tcfp_avg[cycle] = tcfp_all[cycle].mean(axis=0)[idx_class]
        #tcfp_std[cycle] = tcfp_all[cycle].std(axis=0)[idx_class]


    #plt.errorbar(points,rnd_avg,rnd_std,color='b',label='Random',marker='o',capsize=2)
    #plt.errorbar(points,ent_avg,ent_std,color=(1.0,0.5,0),label='Entropy',marker='o',capsize=2)
    #plt.errorbar(points,drp_avg,drp_std,color='r',label='Dropout',marker='o',capsize=2)
    #plt.errorbar(points,tcfp_avg,tcfp_std,color='c',label='TCFP-3N',marker='o',capsize=2)

    #plt.legend()

    #plt.xticks(points,['1 f/v','2 f/v', '3 f/v'])
    #plt.xlabel('Number of frames')
    #plt.ylabel('AP')
    #plt.title('Class {}'.format(class_names[idx_class]))

    #plt.savefig(data_dir+'3cycles-{}'.format(class_names[idx_class]))

    #plt.show()

#else:
    #for cycle in range(3):

        #mAP = rnd_all[cycle].mean(axis=1)
        #rnd_avg[cycle] = mAP.mean()
        #rnd_std[cycle] = mAP.std()
        #mAP = ent_all[cycle].mean(axis=1)
        #ent_avg[cycle] = mAP.mean()
        #ent_std[cycle] = mAP.std()
        #mAP = drp_all[cycle].mean(axis=1)
        #drp_avg[cycle] = mAP.mean()
        #drp_std[cycle] = mAP.std()
        #mAP = tcfp_all[cycle].mean(axis=1)
        #tcfp_avg[cycle] = mAP.mean()
        #tcfp_std[cycle] = mAP.std()

    #plt.errorbar(points,rnd_avg,rnd_std,color='b',label='Random',marker='o',capsize=2)
    #plt.errorbar(points,ent_avg,ent_std,color=(1.0,0.5,0),label='Entropy',marker='o',capsize=2)
    #plt.errorbar(points,drp_avg,drp_std,color='r',label='Dropout',marker='o',capsize=2)
    #plt.errorbar(points,tcfp_avg,tcfp_std,color='c',label='TCFP-3N',marker='o',capsize=2)

    #plt.legend()

    #plt.xticks(points,['1 f/v','2 f/v', '3 f/v'])
    #plt.xlabel('Number of frames')
    #plt.ylabel('mAP')

    #plt.savefig(data_dir+'3cycles-all')

    #plt.show()


