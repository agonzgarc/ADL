import matplotlib.pyplot as plt
import numpy as np
import os
from shutil import copyfile
import pdb

data_dir = '/home/abel/DATA/ILSVRC/'
checkpoint_dir = '/home/abel/DATA/faster_rcnn/resnet101_coco/checkpoints/'

name = 'EntAllVideos-NeighCycle'


run = 1

data_info = {'data_dir': data_dir,
          'annotations_dir':'Annotations',
          'label_map_path': './data/imagenetvid_label_map.pbtxt',
          'set': 'train_150K_clean'}


def augment_active_set(dataset,videos,active_set,num_neighbors=5):
    """ Augment set of indices in active_set by adding a given number of neighbors
    Arg:
        dataset: structure with information about each frames
        videos: list of video names
        active_set: list of indices of active_set
        num_neighbors: number of neighbors to include
    Returns:
        aug_active_set: augmented list of indices with neighbors
    """
    aug_active_set = []

    # We need to do this per video to keep limits in check
    for v in videos:
        frames_video = [f['idx'] for f in dataset if f['video'] == v]
        max_frame = np.max(frames_video)
        idx_videos_active_set = [idx for idx in frames_video if idx in active_set]
        idx_with_neighbors = [i for idx in idx_videos_active_set for i in range(idx-num_neighbors,idx+num_neighbors+1) if i >= 0 and i
         <= max_frame ]
        aug_active_set.extend(idx_with_neighbors)

    return aug_active_set

def get_dataset(data_info):
    """ Gathers information about the dataset given and stores it in a
    structure at the frame level.
    Args:
        data_info: dictionary with information about the dataset
    Returns:
        dataset: structure in form of list, each element corresponds to a
            frame and its a dictionary with multiple keys
        videos: list of videos
    """
    dataset = []
    path_file = os.path.join(data_info['data_dir'],'AL', data_info['set'] + '.txt')
    with open(path_file,'r') as pF:
        idx = 0
        for line in pF:
            # Separate frame path and clean annotation flag
            split_line = line.split(' ')
            # Remove trailing \n
            verified = True if split_line[1][:-1] == '1' else False
            path = split_line[0]
            split_path = path.split('/')
            filename = split_path[-1]
            video = split_path[-3]+'/'+split_path[-2]
            dataset.append({'idx':idx,'filename':filename,'video':video,'verified':verified})
            idx+=1
    videos = set([d['video'] for d in dataset])
    return dataset,videos


def save_frames(dataset,videos):
    prev_set = 0
    graphics_dir = '/home/abel/Documents/graphics/ADL/selectedFrames/'
    if not os.path.exists(graphics_dir+name):
        os.makedirs(graphics_dir+name)

    for cycle in range(0,4):
        active_set = []
        with open(os.path.join(checkpoint_dir,name+'R'+str(run)+'cycle'+str(cycle),'active_set.txt'),'r') as f:
            for line in f:
                active_set.append(int(line))
        #for frame in active_set:
        #idx = active_set[0]
        size_active_set = len(active_set)
        active_set = active_set[prev_set:]
        prev_set = size_active_set
        videos_cycle = []
        for idx in active_set:
            frame = dataset[idx]
            src = os.path.join(data_dir,'Data','VID','train',frame['video'],frame['filename'])
            dst = os.path.join(graphics_dir,name,'c'+str(cycle)+'-'+ str(idx)+'.jpg')
            #copyfile(src,dst)
            videos_cycle.append(frame['video'])

        print('Videos in cycle: ' + str(len(set(videos_cycle))))


if __name__ == '__main__':
    dataset,videos = get_dataset(data_info)
    
    save_frames(dataset,videos)

    #cycle = 1
    #ent = np.load(os.path.join(checkpoint_dir,name+'R'+str(run)+'cycle'+str(cycle),'entropies.npy'))
    #active_set = []
    #with open(os.path.join(checkpoint_dir,name+'R'+str(run)+'cycle'+str(cycle-1),'active_set.txt'),'r') as f:
        #for line in f:
            #active_set.append(int(line))

    #aug_active_set =  augment_active_set(dataset,videos,active_set,num_neighbors=5)

    #unlabeled_set = [f['idx'] for f in dataset if f['idx'] not in aug_active_set and f['verified']]

    #ent_videos_max = []
    #ent_videos_mean = []

    #for v in videos:

        ## Get indices of frames in unlabeled set and in the current video
        #frames_video = [f['idx'] for f in dataset if f['video'] == v]
        #unlabeled_frames_video = [i for i in range(len(unlabeled_set)) if dataset[unlabeled_set[i]]['video']  == v]
        #ent_video = ent.take(unlabeled_frames_video)

        #if len(ent_video) > 0:
            #ent_videos_max.append(ent_video.max())
            #ent_videos_mean.append(ent_video.mean())
        #else:
            #ent_videos_max.append(0)
            #ent_videos_mean.append(0)

    #pdb.set_trace()
    #plt.plot(range(len(videos)),ent_videos_max,label='max',color='r')
    #plt.plot(range(len(videos)),ent_videos_mean,label='mean',color='b')

    #mean_mean = np.asarray(ent_videos_mean).mean()
    #plt.plot(range(len(videos)), [mean_mean for i in range(len(videos))],color='k',linestyle=':',label='Avg. mean')
    #mean_max = np.asarray(ent_videos_max).mean()
    #plt.plot(range(len(videos)), [mean_max for i in range(len(videos))],color='k',linestyle='-',label='Avg. max')

    #plt.xlabel('Video idx')
    #plt.ylabel('Entropy')
    #plt.legend()

        #graphics_dir = '/home/abel/Documents/graphics/ADL/inspection/EntAllVideos/' + v[-25:]
        #if not os.path.exists(graphics_dir):
            #os.makedirs(graphics_dir)


        #idx_unlabeled_frames = [unlabeled_set[i]-frames_video[0] for i in unlabeled_frames_video]

        #all_ent_video = np.zeros(len(frames_video))
        #all_ent_video[idx_unlabeled_frames] = ent_video

        ## Save entropy plot
        #plt.plot(range(len(frames_video)),all_ent_video)
        #plt.xlabel('Frame idx')
        #plt.ylabel('Entropy')
        #plt.title('Video: '+v)
        #plt.savefig(graphics_dir+'/ent.png')
        #plt.clf()

        ## Copy some frames
        #for idx in idx_unlabeled_frames:
            #frame = dataset[idx + frames_video[0]]
            #src = os.path.join(data_dir,'Data','VID','train',frame['video'],frame['filename'])
            #dst = os.path.join(graphics_dir,"{}-{:.2f}.jpg".format(idx,all_ent_video[idx]))
            #copyfile(src,dst)








