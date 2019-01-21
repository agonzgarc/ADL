import matplotlib.pyplot as plt
import numpy as np



data_dir = '/home/abel/Documents/graphics/ADL/efficiency/'


#num_frames_train = [1567, 2323, 3041, 3727, 4376, 4982, 5541, 6050, 6525, 6971]
num_frames_train = [1.567, 2.323, 3.041, 3.727, 4.376, 4.982, 5.541, 6.050, 6.525, 6.971]

num_frames_eval = [118, 112, 106, 100, 94, 88, 82, 76, 70, 64]


time_train = [1.1, 1.7, 2.2, 2.7, 3.2, 3.7, 4.1, 4.5, 4.8, 5.2]
time_eval = [4.7, 4.4, 4.2, 4, 3.7, 3.5, 3.2, 3.0, 2.8, 2.5]
time_track = np.linspace(5,4,10)
time_total = time_train + time_eval

num_cycles = len(num_frames_train)

index = np.arange(num_cycles)

### Timings
bar_width = 0.2

fig, ax = plt.subplots()

rects1 = ax.bar(index, time_train, bar_width, color='b',label='Train on labeled')
rects2 = ax.bar(index+bar_width, time_eval, bar_width, color='r',label='Evaluate on unlabeled')
rects3 = ax.bar(index+2*bar_width, time_track, bar_width, color='g',label='Track detections')


ax.set_xlabel('Cycle')
ax.set_ylabel('Time (hours)')
ax.set_title('Active learning timing distribution')
ax.set_xticks(index+bar_width/3)
ax.set_xticklabels(range(1,11))
ax.legend()

plt.savefig(data_dir+'times')

plt.show()



### Number of frames
bar_width = 0.3

fig, ax = plt.subplots()
rects1 = ax.bar(index,num_frames_train, bar_width, color='b', label='Labeled')
rects2 = ax.bar(index+bar_width,num_frames_eval, bar_width, color='r', label='Unlabeled')

ax.set_xlabel('Cycle')
ax.set_ylabel('Number of frames (K)')
ax.set_title('Number of frames')
ax.set_xticks(index+bar_width/2)
ax.set_xticklabels(range(1,11))
ax.legend()

plt.savefig(data_dir+'frames')

plt.show()

