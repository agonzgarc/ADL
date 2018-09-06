
% Add paths with code to be used
addpath('/home/abel/calvin/matlab/released/draw_on_matlab_images/');
addpath('/home/abel/calvin/matlab/abel_matlab/UsefulCode/');

% Define directories
dataDir = '/home/abel/DATA/ILSVRC/AL/';
annoDir = '/home/abel/DATA/ILSVRC/Annotations/VID/';
graphicsDir = '/home/abel/Documents/graphics/ADL/';
imDir = '/home/abel/DATA/ILSVRC/Data/VID/train/';

classNames = {'bike','car','motorbike'};

fileName = 'train_ALL.txt';

% Read file with frames
fid = fopen([dataDir fileName],'r');

line=0;
tline = fgetl(fid);
while ischar(tline) % read all the lines and copy to address struct
    line=line+1;
    frames_local{line,1}=tline;
    tline = fgetl(fid);
end
fclose(fid);

numTotalFrames = size(frames,1);

fid = fopen([dataDir fileName(1:end-4) '_cluster.txt'],'r');
line=0;
tline = fgetl(fid);
while ischar(tline) % read all the lines and copy to address struct
    line=line+1;
    frames_cluster{line,1}=tline;
    tline = fgetl(fid);
end
fclose(fid);


% Overshoot space for name video cell
idxVideo = zeros(numTotalFrames,1);
allVideoNames = cell(400,1);

currVideo = '';
idxCurr = 0;

for ii = 1:numTotalFrames
    % Select video id from frame str
    if ~strcmp(currVideo,frames{ii}(end-62:end-12))
        currVideo = frames{ii}(end-62:end-12);
        idxCurr = idxCurr + 1;
        allVideoNames{idxCurr} = currVideo;
    end
    idxVideo(ii) = idxCurr;
end

numVideos = idxCurr;

% Trim if there are fewer videos
allVideoNames = allVideoNames(1:numVideos);


