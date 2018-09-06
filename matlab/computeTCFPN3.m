
initializeADL

% Number of forward/backward frames to take into account
fF = 3;
fB = 3;

% Threshold for matching detections
threshTrack = 0.7;

% Load tracking information (it also contains detections)
trash = load([dataDir '/tracks/Rdets_trax10.mat']);
R = trash.R;
clear trash;

% Get image names
numFrames = size(R.frame,1);
imNames = cell(numFrames,1);

assert(numFrames == numTotalFrames);

for ii = 1:numFrames
    imNames{ii} = R.frame{ii}(end-62:end);
end


tcScore = cell(numTotalFrames,1);

tcVideoScore = cell(numVideos,1);

draw = false;


% Processing per video
for idxV = 1:numVideos
% for idxV = 7
    idxV
    % Get frames for video
    idxFramesVideo = idxVideo == idxV;
    
    % Get offset of video for indexing
    idxOffset = find(idxFramesVideo);
    idxOffset = idxOffset(1);
    
    detections = R.detection(idxFramesVideo);
    classes = R.classes(idxFramesVideo);
    forward = R.forward(idxFramesVideo);
    backward = R.backward(idxFramesVideo);
    
    numFramesVideo = size(detections,1);
    
    if draw
        videoDir = sprintf('%s/slides/video-%03d/',graphicsDir,idxV);
        mkdir_if_missing(videoDir);
    end
    
    for f = [1:3 (numFramesVideo-3):numFramesVideo]
        idxFrame = idxOffset + f -1;
        tcScore{idxFrame} = -1;
    end

    for f = 4:numFramesVideo-4
        
        % Absolute indexing of the frame
        idxFrame = idxOffset + f -1;
        
        if draw
            frameDir = sprintf('%s/slides/frame-%04d/',graphicsDir,idxFrame);
            mkdir_if_missing(frameDir);
            currIm = imread(sprintf('%s/%s',imDir,imNames{idxFrame}));
            imBoxes = imread(sprintf('%s/%s',imDir,imNames{idxFrame}));
        end
        
        detsF = detections{f}(:,[1 3 2 4]);
        classesF = classes{f}(:);
        numDets = size(detsF,1);
        
        tcScore{idxFrame} = zeros(numDets,1);
        
        

        % Why is this lower?
        for d = 1:min(numDets,size(forward{f},1))
            if draw
               imBoxes = ap_drawbox(currIm, detsF(d,:),[0 1 0], classNames{classesF(d)-1}, false, 5);  
               imwrite(imBoxes,sprintf('%sdet-%d.png',frameDir,d));
            end

            % Forward
            forwardF = squeeze(forward{f}(d,:,[1 3 2 4]));
            for ff = 1:fF
                detsFF = detections{f+ff}(:,[1 3 2 4]); 
                classesFF = classes{f+ff}(:);

                if draw
                    currIm = imread(sprintf('%s/%s',imDir,imNames{idxFrame+ff}));
                    imBoxes = ap_drawbox(currIm, detsFF ,[0 1 0], '', false, 5);  
                end
                
                % Compare matched detection with tracked detection
                ovTr = computeOverlapTableSingle(forwardF(ff,:),detsFF);

                [maxOvTr, idxOvTr] = max(ovTr);

                if maxOvTr >= threshTrack
                 % Good --> add positive evidence for current detection
                 tcScore{idxFrame}(d) = tcScore{idxFrame}(d) + 1;
                 if draw
                    imBoxes = ap_drawbox(imBoxes, detsFF(idxOvTr(1),:),[1 0.5 0], '', false, 5);  
                 end
                end

                if draw
                    imBoxes = ap_drawbox(imBoxes, forwardF(ff,:),[1 0 0], '', false, 5);
                    imwrite(imBoxes,sprintf('%sFF-%d-f-%d-ov-%.2f.png',frameDir,d,ff, maxOvTr));
                end


            end

           % Backward
           backwardF = squeeze(backward{f}(d,:,[1 3 2 4]));

           for fb = 1:fB
                detsFB = detections{f-fb}(:,[1 3 2 4]); 
                classesFF = classes{f-fb}(:);

                if draw
                    currIm = imread(sprintf('%s/%s',imDir,imNames{idxFrame-fb}));
                    imBoxes = ap_drawbox(currIm, detsFB ,[0 1 0], '', false, 5);  
                end

                % Compare matched detection with tracked detection
                ovTr = computeOverlapTableSingle(backwardF(fb,:),detsFB);
                [maxOvTr, idxOvTr] = max(ovTr);

                if maxOvTr >= threshTrack
                 % Good --> add positive evidence for current detection
                 tcScore{idxFrame}(d) = tcScore{idxFrame}(d) + 1;
                 
                 if draw
                    imBoxes = ap_drawbox(imBoxes, detsFB(idxOvTr,:),[1 0.5 0], '', false, 5);  
                 end

                end

                if draw
                    imBoxes = ap_drawbox(imBoxes, backwardF(fb,:),[1 0 0], '', false, 5);
                    imwrite(imBoxes,sprintf('%sFB-%d-f-%d-ov-%.2f.png',frameDir,d,fb, maxOvTr));
                end


           end

           if draw
               imBoxes = ap_drawbox(imBoxes, [detsF(d,:) tcScore{idxFrame}(d)],[0 1 0], classNames{classesF(d)-1}, true, 5);  
               imwrite(imBoxes,sprintf('%sf-%d-.png',videoDir,f));
           end

        end

    end
    
   
end


scores = cell2mat(tcScore);
histogram(scores);
xlabel('TC score');
title('Overall');

saveas(gcf,sprintf('%s/slides/overallTCscores.png',graphicsDir));

A = cellfun(@(x) size(x,1),tcScore);
tcNonEmpty = tcScore(A>0);

sum(A == 0)/size(tcScore,1)

exFrames = cellfun(@(x) x(1)==-1,tcNonEmpty);

tcDets = tcNonEmpty(~exFrames);
meanTCDets = cellfun(@mean, tcDets);


histogram(meanTCDets);
xlabel('Mean TC score');
title('Average/frame');
saveas(gcf,sprintf('%s/slides/meanTCscores.png',graphicsDir));


idxDets = 1:numTotalFrames;
idxDets = idxDets(A>0);
idxDets = idxDets(~exFrames)';

[sMeanTCDets, idxSort] = sort(meanTCDets);

idxSortedDets = idxDets(idxSort);



%% Select subset top k% 
portion = 5;

% Add another numSamples to the set
numSamples = round(numTotalFrames*portion/100);

fidH=fopen([dataDir 'tempcoherence/train_tcAlg2' num2str(portion) '.txt'],'w');

for s = idxSortedDets(1:numSamples)'
    str = frames{s};
    fprintf(fidH,'%s\n', str);
end
fclose(fidH);


%% Per video

fPerV = 10;

if cluster
    fidH=fopen([dataDir 'tempcoherence/train_tc2A' num2str(fPerV) 'xVid_cluster.txt'],'w');
else
    fidH=fopen([dataDir 'tempcoherence/train_tc2A' num2str(fPerV) 'xVid.txt'],'w');
end


for idxV = 1:numVideos
% for idxV = 7
    % Get frames for video
    idxFramesVideo = idxVideo == idxV;
    tcScoresVideo = tcScore(idxFramesVideo);
    
    detVideo = cellfun(@(x) size(x,1),tcScoresVideo);
    meanTCDetsNonEmpty = cellfun(@mean, tcScoresVideo(detVideo>0));
    
    meanTCDets = Inf*ones(size(tcScoresVideo,1),1);
    meanTCDets(detVideo>0) = meanTCDetsNonEmpty;
    
    meanTCDets(meanTCDets == -1) = Inf;
    
    [smTCD, idxmTCD] = sort(meanTCDets,'ascend');
    
    
    framesVideo = frames(idxVideo == idxV);
    
    for n = 1:min(fPerV,size(tcScoresVideo,1))
        str = framesVideo{idxmTCD(n)};
        fprintf(fidH,'%s\n', str);
    end

end

fclose(fidH);


%% Visualization of per frame selection

graphicsPerFrameDir = '/home/abel/Documents/graphics/ADL/PerFrameVis/TCFPN3';
mkdir_if_missing(graphicsPerFrameDir);

distanceFrames = zeros(numVideos,1);
tieTopFrames = zeros(numVideos,1);


for v = 1:max(idxVideo)
    
    idxFramesVideo = idxVideo == v;
    tcScoresVideo = tcScore(idxFramesVideo);
    
    detVideo = cellfun(@(x) size(x,1),tcScoresVideo);
    meanTCDetsNonEmpty = cellfun(@mean, tcScoresVideo(detVideo>0));
    
    meanTCDets = Inf*ones(size(tcScoresVideo,1),1);
    meanTCDets(detVideo>0) = meanTCDetsNonEmpty;
    
    meanTCDets(meanTCDets == -1) = Inf;
    
    [smTCD, idxmTCD] = sort(meanTCDets,'ascend');
    
    tieTopFrames(v) = sum(smTCD(1) == smTCD);
%     
%     idxOffset = find(idxFramesVideo);
%     idxOffset = idxOffset(1)-1;
%     ii = idxOffset + idxmTCD(1);
%     
%     detf = R.detection{ii}(:,[1 3 2 4]);
%    
%     imBoxes =imread(frames{ii});
%     
%     % Show box with max entropy
%     for d = 1:size(detf,1)
%         % Take average regressed box for our 3 classes (ideally, we should
%         % use the original box without regression, but this is an apprx.
%        imBoxes = ap_drawbox(imBoxes, detf(d,:), [1 0 0], num2str(tcScoresVideo{idxmTCD(1)}(d)), false, 3);  
%     end
%     imBoxes = imresize(imBoxes,[600,1024]);
%     imwrite(imBoxes,sprintf('%s/%03d-%.2f.png',graphicsPerFrameDir,v,smTCD(1)));
end

figure();
histogram(tieTopFrames,[0:1:100]);
xlabel('Number of frames with max value');
title(sprintf('TCFP-N=3 (sum) - Mean: %.2f, median: %d',mean(tieTopFrames), median(tieTopFrames)));
saveas(gcf,'/home/abel/Documents/graphics/ADL/PerFrameVis/TCFPSum3Ties.png');

%% Visualize detection
% 
% for idxFrame = 600:20:8000
%     idxFrame
%     currIm = imread(sprintf('%s/%s',imDir,imNames{idxFrame}));
%     imshow(currIm);
%     waitforbuttonpress
% end

idxFrame = 375;
resultsDir = '/home/abel/Documents/graphics/ADL/tracks/';


% Load frame image
currIm = imread(sprintf('%s/%s',imDir,imNames{idxFrame}));

dets = R.detection{idxFrame}(:,[1 3 2 4]);
% dets = convertWH2BB(R.detection{idxFrame});
% dets = dets(:,[2 1 4 3]);
numDets = size(dets,1);

frameDir = sprintf('%sframe-%04d/',resultsDir,idxFrame);
mkdir_if_missing(frameDir);

k = 1;

lim = 300:1100;
imwrite(imBoxes(:,lim,:),sprintf('%sdet-%d-f-0.png',frameDir,k));
imwrite(imBoxes(:,lim,:),sprintf('%sdet-%d-b-0.png',frameDir,k));

forward = squeeze(R.forward{idxFrame}(k,:,[1 3 2 4]));
numTrackedFrames = size(forward,1);

for l = 1:numTrackedFrames
    nextIm = imread(sprintf('%s/%s',imDir,imNames{idxFrame+l}));
    detf = R.detection{idxFrame+l}(1,[1 3 2 4]);
%     imBoxes = ap_drawbox(nextIm, detf,[0 1 0], '', false, 5);  
    imBoxes = ap_drawbox(nextIm, forward(l,:),[1 0 0], '', false, 5);  
    imwrite(imBoxes(:,lim,:),sprintf('%sdet-%d-f-%d.png',frameDir,k,l));
end

backward = squeeze(R.backward{idxFrame}(k,:,[1 3 2 4]));
numTrackedFrames = size(backward,1);
    
for l = 1:numTrackedFrames
    nextIm = imread(sprintf('%s/%s',imDir,imNames{idxFrame-l}));
    detb = R.detection{idxFrame-l}(1,[1 3 2 4]);
%     imBoxes = ap_drawbox(nextIm, detb,[0 1 0], '', false, 5);  
    imBoxes = ap_drawbox(nextIm, backward(l,:),[0 0 1], '', false, 5);  
    imwrite(imBoxes(:,lim,:),sprintf('%sdet-%d-b-%d.png',frameDir,k,l));
end



%% Visualize tracks

