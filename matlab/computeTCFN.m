
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

trash = load('/home/abel/DATA/faster_rcnn/resnet101_coco/RdetsScores.mat');
detsWithScores = trash.R;
clear trash;


%% Accumulate tracked detections into one structure
tracks = cell(numTotalFrames,1);
trackScores = cell(numTotalFrames,1);

for idxFrame = 1:numTotalFrames
    
    detScores = detsWithScores.scores{idxFrame};
    
    forward = R.forward{idxFrame};
    
    fF = size(forward,2);
    for ff = 1:min(fF,3)
        tracks{idxFrame+ff} =  [tracks{idxFrame+ff}; reshape(forward(:,ff,[1 3 2 4]),[],4)];
        trackScores{idxFrame+ff} =  [trackScores{idxFrame+ff}; detScores];
    end
    
    backward = R.backward{idxFrame};

    fB = size(backward,2);
    for fb = 1:min(fB,3)
        tracks{idxFrame-fb} =  [tracks{idxFrame-fb}; reshape(backward(:,fb,[1 3 2 4]),[],4)];
        trackScores{idxFrame-fb} =  [trackScores{idxFrame-fb}; detScores];

    end

end


%% Processing per video

drawF = false;

drawV = false;

totalRT = zeros(numFrames,1);
groupsRT = zeros(numFrames,1);
mScoreRT = -1*ones(numFrames,1);
mScoreGroupRT = -1*ones(numFrames,1);


for idxV = 1:numVideos 
% for idxV = 5
    idxV
    
    % Get frames for video
    idxFramesVideo = idxVideo == idxV;
    
    % Get offset of video for indexing
    idxOffset = find(idxFramesVideo);
    idxOffset = idxOffset(1);
    
    detections = R.detection(idxFramesVideo);
    classes = R.classes(idxFramesVideo);

    numFramesVideo = size(detections,1);
    
    if drawV
        videoDir = sprintf('%s/slides/video-%03d/',graphicsDir,idxV);
        mkdir_if_missing(videoDir);
    end
        
    for f = 1:numFramesVideo
        
        % Absolute indexing of the frame
        idxFrame = idxOffset + f -1;
        
        if drawF
            frameDir = sprintf('%s/slides/frame-%04d/',graphicsDir,idxFrame);
            mkdir_if_missing(frameDir);
        end
        
        if drawV
            currIm = imread(sprintf('%s/%s',imDir,imNames{idxFrame}));
        end
        
        detsF = detections{f}(:,[1 3 2 4]);
        classesF = classes{f}(:);
        numDets = size(detsF,1);
        
        if drawF
           imBoxes = ap_drawbox(currIm, detsF(d,:),[0 1 0], classNames{classesF(d)-1}, false, 5);  
           imwrite(imBoxes,sprintf('%sdet-%d.png',frameDir,d));
        end
        
        if drawV
           imBoxes = ap_drawbox(currIm, tracks{idxFrame},[1 0 0], '', false, 5);  
           imwrite(imBoxes,sprintf('%sf-%d-.png',videoDir,f));
        end
        
        % Now remove tracks based on detections
        
        if numDets > 0 && size(tracks{idxFrame},1)>0
            iou = max(computeOverlapTableSingle(tracks{idxFrame},detsF),[],2);
            matchTracks = iou > 0.5;
        else
            matchTracks = false(size(tracks{idxFrame},1),1);

        end
        
        restTracks = tracks{idxFrame}(~matchTracks,:);
        restTrackScores = trackScores{idxFrame}(~matchTracks,:);

        numRestTracks = size(restTracks,1);
        
        totalRT(idxFrame) = numRestTracks;
        
        
%         if drawV
%            imBoxes = ap_drawbox(currIm, restTracks,[1 0 0], '', false, 5);     
%            imwrite(imBoxes,sprintf('%sf-%d-.png',videoDir,f));
%         end
        
        if numRestTracks > 1
            

            iouRT = computeOverlapTableSingle(restTracks,restTracks);

            iouRT = iouRT - eye(numRestTracks);
            maxGR = -Inf;
            idxG = [];
            for rT = 1:size(restTracks,1)
                idxRT = iouRT(:,rT) > 0.5;
                countRT = sum(idxRT);
                if countRT > maxGR
                    maxGR = countRT;
                    idxG = idxRT;
                end
            end
            
            restTrackGroup = restTrackScores(idxG);
        else
            countRT = numRestTracks;
            restTrackGroup = restTrackScores; 
        end
        
        groupsRT(idxFrame) = countRT;

        
        mScoreRT(idxFrame) = mean(1-restTrackScores);
        mScoreGroupRT(idxFrame) = mean(1-restTrackGroup);
        
        
        if drawV
            imBoxes = ap_drawbox(currIm, restTracks,[1 0 0], '', false, 5);  

%             imBoxes = ap_drawbox(currIm, restTracks(~idxG,:),[1 0 0], '', false, 5);  
%             imBoxes = ap_drawbox(imBoxes, restTracks(idxG,:),[0.5 0 0.5], '', false, 5);  
            imwrite(imBoxes,sprintf('%sGf-%d-%.2f.png',videoDir,f,mScoreRT(idxFrame)));
        end
        
    end
    
end

TCFN.totalRT = totalRT;
TCFN.groupsRT = groupsRT;
TCFN.mScoreRT = mScoreRT;
TCFN.mScoreGroupRT = mScoreGroupRT;

save([dataDir 'TCFN.mat'],'TCFN');

figure()
tRTFN = totalRT(fN>0);
tRTOther = totalRT(fN==0);
subplot(1,2,1);
histogram(tRTFN);
title('Total RT for FN');
subplot(1,2,2);
histogram(tRTOther);
title('Total RT for Not FN');
saveas(gcf,sprintf('%s/FN/plots/totalRT.png',graphicsDir));

figure()
gRTFN = groupsRT(fN>0);
gRTOther = groupsRT(fN==0);
subplot(1,2,1);
histogram(gRTFN);
title('Max Group RT for FN');
subplot(1,2,2);
histogram(gRTOther);
title('Max Group RT for Not FN');
saveas(gcf,sprintf('%s/FN/plots/maxGroupRT.png',graphicsDir));




xFN = [tRTFN gRTFN];
xOther = [tRTOther gRTOther];

figure();
scatter(xFN(:,1),xFN(:,2),'r','.');
hold on;
scatter(xOther(:,1)+0.1,xOther(:,2)+0.1,'g','+');
xlim([0,51]);
xlabel('Total RT');
ylabel('Max group RT');
legend('FN','No FN');

saveas(gcf,sprintf('%s/FN/plots/scatter.png',graphicsDir));


figure();
subplot(1,2,1);
scatter(mScoreRT(fN>0),mScoreGroupRT(fN>0),'r','.');
subplot(1,2,2);
scatter(mScoreRT(fN==0),mScoreGroupRT(fN==0),'g','+');

mScoreRT

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

