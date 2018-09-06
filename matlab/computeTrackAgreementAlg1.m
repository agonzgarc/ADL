
initializeADL

% Number of forward/backward frames to take into account
fF = 3;
fB = 3;

% Threshold for matching detections
threshMatch = 0.8;

threshTrack = 0.8;

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


tcScore = zeros(numTotalFrames,1);

% Processing per video
% for idxV = 1:numVideos
for idxV = 15
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
    
    for f = 5:numFramesVideo
        
        % Absolute indexing of the frame
        idxFrame = idxOffset + f -1;
        
        frameDir = sprintf('%s/slides/frame-%04d/',graphicsDir,idxFrame);
        mkdir_if_missing(frameDir);
        
        currIm = imread(sprintf('%s/%s',imDir,imNames{idxFrame}));
3
        detsF = detections{f}(:,[1 3 2 4]);
        classesF = classes{f}(:);
        numDets = size(detsF,1);
        for d = 1:numDets
           imBoxes = ap_drawbox(currIm, detsF(d,:),[0 1 0], classNames{classesF(d)-1}, false, 5);  
%            subplot(3,3,1);
%            imshow(imBoxes);
%            title(sprintf('F:%d - det:%d',f,d));
           
           imwrite(imBoxes,sprintf('%sdet-%d.png',frameDir,d));
           
           % Forward
           forwardF = squeeze(forward{f}(d,:,[1 3 2 4]));
           for ff = 1:fF
               currIm = imread(sprintf('%s/%s',imDir,imNames{idxFrame+ff}));
               detsFF = detections{f+ff}(:,[1 3 2 4]); 
               classesFF = classes{f+ff}(:);
               
               imBoxes = ap_drawbox(currIm, detsFF ,[0 1 0], '', false, 5);  

               
               % Matching detection
               overlap = computeOverlapTableSingle(detsF(d,:),detsFF);
               [maxO, idxMaxO] = max(overlap);
               
               if maxO >= threshMatch
                  imBoxes = ap_drawbox(currIm, detsFF(idxMaxO,:) ,[1 0.5 0], '', false, 5);  
                  
                  % Compare matched detection with tracked detection
                  ovTr = computeOverlapTableSingle(detsFF(idxMaxO,:),forwardF(ff,:));
                  
                  if ovTr < threshTrack
                     % Bad --> no temporal coherence with tracker
                     tcScore(idxFrame) = tcScore(idxFrame) + 1;
                  end
               else
                   ovTr = 0;
                   % Bad --> no corresponding detection
                   tcScore(idxFrame) = tcScore(idxFrame) + 1;
               end


               imBoxes = ap_drawbox(imBoxes, forwardF(ff,:),[1 0 0], '', false, 5);  
               
               
               imwrite(imBoxes,sprintf('%sFF-%d-f-%d-ov-%.2f.png',frameDir,d,ff, ovTr));
               

           end
           
           % Backward
           backwardF = squeeze(forward{f}(d,:,[1 3 2 4]));
           for fb = 1:fB
               currIm = imread(sprintf('%s/%s',imDir,imNames{idxFrame-fb}));
               detsFB = detections{f-fb}(:,[1 3 2 4]); 
               classesFB = classes{f-fb}(:);
               
               imBoxes = ap_drawbox(currIm, detsFB ,[0 1 0], '', false, 5);  

               
               % Matching detection
               overlap = computeOverlapTableSingle(detsF(d,:),detsFB);
               [maxO, idxMaxO] = max(overlap);
               
               if maxO >= threshMatch
                  imBoxes = ap_drawbox(currIm, detsFB(idxMaxO,:) ,[1 0.5 0], '', false, 5);  
                  
                  % Compare matched detection with tracked detection
                  ovTr = computeOverlapTableSingle(detsFB(idxMaxO,:),backwardF(fb,:));
                  
                  if ovTr < threshTrack
                     % Bad --> no temporal coherence with tracker
                     tcScore(idxFrame) = tcScore(idxFrame) + 1;
                  end
               else
                   ovTr = 0;
                   % Bad --> no corresponding detection
                   tcScore(idxFrame) = tcScore(idxFrame) + 1;
               end


               imBoxes = ap_drawbox(imBoxes, backwardF(ff,:),[1 0 0], '', false, 5);  
               
               
               imwrite(imBoxes,sprintf('%sFB-%d-f-%d-ov-%.2f.png',frameDir,d,fb, ovTr));
               

           end
           
        end
    end

end





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

