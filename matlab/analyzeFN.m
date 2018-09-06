

initializeADL

draw = false;

% Load tracking information (it also contains detections)
trash = load([dataDir '/tracks/Rdets_trax10.mat']);
R = trash.R;
clear trash;

objGT = getGTVID(frames);

numObj = cellfun(@(x) size(x,1),objGT.boxes);

fN = zeros(numTotalFrames,1);

for idxFrame = 1:numTotalFrames
    % Get GTs boxes
    boxes = single(objGT.boxes{idxFrame});
    
    if size(boxes,1) > 0

        dets = round(R.detection{idxFrame}(:,[1 3 2 4]));
        if size(dets,1) > 0
            iouGT = max(computeOverlapTableSingle(boxes,dets),[],2);

            fN(idxFrame) = sum(iouGT < 0.5);
        else
            fN(idxFrame) = size(boxes,1);
        end

        if draw
            imBoxes = imread(frames{idxFrame});
            imBoxes = ap_drawbox(imBoxes,boxes, [0 0 1], '', false, 3);
            imBoxes = ap_drawbox(imBoxes,dets, [0 1 0], '', false, 3);
            imwrite(imBoxes,sprintf('%s/FN/f-%d-FN-%d.png',graphicsDir,idxFrame,fN(idxFrame)));
        end

    end
        
end


histogram(fN)
xlabel('False Negatives');
ylabel('Frames');

sum(fN >0)
