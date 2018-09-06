function [objGT] =  getGTVID(frames)

initializeADL

addpath('/home/abel/code/Datasets/ImageNet-VID')

draw = 0;

classNames={'bike','car','motorcycle'};

s1='n02834778'; % bike
s2='n02958343'; % car
s3='n03790512'; % motorcycle

numTotalFrames = size(frames,1);

objGT.frames = frames;
objGT.boxes = cell(numTotalFrames,1);
objGT.classes = cell(numTotalFrames,1);
objGT.occluded = cell(numTotalFrames,1);

for idxFrame = 1:numTotalFrames
    annoFile = [annoDir frames{idxFrame}(end-68:end-5) '.xml'];
    res = VOCreadxml(annoFile);
    if isfield(res.annotation, 'object')
        numObjects = numel(res.annotation.object);
        objects.boxes = zeros(numObjects,4);
        objects.classID = zeros(numObjects,1);
        objects.occluded = zeros(numObjects,1);
        for k=1:numObjects
            obj = res.annotation.object(k);
            objects.classID(k) = -1;
            switch obj.name
                case s1
                    objects.classID(k) = 1;
                case s2
                    objects.classID(k) = 2;
                case s3
                    objects.classID(k) = 3;
            end
            if objects.classID(k) > 0
                b = obj.bndbox;
                bb = str2double({b.xmin b.ymin b.xmax b.ymax});
                objects.boxes(k,:) = [bb(1), bb(2), bb(3), bb(4)];
                objects.occluded(k) = str2double(obj.occluded);
%                 objects{k}.trackid = str2double(obj.trackid) + 1; % 1-index
            end
        end
        
        % Remove space of those not in the target classes
        targetClasses = objects.classID > 0;
        objects.boxes = objects.boxes(targetClasses,:);
        objects.classID = objects.classID(targetClasses,:);
        objects.occluded = objects.occluded(targetClasses,:);
        
        objGT.boxes{idxFrame} = objects.boxes;
        objGT.classes{idxFrame} = objects.classID;
        objGT.occluded{idxFrame} = objects.occluded;

        if draw 
            imBoxes = imread(frames{idxFrame});
            for k = 1:size(objects.classID,1)
                if objects.occluded(k)
                    imBoxes = ap_drawbox(imBoxes, objects.boxes(k,:), [0.2 0.8 1], classNames{objects.classID(k)}, false, 2);
                else
                    imBoxes = ap_drawbox(imBoxes, objects.boxes(k,:), [0 0 1], classNames{objects.classID(k)}, false, 3);
                end
            end
            imwrite(imBoxes,sprintf('%s/GT/f-%d.png',graphicsDir,idxFrame));

        end
    end
    
end


