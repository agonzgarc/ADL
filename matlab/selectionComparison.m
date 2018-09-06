name1 = 'MCDropout';

name2 = 'TCFPN3';

graphicsBaseDir = '/home/abel/Documents/graphics/ADL/PerFrameVis/';

imDir1 = dir([graphicsBaseDir name1]);
imDir1(1:2) = [];

imDir2 = dir([graphicsBaseDir name2]);
imDir2(1:2) = [];

numIms = min(numel(imDir1),numel(imDir2));

graphicsTargetDir = [graphicsBaseDir name1 '-' name2 '/'];
mkdir_if_missing(graphicsTargetDir);

for ii = 1:numIms
   im1 = imread([imDir1(ii).folder '/' imDir1(ii).name]);
   im2 = imread([imDir2(ii).folder '/' imDir2(ii).name]);
   
   imC = [im1 im2];
   imwrite(imC,sprintf('%s/%i.png',graphicsTargetDir,ii));
end