
% Depending on your setup the following path has to be set manually.
run([getenv('HOME') '/projects/vlfeat/toolbox/vl_setup'])

% Test rgb2gray
img      = imread('roofs1.jpg');
img_gray = rgb2gray(im2double(img));
dlmwrite('img_gray.txt', img_gray, 'delimiter', '\t', 'precision', 4);

% Test vl_imsmooth
binsize = 8;
magnif  = 3;
img_smooth = vl_imsmooth(img_gray, sqrt((binsize/magnif)^2 - .25));
dlmwrite('img_smooth.txt', img_smooth, 'delimiter', '\t', 'precision', 4);

% Test vl_dsift
img_smooth = single(img_smooth);
[frames, descrs] = vl_dsift(img_smooth, 'size', binsize);
dlmwrite('dsift_frames.txt', frames, 'delimiter', '\t', 'precision', 4);
dlmwrite('dsift_descrs.txt', descrs, 'delimiter', '\t', 'precision', 4);

% Test vl_kmeans
[centers, assigns] = vl_kmeans([1 2 3 10 11 12], 2);
centers, assigns

[centers, assigns] = vl_kmeans([1 2 3 10 11 12;
                                0 0 0  1  1  1], 2);
centers, assigns
