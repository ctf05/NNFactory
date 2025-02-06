cap = 5;
num_images = 400000;
res = 224;

if ~exist('training', 'dir')
    mkdir('training');
end

rng('shuffle');

for img = 1:num_images
    params = round(-cap + (2*cap)*rand(1,8), 6);
    
    image = Interferogram_Plot(params(1), params(2), params(3), params(4), ...
                             params(5), params(6), params(7), params(8), ...
                             res, []);
    
    baseFilename = sprintf('img_D%.6f_C%.6f_B%.6f_G%.6f_F%.6f_J%.6f_E%.6f_I%.6f', ...
        params(1), params(2), params(3), params(4), ...
        params(5), params(6), params(7), params(8));
    
    baseFilename = strrep(baseFilename, '-', 'n');
    baseFilename = strrep(baseFilename, '.', 'p');
    
    writeFile = fullfile('training', [baseFilename '.jpg']);
    imwrite(image, writeFile);
end