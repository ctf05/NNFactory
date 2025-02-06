function NeuralNetworkFactory
    imds = imageDatastore(fullfile('training', '*.jpg'), ...
        'ReadFcn', @(x) preprocessImage(x));
    batchSize = 20000;
    numFiles = length(imds.Files);
    fprintf('Number of files: %d\n', numFiles);
    targets = zeros(numFiles, 8);
    
    for i = 1:batchSize:numFiles
        endIdx = min(i + batchSize - 1, numFiles);
        batch_files = imds.Files(i:endIdx);
        [~, names, ~] = cellfun(@fileparts, batch_files, 'UniformOutput', false);
        for j = 1:length(names)
            name = names{j};
            name = strrep(name, 'n', '-');
            name = strrep(name, 'p', '.');
            parts = split(name, '_');
            params = zeros(1, 8);
            for k = 2:length(parts)
                param_str = parts{k};
                param_val = str2double(param_str(2:end));
                params(k-1) = param_val;
            end
            targets(i+j-1,:) = params;
        end
        fprintf('Processed files %d to %d of %d\n', i, endIdx, numFiles);
    end
    
    validationSize = floor(0.2 * numFiles);
    trainSize = numFiles - validationSize;
    
    trainIdx = 1:trainSize;
    valIdx = (trainSize + 1):numFiles;

    trainFiles = imds.Files(trainIdx);
    trainTargets = targets(trainIdx, :);
    
    trainImds = imageDatastore(trainFiles, 'ReadFcn', @(x) preprocessImage(x));
    trainDs = arrayDatastore(trainTargets);
    trainCds = transform(combine(trainImds, trainDs), @preprocessTrainingData);
    
    valFiles = imds.Files(valIdx);
    valTargets = targets(valIdx, :);
    valImds = imageDatastore(valFiles, 'ReadFcn', @(x) preprocessImage(x));
    valDs = arrayDatastore(valTargets);
    valCds = transform(combine(valImds, valDs), @preprocessTrainingData);
    
    inputSize = [224 224 1];
    layers = [
        imageInputLayer(inputSize)
        convolution2dLayer(3, 32, 'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride',2)
        convolution2dLayer(3, 64, 'Padding','same')
        reluLayer
        fullyConnectedLayer(256)
        reluLayer
        dropoutLayer(0.5)
        fullyConnectedLayer(8)
        regressionLayer
    ];
    
    options = trainingOptions('adam', ...
        'MaxEpochs', 5, ...
        'InitialLearnRate', 1e-3, ...
        'ValidationData', valCds, ...
        'ValidationFrequency', 50, ...
        'ValidationPatience', 5, ...
        'Plots', 'training-progress', ...
        'Verbose', true, ...
        'ExecutionEnvironment', 'gpu', ...
        'MiniBatchSize', 14, ...
        'Shuffle', 'every-epoch', ...
        'DispatchInBackground', true);
    
    net = trainNetwork(trainCds, layers, options);
    
    saveDir = fullfile('models');
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end
    modelFileName = fullfile(saveDir, 'trained_network.mat');
    saveLearnerForCoder(net, modelFileName);
    fprintf('Model saved as: %s\n', modelFileName);
end

function img = preprocessImage(filename)
    img = imread(filename);
    img = single(img)/255;
    img = reshape(img, [size(img,1), size(img,2), 1]);
end

function [X,Y] = preprocessTrainingData(data)
    X = data{1};
    Y = data{2};
end
