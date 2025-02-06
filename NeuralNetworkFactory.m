function NeuralNetworkFactory
    imds = imageDatastore(fullfile('training', '*.jpg'), ...
        'ReadFcn', @(x) preprocessImage(x));
    
    numFiles = length(imds.Files);
    fprintf('Number of files: %d\n', numFiles);
    targets = zeros(numFiles, 8);
    for i = 1:numFiles
        [~, name, ~] = fileparts(imds.Files{i});
        name = strrep(name, 'n', '-');
        name = strrep(name, 'p', '.');
        parts = split(name, '_');
        params = zeros(1, 8);
        for j = 2:length(parts)
            param_str = parts{j};
            param_val = str2double(param_str(2:end));
            params(j-1) = param_val;
        end
        targets(i,:) = params;
    end
    
    ds = arrayDatastore(targets, 'OutputType', 'same');
    cds = combine(imds, ds);
    
    cds.MiniBatchSize = 16;
    cds.ReadSize = 16;
    
    inputSize = [225 225 1];
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
    
    validationFunction = @(net) dynamicValidation(net, cds);
    
    options = trainingOptions('adam', ...
        'MaxEpochs', 5, ...
        'InitialLearnRate', 1e-3, ...
        'ValidationFunction', validationFunction, ...
        'Plots', 'training-progress', ...
        'Verbose', true, ...
        'ExecutionEnvironment', 'gpu', ...
        'MiniBatchSize', 16, ...
        'Shuffle', 'every-epoch', ...
        'DispatchInBackground', true);
    
    net = trainNetwork(cds, layers, options);
    
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

function [validationError, allPredictions] = dynamicValidation(net, cds)
    numFiles = length(cds.UnderlyingDatastores{1}.Files);
    validationSize = floor(0.2 * numFiles);
    
    validationIndices = randperm(numFiles, validationSize);
    
    valFiles = cds.UnderlyingDatastores{1}.Files(validationIndices);
    valTargets = cds.UnderlyingDatastores{2}.Data(validationIndices,:);
    
    valImds = imageDatastore(valFiles, ...
        'ReadFcn', @(x) preprocessImage(x));
    valDs = arrayDatastore(valTargets, 'OutputType', 'same');
    valCds = combine(valImds, valDs);
    valCds.MiniBatchSize = 16;
    
    predictions = [];
    actual = [];
    
    reset(valCds);
    while hasdata(valCds)
        [batch, info] = read(valCds);
        batchPreds = predict(net, batch{1});
        predictions = [predictions; batchPreds];
        actual = [actual; batch{2}];
    end
    
    validationError = mean((predictions - actual).^2, 'all');
    allPredictions = predictions;
end