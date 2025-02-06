function NeuralNetworkFactory
    imgs = dir(fullfile('training', '*.jpg'));
    numFiles = numel(imgs);
    
    randIdx = randperm(numFiles);
    imgs = imgs(randIdx);
    
    inputSize = [224 224 1];
    images = zeros([inputSize, numFiles], 'single');
    targets = zeros(numFiles, 8);
    
    for i = 1:numFiles
        img = imread(fullfile('training', imgs(i).name));
        images(:,:,:,i) = single(img)/255;
        
        [~, name, ~] = fileparts(imgs(i).name);
        params = sscanf(name, 'img_D%f_C%f_B%f_G%f_F%f_J%f_E%f_I%f');
        targets(i,:) = params';
    end
    
    cv = cvpartition(size(images,4), 'HoldOut', 0.2);
    trainImgs = images(:,:,:, cv.training);
    trainTargets = targets(cv.training,:);
    valImgs = images(:,:,:, cv.test);
    valTargets = targets(cv.test,:);
    
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
    
    validationFunction = @(net) customValidation(net, valImgs, valTargets);
    
    options = trainingOptions('adam', ...
        'MaxEpochs',5, ...
        'InitialLearnRate',1e-3, ...
        'ValidationFunction', validationFunction, ...
        'Plots','training-progress', ...
        'Verbose',true, ...
        'ExecutionEnvironment','gpu', ...
        'MiniBatchSize', 16);
    
    net = trainNetwork(trainImgs, trainTargets, layers, options);
end

function [validationError, allPredictions] = customValidation(net, XVal, YVal)
    miniBatchSize = 16;
    numObservations = size(XVal, 4);
    numBatches = ceil(numObservations / miniBatchSize);
    allPredictions = [];
    
    for i = 1:numBatches
        startIdx = (i-1)*miniBatchSize + 1;
        endIdx = min(i*miniBatchSize, numObservations);
        predictions = predict(net, XVal(:,:,:,startIdx:endIdx));
        allPredictions = [allPredictions; predictions];
    end
    
    validationError = mean((allPredictions - YVal).^2, 'all');
end