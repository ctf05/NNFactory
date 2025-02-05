function NeuralNetworkFactory
    imgs = dir(fullfile('training', '*.jpg'));
    allKeys = {};
    for i = 1:numel(imgs)
        [~, name, ~] = fileparts(imgs(i).name);
        jsonFile = fullfile('training', [name '.json']);
        jsonStr = fileread(jsonFile);
        data = jsondecode(jsonStr);
        allKeys = union(allKeys, fieldnames(data));
    end
    sortedKeys = sort(allKeys);
    numOutputs = numel(sortedKeys);
    inputSize = [224 224 1];
    images = zeros([inputSize, numel(imgs)], 'single');
    targets = zeros(numel(imgs), numOutputs);
    for i = 1:numel(imgs)
        img = imread(fullfile('training', imgs(i).name));
        img = imresize(img, inputSize(1:2));
        images(:,:,:,i) = single(img)/255;
        [~, name, ~] = fileparts(imgs(i).name);
        jsonFile = fullfile('training', [name '.json']);
        jsonStr = fileread(jsonFile);
        data = jsondecode(jsonStr);
        for j = 1:numOutputs
            targets(i,j) = data.(sortedKeys{j});
        end
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
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride',2)
        convolution2dLayer(3, 128, 'Padding','same')
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(512)
        reluLayer
        dropoutLayer(0.5)
        fullyConnectedLayer(numOutputs)
        regressionLayer
    ];
    options = trainingOptions('adam', ...
        'MaxEpochs',5, ...
        'InitialLearnRate',1e-3, ...
        'ValidationData',{valImgs, valTargets}, ...
        'Plots','training-progress', ...
        'Verbose',true, ...
        'ExecutionEnvironment','gpu', ...
        'MiniBatchSize', 16);
    net = trainNetwork(trainImgs, trainTargets, layers, options);
end