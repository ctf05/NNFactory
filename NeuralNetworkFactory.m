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
        'DispatchInBackground', true, ...
        'OutputFcn', @(info)outputFcn(info, valCds));
    
    net = trainNetwork(trainCds, layers, options);
    
    saveDir = fullfile('models');
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end
    modelFileName = fullfile(saveDir, 'trained_network.mat');
    save(modelFileName, 'net');
    fprintf('Model saved as: %s\n', modelFileName);

    fprintf('\n=== Testing Some Predictions ===\n');

    testBatch = preview(valCds);
    testImages = testBatch{1};
    actualParams = testBatch{2};
    
    predictions = predict(net, testImages);
    
    for i = 1:min(5, size(predictions,1))
        fprintf('\nSample %d:\n', i);
        fprintf('Predicted: D=%.4f, C=%.4f, B=%.4f, G=%.4f, F=%.4f, J=%.4f, E=%.4f, I=%.4f\n', ...
            predictions(i,:));
        fprintf('Actual:    D=%.4f, C=%.4f, B=%.4f, G=%.4f, F=%.4f, J=%.4f, E=%.4f, I=%.4f\n', ...
            actualParams(i,:));
        
        errors = abs(predictions(i,:) - actualParams(i,:));
        fprintf('Abs Error: D=%.4f, C=%.4f, B=%.4f, G=%.4f, F=%.4f, J=%.4f, E=%.4f, I=%.4f\n', ...
            errors);
    end
end

function img = preprocessImage(filename)
    img = imread(filename);
    img = single(img)/255;
    img = reshape(img, [size(img,1), size(img,2), 1]);
end

function out = preprocessTrainingData(data)
    out = {data{1}, data{2}};  
end

function stop = outputFcn(info, validationData)
    stop = false;
    
    if mod(info.Iteration, 100) == 0
        fprintf('\n=== Training Progress at Iteration %d ===\n', info.Iteration);
        fprintf('Epoch: %d\n', info.Epoch);
        fprintf('Training Loss: %.6f\n', info.TrainingLoss);
        fprintf('Training RMSE: %.6f\n', info.TrainingRMSE);
        if ~isempty(info.ValidationLoss)
            fprintf('Validation Loss: %.6f\n', info.ValidationLoss);
            fprintf('Validation RMSE: %.6f\n', info.ValidationRMSE);
        end
        fprintf('Time Since Start: %.2f minutes\n', info.TimeSinceStart/60);
        fprintf('Current Learning Rate: %.6f\n', info.BaseLearnRate);
        fprintf('===================================\n\n');
    end
end
