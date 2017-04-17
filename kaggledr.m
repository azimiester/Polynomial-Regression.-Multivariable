clear;
training_data = csvread('RegressionTraining.csv',1,1);
errors = [];
feat=50;
[n, variates] = size(training_data);
variates = variates - 1;
for degree = 1:8;
    avError = 0;
    perm = randperm(n);
    for i = 1:n
        training_idx = perm([1:i-1 i+1:end]);
        test_idx = perm(i);
        trainingData = training_data(training_idx, 1:end-1);
        trainingData = trainingData(:, 1:feat);
        validationData = training_data(test_idx, 1:end-1);
        targets = training_data(training_idx,51);
        outputVal = training_data(test_idx,51);

        %%Find the PowerMatrix
        A = getBusted(feat,degree,n);
        repeatMat = repmat(trainingData, 1, degree+1);
        phi = repeatMat.^A;

        %% Find the weights by pseudoinverse technique
        weight = pinv(phi' * phi) * phi' * targets;
        %% Validation
        A = getBusted(feat,degree,2);
        phi_val = repmat(validationData,1,degree+1);
        phi_val = phi_val.^A;
        %% Find the output and the error
        output = phi_val * weight;
        errorTest = mean((outputVal - output).^2);
        avError = avError + errorTest;
    end;
avError = avError;
text = ['The validation mean squared error for degree',num2str(degree), ' is ',num2str(avError)];
disp(text)
errors = [errors avError];
end;
[m,i] = min(errors);
i = 3;
testingData = csvread('RegressionTesting.csv',1,1);
testOutput = csvread('RegressionSolution.csv',1,1);
A = getBusted(feat,i,n);
repeatMat = repmat(trainingData, 1, i+1);
phi = repeatMat.^A;
weight = pinv(phi' * phi) * phi' * targets;
A = getBusted(feat,i,1000+1);
phi = repmat(testingData,1,i+1);
phi = phi.^A;
output = phi * weight;
errorTest = mean((testOutput - output).^2);
text = ['The test mean squared error for degree ',num2str(i), ' is ',num2str(errorTest)];
disp(text);
disp(['Accuracy= ', num2str(100-errorTest *100)]);