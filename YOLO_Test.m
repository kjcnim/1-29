data = load('Turning Inform.mat') % 구조체 데이터 갖고옴

leftDataset = data.data_all.turn_left % 10x2 table인 leftdataset 갖고옴

leftDataset(1:4,:) % 그 중에 처음 4행까지 보여준다

leftDataset.Left_imageFilename = fullfile(pwd,leftDataset.Left_imageFilename) % leftdataset에서 이미지파일경로를 절대경로로 바꿈

rng(0)

shuffledIndices = randperm(height(leftDataset)); %leftdataset의 행의 길이사진갯수)만큼의 숫자를 랜덤으러 섞어 배열한다. 10개의 숫자가 무작위로 배열되있음

idx = floor(0.6 * length(shuffledIndices)); % 트레이닝 데이터는 60%로 잡았기 떄문에 10개 중 6개를 인덱스로 잡는다.

trainingIdx = 1: idx % [1,2,3,4,5,6]

trainingDataTbl = leftDataset(shuffledIndices(trainingIdx),:) %랜덤으로 6개를 갖고와 트레이닝데이터 테이블을 만듬. 여기서 랜덤은 위에서 정햇음

validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) ); % 20퍼센트는 모의고사 데이터로 만듬
validationDataTbl = leftDataset(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices); % 20퍼센트는 테스트 데이터로만듬
testDataTbl = leftDataset(shuffledIndices(testIdx),:);

imdsTrain = imageDatastore(trainingDataTbl{:,'Left_imageFilename'}); % 트레이닝 이미지파일들을 imagedatastore함수를 사용하여 관리하도록함
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'LEFT')); % 라벨링한 녀석들을 boxlabeldatastore함수 사용하여 관리하도록함

imdsValidation = imageDatastore(validationDataTbl{:,'Left_imageFilename'});
bldsValidation = boxLabelDatastore(validationDataTbl(:,'LEFT'));

imdsTest = imageDatastore(testDataTbl{:,'Left_imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:,'LEFT'));

% 영상 데이터저장소와 상자 레이블 데이터저장소를 결합한다.
trainingData = combine(imdsTrain,bldsTrain); % 사진과 라벨링값을 열단위로 concatenating한다.
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);

% 상자 레이블과 함께 훈련 영상 중 하나를 표시한다.
data = read(trainingData); % 트레이닝 데이터의 값들을 읽어와 1x3행렬인데 이걸 data에 저장한다.
I = data{1}; % data의 첫번째 열 값은 이미지데이터 127x51x3녀석인데 이걸 I에 저장
bbox = data{2}; % 두번쨰놈은 라벨링 데이터이다 이를 bbox에 저장
annotatedImage = insertShape(I,'Rectangle',bbox); % I라는 이미지에 bbox에 저장된 녀석을 rectangle form을 가진 모양으로 집어넣음
annotatedImage = imresize(annotatedImage,4); %이미지를 n배확대한다.
figure
imshow(annotatedImage)
inputSize = [508 204 3]; % 신경망 입력 크기

numClasses = width(leftDataset)-1; % 측정할 사물의 갯수

trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 3;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors)

featureExtractionNetwork = resnet50;

featureLayer = 'activation_40_relu';

lgraph = yolov2Layers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);


%%%% 데이터 증대

augmentedTrainingData = transform(trainingData,@augmentData);

% Visualize the augmented images.
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'Rectangle',data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,'BorderSize',10)

%훈련 데이터 전처리하기

preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
preprocessedValidationData = transform(validationData,@(data)preprocessData(data,inputSize));

data = read(preprocessedTrainingData);

I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)


% 사물 검출기 훈련시키기

options = trainingOptions('sgdm', ...
        'MiniBatchSize',16, ....
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',3,...
        'CheckpointPath',tempdir, ...
        'ValidationData',preprocessedValidationData);
    
    
% Train the YOLO v2 detector.
[detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options);
    
I = imread(testDataTbl.Left_imageFilename{2});
I = imresize(I,inputSize(1:2));
[bboxes,scores]  = detect(detector, I);

I = insertObjectAnnotation(I,'rectangle',bboxes,cellstr(labels));
figure
imshow(I)

%테스트 세트를 사용하여 검출기 평가하기

preprocessedTestData = transform(testData,@(data)preprocessData(data,inputSize));

detectionResults = detect(detector, preprocessedTestData);

[ap,recall,precision] = evaluateDetectionPrecision(detectionResults, preprocessedTestData);

figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f',ap))