clc, clear, close all

doTraining = true;
    
%% 데이터셋 불러오기    
data = load('Turning Inform.mat') % 구조체 데이터 갖고옴
leftDataset = data.data_all.turn_left % 10x2 table인 leftdataset 갖고옴

% Display first few rows of the data set
leftDataset(1:4,:) % 그 중에 처음 4행까지 보여준다

% Add the fullpath to the local vehicle data folder
leftDataset.Left_imageFilename = fullfile(pwd,leftDataset.Left_imageFilename) % leftdataset에서 이미지파일경로를 절대경로로 바꿈

% 데이터셋을 훈련 세트, 검증 세트, 테스트 세트로 분할한다.
% 데이터의 60%를 훈련용으로, 데이터의 10%를 검증용으로, 나머지를 훈련된 검출기의 테스트용으로 선택한다.
rng(0)
shuffledIndices = randperm(height(leftDataset)); %leftdataset의 행의 길이사진갯수)만큼의 숫자를 랜덤으러 섞어 배열한다. 10개의 숫자가 무작위로 배열되있음
idx = floor(0.6 * length(shuffledIndices)); % 트레이닝 데이터는 60%로 잡았기 떄문에 10개 중 6개를 인덱스로 잡는다.

trainingIdx = 1: idx % [1,2,3,4,5,6]
trainingDataTbl = leftDataset(shuffledIndices(trainingIdx),:) %랜덤으로 6개를 갖고와 트레이닝데이터 테이블을 만듬. 여기서 랜덤은 위에서 정햇음

validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) ); % 20퍼센트는 모의고사 데이터로 만듬
validationDataTbl = leftDataset(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices); % 20퍼센트는 테스트 데이터로만듬
testDataTbl = leftDataset(shuffledIndices(testIdx),:);

% imageDatastore와 boxLabelDatastore를 사용하여 훈련과 평가 과정에서 영상 및 레이블 데이터를 불러오기 위한
% 데이터장소를 만든다.
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
annotatedImage = imresize(annotatedImage,2); %이미지를 n배확대한다.
figure
imshow(annotatedImage)

%% YOLO v2 객체 검출 신경망 만들기
% 신경망 입력 크기
inputSize = [224 224 3]; 

% 측정할 사물의 갯수
numClasses = width(leftDataset)-1; 

% estimateAnchorBoxes를 사용하여 훈련 데이터의 사물 크기를 기반으로 앵커 상자를 추정한다.
% 훈련 전 이루어지는 영상 크기 조정을 고려하기 위해 앵커 상자 추정에 사용하는 훈련 데이터 크기를 조정한다.
% transform을 사용하여 훈련 데이터를 전처리한 후에 앵커 상자의 개수를 정의하고 앵커 상자를 추정한다.
% 지원 함수 preprocessData를 사용하여 훈련 데이터를 신경망의 입력 영상 크기로 크기 조정한다.
trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 3;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors)

% resnet50을 사용하여 사전 훈련된 Resnet-50모델을 불러온다.
featureExtractionNetwork = resnet50;

% 'activation_40_relu' 뒤에 오는 계층들을 검출 하위 신경망으로 교체할 특징 추출 계층으로 이를 선택한다.
% 이 특징 추출 계층은 16배만큼 다운샘플링된 특징 맵을 출력한다. 
% 이 정도의 다운샘플링은 공간 분해능과 추출된 특징의 강도 사이를 적절히 절충한 값이다.
% 신경망의 더 아래쪽에서 추출된 특징은 더 강한 영상 특징을 인코딩하나 공간 분해능이 줄어들기 때문이다.
% 최적의 특징 추출 계층을 선택하려면 경험적 분석이 필요하다.
featureLayer = 'activation_40_relu';

% YOLOv2 객체 검출 신경망을 만든다.
lgraph = yolov2Layers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);


%% 데이터 증대

augmentedTrainingData = transform(trainingData,@(data)preprocessData(data,inputSize));

% Visualize the augmented images.
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'Rectangle',data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,'BorderSize',10)

%% 훈련 데이터 전처리하기
preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
preprocessedValidationData = transform(validationData,@(data)preprocessData(data,inputSize));

data = read(preprocessedTrainingData);

I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,1);
figure
imshow(annotatedImage)


%% 사물 검출기 훈련시키기

options = trainingOptions('sgdm', ...
        'MiniBatchSize',16, ....
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',10,...
        'CheckpointPath',tempdir, ...
        'ValidationData',preprocessedValidationData);
    
 if doTraining       
    % Train the YOLO v2 detector.
    [detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options);
 end 
 
I = imread(testDataTbl.Left_imageFilename{1});
I = imresize(I,inputSize(1:2));
figure
imshow(I)
[bboxes,scores] = detect(detector,I)

if ~isempty(bboxes)
    I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
    figure
    imshow(I)
end


%% 테스트 세트를 사용하여 검출기 평가하기

preprocessedTestData = transform(testData,@(data)preprocessData(data,inputSize));

detectionResults = detect(detector, preprocessedTestData);

[ap,recall,precision] = evaluateDetectionPrecision(detectionResults, preprocessedTestData);

figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f',ap))