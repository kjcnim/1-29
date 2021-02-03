%%% 이 코드는 입력받은 레이다 데이터를 이미지로 바꾼뒤, 이를 라벨링 시킬때, 
%%% 라벨링 구역(네모 박스)의 왼쪽 위, 오른쪽 아래. 이렇게 2개의 점을 찍으면
%%% 자동적으로 [X축, Y축, 너비, 높이] 데이터 정보가 입력되고
%%% 자동적으로 레이다 데이터 이미지가 저장되며(이때, 폴더 미리 설정)
%%% 자동적으로 data 값들이 하나의 1x1 struct에 입력되어 YOLO알고리즘에 그대로 대입시킬 수 있게 만들었다.

clear;
close all;
clc;

load('Test_Data.mat')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 실제 차량용 FMCW 레이더 데이터 샘플
%%% 좌회전 : 10장, 우회전 : 10장
%%% 각 데이터는 크기가 127 x 51인 행렬
%%% 10 x 127 x 51 : Cube expression of processed radar signal.
%%% 적절한 픽셀 크기로 라벨링 -> YOLO 네트워크 설계 후 돌려보기
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %행렬 초기화
    L = int16.empty(0,4)
    L_A = int16.empty(0,4)
    
    R = int16.empty(0,4)
    R_A = int16.empty(0,4)
    
    data_count = 10; %사용할 레이다 이미지 개수 
   
for ii = 1:data_count
    %%좌회전
 
    subplot(1,2,1)
    imagesc(squeeze(left_turn_Data(ii, :, :).^1.5));   
            
    xlabel('Distance (m)')
    ylabel('Angle (deg.)')
    
    t = squeeze(left_turn_Data(ii, :, :).^1.5)
    letter_left = 'left\\left_%d.jpg';
    str_left = sprintf(letter_left,ii);
    imwrite( ind2rgb(im2uint8(mat2gray(t)), parula(256)), str_left)

    
    % 점을 찍는 순서는 왼쪽 위, 오른쪽 아래
    [X,Y] = ginput(2);

    
    L(1,1) = X(1,1) % X좌표
    L(1,2) = Y(1,1) % Y좌표
    L(1,3) = X(2,1) - X(1,1) % 너비
    L(1,4) = Y(2,1) - Y(1,1) % 높이 
    
    L_A = [L_A; L];
    
    %% 우회전
    subplot(1,2,2)
    imagesc(squeeze(right_turn_Data(ii, :, :).^1.5)); % 우회전
    xlabel('Distance (m)')
    ylabel('Angle (deg.)')
 
    m = squeeze(right_turn_Data(ii, :, :).^1.5)
    letter_right = 'right\\right_%d.jpg';
    str_right = sprintf(letter_right,ii);
    imwrite( ind2rgb(im2uint8(mat2gray(m)), parula(256)), str_right)
    
    [A,B] = ginput(2);   
  
    R(1,1) = A(1,1) % X좌표
    R(1,2) = B(1,1) % Y좌표
    R(1,3) = A(2,1) - A(1,1) % 너비
    R(1,4) = B(2,1) - B(1,1) % 높이

    R_A = [R_A; R];
    
end

%%셀 저장
Left_imageFilename = cell(data_count,1)
Right_imageFilename = cell(data_count,1)
LEFT = cell(data_count,1)
RIGHT = cell(data_count,1)

for ii = 1:data_count
    Left_imageFilename{ii,1} = sprintf('left\\left_%d.jpg',ii)
    Right_imageFilename{ii,1} = sprintf('\right\\right_%d.jpg',ii)
    LEFT{ii,1} = L_A(ii, :)
    RIGHT{ii,1} = R_A(ii, :)
end

T_left = table(Left_imageFilename,LEFT);
T_right = table(Right_imageFilename,RIGHT);
T = table(LEFT,RIGHT);

%구조체 struct
%data_left = struct('turn_left',{T_left})
%data_right = struct('turn_right',{T_right})
data_all = struct('turn_left',{T_left},'turn_right',{T_right} )

save('savefile.mat', 'T')
save('leftfile.mat','T_left')
save('rightfile.mat','T_right')
save('Turning Inform.mat','data_all')