clear variables; clc;

run('D:/vlfeat-0.9.20-bin/vlfeat-0.9.20/toolbox/vl_setup')

imageDir = 'D:/_Repositories/GenderclassificationCAL/LDB/';
listing = dir(imageDir);

% ������� �����������
he = 60;
wi = 60;

cntr = 0;
list =[]; % ������ ����� ��������� �����������
lbl = []; % ��������
for i = 3:length(listing);
    cntr = cntr+1;
    list(cntr).img = double(imread([imageDir listing(i).name]));
    if isempty(findstr(listing(i).name,'SMILE'))
        lbl(cntr) = 1;
    else
        lbl(cntr) = 0;
    end;
end;
B=8;%����� ����� �����������
step = 6; 

testLbl = [lbl'];
% ���� � ��������� 20/80
% load testLblSnap;

% T = 0.8; % ��� ���� ��������
% testLbl_ = rand(size(testLbl)) < T;
testLbl_ = rand(size(testLbl)) < 2;
% TLD = testLbl(testLbl_); % ��������� ����� 0/1
TLD = testLbl;
TLD_2 = TLD*2-1; % ��������� ����� +/- 1
SIFT_SIZE = 128; % ������ ������ SIFT
SIFT_CNT = 300;  % ����� SIFT'��
HOG_CNT = 500;   % ����� HOG'��
HOG_SIZE = 32;   % ������ HOG'�

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ����������� ���� ��� ���������. ����� ����� ��� ��� ������� �� �����
% load 

HOGi = zeros(length(testLbl_),HOG_CNT); % �������, ��� ��������� ����� �������, ���������� ����� ��������� ��� HOG
SIFT_ = zeros(length(testLbl_),SIFT_CNT);% �������, ��� ��������� ����� �������, ���������� ����� ��������� ��� SIFT
fea = []; % ������ ��������� ������ ��� SIFT
feaH = []; % ������ ��������� ������ ��� HOG
% load fea_feaH % �������� ��������� ������ ���������������
% ��������� ������� ��������� ������
for ITER = 1:SIFT_CNT
    startHe = 1 + floor(rand(1)*(he-16));
    he_step = floor(max(rand(1)*min(he-startHe,30),16));

    endHe = startHe + he_step;
    startWi = 1 + floor(rand(1)*(wi-16));
    randmax = rand(1)*400 + 1000;
    endWi = startWi + floor(max(rand(1)*min((wi-startWi),max(1200/he_step,1)),16)); 

    if (rand(1) >0.5)                	
        tmp = he + 1 - endHe;
        endHe = he +1 - startHe;
        startHe = tmp;
    end;
    if (rand(1) > 0.5)
        tmp = wi + 1 - endWi;
        endWi = wi + 1 - startWi;
        startWi = tmp;
    end;    
    fInd = ITER;
    fea{fInd} = [startHe startWi endHe endWi];
end;
% ���������� ���������
alfaSIFT=[];
 for ITER = 1:SIFT_CNT
     %fileName = ['C:\prog\FEATURES_GENERATE\csvs\fea_'  int2str(ITER-1)  '.csv'];
     fileName = ['D:/_Repositories/GenderclassificationCAL/MATLAB Package/FEATURES_GENERATE/fea_'  int2str(ITER-1)  '.csv'];
%     ITER
     for i = 1:length(lbl);        
        A = list(i).img;   
        Acropped = A(fea{ITER}(1):fea{ITER}(3),fea{ITER}(2):fea{ITER}(4)); 
        [ps, f] = vl_dsift(im2single(Acropped),'Step',16,'Size',4 );
        if i==1
            tempS = zeros(length(lbl),length(f(:)));
        end;
        tempS(i,:) = f(:)';
    end;
     X = tempS(testLbl_,:);
     if (rank(X) < size(X,2))
         [U S V] = svd(X,0); % SVD ������������ ��� ������, ����� ������������ �� ����� ���� �������� �������� ����
                             % ��� ��� ������� ����������������
         s = S(logical(eye(size(S))));
         ts = find(s<0.01,1);
         U = U(:,1:ts);
         S = S(1:ts,1:ts);
         V = V(:,1:ts);
         U = [U ones(size(U,1),1)];
         betta = (U'*U) \ (U'*TLD_2);
         alfa = V*(diag(1./s(1:ts)))*betta(1:end-1);
         alfa = [alfa;betta(end)];
     else
         X = [X ones(size(X,1),1)];
         alfa = (X'*X) \ (X'*TLD_2);
     end;
     y = [tempS ones(size(tempS,1),1)]*alfa;
     alfaSIFT{ITER} = alfa;
     SIFT_(:,ITER) = y;
end;
% ���������� ��� HOG � GABOR ��������� ������ ����������� � �������� � feaH 
for ITER = 1:HOG_CNT
    startHe = 1 + floor(rand(1)*(he-8));
    he_step = floor(max(rand(1)*min(he-startHe,30),8));

    endHe = startHe + he_step;
    startWi = 1 + floor(rand(1)*(wi-8));
    randmax = rand(1)*300 + 600;
    endWi = startWi + floor(max(rand(1)*min((wi-startWi),max(randmax/he_step,1)),8)); 

    if (rand(1) >0.5)                	
        tmp = he + 1 - endHe;
        endHe = he +1 - startHe;
        startHe = tmp;
    end;
    if (rand(1) > 0.5)
        tmp = wi + 1 - endWi;
        endWi = wi + 1 - startWi;
        startWi = tmp;
    end;
    fInd = ITER;
    feaH{fInd} = [startHe startWi endHe endWi];
end;
% ����������� �������� ������������� ��������� � �������� �������������
% ���������� ��������� � ������-������� HOGi (��� ��������������� HOG) � FG (��� �������)
alfaHOG=[];
for ITER = 1:HOG_CNT
    ITER
    for i = 1:length(lbl);        
        A = list(i).img;   
        Acropped = A(feaH{ITER}(1):feaH{ITER}(3),feaH{ITER}(2):feaH{ITER}(4)); 
        f = indirectHOG(Acropped);% vl_hog(im2single(Acropped),step,'NumOrientations',B,'Variant','DalalTriggs');
        if i==1
            tempH = zeros(length(lbl),length(f(:)));
        end;
        tempH(i,:) = f(:)';
    end;
    X = tempH(testLbl_,:);
    if (rank(X) < size(X,2))
         [U S V] = svd(X,0);
         s = S(logical(eye(size(S))));
         ts = find(s<0.01,1);
         U = U(:,1:ts);
         S = S(1:ts,1:ts);
         V = V(:,1:ts);
         U = [U ones(size(U,1),1)];
         %betta = (U'*dw*U) \ (U'*dw*y);
         betta = (U'*U) \ (U'*TLD_2);
         alfa = V*(diag(1./s(1:ts)))*betta(1:end-1);
         alfa = [alfa;betta(end)];
     else
        X = [X ones(size(X,1),1)];
        alfa = (X'*X) \ (X'*TLD_2);
    end;   
    y = [tempH ones(size(tempH,1),1)]*alfa;
    alfaHOG{ITER} = alfa;
    HOGi(:,ITER) = y;
end;
%%%%%%%%%%%%%%%%%
% � ���� ����, ��� ��������� GABOR � ������� ���, ����� � OpenCV ��������
% ������� ��� �� �������������� ���������, ������ ������ ����
% �������������� ���������� � OpenCV � �������� ������
GABORT = readtable('D:\_Repositories\GenderclassificationCAL\FACES_PREDICT_14_04\GABORS_smile.csv','ReadVariableNames',false);
GABOR = table2array(GABORT);
%%%%%%%%%%%%%%%%%%
% ������� csv ����� � ���� ����� ������� � ������, ������� � �������� �� ��� 2
%GABOR1 = table2array(GABORT);
%GABORT = readtable('D:/_Repositories/GenderclassificationCAL/FACES_PREDICT_14_04/GABORS2.csv');
%GABOR2 = table2array(GABORT);
%clear GABORT; % � ���� �� ������� ���������, ��� ��� � ����� ������ �������� ��� �������
%GABOR = [GABOR1;GABOR2];
%clear GABOR1;
%clear GABOR2;
%GABOR = GABOR(:,1:30*30*18);

% �� ���������� GABOR ����������� ������� ���������
G_CNT = HOG_CNT; % ����� ��� GABOR �������� �� ��
FHG = zeros(size(HOGi,1),G_CNT); % ���������, ����������� SIFT_ � HOGi
tis = zeros(1,G_CNT); % ��� ��� ���� ������� ������� �� ����� �������������
% - ����� ������� ����� ������������� ��� ������� ������-������� FHG
alfaGABOR=[];
for ITER=1:G_CNT;
    ITER
    ti = featureToGabor(feaH{ITER}, 60); % ������� ��������� �����, � ������� � ������� GABOR
    tis(ITER) = length(ti);
    tempG = GABOR(:,ti);
    X = tempG(testLbl_,:);
    X = [X ones(size(X,1),1)];
    alfa = (X'*X) \ (X'*TLD_2); % ��� GABOR ��� ��� ��� �� �����������, � �������� ��������������� ����������� -
    % �� �� ������������ � ���������� ���������������� ������� X, �������
    % ������������� � SVD - ���
    y = [tempG ones(size(tempG,1),1)]*alfa;
    alfaGABOR{ITER} = alfa;
    FHG(:,ITER) = y;   
end;

% �������� ��� �������� ������
% lbl2 = testLbl(~testLbl_);
%% RealAdaBoost
% ������� �������� lbl2 ������������ ��� �������� �������� ����������
% � ��������� �� ���������. ��� ��������� - ����� ����������������
% ��������� ���� � RAB8 � ������ *0 � lbl2

P = RAB(SIFT_,HOGi,FHG,tis,testLbl_,fea,feaH, TLD, 0);
gaborCount = 0;
hogCount=0;
FH2 = zeros(size(HOGi,1),length(P)); % ������ �� ������������, ����� ���������, ���������� RAB
for i = 1:length(P)
    i2 = P{i}.fInd;
    if (P{i}.type == 1)
        FH2(:,i) = SIFT_(:,i2);
		P{i}.alfa = alfaSIFT{i2};
    elseif P{i}.type == 2
        hogCount = hogCount + 1;
        FH2(:,i) =  HOGi(:,i2);
		P{i}.alfa = alfaHOG{i2};
    else
        gaborCount = gaborCount + 1;
        FH2(:,i) =  FHG(:,i2);
		P{i}.alfa = alfaGABOR{i2};
    end
end;
% �������� SVM �������� �� ��� �� 80% ������
param = ['-t 0 -q'];
%disp(FH2(testLbl_,:));
model2 = svmtrain(TLD, FH2(testLbl_,:), param);
%model2 = svmtrain(FH2(testLbl_,:), TLD, param);
% model2.rho = -2.6;
% ������������
% [pl3 ac pr] = svmpredict(lbl2, FH2(~testLbl_,:), model2);
% ����������
% [ac(1) hogCount gaborCount (1-[mean(1-pl3(lbl2==1)) mean(pl3(lbl2==0))])]
% ���� ��� ���������� - ��������� P � model2

% ��� ��� ��������� ��������� ������
% I = imresize(A,[240,240]);
% imshow(mat2gray(I))
% hold on;
% for i = 1:17
%     p = P{i};
%     if (p.type == 1)
%         rectangle('EdgeColor',[0 0 1],'Position',4*[fea{p.fInd}(2),fea{p.fInd}(1) ...
%             ,(fea{p.fInd}(4)-fea{p.fInd}(2)+1),(fea{p.fInd}(3)-fea{p.fInd}(1)+1)]);    
%     elseif (p.type == 2)
%         rectangle('EdgeColor',[0 1 0],'Position',4*[feaH{p.fInd}(2),feaH{p.fInd}(1) ...
%             ,(feaH{p.fInd}(4)-feaH{p.fInd}(2)+1),(feaH{p.fInd}(3)-feaH{p.fInd}(1)+1)]);    
%     else
%         rectangle('EdgeColor',[1 0 0],'Position',4*[feaH{p.fInd}(2),feaH{p.fInd}(1) ...
%             ,(feaH{p.fInd}(4)-feaH{p.fInd}(2)+1),(feaH{p.fInd}(3)-feaH{p.fInd}(1)+1)]);    
%     end;
% end;
% hold off;