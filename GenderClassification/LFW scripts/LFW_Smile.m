clear variables; clc;

run('D:/vlfeat-0.9.20-bin/vlfeat-0.9.20/toolbox/vl_setup')

imageDir = 'D:/_Repositories/GenderclassificationCAL/LDB/';
listing = dir(imageDir);

% Размеры изображения
he = 60;
wi = 60;

cntr = 0;
list =[]; % Список будет содержать изображения
lbl = []; % Разметка
for i = 3:length(listing);
    cntr = cntr+1;
    list(cntr).img = double(imread([imageDir listing(i).name]));
    if isempty(findstr(listing(i).name,'SMILE'))
        lbl(cntr) = 1;
    else
        lbl(cntr) = 0;
    end;
end;
B=8;%Число бинов гистограммы
step = 6; 

testLbl = [lbl'];
% Путь к разбиению 20/80
% load testLblSnap;

% T = 0.8; % Так было получено
% testLbl_ = rand(size(testLbl)) < T;
testLbl_ = rand(size(testLbl)) < 2;
% TLD = testLbl(testLbl_); % Обучающие метки 0/1
TLD = testLbl;
TLD_2 = TLD*2-1; % Обучающие метки +/- 1
SIFT_SIZE = 128; % Размер одного SIFT
SIFT_CNT = 300;  % Число SIFT'ов
HOG_CNT = 500;   % Число HOG'ов
HOG_SIZE = 32;   % Размер HOG'а

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Особенности один раз обучаются. После этого уже они берутся из файла
% load 

HOGi = zeros(length(testLbl_),HOG_CNT); % Матрица, где столбцами будут векторы, полученные путем регрессии для HOG
SIFT_ = zeros(length(testLbl_),SIFT_CNT);% Матрица, где столбцами будут векторы, полученные путем регрессии для SIFT
fea = []; % Массив положений блоков для SIFT
feaH = []; % Массив положений блоков для HOG
% load fea_feaH % Загрузка положений блоков предвычисленных
% Получение массива положений блоков
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
% Вычисление регрессий
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
         [U S V] = svd(X,0); % SVD используется для случая, когда коэффициенты не могут быть получены решением СЛАУ
                             % так как матрица недоопределенная
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
% Аналогично для HOG и GABOR положение блоков вычисляется и хранится в feaH 
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
% Вычисляется значение коэффициентов регрессии и значения регрессионной
% переменной заносятся в вектор-столбец HOGi (для ненаправленного HOG) и FG (для Габоров)
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
% В силу того, что вычислить GABOR в матлабе так, чтобы в OpenCV значения
% совпали мне не представляется возможным, Габоры должны быть
% предварительно вычисленны в OpenCV и читаться оттуда
GABORT = readtable('D:\_Repositories\GenderclassificationCAL\FACES_PREDICT_14_04\GABORS_smile.csv','ReadVariableNames',false);
GABOR = table2array(GABORT);
%%%%%%%%%%%%%%%%%%
% большие csv файлы у меня плохо залазят в память, поэтому я сохраняю их как 2
%GABOR1 = table2array(GABORT);
%GABORT = readtable('D:/_Repositories/GenderclassificationCAL/FACES_PREDICT_14_04/GABORS2.csv');
%GABOR2 = table2array(GABORT);
%clear GABORT; % У меня не хватает опреативы, так что я сразу удаляю ненужные мне массивы
%GABOR = [GABOR1;GABOR2];
%clear GABOR1;
%clear GABOR2;
%GABOR = GABOR(:,1:30*30*18);

% По полученным GABOR вычисляются вектора регрессии
G_CNT = HOG_CNT; % Блоки для GABOR беруться те же
FHG = zeros(size(HOGi,1),G_CNT); % Структура, аналогичная SIFT_ и HOGi
tis = zeros(1,G_CNT); % Так как веса габоров зависят от числа коэффициентов
% - нужно хранить число коэффициентов для каждого вектор-столбца FHG
alfaGABOR=[];
for ITER=1:G_CNT;
    ITER
    ti = featureToGabor(feaH{ITER}, 60); % Перевод положения блока, в индексы в массиве GABOR
    tis(ITER) = length(ti);
    tempG = GABOR(:,ti);
    X = tempG(testLbl_,:);
    X = [X ones(size(X,1),1)];
    alfa = (X'*X) \ (X'*TLD_2); % Для GABOR так как это не гистограммы, а значения отфильтрованных изображений -
    % Мы не сталкиваемся с проблеммой недоопределенной матрицы X, поэтому
    % необходимости в SVD - нет
    y = [tempG ones(size(tempG,1),1)]*alfa;
    alfaGABOR{ITER} = alfa;
    FHG(:,ITER) = y;   
end;

% Разметка для тестовых данных
% lbl2 = testLbl(~testLbl_);
%% RealAdaBoost
% Входной параметр lbl2 используется для контроля процесса сходимости
% в рассчетах не участвует. Для просмотра - нужно раскоментировать
% последний блок в RAB8 и убрать *0 у lbl2

P = RAB(SIFT_,HOGi,FHG,tis,testLbl_,fea,feaH, TLD, 0);
gaborCount = 0;
hogCount=0;
FH2 = zeros(size(HOGi,1),length(P)); % Массив из особенностей, после регрессии, отобранных RAB
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
% Линейный SVM обученый на тех же 80% данных
param = ['-t 0 -q'];
%disp(FH2(testLbl_,:));
model2 = svmtrain(TLD, FH2(testLbl_,:), param);
%model2 = svmtrain(FH2(testLbl_,:), TLD, param);
% model2.rho = -2.6;
% Предсказание
% [pl3 ac pr] = svmpredict(lbl2, FH2(~testLbl_,:), model2);
% Результаты
% [ac(1) hogCount gaborCount (1-[mean(1-pl3(lbl2==1)) mean(pl3(lbl2==0))])]
% Если все устраивает - сохраняем P и model2

% Код для просмотра положений блоков
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