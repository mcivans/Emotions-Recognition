function P = RAB(SIFT,HOG,GABOR,tis,testLbl_,fea, feaH, y, testY)

trainY = y; % Обучающий набор
y = y*2-1;  % у - из статьи (+/-1)

[N] = length(y); % Число обучающий примеров
Nb = 32;         % Число бинов между -1 и 1
intervals = (-1:2/Nb:1)+1/Nb;
intervals = intervals(1:end-1); % интервалы
CA = 0.03;      % Complex Awere Factor
M = 800;        % Число иттераций поиска блоков
T = 25;         % Число иттераций RelaAdaBoost
NP_HOG = size(HOG,2); % Число коэффициентов HOG
NP_SIFT =size(SIFT,2);% Число коэффициентов SIFT
NP_GABOR =size(GABOR,2);% Число коэффициентов GABOR
weights = ones(length(y),1)/length(y); % Веса, соответствующие примерам
h = zeros(N,1); % Массив порогов для каждого из примеров
eps = 0.003; % Сглаживающий параметр
falsePos = 0.5; % Стартовое FP
siftCnt = 0;
for t = 1:T
    weights = weights.*exp(-(y.*h));
    weights = weights / sum(weights); 
    Z = zeros(1,M);
    types = [];
    for m = 1 : M
        if (mod(m,3) == 1)
            fInd = floor(NP_SIFT*rand(1)+1);
            type = 1;
        elseif (mod(m,3) == 2)
            fInd = floor(NP_HOG*rand(1)+1);
            type = 2;
        else
            fInd = floor(NP_GABOR*rand(1)+1);
            type = 3;
        end; 
        types{m} = type;
        if (type == 1)
            y1 = SIFT(testLbl_,fInd);            
            ft = fea{fInd};           
            Complex_Aware = 10;
            if (siftCnt>=2)
                Complex_Aware = 40;
            end;
        elseif (type == 2)
            y1 = HOG(testLbl_,fInd); 
            ft = feaH{fInd};
            Complex_Aware = 2;
         elseif (type == 3)
            y1 = GABOR(testLbl_,fInd); 
            ft = feaH{fInd};
%             Complex_Aware = 1 + tis(fInd)/90; % Для DOG
            Complex_Aware = 2 + tis(fInd)/150;% без DOG
        end;     
        cas{m} = Complex_Aware; % Значение фактора, в зависимости от типа
        fIndxs{m} = fInd; % Индекс блока особенности
        % Вычисляем гистограммы распределения для данного блока
        cntr = 1;
        indxs{1} = y1 < intervals(1);
        for thr = 1:Nb-1
            indxs{thr+1} = (y1 >= intervals(thr)) & (y1 < intervals(thr+1));
        end;
        indxs{Nb+1} = y1 >= intervals(Nb);   

        for cntr = 1:Nb+1
            Wp(cntr) = sum(weights(indxs{cntr}).*(y(indxs{cntr}) + 1)/2);
            Wm(cntr) = sum(weights(indxs{cntr}).*(1 - y(indxs{cntr}))/2);
        end;

        Wps{m} = Wp;
        Wms{m} = Wm;
        % Значение весовой функции для m-го блока
        Z(m) = 2*sum(sqrt(Wp.*Wm)) + Complex_Aware*falsePos*CA;
    end;
    % Находим минимум весовой функции 
    [~, minInd] = min(Z);
    p = [];    
    p.type = types{minInd}; % Сохраняем тип и индекс особенности
    p.fInd = fIndxs{minInd};
    
    if (p.type == 1)
        siftCnt = siftCnt+1;
       y1 = SIFT(testLbl_, p.fInd); 
    elseif p.type == 2
       y1 = HOG(testLbl_, p.fInd); 
    elseif p.type == 3
       y1 = GABOR(testLbl_,p.fInd); 
    end;
    % рассчитываем значение h
    p.h = 1/2*log( (Wps{minInd} + eps)./((Wms{minInd}+eps)));
    % Сохраняем в общую структуру
    P{t} = p;
    % Проставляем h - значение порога для каждого из примеров
    indxs{1} = y1 < intervals(1);
    for thr = 1:Nb-1
        indxs{thr+1} = (y1 >= intervals(thr)) & (y1 < intervals(thr+1));
    end;
    indxs{Nb+1} = y1 >= intervals(Nb);   
    for cntr = 1:Nb+1
         h(indxs{cntr}) = p.h(cntr);
    end;
    % Предсказание на основе RealAdaBoost
     yRab = testRAB8(P,SIFT,HOG,GABOR,testLbl_);
     % Я использовал следующую формулу для оценки FP 
     falsePos = 0.03 + 1.5*(sum(abs(yRab(trainY==1)-trainY(trainY==1))) + sum(abs(yRab(trainY==0)-trainY(trainY==0))))/length(trainY);
% Можно раскоментировать для анализа сходимости в рассчете не участвует
     % yRab2 = testRAB8(P,SIFT,HOG,GABOR,~testLbl_);
     % [t p.type falsePos sum(abs(yRab2(testY==1)-testY(testY==1)))/length(testY(testY==1)) ...
     % sum(abs(yRab2(testY==0)-testY(testY==0)))/length(testY(testY==0))]
end;        