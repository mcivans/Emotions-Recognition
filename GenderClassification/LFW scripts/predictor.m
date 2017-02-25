% predictor
W = model2.sv_coef'*model2.SVs;
addpath 'LFW2/Most Quality';
listing = dir('LFW2/Most Quality');
cntr = 0;
list =[]; % Список будет содержать изображения
lbl = []; % Разметка
for i = 3:length(listing);
    cntr = cntr+1;
    list(cntr).img = double(imread(listing(i).name));
    if isempty(findstr(listing(i).name,'F'))
        lbl(cntr) = 1;
    else
        lbl(cntr) = 0;
    end;
end;
for iter = 1:length(list)
    A = list(iter).img;
    pred = zeros(1,length(P));
    for i = 1:length(P)
        p = P{i};
        type = p.type;
        if type == 1
            ft = fea{p.fInd};
            Acropped = A(ft(1):ft(3),ft(2):ft(4));
            [ps, f] = vl_dsift(im2single(Acropped),'Step',16,'Size',4);
            pred(i) = double([f(:)' 1])*p.alfa;
        elseif type == 2
            ft = feaH{p.fInd};
            Acropped = A(ft(1):ft(3),ft(2):ft(4));
            f = indirectHOG(Acropped);
            pred(i) = double([f(:)' 1])*p.alfa;
        else
             ft = feaH{p.fInd};
             ti = featureToGabor(ft, 60);
             tempg = [GABOR(i,ti) 1];
             pred(i)=tempg*p.alfa;
        end;
    end;
    pr(iter) = svmpredict(lbl(iter),pred,model2)
end;

[sum(abs(pr(lbl==1)-lbl(lbl==1)))/length(lbl(lbl==1)) sum(abs(pr(lbl==0)-lbl(lbl==0)))/length(lbl(lbl==0))]
[sum(abs(pr(lbl==1)-lbl(lbl==1))) sum(abs(pr(lbl==0)-lbl(lbl==0)))]
