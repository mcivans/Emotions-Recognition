function yRab = testRAB8(P,SIFT,HOG,GABOR,testLbl_)
H = zeros(sum(testLbl_),1);
[N] = length(H);
Nb = 32;
intervals = (-1:2/Nb:1)+1/Nb;
intervals = intervals(1:end-1);

for t = 1:length(P)
    p = P{t};   
     if (p.type == 1)
       y1 = SIFT(testLbl_,p.fInd); 
    elseif p.type == 2
       y1 = HOG(testLbl_,p.fInd); 
    elseif p.type == 3
       y1 = GABOR(testLbl_,p.fInd); 
    end;    
    h = p.h;
    indxs{1} = y1 < intervals(1);
    for thr = 1:Nb-1
        indxs{thr+1} = (y1 >= intervals(thr)) & (y1 < intervals(thr+1));
    end;
    indxs{Nb+1} = y1 >= intervals(Nb);           
    for cntr = 1:Nb+1
         H(indxs{cntr}) = H(indxs{cntr}) + h(cntr);
    end;
end;

yRab = (sign(H)+1)/2;
   