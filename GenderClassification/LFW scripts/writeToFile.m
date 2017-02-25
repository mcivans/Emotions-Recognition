% clear
%load dataForLFW7
%load PforLFW7
W = model2.sv_coef'*model2.SVs; 
fileID = fopen('D:\_Repositories\GenderclassificationCAL\FACES_PREDICT_14_04\exp_smile.csv','w');
fprintf(fileID,'%i',length(P)); 
fprintf(fileID,'\n');
for i = 1:length(P)
    p = P{i};
    fprintf(fileID,'%i',p.type);
    if (p.type == 1)
        feature = fea{p.fInd};
    else
        feature = feaH{p.fInd};
    end;
    for j = 1:4
        fprintf(fileID,',%i',feature(j));      
    end;
    fprintf(fileID,'\n');
end;
fprintf(fileID,'%.7f',-model2.rho);
for i = 1:length(W)
     fprintf(fileID,',%.7f',W(i));
end;
fprintf(fileID,'\n');
for i = 1:length(P)
    p = P{i};
    alfa = p.alfa;
    for j = 1:length(alfa)
        if (j == 1)
            fprintf(fileID,'%.7f',alfa(j));
        else
            fprintf(fileID,',%.7f',alfa(j));
        end;
    end;
     fprintf(fileID,'\n');
end;

fclose(fileID);