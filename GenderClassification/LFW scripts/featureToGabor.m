function [ totalInd ] = featureToGabor(ft,SZ)
    % GABOR ��� �������������� ������������ �� 2 �� ����������� � ���������
    % ������� � ��������� ������ ����� �������� �� 2
    ft = floor(ft/2);
    ft(ft==0) = 1;
    % ��� ��� ������ ��������� �������� � ����������� �� ������� �����, ��
    % �������� ����������� ���� �� ������ �� ����������
    subFea = zeros(SZ/2,SZ/2);
    step2 = floor((ft(4)-ft(2)+1)/3);
    step2 = min(max(step2,1),3);
    step1 = floor((ft(3)-ft(1)+1)/3);
    step1 = min(max(step1,1),3);
    subFea(ft(2):step2:ft(4),ft(1):step1:ft(3)) = 1;  
    % ������� �������� ��������
    ind = find(subFea(:) == 1);  
    % ���������� �� 18=3*6 ��������� � ���������
    totalInd = zeros(length(ind)*18,1);
    for scaleRot = 1:18
        totalInd((scaleRot-1)*length(ind)+1:scaleRot*length(ind)) = (ind + (scaleRot-1)*SZ*SZ/4);
    end;
end

