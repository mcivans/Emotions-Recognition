function H=indirectHOG(Im)
nwin_x=2;%set here the number of HOG windows per bound box
nwin_y=2;
B=8;%set here the number of histogram bins
[L,C]=size(Im); % L num of lines ; C num of columns
H=zeros(nwin_x*nwin_y*B,1); % column vector with zeros
m=sqrt(L/2);
if C==1 % if num of columns==1
    Im=im_recover(Im,m,2*m);%verify the size of image, e.g. 25x50
    L=2*m;
    C=m;
end
Im=double(Im);
step_x=floor(C/(nwin_x));
step_y=floor(L/(nwin_y));
cont=0;
hx = [-1,0,1];
hy = -hx';

% grad_xr = imfilter(double(Im),hx,'replicate');
% grad_yu = imfilter(double(Im),hy,'replicate');
grad_xr = imfilter(double(Im),hx);
grad_yu = imfilter(double(Im),hy);
angles=atan2(grad_yu,grad_xr);
magnit=((grad_yu.^2)+(grad_xr.^2)).^.5;
for n=0:nwin_y-1
    for m=0:nwin_x-1
        cont=cont+1;
        angles2=angles(n*step_y+1:(n+1)*step_y,m*step_x+1:(m+1)*step_x)'; 
        magnit2=magnit(n*step_y+1:(n+1)*step_y,m*step_x+1:(m+1)*step_x)';
        v_angles=angles2(:); 
        inter = zeros(size(v_angles)) - 1;
        v_magnit=magnit2(:);
        K=max(size(v_angles));
        %assembling the histogram with 9 bins (range of 20 degrees per bin)
        bin=0;
        H2=zeros(B*2,1);
%         hist=zeros(B*2,1);
        for ang_lim=-pi+2*pi/(B*2):2*pi/(B*2):pi;
            bin=bin+1;
            for k=1:K
                if v_angles(k)<ang_lim
                    v_angles(k)=100;
                    inter(k) = bin - 1;
                    H2(bin)=H2(bin)+v_magnit(k);
%                     hist(bin) = hist(bin) + 1;
                end
            end
        end 
%         H2(1) = H2(1) + sum(v_magnit(v_angles~=100));
%         for k = 1:900
%             [k angles2(k) inter(k)]
%         end;
        H((cont-1)*B+1:cont*B,1)=H2(1:B)+H2(B+1:2*B);
    end
end
H=H/(norm(H)+0.01);   
