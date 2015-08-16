//CellImageProcessing.m
//Created by Jizhizi Li on 01/06/2014
//
//Copyright (c) 2014 Jizhizi Li.
//All rights reserved.




%% READ THE PICTURE
clear all;
close all;
I=imread('1.tif');
m=24;%�趨ĳ����ϸ��


%% CELL PROCESSING

figure;
subplot(221);imshow(I);
title('ԭͼ')

background=imopen(I,strel('disk',25));
I2=I-background;
subplot(222);imshow(I2);
title('ȥ������');

I3=imadjust(I2);
subplot(223);imshow(I3);
title('��ǿ�ԱȶȻҶ�');

[level,EM]=graythresh(I3);
bw=im2bw(I3,level);
subplot(224);imshow(bw);
title('OTSU�ָ�ͼ���ֵ');

BW=bwfill(bw,'holes');
figure;
subplot(131);imshow(BW);
title('��������')

SE=strel('disk',9);
BW4=imclose(BW,SE);
BW5=bwfill(BW4,'holes');
subplot(132);imshow(BW4);
title('�պ������');
subplot(133);imshow(BW5);
title('�ٴ����')



%��ͼ����б���Ա�֮��ʹ��regionprops����
%%%%%%%%%%�˲�Ҳ������ϸ��������%%%%%%%%%
A=im2bw(BW5);%��Ϊ��ֵͼ��
%����������������������ƽ�����
area=regionprops(A,'area');
area2matrix=cat(1,area.Area);%���ṹ����任Ϊ������ʽ
s=0;
for n=1:1:length(area);
s=area2matrix(n,1)+s;
end
area_ave=s/length(area);%sΪͼ�����������
display(area_ave)%area_aveΪͼ���������ֵ��ƽ��ֵ


[height,width,flag]=size(A);%��ͼ����гߴ�궨���������ʹ��

%%%%%����С���ȥ������������0.3*averageΪ�����ֵ
small=0.3*area_ave
small=floor(small)
B=bwareaopen(A,small);
figure;
subplot(121);imshow(A);
title('ȥ��֮ǰ');
subplot(122);imshow(B)
title('ȥ��֮��');


%%%%%%%%%%%%%%%%�÷�ˮ�뷨����ճ��ϸ���ķָ�%%%%%%%%%%%%%%
%%%%%%%%��Ҫһ��%%%%%%%%%%%%%%%%

%��Bȥ��
gc = ~B;
%���о���任���ҵ�ÿ�����������ڷ�0����֮���ŷ����þ���
d = bwdist(gc);
figure;
subplot(121);
imshow(d);
title('����任')
% H = fspecial('gaussian',HSIZE,SIGMA) returns a rotationally
%symmetric Gaussian lowpass filter  of size HSIZE with standard
%deviation SIGMA (positive). HSIZE can be a vector specifying the
%number of rows and columns in H or a scalar, in which case H is a
%square matrix.
h = fspecial('gaussian',[40 40],6.5);%%���и�˹�˲�
d = imfilter(d,h);%%����һά�����˲�
%%�����ⲿԼ��
L=watershed(-d);
w= L==0;
subplot(122);
imshow(~w);
title('������Լ��')
%%%%��B��~w��ͣ�&����˼�ǶԾ������and���㣬�������Ϊ1&1=1,1&0=0,0&1=0,0&0=0,
%%%%Ҳ����ͼ��������Ϊ�ף���ɫ����1������ϳ�ͼҲΪ�ף�����ͼ���ض�Ϊ�ڣ���ɫ����0������ϳ�ͼҲΪ��
%%%%����ͼ����һ��һ�ף���ϳ�ͼҲΪ�ڡ��Ӷ�ʵ��ճ��ϸ���ķָ
g2 = B & ~w;
figure;
imshow(g2);
title('ճ��ϸ���ָ��');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%ȥ��С���������ڶ��ַ���
%B=bwlabel(A);%����ȥС���֮ǰ�ı��
%for j=1:length(area);
%    if area2matrix(j,1)<0.3*area_ave;
%        for x=1:height;
%         for y=1:width;
%          if B(x,y)==j;
%             B(x,y)==0;
%           end
%         end
%        end
%    end
%end
%B1=B
%figure;
%subplot(121);imshow(A);
%subplot(122);imshow(B1)



%%%%%%%%%%%%��ճ��ϸ���ָ�֮ǰ��ͼ��������Ķ�λ%%%%%%%%%%%%%%
%��ͼ�����ϸ�����Ķ�λ��һ�ַ�����ͨ��ѭ���͵���
label1=bwlabel(B);%��ͼ����б�ǣ������ִ����������
centroid1=regionprops(label1,'Centroid');
figure;
imshow(label1);
title('ճ��ϸ��ǰ�����Ķ�λ');
hold on;
for i=1:1:length(centroid1);
plot(centroid1(i).Centroid(1),centroid1(i).Centroid(2),'*');
end
display(max(max(label1)))%��ʾͼ�����������Ҳ��ϸ������




%%%%%%%%%%%%��ճ��ϸ���ָ�֮���ͼ��������Ķ�λ%%%%%%%%%%%%%%

%��ͼ�����ϸ�����Ķ�λ��һ�ַ�����ͨ��ѭ���͵���
label=bwlabel(g2);%��ͼ����б�ǣ������ִ����������
centroid=regionprops(label,'Centroid');
figure;
imshow(label);
title('ճ��ϸ��������Ķ�λ');
hold on;
for i=1:1:length(centroid);
plot(centroid(i).Centroid(1),centroid(i).Centroid(2),'*');
end
display(max(max(label)))%��ʾͼ�����������Ҳ��ϸ������
centroids=cat(1,centroid.Centroid);%���ṹ����ת���ɾ���
display(centroids)%��ʾ��ͼ����ϸ�����ĵ������



perimeter=regionprops(label,'Perimeter');%��ø���������ܳ�
perimeters=cat(1,perimeter.Perimeter);%��Ӧ������ܳ���������ʽ


%��ͼ�����ϸ�����Ķ�λ�ڶ��ַ�����ͨ��ƥ���������
%figure;
%imshow(B);
%centroids=cat(1,centroid.Centroid)��
%hold on;
%plot(centroid2matrix(:,1),centroid2matrix(:,2),'+');
%hold off


%figure;
%m=24;%��λĳ������ϸ��
%imshow(B);
%title('ĳ���������')
%hold on;
%plot(centroid(m).Centroid(1),centroid(m).Centroid(2),'*');


%figure;
%imshow(B);
%hold on;
%plot(262,1314,'*');%������ʵ���������ҵ�


%figure;
%imshow(B);
%hold on;
%for i=1:1:length(centroid);
%plot(centroid(i).Centroid(1),centroid(i).Centroid(2),'*');
%end
%[x,y]=textread('1.txt','%f %f');%������Ϊ�������ṩ��ʵϸ�����꣬���xml�ļ����ֶ����
%plot(x,y,'ro');%�����ڷ�������ϸ�����ı��ͼ�ϣ�������бȽ�
%grid on


%for m=20:1:25;
%figure;
%imshow(B);
%hold on;
%plot(centroid(m).Centroid(1),centroid.Centroid(2),'*');
%end
%�趨��Χ������������Ӷ��ҵ�����ĳ��


%%%%%%                                             %%%%%%
%%%%%%                                             %%%%%%
%%%%%%                                             %%%%%%
%%%%%%%%%%%%%%%%B�Ƕ�λǰ�����������ͼ��%%%%%%%%%%%%%%%%%%
%%%%%%%%%%centroid�ǰ���ϸ����������Ľṹ����%%%%%%%%%%%%%
%%%%%%%%%%%%%centroids�ǰ���ϸ����������ľ���%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%���������ܽ�%%%%%%%%%%%%%%%%%%%%%%%



%% CELL TRACKING

%%%%��ʼ�����趨%%%%%%
t=0.43;  %֡��ʱ����

xv=1.5*ones(1,length(centroid))%����1*������1����ÿ������X�᷽���ٶ�
yv=2*ones(1,length(centroid))%����1*������1����ÿ������Y�᷽���ٶ�
det=centroids';%���centroids�����ת�þ���
X=cat(1,det(1,:),xv,det(2,:),yv);%�����µ�X����ÿ��Ϊ(x,xv,y,yv),����Ϊϸ������
display(X)%��ʾX����


%%%��һ���˶�״̬�������˶�%%%

F1=[1 t 0 0;
    0 1 0 0;
    0 0 1 t;
    0 0 0 1]%״̬ת�ƾ���
H1=[1 0 0 0;
    0 0 1 0]%�������
T1=[t^2/2 0
    t 0;
    0 t^2/2;
    0 t];%��������

%%%�ڶ����˶�״̬������Բ���˶�%%%
F2=[1,t,0,0,t^2,0;
    0,1,0,0,t,0;
    0,0,1,t,0,t^2/2;
    0,0,0,1,0,t;
    0,0,0,0,1,0;
    0,0,0,0,0,1];%״̬ת�ƾ���
H2=[1,0,0,0,0,0;0,0,1,0,0,0];%�������
T2=[t^2/4,0;t/2,0;0,t^2/4;0,t/2;1,0;0,1];%��������


P1=[0.2^2 0 0 0;
    0 0.2^2 0 0;
    0 0 0.2^2 0;
    0 0 0 0.2^2]%���Э�������


%%%%%%�������˲�������״̬���ƺ�Ԥ��%%%%%%%

X1=F1*X;
XX=X1';

PP=F1*P1*F1';
%[l1,l2]=find(B==1)%�ҵ�ͼ������ֵΪ1����������
%zero=zeros(1,length(l1));%����1*�����������
%l1zhuan=l1';
%l2zhuan=l2';
%YAOCE=cat(1,l2zhuan,zero,l1zhuan,zero);%�˴�Ҫ����д= =


%for xx=1:height;
%for yy=1:width;
%if B(xx,yy)==1;
%end
%end 
%end


%% TESTING THE NEXT FRAME PICTURE

II=imread('6.tif')

figure;
subplot(221);imshow(II);
title('ԭͼ')
background=imopen(II,strel('disk',25));
II2=II-background;
subplot(222);imshow(II2);
title('ȥ������');

II3=imadjust(II2);
subplot(223);imshow(II3);
title('��ǿ�ԱȶȻҶ�');

[level,EM]=graythresh(II3);
bbw=im2bw(II3,level);
subplot(224);imshow(bbw);
title('OTSU�ָ�ͼ���ֵ');

BBW=bwfill(bbw,'holes');
%figure;
%subplot(131);imshow(BBW);
%title('��������')

SSE=strel('disk',9);
BBW4=imclose(BBW,SSE);
BBW5=bwfill(BBW4,'holes');
%subplot(132);imshow(BBW4);
%title('�պ������');
%subplot(133);imshow(BBW5);
%title('�ٴ����')



%��ͼ����б���Ա�֮��ʹ��regionprops����
%%%%%%%%%%�˲�Ҳ������ϸ��������%%%%%%%%%
AA=im2bw(BBW5);%��Ϊ��ֵͼ��
%����������������������ƽ�����
areaarea=regionprops(AA,'area');
area2matrix=cat(1,areaarea.Area);%���ṹ����任Ϊ������ʽ
s=0;
for n=1:1:length(areaarea);
s=area2matrix(n,1)+s;
end
area_ave2=s/length(areaarea);%sΪͼ�����������
display(area_ave2)%area_aveΪͼ���������ֵ��ƽ��ֵ

%%%%%����С���ȥ������������0.3*averageΪ�����ֵ
small2=0.3*area_ave2
small2=floor(small2)
BB=bwareaopen(AA,small2);
figure;
subplot(121);imshow(AA);
title('ȥ��֮ǰ');
subplot(122);imshow(BB)
title('ȥ��֮��');



%%%%%%%%%%%%%%%%ճ��ϸ���ָ��%%%%%%%%%%%%%%


ggc = ~BB;
dd = bwdist(ggc);
hh = fspecial('gaussian',[40 40],6.5);
dd = imfilter(dd,hh);
LL=watershed(-dd);
ww= LL==0;
gg2 = BB & ~ww;
figure;
imshow(gg2);
title('ճ��ϸ���ָ��');




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%��ͼ�����ϸ�����Ķ�λ��һ�ַ�����ͨ��ѭ���͵���
label1=bwlabel(gg2);%��ͼ����б�ǣ������ִ����������
centroid1=regionprops(label1,'Centroid');
figure;
imshow(label1);
title('���Ķ�λ��');
hold on;
for i=1:1:length(centroid1);
plot(centroid1(i).Centroid(1),centroid1(i).Centroid(2),'*');
end
display(max(max(label1)))%��ʾͼ�����������Ҳ��ϸ������
centroids1=cat(1,centroid1.Centroid);%���ṹ����ת���ɾ���
display(centroids1)%��ʾ��ͼ����ϸ�����ĵ������



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% START USING K-MEANS ALGORITHM

% CSKMEANS K-Means clustering - general method.
%
% This implements the more general k-means algorithm, where
% HMEANS is used to find the initial partition and then each
% observation is examined for further improvements in minimizing
% the within-group sum of squares.
%
% [CID,NR,CENTERS] = CSKMEANS(X,K,NC) Performs K-means
% clustering using the data given in X.
%
% INPUTS: X is the n x d matrix of data,
% where each row indicates an observation. K indicates
% the number of desired clusters. NC is a k x d matrix for the
% initial cluster centers. If NC is not specified, then the
% centers will be randomly chosen from the observations.
%
% OUTPUTS: CID provides a set of n indexes indicating cluster
% membership for each point. NR is the number of observations
% in each cluster. CENTERS is a matrix, where each row
% corresponds to a cluster center.

%%%%%Ӧѡȡ��һ�Ŵ�����ͼ����ΪҪ����Ķ���
[l1,l2]=find(BB==1)%�ҵ�ͼ������ֵΪ1����������
zero=zeros(1,length(l1));%����1*�����������
l1zhuan=l1';
l2zhuan=l2';
YAOCE=cat(1,l2zhuan,zero,l1zhuan,zero);%�˴�Ҫ����д= =

%%%%%%%%%%%%%%%%%%%��ʼ����K-means�㷨����%%%%%%%%%%%%%%%%%%%%%%%

x=YAOCE';
k=length(centroid);
nc=XX;

[n,d] = size(x);
if nargin < 3
    % Then pick some observations to be the cluster centers.
    ind = ceil(n*rand(1,k));
    % We will add some noise to make it interesting.
    nc = x(ind,:)+ randn(k,d);
end
% set up storage
% integer 1,...,k indicating cluster membership
cid = zeros(1,n);
% Make this different to get the loop started.
oldcid = ones(1,n);
% The number in each cluster.
nr = zeros(1,k);
% Set up maximum number of iterations.
maxiter =8;
iter = 1;
while ~isequal(cid,oldcid) && iter < maxiter
    % Implement the hmeans algorithm
    % For each point, find the distance to all cluster centers
    for i = 1:n
        dist = sum((repmat(x(i,:),k,1)-nc).^2,2);
        [m,ind] = min(dist); % assign it to this cluster center
        cid(i) = ind;
    end
    % Find the new cluster centers
    for i = 1:k
        % find all points in this cluster
        ind = find(cid==i);
        % find the centroid
        nc(i,:) = mean(x(ind,:));
        % Find the number in each cluster;
        nr(i) = length(ind);
    end
    iter = iter + 1;
end
% Now check each observation to see if the error can be minimized some more.
% Loop through all points.
maxiter =8;
iter = 1;
move = 1;
while iter < maxiter && move ~= 0
    move = 0;
    % Loop through all points.
    for i = 1:n
        % find the distance to all cluster centers
        dist = sum((repmat(x(i,:),k,1)-nc).^2,2);
        r = cid(i); % This is the cluster id for x
        %%nr,nr+1;
        dadj = nr./(nr+1).*dist'; % All adjusted distances
        [m,ind] = min(dadj); % minimum should be the cluster it belongs to
        if ind ~= r % if not, then move x
            cid(i) = ind;
            ic = find(cid == ind);
            nc(ind,:) = mean(x(ic,:));
            move = 1;
        end
    end
    iter = iter+1;
end
centers = nc;
if move == 0
   disp('No points were moved after the initial clustering procedure.')
else
    disp('Some points were moved after the initial clustering procedure.')
end

 
%% PLOT FINAL RESULT
%clear all;
%[x,y,z]=textread('result for 28.txt','%f %f %f');%%%�����㷨Ԥ�⵽����ֵ���ڴ˴�
%plot3(x,y,z,'-ro','linewidth',1,'MarkerEdgeColor','r','MarkerFace','b','MarkerSize',2);
%xlabel('ϸ��x����'),ylabel('ϸ��y����'),zlabel('֡��');
%grid on
%text(636.7168,291.0874,1,'start');
%text(683.6,245.8,22,'end')
%title('���ٽ���Ƚ�')

%hold on;
%[x,y,z]=textread('real result for 28.txt','%f %f %f');%%%����ʵ��ֵ���ڴ˴�
%plot3(x,y,z,'-bo','linewidth',1,'MarkerEdgeColor','g','MarkerFace','g','MarkerSize',2);
%xlabel('ϸ��x����'),ylabel('ϸ��y����'),zlabel('֡��');
%grid on
%%text(636.671000,290.239000,1,'start');
%%text(683.909000,246.904000,22,'end')







