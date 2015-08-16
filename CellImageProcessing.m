//CellImageProcessing.m
//Created by Jizhizi Li on 01/06/2014
//
//Copyright (c) 2014 Jizhizi Li.
//All rights reserved.




%% READ THE PICTURE
clear all;
close all;
I=imread('1.tif');
m=24;%设定某具体细胞


%% CELL PROCESSING

figure;
subplot(221);imshow(I);
title('原图')

background=imopen(I,strel('disk',25));
I2=I-background;
subplot(222);imshow(I2);
title('去背景后');

I3=imadjust(I2);
subplot(223);imshow(I3);
title('增强对比度灰度');

[level,EM]=graythresh(I3);
bw=im2bw(I3,level);
subplot(224);imshow(bw);
title('OTSU分割图像二值');

BW=bwfill(bw,'holes');
figure;
subplot(131);imshow(BW);
title('进行填充后')

SE=strel('disk',9);
BW4=imclose(BW,SE);
BW5=bwfill(BW4,'holes');
subplot(132);imshow(BW4);
title('闭合运算后');
subplot(133);imshow(BW5);
title('再次填充')



%对图像进行标记以便之后使用regionprops函数
%%%%%%%%%%此步也可用在细胞计数上%%%%%%%%%
A=im2bw(BW5);%变为二值图像；
%测出区域各个面积及总面积及平均面积
area=regionprops(A,'area');
area2matrix=cat(1,area.Area);%将结构数组变换为矩阵形式
s=0;
for n=1:1:length(area);
s=area2matrix(n,1)+s;
end
area_ave=s/length(area);%s为图中区域总面积
display(area_ave)%area_ave为图中区域面积值的平均值


[height,width,flag]=size(A);%对图像进行尺寸标定，方便后面使用

%%%%%进行小面积去除，本例中以0.3*average为面积阈值
small=0.3*area_ave
small=floor(small)
B=bwareaopen(A,small);
figure;
subplot(121);imshow(A);
title('去噪之前');
subplot(122);imshow(B)
title('去噪之后');


%%%%%%%%%%%%%%%%用分水岭法进行粘连细胞的分割%%%%%%%%%%%%%%
%%%%%%%%重要一步%%%%%%%%%%%%%%%%

%对B去反
gc = ~B;
%进行距离变换，找到每个像素与相邻非0像素之间的欧几里得距离
d = bwdist(gc);
figure;
subplot(121);
imshow(d);
title('距离变换')
% H = fspecial('gaussian',HSIZE,SIGMA) returns a rotationally
%symmetric Gaussian lowpass filter  of size HSIZE with standard
%deviation SIGMA (positive). HSIZE can be a vector specifying the
%number of rows and columns in H or a scalar, in which case H is a
%square matrix.
h = fspecial('gaussian',[40 40],6.5);%%进行高斯滤波
d = imfilter(d,h);%%进行一维数字滤波
%%计算外部约束
L=watershed(-d);
w= L==0;
subplot(122);
imshow(~w);
title('标记外的约束')
%%%%对B和~w求和，&的意思是对矩阵进行and运算，运算规则为1&1=1,1&0=0,0&1=0,0&0=0,
%%%%也即两图像素若都为白（白色等于1），则合成图也为白，若两图像素都为黑（黑色等于0），则合成图也为黑
%%%%若两图像素一黑一白，则合成图也为黑。从而实现粘连细胞的分割。
g2 = B & ~w;
figure;
imshow(g2);
title('粘连细胞分割后');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%去除小区域的面积第二种方法
%B=bwlabel(A);%进行去小面积之前的标记
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



%%%%%%%%%%%%对粘连细胞分割之前的图像进行中心定位%%%%%%%%%%%%%%
%对图像进行细胞中心定位第一种方法，通过循环和叠加
label1=bwlabel(B);%对图像进行标记，用数字代表具体区域
centroid1=regionprops(label1,'Centroid');
figure;
imshow(label1);
title('粘连细胞前的中心定位');
hold on;
for i=1:1:length(centroid1);
plot(centroid1(i).Centroid(1),centroid1(i).Centroid(2),'*');
end
display(max(max(label1)))%显示图中区域个数，也即细胞个数




%%%%%%%%%%%%对粘连细胞分割之后的图像进行中心定位%%%%%%%%%%%%%%

%对图像进行细胞中心定位第一种方法，通过循环和叠加
label=bwlabel(g2);%对图像进行标记，用数字代表具体区域
centroid=regionprops(label,'Centroid');
figure;
imshow(label);
title('粘连细胞后的中心定位');
hold on;
for i=1:1:length(centroid);
plot(centroid(i).Centroid(1),centroid(i).Centroid(2),'*');
end
display(max(max(label)))%显示图中区域个数，也即细胞个数
centroids=cat(1,centroid.Centroid);%将结构数组转化成矩阵
display(centroids)%显示出图像中细胞中心点的坐标



perimeter=regionprops(label,'Perimeter');%算得各个区域的周长
perimeters=cat(1,perimeter.Perimeter);%对应区域的周长，矩阵形式


%对图像进行细胞中心定位第二种方法，通过匹配矩阵两列
%figure;
%imshow(B);
%centroids=cat(1,centroid.Centroid)；
%hold on;
%plot(centroid2matrix(:,1),centroid2matrix(:,2),'+');
%hold off


%figure;
%m=24;%定位某个具体细胞
%imshow(B);
%title('某具体点坐标')
%hold on;
%plot(centroid(m).Centroid(1),centroid(m).Centroid(2),'*');


%figure;
%imshow(B);
%hold on;
%plot(262,1314,'*');%按照真实库中坐标找点


%figure;
%imshow(B);
%hold on;
%for i=1:1:length(centroid);
%plot(centroid(i).Centroid(1),centroid(i).Centroid(2),'*');
%end
%[x,y]=textread('1.txt','%f %f');%其内容为数据所提供真实细胞坐标，需从xml文件中手动获得
%plot(x,y,'ro');%叠加在分析出的细胞中心标记图上，方便进行比较
%grid on


%for m=20:1:25;
%figure;
%imshow(B);
%hold on;
%plot(centroid(m).Centroid(1),centroid.Centroid(2),'*');
%end
%设定范围，单个输出，从而找到具体某点


%%%%%%                                             %%%%%%
%%%%%%                                             %%%%%%
%%%%%%                                             %%%%%%
%%%%%%%%%%%%%%%%B是定位前所处理的最终图像%%%%%%%%%%%%%%%%%%
%%%%%%%%%%centroid是包含细胞中心坐标的结构数组%%%%%%%%%%%%%
%%%%%%%%%%%%%centroids是包含细胞中心坐标的矩阵%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%产生变量总结%%%%%%%%%%%%%%%%%%%%%%%



%% CELL TRACKING

%%%%初始参数设定%%%%%%
t=0.43;  %帧间时间间隔

xv=1.5*ones(1,length(centroid))%建立1*个数的1矩阵，每个代表X轴方向速度
yv=2*ones(1,length(centroid))%建立1*个数的1矩阵，每个代表Y轴方向速度
det=centroids';%获得centroids矩阵的转置矩阵
X=cat(1,det(1,:),xv,det(2,:),yv);%构建新的X矩阵，每列为(x,xv,y,yv),列数为细胞个数
display(X)%显示X矩阵


%%%第一种运动状态，匀速运动%%%

F1=[1 t 0 0;
    0 1 0 0;
    0 0 1 t;
    0 0 0 1]%状态转移矩阵
H1=[1 0 0 0;
    0 0 1 0]%量测矩阵
T1=[t^2/2 0
    t 0;
    0 t^2/2;
    0 t];%噪声矩阵

%%%第二种运动状态，匀速圆周运动%%%
F2=[1,t,0,0,t^2,0;
    0,1,0,0,t,0;
    0,0,1,t,0,t^2/2;
    0,0,0,1,0,t;
    0,0,0,0,1,0;
    0,0,0,0,0,1];%状态转移矩阵
H2=[1,0,0,0,0,0;0,0,1,0,0,0];%量测矩阵
T2=[t^2/4,0;t/2,0;0,t^2/4;0,t/2;1,0;0,1];%噪声矩阵


P1=[0.2^2 0 0 0;
    0 0.2^2 0 0;
    0 0 0.2^2 0;
    0 0 0 0.2^2]%误差协方差矩阵


%%%%%%卡尔曼滤波器进行状态估计和预测%%%%%%%

X1=F1*X;
XX=X1';

PP=F1*P1*F1';
%[l1,l2]=find(B==1)%找到图中像素值为1的所有坐标
%zero=zeros(1,length(l1));%建立1*个数的零矩阵
%l1zhuan=l1';
%l2zhuan=l2';
%YAOCE=cat(1,l2zhuan,zero,l1zhuan,zero);%此处要倒着写= =


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
title('原图')
background=imopen(II,strel('disk',25));
II2=II-background;
subplot(222);imshow(II2);
title('去背景后');

II3=imadjust(II2);
subplot(223);imshow(II3);
title('增强对比度灰度');

[level,EM]=graythresh(II3);
bbw=im2bw(II3,level);
subplot(224);imshow(bbw);
title('OTSU分割图像二值');

BBW=bwfill(bbw,'holes');
%figure;
%subplot(131);imshow(BBW);
%title('进行填充后')

SSE=strel('disk',9);
BBW4=imclose(BBW,SSE);
BBW5=bwfill(BBW4,'holes');
%subplot(132);imshow(BBW4);
%title('闭合运算后');
%subplot(133);imshow(BBW5);
%title('再次填充')



%对图像进行标记以便之后使用regionprops函数
%%%%%%%%%%此步也可用在细胞计数上%%%%%%%%%
AA=im2bw(BBW5);%变为二值图像；
%测出区域各个面积及总面积及平均面积
areaarea=regionprops(AA,'area');
area2matrix=cat(1,areaarea.Area);%将结构数组变换为矩阵形式
s=0;
for n=1:1:length(areaarea);
s=area2matrix(n,1)+s;
end
area_ave2=s/length(areaarea);%s为图中区域总面积
display(area_ave2)%area_ave为图中区域面积值的平均值

%%%%%进行小面积去除，本例中以0.3*average为面积阈值
small2=0.3*area_ave2
small2=floor(small2)
BB=bwareaopen(AA,small2);
figure;
subplot(121);imshow(AA);
title('去噪之前');
subplot(122);imshow(BB)
title('去噪之后');



%%%%%%%%%%%%%%%%粘连细胞分割后%%%%%%%%%%%%%%


ggc = ~BB;
dd = bwdist(ggc);
hh = fspecial('gaussian',[40 40],6.5);
dd = imfilter(dd,hh);
LL=watershed(-dd);
ww= LL==0;
gg2 = BB & ~ww;
figure;
imshow(gg2);
title('粘连细胞分割后');




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%对图像进行细胞中心定位第一种方法，通过循环和叠加
label1=bwlabel(gg2);%对图像进行标记，用数字代表具体区域
centroid1=regionprops(label1,'Centroid');
figure;
imshow(label1);
title('中心定位后');
hold on;
for i=1:1:length(centroid1);
plot(centroid1(i).Centroid(1),centroid1(i).Centroid(2),'*');
end
display(max(max(label1)))%显示图中区域个数，也即细胞个数
centroids1=cat(1,centroid1.Centroid);%将结构数组转化成矩阵
display(centroids1)%显示出图像中细胞中心点的坐标



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

%%%%%应选取下一张处理后的图像作为要聚类的对象
[l1,l2]=find(BB==1)%找到图中像素值为1的所有坐标
zero=zeros(1,length(l1));%建立1*个数的零矩阵
l1zhuan=l1';
l2zhuan=l2';
YAOCE=cat(1,l2zhuan,zero,l1zhuan,zero);%此处要倒着写= =

%%%%%%%%%%%%%%%%%%%开始进行K-means算法跟踪%%%%%%%%%%%%%%%%%%%%%%%

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
%[x,y,z]=textread('result for 28.txt','%f %f %f');%%%将用算法预测到的数值放在此处
%plot3(x,y,z,'-ro','linewidth',1,'MarkerEdgeColor','r','MarkerFace','b','MarkerSize',2);
%xlabel('细胞x坐标'),ylabel('细胞y坐标'),zlabel('帧数');
%grid on
%text(636.7168,291.0874,1,'start');
%text(683.6,245.8,22,'end')
%title('跟踪结果比较')

%hold on;
%[x,y,z]=textread('real result for 28.txt','%f %f %f');%%%将真实的值放在此处
%plot3(x,y,z,'-bo','linewidth',1,'MarkerEdgeColor','g','MarkerFace','g','MarkerSize',2);
%xlabel('细胞x坐标'),ylabel('细胞y坐标'),zlabel('帧数');
%grid on
%%text(636.671000,290.239000,1,'start');
%%text(683.909000,246.904000,22,'end')







