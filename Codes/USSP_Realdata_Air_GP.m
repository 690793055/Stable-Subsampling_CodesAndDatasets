%%
% Project Name: USSP
% Description: main function of experiment 6.2, the GP model in The urban air quality dataset of China
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2024-09-14
%%


clc
clear
sample = readmatrix('..\Real dataset\The urban air quality dataset of China\xizang.xlsx','Range','D:J');   %%The train data of xizang province
sample(1,:)=[];
sample=sample(all(~isnan(sample),2),:);

%%Full data
[~,sa_1,~]=unique(sample(:,2:7),'rows','stable');
train_sample_unique=sample(sa_1,:);

x_train_unique=train_sample_unique(:,2:7);
y_train_unique=train_sample_unique(:,1);

theta = [10 10 10 10 10 10]; 
lob = [1e-1 1e-1 1e-1 1e-1 1e-1 1e-1];
upb = [20 20 20 20 20 20];
dmodel = dacefit(x_train_unique, y_train_unique, @regpoly1, @corrgauss, theta, lob, upb);   %%GPR fit using  dacefit
y_train_unique_gp=predictor(x_train_unique, dmodel);

 %%USSP
 d=readmatrix('UD_100_6.csv','Range',[2 2]);    %%Load the uniform design points
[cm,id]=USSP(x_train_unique,d);
id=id{1,1};
subsample=sample(id,:);
[~,sa_1,~]=unique(subsample(:,2:7),'rows','stable');
train_subsample_unique=subsample(sa_1,:);


x_subsample_train=train_subsample_unique(:,2:7);
y_subsample_train=train_subsample_unique(:,1);
dmodel_subsample1 = dacefit(x_subsample_train, y_subsample_train, @regpoly1, @corrgauss, theta, lob, upb);
y_sub_train_unique_gp=predictor(x_subsample_train, dmodel_subsample1);
[GOF_sub,RMSE_sub]=rsquare(y_subsample_train,y_sub_train_unique_gp);
 
 %% test
rmse_all=zeros(25,1);
rmse_all_2=zeros(25,1);
rmse_subsample=zeros(25,1);
test_sample=cell(25,1);


%%load the 22 test datasets
test_sample(1) = {readmatrix('..\Real dataset\The urban air quality dataset of China\hainan.xlsx','Range','D:J')};
test_sample(2) = {readmatrix('..\Real dataset\The urban air quality dataset of China\fujian.xlsx','Range','D:J')};
test_sample(3) = {readmatrix('..\Real dataset\The urban air quality dataset of China\yunnan.xlsx','Range','D:J')};
test_sample(4) = {readmatrix('..\Real dataset\The urban air quality dataset of China\guizhou.xlsx','Range','D:J')};
test_sample(5) = {readmatrix('..\Real dataset\The urban air quality dataset of China\heilongjiang.xlsx','Range','D:J')};
test_sample(6) = {readmatrix('..\Real dataset\The urban air quality dataset of China\guangdong.xlsx','Range','D:J')};
test_sample(7) = {readmatrix('..\Real dataset\The urban air quality dataset of China\jiangxi.xlsx','Range','D:J')};
test_sample(8) = {readmatrix('..\Real dataset\The urban air quality dataset of China\guangxi.xlsx','Range','D:J')};
test_sample(9) = {readmatrix('..\Real dataset\The urban air quality dataset of China\jilin.xlsx','Range','D:J')};
test_sample(10) = {readmatrix('..\Real dataset\The urban air quality dataset of China\liaoning.xlsx','Range','D:J')};
test_sample(11) = {readmatrix('..\Real dataset\The urban air quality dataset of China\zhejiang.xlsx','Range','D:J')};
test_sample(12) = {readmatrix('..\Real dataset\The urban air quality dataset of China\gansu.xlsx','Range','D:J')};
test_sample(13) = {readmatrix('..\Real dataset\The urban air quality dataset of China\sichuan.xlsx','Range','D:J')};
test_sample(14) = {readmatrix('..\Real dataset\The urban air quality dataset of China\hunan.xlsx','Range','D:J')};
test_sample(15) = {readmatrix('..\Real dataset\The urban air quality dataset of China\tianjin.xlsx','Range','D:J')};
test_sample(16) = {readmatrix('..\Real dataset\The urban air quality dataset of China\shandong.xlsx','Range','D:J')};
test_sample(17) = {readmatrix('..\Real dataset\The urban air quality dataset of China\shanaxi.xlsx','Range','D:J')};
test_sample(18) = {readmatrix('..\Real dataset\The urban air quality dataset of China\shanxi.xlsx','Range','D:J')};
test_sample(19) = {readmatrix('..\Real dataset\The urban air quality dataset of China\hebei.xlsx','Range','D:J')};
test_sample(20) = {readmatrix('..\Real dataset\The urban air quality dataset of China\ningxia.xlsx','Range','D:J')};
test_sample(22) = {readmatrix('..\Real dataset\The urban air quality dataset of China\hubei.xlsx','Range','D:J')};
test_sample(22) = {readmatrix('..\Real dataset\The urban air quality dataset of China\chongqing.xlsx','Range','D:J')};




for i=1:22
    test_data=test_sample(i);
    test_data=test_data{1,1};
    test_data(1,:)=[];
    test_data=test_data(all(~isnan(test_data),2),:);
    x_test=test_data(:,2:7);
    y_test=test_data(:,1);
    data=[x_test,y_test];
    y_test_ols=predictor(x_test, dmodel);
    rmse_all(i)=sqrt(mean((y_test-y_test_ols).^2));
    y_test_subsample_ols=predictor(x_test, dmodel_subsample1);
    rmse_subsample(i)=sqrt(mean((y_test-y_test_subsample_ols).^2));   
end

%% figure of RMSE
X=1:22;
Y=[rmse_all,rmse_subsample];
h=bar(X,Y);
set(h(1),'FaceColor','#F7903D');    
set(h(2),'FaceColor','#4D85BD');  
xticks([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26]);
set(gca,'FontSize',15)
xticklabels({'Hainan','Fujian', 'Yunnan','Guizhou','Heilongjiang',  'Guangdong', 'Jiangxi','Guangxi', 'Jilin', 'Liaoning','Zhejiang',  'Gansu','Sichuan','Hunan','Tianjin','Shandong','Shaanxi','Shanxi','Hebei','Jiangsu','Ningxia','Hubei', 'Chongqing'   } );%定义标尺标签内容
ylabel('RMSE','FontSize',30);
xlabel('Dataset Name','FontSize',30);
legend('Baseline','SPAS','location','northwest','FontSize',30);