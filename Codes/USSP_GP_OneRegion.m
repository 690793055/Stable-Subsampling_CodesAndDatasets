%%
% Project Name: USSP
% Description: main function of experiment 3.2 in supplement materials when we consider consistent and inconsistent environments
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2024-09-14
%%

clear
N=100;  %%Number of experiments

    train_numb=20000;  %% Number of full samples 
    %% Train data

    d2=readmatrix('UD_200_5.csv','Range',[2 2]);   %%Load the uniform design points 200points
    d3=readmatrix('UD_500_5.csv','Range',[2 2]);   %%Load the uniform design points 500points
    d4=readmatrix('UD_1000_5.csv','Range',[2 2]);  %%Load the uniform design points 1000points
    d5=readmatrix('UD_2000_5.csv','Range',[2 2]);  %%Load the uniform design points 2000points

    [d1row,d1col]=size(d1);
    [d2row,d2col]=size(d2);
    [d3row,d3col]=size(d3);
    [d4row,d4col]=size(d4);
    [d5row,d5col]=size(d5);


    rmse_a=zeros(1,N);
    rmse_sub1=zeros(1,N);
    rmse_sub2=zeros(1,N);
    rmse_sub3=zeros(1,N);
    rmse_sub4=zeros(1,N);
    rmse_sub5=zeros(1,N);

    
    for i=1:N  
        seed=i;
        rng(seed,'twister');
       [sample]=Sample_Generate_GP_OneRegion(train_numb,i);     %%Generate normal distribution train samples                                                        
        
        x_train=sample(:,1:5);
        y_train=sample(:,6);
        
        sample3=sample;
        x_train2=sample3(:,1:5);
        y_train2=sample3(:,6);
        %% Full data
        theta = [10 10 10 10 10]; 
        lob = [1e-1 1e-1 1e-1 1e-1 1e-1];
        upb = [20 20 20 20 20];         
        dmodel = dacefit(x_train, y_train, @regpoly2, @corrgauss, theta, lob, upb);  %%GPR fit using  dacefit
            
           
         %% UD subsample
 
        [cm2,id2]=USSP(x_train2,d2);
        id2=id2{1,1};
        subsample2=unique(sample3(id2,:),'rows','stable');
        x_subsample_train2=subsample2(:,1:5);
        y_subsample_train2=subsample2(:,6);
        dmodel_subsample2 = dacefit(x_subsample_train2, y_subsample_train2, @regpoly2, @corrgauss, theta, lob, upb);

        [cm3,id3]=USSP(x_train2,d3);
        id3=id3{1,1};
        subsample3=unique(sample3(id3,:),'rows','stable');
        x_subsample_train3=subsample3(:,1:5);
        y_subsample_train3=subsample3(:,6);
        dmodel_subsample3 = dacefit(x_subsample_train3, y_subsample_train3, @regpoly2, @corrgauss, theta, lob, upb);

        [cm4,id4]=USSP(x_train2,d4);
        id4=id4{1,1};
        subsample4=unique(sample3(id4,:),'rows','stable');
        x_subsample_train4=subsample4(:,1:5);
        y_subsample_train4=subsample4(:,6);
        dmodel_subsample4 = dacefit(x_subsample_train4, y_subsample_train4, @regpoly2, @corrgauss, theta, lob, upb);

        [cm5,id5]=USSP(x_train2,d5);
        id5=id5{1,1};
        subsample5=unique(sample3(id5,:),'rows','stable');
        x_subsample_train5=subsample5(:,1:5);
        y_subsample_train5=subsample5(:,6);
        dmodel_subsample5 = dacefit(x_subsample_train5, y_subsample_train5, @regpoly2, @corrgauss, theta, lob, upb);


       %% Test_data
   
            seed=i;
            rng(seed,'twister');
            [sample_test]=Sample_Generate_GP_OneRegion_Test(2000,i);    %%Generate uniformly distribution test samples
            x_test=sample_test(:,1:5);
            y_test=sample_test(:,6);  
            
            y_test_ols = predictor(x_test, dmodel);
            rmse_a(i)=sqrt(mean((y_test-y_test_ols).^2));
           
            y_test_subsample_ols2=predictor(x_test, dmodel_subsample2);
            rmse_sub2(i)=sqrt(mean((y_test-y_test_subsample_ols2).^2));
            
            y_test_subsample_ols3=predictor(x_test, dmodel_subsample3);
            rmse_sub3(i)=sqrt(mean((y_test-y_test_subsample_ols3).^2));

            y_test_subsample_ols4=predictor(x_test, dmodel_subsample4);
            rmse_sub4(i)=sqrt(mean((y_test-y_test_subsample_ols4).^2));

            y_test_subsample_ols5=predictor(x_test, dmodel_subsample5);
            rmse_sub5(i)=sqrt(mean((y_test-y_test_subsample_ols5).^2));
    end



figure1=figure;  %% boxplot of RMSE
edgecolor1=[0,0,0]; % black color
edgecolor2=[0,0,0]; % black color
position_1 = [0.5];  % define position for first boxplots
position_2 = [1.5];  % define position for second boxplots 
position_3 = [2.5]; 
position_4 = [3.5];  
position_5 = [4.5];   
fillcolor1=[247, 144, 61]/255; 
fillcolor2=[77, 155, 189]/255;
box_1 = boxplot(rmse_a','positions',position_1,'colors',edgecolor1,'width',0.4,'symbol','r+','outliersize',5);
hold on;
box_2 = boxplot(rmse_sub2','positions',position_2,'colors',edgecolor2,'width',0.4,'symbol','r+','outliersize',5);
box_3 = boxplot(rmse_sub3','positions',position_3,'colors',edgecolor2,'width',0.4,'symbol','r+','outliersize',5);
box_4 = boxplot(rmse_sub4','positions',position_4,'colors',edgecolor2,'width',0.4,'symbol','r+','outliersize',5);
box_5 = boxplot(rmse_sub5','positions',position_5,'colors',edgecolor2,'width',0.4,'symbol','r+','outliersize',5);
boxobj = findobj(gca,'Tag','Box');
patch(get(boxobj(5),'XData'),get(boxobj(5),'YData'),fillcolor1);
patch(get(boxobj(4),'XData'),get(boxobj(4),'YData'),fillcolor2);
patch(get(boxobj(3),'XData'),get(boxobj(3),'YData'),fillcolor2);
patch(get(boxobj(2),'XData'),get(boxobj(2),'YData'),fillcolor2);
patch(get(boxobj(1),'XData'),get(boxobj(1),'YData'),fillcolor2);
set(gca,'XTick',[0.5,1.5,2.5,3.5,4.5],'XTicklabel',{'Baseline','SPAS(m=200)','SPAS(m=500)','SPAS(m=1000)','SPAS(m=2000)'},'FontSize',20)
ylabel('IRMSE','FontSize',30)


figure3=figure;   %% 2-dimensionaal projection comparsion
tiledlayout(2,2,'TileSpacing','tight','Padding','tight');
nexttile
plot(sample(:,1),sample(:,2),'.','color',[.7 .7 .7],'LineWidth', 0.1);
hold on
plot(x_subsample_train2(:,1),x_subsample_train2(:,2),'r*','LineWidth', 1.5)
hold off
set(gca,'xtick',-3:1:3,'FontSize',15);
set(gca,'ytick',-3:1:3,'FontSize',15);
xlabel('S_1','FontSize',15);
ylabel('S_2','FontSize',15);
title('A','FontSize',20)
ax = gca;
ax.TitleHorizontalAlignment = 'left';
nexttile
plot(sample(:,2),sample(:,3),'.','color',[.7 .7 .7],'LineWidth', 0.1);
hold on
plot(x_subsample_train2(:,2),x_subsample_train2(:,3),'r*','LineWidth', 1.5)
hold off
set(gca,'xtick',-3:1:3,'FontSize',15);
set(gca,'ytick',-3:1:3,'FontSize',15);
xlabel('S_2','FontSize',15);
ylabel('S_3','FontSize',15);
title('B','FontSize',20)
ax = gca;
ax.TitleHorizontalAlignment = 'left';
nexttile
plot(sample(:,3),sample(:,4),'.','color',[.7 .7 .7],'LineWidth', 0.1);
hold on
plot(x_subsample_train2(:,3),x_subsample_train2(:,4),'r*','LineWidth', 1.5)
hold off
set(gca,'xtick',-3:1:3,'FontSize',15);
set(gca,'ytick',-3:1:3,'FontSize',15);
xlabel('S_3','FontSize',15);
ylabel('S_4','FontSize',15);
title('C','FontSize',20)
ax = gca;
ax.TitleHorizontalAlignment = 'left';
nexttile
plot(sample(:,4),sample(:,5),'.','color',[.7 .7 .7],'LineWidth', 0.1);
hold on
plot(x_subsample_train2(:,4),x_subsample_train2(:,5),'r*','LineWidth', 1.5)
hold off
set(gca,'xtick',-3:1:3,'FontSize',15);
set(gca,'ytick',-3:1:3,'FontSize',15);
xlabel('S_4','FontSize',15);
ylabel('S_5','FontSize',15);
title('D','FontSize',20)
ax = gca;
ax.TitleHorizontalAlignment = 'left';


