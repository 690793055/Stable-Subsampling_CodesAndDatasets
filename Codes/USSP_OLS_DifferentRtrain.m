%%
% Project Name: USSP
% Description: main function of experiment 5.1
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2025-04-19
%%
clc
clear
N=20;  %%Number of experiments
r_train_all=[-0.8;-0.6;-0.4;-0.2;0;0.2;0.4;0.6;0.8];  %% Value of r_train
rmse_a=zeros(length(r_train_all),19);
rmse_sub_our=zeros(length(r_train_all),19);
rmse_sub_lowcon=zeros(length(r_train_all),19);
rmse_sub_iboss=zeros(length(r_train_all),19);
rmse_sub_srs=zeros(length(r_train_all),19);

 for ii=1:1:length(r_train_all)
    train_numb=20000;    %% Number of full samples 
    %% Train data
    r_train=r_train_all(ii);
    d1=readmatrix('UD_200_8.csv','Range',[2 2]); %%Load the uniform design points
    d2=readmatrix('OLHD_199_8.csv','Range',[2 2]);
    beta_train=zeros(9,N);   
    beta_s_error=zeros(5,N); 
    beta_v_error=zeros(3,N);  
    beta_subsample_train_our=zeros(9,N);   
%     beta_s_subsample_error1=zeros(5,N);  
%     beta_v_subsample_error1=zeros(3,N);  
    
    beta_subsample_train_lowcon=zeros(9,N);   %%每一列表示每一次训练beta值    
    beta_subsample_train_iboss=zeros(9,N);   %%每一列表示每一次训练beta值
    beta_subsample_train_srs=zeros(9,N);   %%每一列表示每一次训练beta值
    GOF_all=zeros(1,N);

    for i=1:N  %%Repeat N times
        seed=i;
        rng(seed,'twister');
        [sample,beta_s,beta_v]=Sample_Generate_OLS_8dimension(r_train,train_numb,i,'exp');  %%Generate normal distribution train samples
        
        x_train=sample(:,1:8);
        s_train=x_train(:,1:5);
        v_trian=x_train(:,6:8);
        y_train=sample(:,9);
       
        %% full data

        beta_train(:,i) = regress(y_train,[x_train,ones(train_numb,1)]);
        
        % [B,FitInfo] = lasso(x_train,y_train,'Alpha',0.8,'CV',5);
        % IndexMinMSE = FitInfo.IndexMinMSE;
        % beta_train(1:8,i) = B(:,IndexMinMSE);
        % beta_train(9,i) = FitInfo.Intercept(IndexMinMSE);

%         beta_s_error(:,i)=abs(beta_train(1:5,i)-beta_s);
%         beta_v_error(:,i)=abs(beta_train(6:8,i)-beta_v);
%         y_train_ols=x_train*beta_train(1:8,i)+beta_train(9,i);
  
        %% UD subsample
         k_srs=200;
         k_iboss=192;
         id_srs = randperm(train_numb, k_srs);
         [cm,id_our]=USSP(x_train,d1);
         id_our=id_our{1,1};
         [id_lowcon]=LowCon(x_train,d2);
         [id_iboss,~]=IBOSS(x_train,k_iboss);

        
        subsample_our=sample(id_our,:);
        subsample_lowcon=sample(id_lowcon,:);
        subsample_iboss=sample(id_iboss,:);
        subsample_srs=sample(id_srs,:);


        x_subsample_train_our=subsample_our(:,1:8);
        x_subsample_train_lowcon=subsample_lowcon(:,1:8);
        x_subsample_train_iboss=subsample_iboss(:,1:8);
        x_subsample_train_srs=subsample_srs(:,1:8);
        y_subsample_train_our=subsample_our(:,9);
        y_subsample_train_lowcon=subsample_lowcon(:,9);
        y_subsample_train_iboss=subsample_iboss(:,9);
        y_subsample_train_srs=subsample_srs(:,9);


        beta_subsample_train_our(:,i) = regress(y_subsample_train_our,[x_subsample_train_our,ones(size(d1,1),1)]);
        beta_subsample_train_lowcon(:,i) = regress(y_subsample_train_lowcon,[x_subsample_train_lowcon,ones(size(d2,1),1)]);
        beta_subsample_train_iboss(:,i) = regress(y_subsample_train_iboss,[x_subsample_train_iboss,ones(k_iboss,1)]);
        beta_subsample_train_srs(:,i) = regress(y_subsample_train_srs,[x_subsample_train_srs,ones(k_srs,1)]);
%         beta_s_subsample_error1(:,i)=abs(beta_subsample_train1(1:5,i)-beta_s);
%         beta_v_subsample_error1(:,i)=abs(beta_subsample_train1(6:8,i)-beta_v);  
%         y_subsample_train_ols_our=x_subsample_train_our*beta_subsample_train1(1:8,i)+beta_subsample_train1(9,i);            
    end


    %% Test_data
    r_test_all=[-0.9;-0.8;-0.7;-0.6;-0.5;-0.4;-0.3;-0.2;-0.1;0;0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9];  %% Value of r_test
    rmse_all=zeros(N,19);
    rmse_subsample_our=zeros(N,19);
    rmse_subsample_lowcon=zeros(N,19);
    rmse_subsample_iboss=zeros(N,19);
    rmse_subsample_srs=zeros(N,19);
    GOF_test_all=zeros(N,19);
    for k=1:19
        r_test=r_test_all(k);
        for i=1:N  %%Repeat N times
            seed=i;
            rng(seed,'twister');
            [sample_test,beta_s,beta_v]=Sample_Generate_OLS_8dimension_Test(r_test,2000,i,'normal','exp'); %%Generate uniformly distribution test samples
            x_test=sample_test(:,1:8);
            s_test=x_test(:,1:5);
            v_test=x_test(:,6:8);
            y_test=sample_test(:,9);        
             y_test_ols=x_test*beta_train(1:8,i)+beta_train(9,i);
            rmse_all(i,k)=sqrt(mean((y_test-y_test_ols).^2));
            GOF_test_all(i,k)=rsquare(y_test,y_test_ols);
            y_test_subsample_ols_our=x_test*beta_subsample_train_our(1:8,i)+beta_subsample_train_our(9,i);
            rmse_subsample_our(i,k)=sqrt(mean((y_test-y_test_subsample_ols_our).^2)); 
            y_test_subsample_ols_lowcon=x_test*beta_subsample_train_lowcon(1:8,i)+beta_subsample_train_lowcon(9,i);
            rmse_subsample_lowcon(i,k)=sqrt(mean((y_test-y_test_subsample_ols_lowcon).^2)); 
            y_test_subsample_ols_iboss=x_test*beta_subsample_train_iboss(1:8,i)+beta_subsample_train_iboss(9,i);
            rmse_subsample_iboss(i,k)=sqrt(mean((y_test-y_test_subsample_ols_iboss).^2)); 
            y_test_subsample_ols_srs=x_test*beta_subsample_train_srs(1:8,i)+beta_subsample_train_srs(9,i);
            rmse_subsample_srs(i,k)=sqrt(mean((y_test-y_test_subsample_ols_srs).^2)); 

         end
    end

    rmse_a(ii,:)=mean(rmse_all,1);
    rmse_sub_our(ii,:)=mean(rmse_subsample_our,1);
    rmse_sub_lowcon(ii,:)=mean(rmse_subsample_lowcon,1);
    rmse_sub_iboss(ii,:)=mean(rmse_subsample_iboss,1);
    rmse_sub_srs(ii,:)=mean(rmse_subsample_srs,1);
 end 
 
%% figure of RMSE

figure1=figure;  
t=tiledlayout(2,2,'TileSpacing','Compact');
x=1:19;
nexttile
plot(x,rmse_a(1,:),'o-','Color','#FD6D5A','linewidth',2,'MarkerSize', 10);
hold on
plot(x,rmse_sub_lowcon(1,:),'*-','Color','#6DC354','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_iboss(1,:),'pentagram-','Color','#994487','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_srs(1,:),'d-','Color','#518CD8','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_our(1,:),'^-','Color','#FEB40B','linewidth',2,'MarkerSize', 10);
set(gca,'xTick',[1,4,7,10,13,16,19],'xTickLabel',r_test_all([1,4,7,10,13,16,19]),'FontSize',12)
% legend('Baseline','LowCon', 'IBOSS','SRS','USSP','FontSize',14,'location','northwest','NumColumns',2);
title('(a)', 'Units', 'normalized', 'Position', [0.5, -0.15],'FontSize',15);
hold off
nexttile
plot(x,rmse_a(3,:),'o-','Color','#FD6D5A','linewidth',2,'MarkerSize', 10);
hold on
plot(x,rmse_sub_lowcon(3,:),'*-','Color','#6DC354','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_iboss(3,:),'pentagram-','Color','#994487','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_srs(3,:),'d-','Color','#518CD8','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_our(3,:),'^-','Color','#FEB40B','linewidth',2,'MarkerSize', 10);
set(gca,'xTick',[1,4,7,10,13,16,19],'xTickLabel',r_test_all([1,4,7,10,13,16,19]),'FontSize',12)
% legend('Baseline','LowCon','IBOSS','SRS','USSP','FontSize',14,'location','northwest','NumColumns',2);
title('(b)', 'Units', 'normalized', 'Position', [0.5, -0.15],'FontSize',15);
% ylim([0.28,0.34])
% ylim([0.25,0.35])
hold off
nexttile
plot(x,rmse_a(5,:),'o-','Color','#FD6D5A','linewidth',2,'MarkerSize', 10);
hold on
plot(x,rmse_sub_lowcon(5,:),'*-','Color','#6DC354','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_iboss(5,:),'pentagram-','Color','#994487','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_srs(5,:),'d-','Color','#518CD8','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_our(5,:),'^-','Color','#FEB40B','linewidth',2,'MarkerSize', 10);
set(gca,'xTick',[1,4,7,10,13,16,19],'xTickLabel',r_test_all([1,4,7,10,13,16,19]),'FontSize',12)
% legend('Baseline','LowCon','IBOSS','SRS','USSP','FontSize',14,'location','northwest','NumColumns',2);
title('(c)', 'Units', 'normalized', 'Position', [0.5, -0.15],'FontSize',15);
% ylim([0.25,0.35])
hold off
nexttile
plot(x,rmse_a(6,:),'o-','Color','#FD6D5A','linewidth',2,'MarkerSize', 10);
hold on
plot(x,rmse_sub_lowcon(6,:),'*-','Color','#6DC354','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_iboss(6,:),'pentagram-','Color','#994487','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_srs(6,:),'d-','Color','#518CD8','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_our(6,:),'^-','Color','#FEB40B','linewidth',2,'MarkerSize', 10);
set(gca,'xTick',[1,4,7,10,13,16,19],'xTickLabel',r_test_all([1,4,7,10,13,16,19]),'FontSize',12)
% legend('Baseline','LowCon','IBOSS','SRS','USSP','FontSize',14,'location','northwest','NumColumns',2);
title('(d)', 'Units', 'normalized', 'Position', [0.5, -0.15],'FontSize',15);
hold off
xlabel(t,'r on test data','FontSize',25)
ylabel(t,'IRMSE','FontSize',25)
lgd = legend('Baseline','LowCon','IBOSS','SRS','USSP','FontSize',25,'NumColumns',5);
lgd.Layout.Tile = 4;
lgd.Layout.Tile = 'north';
t.TileSpacing = 'compact';
t.Padding = 'compact';
% ylim([0.25,0.35])


%% %% boxpolt of RMSE
figure3=figure;    
edgecolor=[0,0,0]; % black color
position_1 = 0.5:2:16.5;  
position_2 = 0.75:2:16.75; 
position_3 = 1:2:17;
position_4 = 1.25:2:17.25;
position_5 = 1.5:2:17.5;

fillcolor1=[253, 109, 90]/255; 
fillcolor2=[109, 195, 84]/255; 
fillcolor3=[153, 68, 135]/255;
fillcolor4=[81, 140, 216]/255;
fillcolor5=[254, 180, 11]/255;

fillcolors=[repmat(fillcolor1,9,1);repmat(fillcolor2,9,1);repmat(fillcolor3,9,1);repmat(fillcolor4,9,1);repmat(fillcolor5,9,1)];
box_1 = boxplot(rmse_a','positions',position_1,'colors',edgecolor,'width',0.25,'symbol','r+','outliersize',5);
hold on;
box_2 = boxplot(rmse_sub_lowcon','positions',position_2,'colors',edgecolor,'width',0.25,'symbol','r+','outliersize',5);
box_3 = boxplot(rmse_sub_iboss','positions',position_3,'colors',edgecolor,'width',0.25,'symbol','r+','outliersize',5);
box_4 = boxplot(rmse_sub_srs','positions',position_4,'colors',edgecolor,'width',0.25,'symbol','r+','outliersize',5);
box_5 = boxplot(rmse_sub_our','positions',position_5,'colors',edgecolor,'width',0.25,'symbol','r+','outliersize',5);
boxobj = findobj(gca,'Tag','Box');
for j=1:length(boxobj) 
    patch(get(boxobj(length(boxobj)-j+1),'XData'),get(boxobj(length(boxobj)-j+1),'YData'),fillcolors(j,:));
end
set(gca,'XTick',[1,3,5,7,9,11,13,15,17],'XTicklabel',{'R=-0.8','R=-0.6','R=-0.4','R=-0.2', ...
    'R=0','R=0.2','R=0.4','R=0.6','R=0.8'},'FontSize',20)
xlabel('R on training data','FontSize',30);
ylabel('IRMSE','FontSize',30)
boxchi = get(gca, 'Children');
legend([boxchi(37),boxchi(28),boxchi(19),boxchi(10),boxchi(1)], ["Baseline","LowCon","IBOSS","SRS","USSP"],'FontSize',30,'location','north','NumColumns',5);


