%%
% Project Name: USSP
% Description: main function of experiment 5.2.1 in supplement materials when we take nonlinearly separable data shifts.
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2025-04-19
%%

clear
N=100;  %%Number of experiments
r_train_all=[-0.8;-0.6;-0.4;-0.2;0;0.2;0.4;0.6;0.8];    %% Value of r_train

rmse_a=zeros(length(r_train_all),19);
rmse_sub1=zeros(length(r_train_all),19);
 for ii=1:length(r_train_all)
    train_numb=20000;   %% Number of full samples 
    rmse_all=zeros(N,19);
    rmse_subsample1=zeros(N,19);
    %% Train data
    r_train=r_train_all(ii);
    d3=readmatrix('UD_1000_8.csv','Range',[2 2]);    %%Load the uniform design points
    

    for i=1:N  %%Repeat N times
        seed=i;
        rng(seed,'twister');
        [sample]=Sample_Generate_GP_NoninearlySeparableTerm(r_train,train_numb,i);  %%Generate normal distribution test samples

        x_train=sample(:,1:8);
        y_train=sample(:,9);
        
       sample3=sample;
        x_train1=sample3(:,1:8);
        s_train1=x_train1(:,1:5);
        v_trian1=x_train1(:,6:8);
        y_train1=sample3(:,9);
        %% Full data
            theta = [10 10 10 10 10 10 10 10]; 
            lob = [1e-1 1e-1 1e-1 1e-1 1e-1 1e-1 1e-1 1e-1];
            upb = [20 20 20 20 20 20 20 20];
            % dmodel = dacefit(x_train, y_train, @regpoly2, @corrgauss, theta, lob, upb);   %%GPR fit using  dacefit
 
        
        %% UD subsample
        [cm,id]=USSP(x_train1,d3);
        id=id{1,1};
        subsample1=unique(sample3(id,:),'rows','stable');
        x_subsample_train1=subsample1(:,1:8);
        y_subsample_train1=subsample1(:,9);
        dmodel_subsample1 = dacefit(x_subsample_train1, y_subsample_train1, @regpoly2, @corrgauss, theta, lob, upb);
        
        dmodel=dmodel_subsample1;
    %% Test_data
    %r_test_all=[-3;-2;-1.7;-1.5;-1.3;1.3;1.5;1.7;2;3];
    r_test_all=[-0.9;-0.8;-0.7;-0.6;-0.5;-0.4;-0.3;-0.2;-0.1;0;0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9];
%     r_test_all=r_test_all*10;

    
        for k=1:19
            r_test=r_test_all(k);
            seed=i+1;
            rng(seed,'twister');
            [sample_test]=Sample_Generate_GP_NoninearlySeparableTerm_Test(r_test,2000,i+1);    %%Generate uniformly distribution test samples
            x_test=sample_test(:,1:8);
            s_test=x_test(:,1:5);
            v_test=x_test(:,6:8);
            y_test=sample_test(:,9);        
            y_test_ols = predictor(x_test, dmodel);
            rmse_all(i,k)=sqrt(mean((y_test-y_test_ols).^2));
                
            y_test_subsample_ols1=predictor(x_test, dmodel_subsample1);
            rmse_subsample1(i,k)=sqrt(mean((y_test-y_test_subsample_ols1).^2));   
        
         end
     end

    rmse_a(ii,:)=mean(rmse_all,1);
    rmse_sub1(ii,:)=mean(rmse_subsample1,1);
        
    
    
 end 
 
figure1=figure;    %% figure of RMSE
x=1:19;
color1=winter(50);
color2=autumn(50);
for i=1:1:ceil(length(r_train_all)/2)
    plot(x,rmse_a(i,:),'^-','Color',color1(i*10,:),'linewidth',2);
    hold on
end
for i=1:1:ceil(length(r_train_all)/2)
    hold on 
    plot(x,rmse_sub1(i,:),'o-','Color',color2(i*10,:),'linewidth',2);
end
set(gca,'xTick',x,'xTickLabel',r_test_all,'FontSize',15)
 legend('Baseline(R=-0.8)','Baseline(R=-0.6)','Baseline(R=-0.4)','Baseline(R=-0.2)', 'Baseline(R=0)',...
     'SPAS(R=-0.8)','SPAS(R=-0.6)','SPAS(R=-0.4)','SPAS(R=-0.2)','SPAS(R=0)','NumColumns',2,'FontSize',15,'location','north');
xlabel('r on test data','FontSize',25);
 ylabel('IRMSE','FontSize',25);
hold off




figure2=figure;    %% boxpolt of RMSE
x=1:19;
color1=winter(50);
color2=autumn(50);
for i=ceil(length(r_train_all)/2)+1:1:length(r_train_all)
plot(x,rmse_a(i,:),'^-','Color',color1((i-5)*10,:),'linewidth',2);
hold on
end
for i=ceil(length(r_train_all)/2)+1:1:length(r_train_all)
hold on 
plot(x,rmse_sub1(i,:),'o-','Color',color2((i-5)*10,:),'linewidth',2);
end
set(gca,'xTick',x,'xTickLabel',r_test_all,'FontSize',15)
legend('Baseline(R=0.2)','Baseline(R=0.4)','Baseline(R=0.6)','Baseline(R=0.8)', ...
     'SPAS(R=0.2)','SPAS(R=0.4)','SPAS(R=0.6)','SPAS(R=0.8)','NumColumns',2,'FontSize',15,'location','north');
xlabel('r on test data','FontSize',25);
ylabel('IRMSE','FontSize',25)
hold off
category=[ones(size(rmse_a',1)*size(rmse_a',2),1);2*ones(size(rmse_sub1',1)*size(rmse_sub1',2),1)];
Rtrain=[ones(size(rmse_a',1),1);2*ones(size(rmse_a',1),1);3*ones(size(rmse_a',1),1);
    4*ones(size(rmse_a',1),1);5*ones(size(rmse_a',1),1);6*ones(size(rmse_a',1),1);
    7*ones(size(rmse_a',1),1);8*ones(size(rmse_a',1),1);9*ones(size(rmse_a',1),1);
    ones(size(rmse_sub1',1),1);2*ones(size(rmse_sub1',1),1);3*ones(size(rmse_sub1',1),1);
    4*ones(size(rmse_sub1',1),1);5*ones(size(rmse_sub1',1),1);6*ones(size(rmse_sub1',1),1);
    7*ones(size(rmse_sub1',1),1);8*ones(size(rmse_sub1',1),1);9*ones(size(rmse_sub1',1),1);];
value=[reshape(rmse_a',[size(rmse_a',1)*size(rmse_a',2),1]);reshape(rmse_sub1',[size(rmse_sub1',1)*size(rmse_sub1',2),1])];
tb1=table(category,Rtrain,value);
