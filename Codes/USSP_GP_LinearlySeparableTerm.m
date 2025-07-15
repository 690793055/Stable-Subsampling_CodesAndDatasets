%%
% Project Name: USSP
% Description: main function of experiment 5.2.1
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2025-05-19
%%

clear
N=3;  %%Number of experiments

train_numb=20000;  %% Number of full samples 
r_train_all=[-3;-2;-1;0;1;2;3];   %% Value of r_train
r_test_all=[-3;-2.5;-2;-1.5;-1;-0.5;0;0.5;1;1.5;2;2.5;3];  %% Value of r_test

rmse_sub_our=zeros(length(r_train_all),length(r_test_all));
rmse_sub_lowcon=zeros(length(r_train_all),length(r_test_all));
rmse_sub_iboss=zeros(length(r_train_all),length(r_test_all));
rmse_sub_srs=zeros(length(r_train_all),length(r_test_all));



 for ii=1:length(r_train_all)
    % train_numb=20000;   %% Number of full samples 
    %% Train data
    r_train=r_train_all(ii);
    d1=readmatrix('UD_200_5.csv','Range',[2 2]);   %%Load the uniform design points
    d2=readmatrix('OLHD_199_5.csv','Range',[2 2]);   
        rmse_subsample_our=zeros(N,length(r_test_all));
       rmse_subsample_lowcon=zeros(N,length(r_test_all));
       rmse_subsample_iboss=zeros(N,length(r_test_all));
       rmse_subsample_srs=zeros(N,length(r_test_all));


    for i=1:N  %%Repeat N times
        seed=i;
        rng(seed,'twister');
        [sample]=Sample_Generate_GP_LinearlySeparableTerm(r_train,train_numb,i,'linear');  %%Generate normal distribution train samples     
        x_train=sample(:,1:5);
        y_train=sample(:,6);
        k_srs=200;
        k_iboss=200;
        %% srs data

            theta = repmat(1,1,5); 
            lob = repmat(1e-2,1,5);
            upb = repmat(40,1,5); 
            
           
        %% ud subsample
        [cm,id_our]=USSP(x_train,d1);
        id_our=id_our{1,1};
        [id_lowcon]=LowCon(x_train,d2(:,1:5));
        [id_iboss,~]=IBOSS(x_train,k_iboss);
        id_srs = randperm(train_numb, k_srs);

        subsample_our=unique(sample(id_our,:),'rows','stable');
        subsample_lowcon=unique(sample(id_lowcon,:),'rows','stable');
        subsample_iboss=unique(sample(id_iboss,:),'rows','stable');
        subsample_srs=unique(sample(id_srs,:),'rows','stable');
        
        x_subsample_train_our=subsample_our(:,1:5);
        x_subsample_train_lowcon=subsample_lowcon(:,1:5);
        x_subsample_train_iboss=subsample_iboss(:,1:5);
        x_subsample_train_srs=subsample_srs(:,1:5);
        y_subsample_train_our=subsample_our(:,6);
        y_subsample_train_lowcon=subsample_lowcon(:,6);
        y_subsample_train_iboss=subsample_iboss(:,6);
        y_subsample_train_srs=subsample_srs(:,6);
        

       dmodel_subsample_our = dacefit(x_subsample_train_our, y_subsample_train_our, @regpoly1, @corrgauss, theta, lob, upb);  
       dmodel_subsample_lowcon = dacefit(x_subsample_train_lowcon, y_subsample_train_lowcon, @regpoly1, @corrgauss, theta, lob, upb);  
       dmodel_subsample_iboss = dacefit(x_subsample_train_iboss, y_subsample_train_iboss, @regpoly1, @corrgauss, theta, lob, upb);  
       dmodel_subsample_srs = dacefit(x_subsample_train_srs, y_subsample_train_srs, @regpoly1, @corrgauss, theta, lob, upb);  %%GPR fit using  dacefit

    
       


    for k=1:length(r_test_all)
            r_test=r_test_all(k);
            seed=i+1;
            rng(seed,'twister');
            [sample_test]=Sample_Generate_GP_LinearlySeparableTerm_Test(r_test,2000,i,'normal','linear');   %%Generate uniformly distribution test samples
            x_test=sample_test(:,1:5);
            y_test=sample_test(:,6);    

            y_test_subsample_our = predictor(x_test, dmodel_subsample_our);
            rmse_subsample_our(i,k)=sqrt(mean((y_test-y_test_subsample_our).^2));
            y_test_subsample_lowcon=predictor(x_test, dmodel_subsample_lowcon);
            rmse_subsample_lowcon(i,k)=sqrt(mean((y_test-y_test_subsample_lowcon).^2));  
            y_test_subsample_iboss=predictor(x_test, dmodel_subsample_iboss);
            rmse_subsample_iboss(i,k)=sqrt(mean((y_test-y_test_subsample_iboss).^2));  
            y_test_subsample_srs=predictor(x_test, dmodel_subsample_srs);
            rmse_subsample_srs(i,k)=sqrt(mean((y_test-y_test_subsample_srs).^2));  
    end

        
              
    end


    
    rmse_sub_our(ii,:)=mean(rmse_subsample_our,1);
    rmse_sub_lowcon(ii,:)=mean(rmse_subsample_lowcon,1);
    rmse_sub_iboss(ii,:)=mean(rmse_subsample_iboss,1);
    rmse_sub_srs(ii,:)=mean(rmse_subsample_srs,1);    
    
    
 end 
 
figure1=figure;     %% figure of RMSE
t=tiledlayout(2,2,'TileSpacing','Compact');
x=1:length(r_test_all);
nexttile
% plot(x,rmse_a(1,:),'o-','Color','#FD6D5A','linewidth',2,'MarkerSize', 10);
hold on
plot(x,rmse_sub_lowcon(1,:),'*-','Color','#6DC354','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_iboss(1,:),'pentagram-','Color','#994487','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_srs(1,:),'d-','Color','#518CD8','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_our(1,:),'^-','Color','#FEB40B','linewidth',2,'MarkerSize', 10);
set(gca,'xTick',[1,4,7,10,13],'xTickLabel',r_test_all([1,4,7,10,13]),'FontSize',12)
% legend('Baseline','LowCon', 'IBOSS','SRS','USSP','FontSize',14,'location','northwest','NumColumns',2);
title('(a)', 'Units', 'normalized', 'Position', [0.5, -0.15],'FontSize',15);
hold off
nexttile
% plot(x,rmse_a(3,:),'o-','Color','#FD6D5A','linewidth',2,'MarkerSize', 10);
hold on
plot(x,rmse_sub_lowcon(3,:),'*-','Color','#6DC354','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_iboss(3,:),'pentagram-','Color','#994487','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_srs(3,:),'d-','Color','#518CD8','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_our(3,:),'^-','Color','#FEB40B','linewidth',2,'MarkerSize', 10);
set(gca,'xTick',[1,4,7,10,13],'xTickLabel',r_test_all([1,4,7,10,13]),'FontSize',12)
% legend('Baseline','LowCon','IBOSS','SRS','USSP','FontSize',14,'location','northwest','NumColumns',2);
title('(b)', 'Units', 'normalized', 'Position', [0.5, -0.15],'FontSize',15);
hold off
nexttile
% plot(x,rmse_a(5,:),'o-','Color','#FD6D5A','linewidth',2,'MarkerSize', 10);
hold on
plot(x,rmse_sub_lowcon(5,:),'*-','Color','#6DC354','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_iboss(5,:),'pentagram-','Color','#994487','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_srs(5,:),'d-','Color','#518CD8','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_our(5,:),'^-','Color','#FEB40B','linewidth',2,'MarkerSize', 10);
set(gca,'xTick',[1,4,7,10,13],'xTickLabel',r_test_all([1,4,7,10,13]),'FontSize',12)
% legend('Baseline','LowCon','IBOSS','SRS','USSP','FontSize',14,'location','northwest','NumColumns',2);
title('(c)', 'Units', 'normalized', 'Position', [0.5, -0.15],'FontSize',15);
hold off
nexttile
% plot(x,rmse_a(6,:),'o-','Color','#FD6D5A','linewidth',2,'MarkerSize', 10);
hold on
plot(x,rmse_sub_lowcon(6,:),'*-','Color','#6DC354','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_iboss(6,:),'pentagram-','Color','#994487','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_srs(6,:),'d-','Color','#518CD8','linewidth',2,'MarkerSize', 10);
plot(x,rmse_sub_our(6,:),'^-','Color','#FEB40B','linewidth',2,'MarkerSize', 10);
set(gca,'xTick',[1,4,7,10,13],'xTickLabel',r_test_all([1,4,7,10,13]),'FontSize',12)
% legend('Baseline','LowCon','IBOSS','SRS','USSP','FontSize',14,'location','northwest','NumColumns',2);
title('(d)', 'Units', 'normalized', 'Position', [0.5, -0.15],'FontSize',15);
hold off
xlabel(t,'r on test data','FontSize',25)
ylabel(t,'IRMSE','FontSize',25)
lgd = legend('LowCon','IBOSS','SRS','USSP','FontSize',25,'NumColumns',5);
lgd.Layout.Tile = 4;
lgd.Layout.Tile = 'north';
t.TileSpacing = 'compact';
t.Padding = 'compact';



%% %% boxpolt of RMSE
figure3=figure;    
edgecolor=[0,0,0]; % black color
% position_1 = [0.5:2:12.5];  
position_2 = [0.75:2:12.75]; 
position_3 = [1:2:13];
position_4 = [1.25:2:13.25];
position_5 = [1.5:2:13.5];

fillcolor1=[253, 109, 90]/255; 
fillcolor2=[109, 195, 84]/255; 
fillcolor3=[153, 68, 135]/255;
fillcolor4=[81, 140, 216]/255;
fillcolor5=[254, 180, 11]/255;

% fillcolors=[repmat(fillcolor1,7,1);repmat(fillcolor2,7,1);repmat(fillcolor3,7,1);repmat(fillcolor4,7,1);repmat(fillcolor5,7,1)];
fillcolors=[repmat(fillcolor2,7,1);repmat(fillcolor3,7,1);repmat(fillcolor4,7,1);repmat(fillcolor5,7,1)];
% box_1 = boxplot(rmse_sub_lowcon','positions',position_1,'colors',edgecolor,'width',0.25,'symbol','r+','outliersize',5);
hold on;
box_2 = boxplot(rmse_sub_lowcon','positions',position_2,'colors',edgecolor,'width',0.25,'symbol','r+','outliersize',5);
box_3 = boxplot(rmse_sub_iboss','positions',position_3,'colors',edgecolor,'width',0.25,'symbol','r+','outliersize',5);
box_4 = boxplot(rmse_sub_srs','positions',position_4,'colors',edgecolor,'width',0.25,'symbol','r+','outliersize',5);
box_5 = boxplot(rmse_sub_our','positions',position_5,'colors',edgecolor,'width',0.25,'symbol','r+','outliersize',5);
boxobj = findobj(gca,'Tag','Box');
for j=1:length(boxobj) 
    patch(get(boxobj(length(boxobj)-j+1),'XData'),get(boxobj(length(boxobj)-j+1),'YData'),fillcolors(j,:));
end
set(gca,'XTick',[1,3,5,7,9,11,13],'XTicklabel',{'R=-3','R=-2','R=-1','R=0', ...
    'R=1','R=2','R=3'},'FontSize',20)
xlabel('R on training data','FontSize',30);
ylabel('IRMSE','FontSize',30)
boxchi = get(gca, 'Children');
% ylim([2,12])
legend([boxchi(22),boxchi(15),boxchi(8),boxchi(1)], ["LowCon","IBOSS","SRS","USSP"],'FontSize',30,'location','northeast','NumColumns',5);







