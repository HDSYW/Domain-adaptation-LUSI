clc;
clear;
%% 数据导入
load heart;
Y(Y==2)=-1;
[X,PS]=mapminmax(X,0,1);
X=mapminmax(X',0,1)';%对X进行归一化处理
%% 交叉验证
k = 10;%
indices = crossvalind('Kfold',X(:,1),k); %矩阵k折分类后的索引值
%%  模型训练
kernel='gaussian';%核函数
ac=0;
J=[];
for p1=-5:5
    Para.p1=2.^p1; 
    C=2.^(p1);
    for power=-8:2:8
           Para.kpar.kp1 =2.^power; Para.kpar.kp2 = 0;  
           kc=2.^power;
           for i = 1:k
            test = (indices == i); train = ~test;%分出训练集和测试集
            DataTrain.X = X(train,:);
            DataTrain.Y = Y(train,:);
            DataTest.X = X(test,:);
            DataTest.Y = Y(test,:);
            svm=train_svm(DataTrain.X',DataTrain.Y',kernel,kc,6);
            [result,wb]=test_svm(svm,DataTest.X',DataTest.Y',kernel,kc);%
            J(i)=1-result.ero;
            if mean(J)>ac
                ac=mean(J);
                Best_C=C;
                Best_kc=kc;
            end
        end
    end
end
fprintf('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
disp(['准确率：',num2str(ac)])
disp(['Best_C:',num2str(Best_C)])
disp(['Best_kc:',num2str(Best_kc)])
fprintf('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
% [m,n]=find(J==max(J));
% AC=max(max(J));
% fprintf('准确率：%f\n',AC);