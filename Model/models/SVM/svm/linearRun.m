clc;
clear;
%% 数据导入
load heart
[X,PS]=mapminmax(X,0,1);
X=mapminmax(X',0,1)';%对X进行归一化处理
%% 交叉验证
k = 10;%
indices = crossvalind('Kfold',X(:,1),k); %矩阵k折分类后的索引值
%%  模型训练
ac=[];
best_C=0;
kc=0;
AC=0;
kernel='linear';
    for C=1:10
        for i = 1:k
            test = (indices == i); train = ~test;
            DataTrain.X = X(train,:);
            DataTrain.Y = Y(train,:);
            DataTest.X = X(test,:);
            DataTest.Y = Y(test,:);
            svm=train_svm(DataTrain.X',DataTrain.Y',kernel,kc,1);
            [result,wb]=test_svm(svm,DataTest.X',DataTest.Y',kernel,kc);
            ac(i)=1-result.ero;
        end
            fprintf('第%d次',C)
            fprintf('准确率：%f\n',mean(ac))
            fprintf('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
        if AC<mean(ac)
            AC=mean(ac);
            best_C=C;
        end
    end
fprintf('最高准确率：%f\n',AC);
fprintf('C：%f\n',best_C);
fprintf('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')