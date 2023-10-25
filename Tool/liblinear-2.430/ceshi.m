% data = load('skin.txt');       %Input a matrix
% x = data(:, 2:end); y = data(:, 1);
rng(869);
% load 'Ecoli.mat';
% load 'Echo.mat';
% load('Hourse.mat')
% load('Cleveland.mat')
% load('Creadit.mat')
% load 'Yeast.mat';
% load('Vowel.mat')
% load('Shuttle.mat')
% load('Titanic.mat')
load 'n1000000.mat';
% load 'Waveform.mat';
% load('TwoNorm.mat')
% load('a6atrain.mat')
% load('covtype.mat')
% load('WPBC.mat')
% load('Haberman.mat')
% X=X(1:100000,:);Y=Y(1:100000,:);
% load 'Echo1.mat';
% load 'phishing.mat';
% training_label_vector = Y;
% training_instance_matrix = X;
% [label_vector, instance_matrix] = libsvmread('Ecoli.mat');
% label_vector = Y;
% instance_matrix = X;
% model = train(Y, X, '-s 3 -B 1','col');
% [predicted_label, accuracy, ~] = predict(Y, X, model, '-s 3 -B 1','col')
%%
% [heart_scale_label, heart_scale_inst] = libsvmread('heart_scale'); %读数据
% model=train(heart_scale_label, heart_scale_inst, '-c 1');
% [predict_label, accuracy, dec_values] = predict(heart_scale_label, heart_scale_inst, model);
%
% X(X > 0.3) = 0; 
k=3;
[m,~] = size(Y);
mBest_AC = 0;
Best_C = 0;
% indices_X1 = crossvalind('Kfold',X(:,1),k);
% for C = 1:10
% for t1=1:k  %十折交叉验证
%     test_X1 = (indices_X1 == t1);      %样本测试集索引
%     train_X1 = ~test_X1;               %的样本训练集索引
%     DataTrain.X = X(train_X1,:);%样本训练集
%     DataTrain.Y = Y(train_X1,:);%标签训练集
%     DataTest.X = X(test_X1,:);  %样本测试集
%     DataTest.Y = Y(test_X1,:);  %测试集
%     TestX = DataTest.X;
%     TestY = DataTest.Y;
%     X1 = DataTrain.X;
%     Y1 = DataTrain.Y;
%     X_sparse = sparse(X1);
%     TestX = sparse(TestX);
%     heart_scale_label = Y1;%训练标签
%     heart_scale_inst = X_sparse;%训练样本稀疏化
%     option = sprintf('-s 3 -B 1 -c %f',C);
%     model=train(heart_scale_label, heart_scale_inst, option);
%     [predict_label,~,~] = predict(TestY, TestX, model);
%     ac=size(find(predict_label==DataTest.Y))/size(DataTest.Y);
%     J(1,t1)=ac;
% end
% sm_Ac=mean(J);
% if sm_Ac> mBest_AC
%     mBest_AC =  sm_Ac;
%     Best_C = C;
% end
% end
% C = Best_C;
C = 1;
tic;
for t2 =1:10
indices_X1 = crossvalind('Kfold',X(:,1),k);
for t1=1:k  %十折交叉验证
    test_X1 = (indices_X1 == t1);      %样本测试集索引
    train_X1 = ~test_X1;               %的样本训练集索引
    DataTrain.X = X(train_X1,:);%样本训练集
    DataTrain.Y = Y(train_X1,:);%标签训练集
    DataTest.X = X(test_X1,:);  %样本测试集
    DataTest.Y = Y(test_X1,:);  %测试集
    TestX = DataTest.X;
    TestY = DataTest.Y;
    X1 = DataTrain.X;
    Y1 = DataTrain.Y;
    X_sparse = sparse(X1);
    TestX = sparse(TestX);
    heart_scale_label = Y1;%训练标签
    heart_scale_inst = X_sparse;%训练样本稀疏化
    option = sprintf('-s 3 -B 1 -c %f',C);
    model=train(heart_scale_label, heart_scale_inst, option);
    [predict_label, accuracy, dec_values] = predict(TestY, TestX, model);
    ac=size(find(predict_label==DataTest.Y))/size(DataTest.Y);
    J(1,t1)=ac;
end
sm_Ac=mean(J);
JJ(t2,1) = sm_Ac;
fprintf('sm_Ac:%.4f\n',sm_Ac);
end
Ssm_Ac = mean(JJ);
SS = var(100*JJ);
fprintf('Ssm_Ac:%.4f,SS:%.4f\n',Ssm_Ac,SS);
toc;