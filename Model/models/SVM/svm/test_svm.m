function [result,wb] = test_svm(svm,test_data,test_label,kerneltype,kc,f_A)
%X = DataTest.X;
%tlabel = DataTest.Y;% input:
% svm: train_svm函数返回的支持向量
% test_data: 测试数据
% test_label测试集标签
% kerneltype和技巧种类，形式参数；可选：inear gaussian mullinear
% output:
% result.Y:测试集中的预测类别? result.Y ∈[+1,-1]
% result.accuracy:测试集的准确率
% b的求法：选择对所有满足0＜ ai ＜C求得bi，并对b进行取平均运算

sum_b = svm.label - (svm.a'.* svm.label)*kernel(svm.data,svm.data,kerneltype,kc); % bj = yj-sum(ai*yi*<xi,xj>)
b = mean(sum_b);
wx = (svm.a'.* svm.label)*kernel(svm.data,test_data,kerneltype,kc);% w = sum(ai*yi*K(x,xi))

%result.y = wx+b;      %fx
result.Y = sign(wx+b+f_A');% 分类
result.ero = 1-length(find(result.Y==test_label))/length(test_label);%mm为正确分类的个数

% result.accuracy = length(find(result.Y==test_label))/length(test_label);% 预测准确率
% index = find(result.Y==1);
% result.Y1 = result.Y(index);
% test_label1 = test_label(index);
% result.pr =length(find(result.Y1==test_label1)) /length(result.Y>0);%pr
wb.B = b;
wb.wx=wx;
end