function [beta,MMD] = KMM(X_train,X_test,Para)
n=size(X_train,1);  
m=size(X_test,1);
B=Para.B;

eps=(sqrt(n)-1)/sqrt(n);
Knp= KerF(X_train,Para.kpar,X_test);
Knn= KerF(X_train,Para.kpar,X_train);
Kpp= KerF(X_test,Para.kpar,X_test);
options=optimoptions('quadprog','Display','off');
H=Knn;
f=-n/m.*sum((KerF(X_test,Para.kpar,X_train)'),2);
A=[ones(1,n);-ones(1,n)];
bb=[n*(eps+1),n*(eps-1)];
lb=[zeros(n,1)];
ub=[B*ones(n,1)];
beta =quadprog(H,f,A,bb,[],[],lb,ub,[],options);
MMD=sum(sum((1/n).^2.*Knn))+ sum(sum((1/m).^2.*Kpp))- 2.*sum(sum((1/(n*m)).*Knp));
end

