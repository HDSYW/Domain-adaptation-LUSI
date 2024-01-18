function [PredY,model] = DLUSI_VSVM(X_test,trn,Para)
%---------- Data ----------
X_train_A=trn.Ax      ; Y_train_A=trn.Ay;
X_train_T=trn.Tx       ; Y_train_T=trn.Ty;
X_train_AT=trn.ATx ; Y_train_AT=trn.ATy;
%---------- Initiation ----------
tao_1 =Para.tao_1   ; tao_2 =Para.tao_2  ;
n=size(X_train_A,1) ; p=size(X_train_T,1) ; m=size(X_test,1);
ptype=Para.ptype    ; beta=Para.beta     ;  beta_=Para.beta_;
lam1=Para.lam1      ; lam2=Para.lam2      ; 
lam =[lam1.*ones(n,1);lam2.*ones(p,1)];
%---------- Q ----------
Qb=[ones(n,1),zeros(n,1);zeros(p,1),ones(p,1)];
%---------- K ---------
Knn= KerF(X_train_A,Para.kpar,X_train_A);
Kpp= KerF(X_train_T,Para.kpar,X_train_T);
Knp= [Knn,zeros(n,p);zeros(p,n),Kpp];
KerTstX = KerF(X_test,Para.kpar,X_train_T);
KerTstX2 = KerF(X_test,Para.kpar,X_train_A);
%---------- V &P -----------
V_n=Para.V_n ; V_p=Para.V_p;
V_np=tao_1.*[V_n,zeros(n,p);zeros(p,n),V_p];
% P_n=Pcaculate(X_train_A,Y_train_A,ptype);
% P_p=Pcaculate(X_train_T,Y_train_T,ptype);
P_n=zeros(size(X_train_A,1));
P_p=zeros(size(X_train_T,1));
P_np =[tao_1.*P_n,zeros(n,p);zeros(p,n),(1-tao_1).*P_p];
%----------transfer invariant ----------
PP_n1=beta*beta';
PP_n2=beta_*beta_';
% PP_np =(1-tao_1).*PP_n1;
PP_np =(1-tao_1).*[PP_n1,zeros(n,p);zeros(p,n),PP_n2];
% PP_np=(1-tao_1).*eye(n+p,n+p);
% PP_np =(1-tao_1).*[PP_n1,zeros(n,p);zeros(p,n),zeros(p,p)];
%---------- AY & AQ ----------
AY=((V_np+P_np+PP_np)*Knp+lam.*eye(n+p,n+p))\(V_np+P_np+PP_np)*Y_train_AT;
Ab=((V_np+P_np+PP_np)*Knp+lam.*eye(n+p,n+p))\(V_np+P_np+PP_np)*Qb;
%---------- Solve A & b ----------
b=(Qb'*(V_np+P_np+PP_np)*(Qb-Knp*Ab))\(Qb'*(V_np+P_np+PP_np)*(Y_train_AT-Knp*AY));
A = AY-Ab*b;
%---------- Output----------
A_t=A((n+1):end,:);
A_a=A(1:n,:);
FTst = (A_t'*KerTstX')'+b(2)*ones(m,1);
% FTst =(A_t'*KerTstX')'+b(2)*ones(m,1)+(A_a'*KerTstX2')'+b(1)*ones(m,1) ;
FTst_prob= mapminmax(FTst',0,1)';
ftst = FTst_prob-0.5;
PredY.tst = sign(ftst);  
model.ftst = ftst;
model.beta = beta;
model.prob=FTst;
model.alpha = A_t;
model.b = b;
end
