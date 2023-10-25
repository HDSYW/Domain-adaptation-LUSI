function [P,tao] = DPcaculate(X,A_X,Y,A_Y,DV,KP,ptype,paratao)
Ptype = num2str(ptype);
tao=paratao;
Y(Y==-1,1)=0;%%
switch  Ptype
    case 'A_Y'
        P = Y*Y';
    case 'DV'
        P = DV*DV';
    case 'Zero'
        P=zeros(size(Y,1),size(Y,1));
        tao=0;
    case 'Kernel'
        K=KerF(X,KP,A_X);
        P_=zeros(size(K,1),size(K,1));
        for i = 1:size(K,2)
          P_=P_+K(:,i)*K(:,i)';
        end
        P=P_./size(K,2);
    case  'PCA'
        relation=corr(A_X,A_Y,'type',"Spearman");
        ind=find(relation==max(relation));
        P=X(:,ind)*X(:,ind)';        
    case  'A_Y+DV+Kernel'
        K=KerF(X,KP,A_X);
        P_=zeros(size(K,1),size(K,1));
        for i = 1:size(K,2)
          P_=P_+K(:,i)*K(:,i)';
        end
        P=(Y*Y'+ DV*DV'+P_./size(K,2))./3;
    case'A_Y+Kernel+PCA'
        K=KerF(X,KP,A_X);
        P_=zeros(size(K,1),size(K,1));
        for i = 1:size(K,2)
          P_=P_+K(:,i)*K(:,i)';
        end
        P1=P_./size(K,2);
        relation=corr(A_X,A_Y,'type',"Spearman");
        ind=find(relation==max(relation));
        P2=X(:,ind)*X(:,ind)';
        P = (Y*Y'+P1+P2)./3;
%     case'9'
%         P =   ones(length(Y),1)*ones(length(Y),1)';
%     case'10'
%         P =   Y*Y';
end
    end