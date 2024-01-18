function phi = DPcaculate_all(X , T_Y ,A_X,P_Y,A_Y,DV,KP,ptype,paratao)
Ptype = num2str(ptype);
tao=paratao;
P_Y(P_Y==-1)=0;%%
A_Y(A_Y==-1)=0;%%
T_Y(T_Y==-1)=0;%%
switch Ptype
    case 'A_Y'
        P = P_Y*P_Y';
        phi=P_Y;
    case 'DV'
        P = DV*DV';
        phi=DV;
    case 'Kernel'
        K=KerF(X,KP,A_X);
        P_=zeros(size(K,1),size(K,1));
        for i = 1:size(K,2)
            P_=P_+K(:,i)*K(:,i)';
        end
        P=P_./size(K,2);
        phi=K;
    case  'Spearman1'
        relation=corr(A_X,A_Y,'type',"Spearman");
        ind=find(relation==max(relation));
        P=X(:,ind)*X(:,ind)'; 
        phi=X(:,ind);
    case  'Kendall1'
        relation=corr(A_X,A_Y,'type',"Kendall");
        ind=find(relation==max(relation));
        P=X(:,ind)*X(:,ind)'; 
        phi=X(:,ind);
    case  'pearson1'
        relation=corr(A_X,A_Y,'type',"pearson");
        ind=find(relation==max(relation));
        P=X(:,ind)*X(:,ind)'; 
        phi=X(:,ind);
    case  'Spearman2'
        relation=corr(A_X,A_Y,'type',"Spearman");
        [secondorder,secondidx]=sort(relation,'descend');
        nancount=sum(isnan(relation));
        idx=secondidx(nancount+2);
        P=X(:,idx)*X(:,idx)'; 
        phi=X(:,idx);
    case  'Kendall2'
        relation=corr(A_X,A_Y,'type',"Kendall");
        [secondorder,secondidx]=sort(relation,'descend');
        nancount=sum(isnan(relation));
        idx=secondidx(nancount+2);
        P=X(:,idx)*X(:,idx)'; 
        phi=X(:,idx);
    case  'pearson2'
        relation=corr(A_X,A_Y,'type',"pearson");
        [secondorder,secondidx]=sort(relation,'descend');
        nancount=sum(isnan(relation));
        idx=secondidx(nancount+2);
        P=X(:,idx)*X(:,idx)'; 
        phi=X(:,idx);
    case  'Spearman_T'
        relation=corr(X,T_Y,'type',"Spearman");
        ind=find(relation==max(relation));
        P=X(:,ind)*X(:,ind)'; 
        phi=X(:,ind);
    case  'T_Y'
        P=T_Y*T_Y';
        phi=T_Y;
    case 'X'
        P_X=zeros(size(X,1),size(X,1));
        for i = 1:size(X,2)
          P_X=P_X+X(:,i)*X(:,i)';
        end
        P=P_X./size(X,2);
        phi=X;
    case 'XX'
        XX=X*X';
        P_XX=zeros(size(XX,1),size(XX,1));
        for i = 1:size(XX,2)
          P_XX=P_XX+XX(:,i)*XX(:,i)';
        end
        P=P_XX./size(XX,2);
        phi=X*X';
    case 'one'  
        P=ones(size(X,1),1)*ones(size(X,1),1)';
        phi=ones(size(X,1),1);
end
end