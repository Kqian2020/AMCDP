function model = train(X, Y, parameter)
%% optimization parameters
lambda1    = parameter.lambda1;
lambda2    = parameter.lambda2;
lambda3    = parameter.lambda3;
lambda4    = parameter.lambda4;
lambda5    = parameter.lambda5;
maxIter    = parameter.maxIter;
minLoss    = parameter.minLoss;
[num_instance, num_dim] = size(X);
[~, num_class] = size(Y);
% representative
paraDc = 1.2;
[tempIndex,instanceSimilarMatrix] = seekMaster(X, paraDc);
Ymaster = Y(tempIndex,:);
tempN = zeros(num_instance,num_instance);
for i=1:num_instance
    tempN(i,tempIndex(1,i)) = 1;
end

N = tempN.*instanceSimilarMatrix;

% label correlation M
M = 1-pdist2((Ymaster+Y)', Y', 'cosine');

XTX = X'*X;
W = eye(num_dim, num_class);
W_1 = W;
C = M;
Lip1 = 2*norm(XTX)^2;
bk = 1; 
bk_1 = 1; 
iter = 1;
obj_loss = zeros(1,maxIter);
while iter <= maxIter
%     L = diag(sum(C,1)) - C;
    L = diag(sum(C,2)) - C;
    %% Lip
    Lip = sqrt(Lip1 + 2*norm(lambda5*(L+L'))^2);
    %% update W
    W_k    = W + (bk_1 - 1)/bk * (W - W_1);
    Gw_x_k = W_k - 1/Lip * gradientOfW(X,Y,W,C,lambda5);
    W_1    = W;
    W      = softthres(Gw_x_k,lambda1/Lip);
    bk_1   = bk;
    bk     = (1 + sqrt(4*bk^2 + 1))/2;
    
    %% update C
    C = ((lambda2+lambda3)*eye(num_class) + lambda4*(Y-N*Y)'*(Y-N*Y))\(lambda2*M);
    
    %% stop conditions
    loss = 0.5*norm((X*W-Y),'fro')^2 + 0.5*lambda2*norm(C-M,'fro')^2 + 0.5*lambda4*norm(N*Y - N*Y*C,'fro')^2;
    loss = loss + lambda5*trace(W*L*W') + 0.5*lambda3*norm(C,'fro')^2 + lambda1*norm(W,1);
    if loss < minLoss
        break
    end
    obj_loss(1,iter) = loss;
    iter=iter+1;
end
model.W = W;
model.M = M;
model.C = C;
model.obj_loss = obj_loss;
model.iter = iter;
end

%% seek instance master
function [tempMasterIndex, instanceSimilarMatrix] = seekMaster(X, paraDc)
[num_instance, ~] = size(X);
tempDistanceMatrix = pdist2(X, X,'euclidean');
instanceSimilarMatrix = exp(-tempDistanceMatrix.^2./ paraDc^2);
tempDensityArray = sum(instanceSimilarMatrix);

tempDistanceToMasterArray = zeros(1, num_instance);
tempMasterIndex = zeros(1, num_instance);
for i = 1:num_instance
    tempMasterIndex(1,i) = i;
    tempIndex = tempDensityArray>tempDensityArray(i);
    [~,tempSelectIndex]=find(tempIndex==1);
    if sum(tempIndex) > 0
        [tempDistanceToMasterArray(1,i),tempMostMasterIndex] = min(tempDistanceMatrix(i,tempIndex)); 
        tempMasterIndex(1,i) = tempSelectIndex(1,tempMostMasterIndex);
    end
end
end
%% soft thresholding operator
function Ws = softthres(W,lambda)
Ws = max(W-lambda,0) - max(-W-lambda,0);
end
%% gradient W
function gradient = gradientOfW(X,Y,W,C,lambda3)
L = diag(sum(C,2)) - C;
% L = diag(sum(C,1)) - C;
gradient =X'*(X*W - Y);
gradient = gradient + lambda3*W*(L + L');
end