%%Matlab implementation of Westwick et al. (2006) "Identification of
%%Multiple-Input Systems with Highly Coupled Inputs: Application to EMG
%%Prediction from Multiple Intracortical Electrodes" Neural Computation
%%18(2)

%2014 Alireza Hashemi

for i = 1:nChannels
    Ch{i} = reshape(Ch{i},nTrials,trialLength);
    %save(['Ch',num2str(i)], 'Ch{i}')
end

stim = reshape(stim,nTrials,trialLength);


for i = 1:nChannels
    
    x = Ch{i};
    y = stim;    
    nTrials = size(x, 1);
    L = size(x, 2);
    nEqsPerTrial = (L - Lf + 1) * ones(nTrials,1);
    nEqs = nEqsPerTrial(1) * nTrials;
    A = zeros(nEqs, Lf);
    Y = zeros(nEqs,1);

    j = 1;

for k = 1 : nTrials
    
    at = toeplitz(x(k,:), [x(k,1) zeros(1,Lf-1)]);
    at = at(Lf : end, :);
    yt = y(k, Lf : end)';
    A(j : j+nEqsPerTrial(k)-1, :) = at;
    Y(j : j+nEqsPerTrial(k)-1) = yt;
    j = j + nEqsPerTrial(k);

end
Ch{i} = NaN(nEqs,Lf);
Ch{i} = A;
 if j ~= nEqs+1
        error('j ~= nEqs+1');
 end

end

%create matrices for QR decomposition using indices & perform QR

count = nChannels
allValues = []; %all min values for each iteration
exclude = []; %excluded channels list
allmin = [];
rsq = [];
decomp = cell(count,count);
copyCh = Ch;
allFest = [];

while count>1
    set = [1:count]; 
    setB = fliplr(set); %create index for QR matrix using toeplitz
    A1 = triu(toeplitz(set)); B1 = tril(toeplitz(setB));
    A2  = [A1;zeros(1,numel(set))]; B2 = [zeros(1,numel(set));B1];
    idx = B2+A2; idx = idx(1:numel(set),:);
    X = cell2mat(copyCh); X = [X ones(size(X,1),1)];
    [Fest,Fint,r,rint,stats] = regress(Y,X);
    rsq = [rsq stats(1)];
    for i = 1:count
        for j = 1:count
            decomp{i,j} = [copyCh{idx(i,j)}]; %create the matrix for QR using index
        end
        kyooR(i,:,:) = [decomp{i,[1:count]},Y]; %cell2mat alternative (QR only works with matrix)
        [Q,R] = qr(reshape(kyooR(i,:,:),length(A(:,1)),[]),0);
        val = [R(end-Lf-1:end-1,end)]; 
        val = dot(val,val)/trialLength; 
        allValues = [allValues, val]; 
    end
    c = find(allValues == min(allValues)); %find minimum
    allmin = [allmin min(allValues)];
    exclude = [exclude c]; %store the location of all the minimum values from each iteration
    aux = ~ismember(set,c); 
    idx2 = set(aux); %new index excluding min value
    copyCh = copyCh(idx2); 
    count = count-1
    if count>1
        kyooR = NaN(i,length(Y),(Lf*(count))+1);
        decomp = cell(count,count); %preallocate cell for next iteration
        allValues = [];
        clear c
    end
end

clear set
