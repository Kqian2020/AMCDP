clear;clc

our_path = '';
compare_path = '';

our_matFiles = dir(strcat(our_path,'*.mat'));
compare_matFiles = dir(strcat(compare_path,'*.mat'));

mat_num = length(our_matFiles);
assert(mat_num == length(compare_matFiles), 'mat number is unequal.');

AveragePrecision = zeros(3,1);
AvgAuc = zeros(3,1);
HammingLoss = zeros(3,1);
Coverage = zeros(3,1);
OneError = zeros(3,1);
RankingLoss = zeros(3,1);

for i = 1:mat_num
    mat_name = our_matFiles(i).name;
    disp([mat_name,': start ']);
    % load our data
    load(strcat(our_path,mat_name));
    our_AveragePrecision = All_results(1,:);
    our_AvgAuc = All_results(2,:);
    our_HammingLoss = All_results(3,:);
    our_Coverage = All_results(4,:);
    our_OneError = All_results(5,:);
    our_RankingLoss = All_results(6,:);
    % load compare data
    load(strcat(compare_path,mat_name));
    compare_AveragePrecision = All_results(1,:);
    compare_AvgAuc = All_results(2,:);
    compare_HammingLoss = All_results(3,:);
    compare_Coverage = All_results(4,:);
    compare_OneError = All_results(5,:);
    compare_RankingLoss = All_results(6,:);
    
    % t test
    [h_AveragePrecision, p_AP] = ttest2(our_AveragePrecision, compare_AveragePrecision);
    [h_AvgAuc, p_Auc] = ttest2(our_AvgAuc, compare_AvgAuc);
    [h_HammingLoss, p_HL] = ttest2(our_HammingLoss, compare_HammingLoss);
    [h_Coverage, p_CV] = ttest2(our_Coverage, compare_Coverage);
    [h_OneError, p_OE] = ttest2(our_OneError, compare_OneError);
    [h_RankingLoss, p_RL] = ttest2(our_RankingLoss, compare_RankingLoss);
    
    % count
    % AP
    if h_AveragePrecision == 0
        AveragePrecision(2,1) = AveragePrecision(2,1) + 1;
    elseif mean(our_AveragePrecision) < mean(compare_AveragePrecision)
        AveragePrecision(3,1) = AveragePrecision(3,1) + 1;
    elseif mean(our_AveragePrecision) > mean(compare_AveragePrecision)
        AveragePrecision(1,1) = AveragePrecision(1,1) + 1;
    elseif mean(our_AveragePrecision) == mean(compare_AveragePrecision) && std(our_AveragePrecision) < std(compare_AveragePrecision)
        AveragePrecision(1,1) = AveragePrecision(1,1) + 1;
    elseif mean(our_AveragePrecision) == mean(compare_AveragePrecision) && std(our_AveragePrecision) == std(compare_AveragePrecision)
        AveragePrecision(2,1) = AveragePrecision(2,1) + 1;
    else
        AveragePrecision(3,1) = AveragePrecision(3,1) + 1;
    end
    % AvgAuc
    if h_AvgAuc == 0
        AvgAuc(2,1) = AvgAuc(2,1) + 1;
    elseif mean(our_AvgAuc) < mean(compare_AvgAuc)
        AvgAuc(3,1) = AvgAuc(3,1) + 1;
    elseif mean(our_AvgAuc) > mean(compare_AvgAuc)
        AvgAuc(1,1) = AvgAuc(1,1) + 1;
    elseif mean(our_AvgAuc) == mean(compare_AvgAuc) && std(our_AvgAuc) < std(compare_AvgAuc)
        AvgAuc(1,1) = AvgAuc(1,1) + 1;
    elseif mean(our_AvgAuc) == mean(compare_AvgAuc) && std(our_AvgAuc) == std(compare_AvgAuc)
        AvgAuc(2,1) = AvgAuc(2,1) + 1;
    else
        AvgAuc(3,1) = AvgAuc(3,1) + 1;
    end
    % HammingLoss
    if h_HammingLoss == 0
        HammingLoss(2,1) = HammingLoss(2,1) + 1;
    elseif mean(our_HammingLoss) > mean(compare_HammingLoss)
        HammingLoss(3,1) = HammingLoss(3,1) + 1;
    elseif mean(our_HammingLoss) < mean(compare_HammingLoss)
        HammingLoss(1,1) = HammingLoss(1,1) + 1;
    elseif mean(our_HammingLoss) == mean(compare_HammingLoss) && std(our_HammingLoss) < std(compare_HammingLoss)
        HammingLoss(1,1) = HammingLoss(1,1) + 1;
    elseif mean(our_HammingLoss) == mean(compare_HammingLoss) && std(our_HammingLoss) == std(compare_HammingLoss)
        HammingLoss(2,1) = HammingLoss(2,1) + 1;
    else
        HammingLoss(3,1) = HammingLoss(3,1) + 1;
    end
    % Coverage
    if h_Coverage == 0
        Coverage(2,1) = Coverage(2,1) + 1;
    elseif mean(our_Coverage) > mean(compare_Coverage)
        Coverage(3,1) = Coverage(3,1) + 1;
    elseif mean(our_Coverage) < mean(compare_Coverage)
        Coverage(1,1) = Coverage(1,1) + 1;
    elseif mean(our_Coverage) == mean(compare_Coverage) && std(our_Coverage) < std(compare_Coverage)
        Coverage(1,1) = Coverage(1,1) + 1;
    elseif mean(our_Coverage) == mean(compare_Coverage) && std(our_Coverage) == std(compare_Coverage)
        Coverage(2,1) = Coverage(2,1) + 1;
    else
        Coverage(3,1) = Coverage(3,1) + 1;
    end
    % OneError
    if h_OneError == 0
        OneError(2,1) = OneError(2,1) + 1;
    elseif mean(our_OneError) > mean(compare_OneError)
        OneError(3,1) = OneError(3,1) + 1;
    elseif mean(our_OneError) < mean(compare_OneError)
        OneError(1,1) = OneError(1,1) + 1;
    elseif mean(our_OneError) == mean(compare_OneError) && std(our_OneError) < std(compare_OneError)
        OneError(1,1) = OneError(1,1) + 1;
    elseif mean(our_OneError) == mean(compare_OneError) && std(our_OneError) == std(compare_OneError)
        OneError(2,1) = OneError(2,1) + 1;
    else
        OneError(3,1) = OneError(3,1) + 1;
    end
    % RankingLoss
    if h_RankingLoss == 0
        RankingLoss(2,1) = RankingLoss(2,1) + 1;
    elseif mean(our_RankingLoss) > mean(compare_RankingLoss)
        RankingLoss(3,1) = RankingLoss(3,1) + 1;
    elseif mean(our_RankingLoss) < mean(compare_RankingLoss)
        RankingLoss(1,1) = RankingLoss(1,1) + 1;
    elseif mean(our_RankingLoss) == mean(compare_RankingLoss) && std(our_RankingLoss) < std(compare_RankingLoss)
        RankingLoss(1,1) = RankingLoss(1,1) + 1;
    elseif mean(our_RankingLoss) == mean(compare_RankingLoss) && std(our_RankingLoss) == std(compare_RankingLoss)
        RankingLoss(2,1) = RankingLoss(2,1) + 1;
    else
        RankingLoss(3,1) = RankingLoss(3,1) + 1;
    end
    disp([mat_name,': end ']);
end

All_rank = [AveragePrecision, AvgAuc, HammingLoss, Coverage, OneError, RankingLoss]';
