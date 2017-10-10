
%% initialization
paths = genpath(cd);
path(path, paths);
load('ucf101.mat')
knn = 7;
sig = cal_sigma (data,knn)

options=make_options('gamma_I',1,'gamma_A',1e-5,'KernelParam',sig);
%options=make_options('gamma_I',1,'gamma_A',1e-5,'Kernel','linear');
options.Verbose=1;
options.UseBias=0;
options.UseHinge=1;
options.LaplacianNormalize=1;
options.NewtonLineSearch=0;

options.Cg=1; % PCG
options.MaxIter=1000; % upper bound
options.CgStopType=1; % 'stability' early stop
options.CgStopParam=0.0001; % tolerance: 1.5%
options.CgStopIter=3; % check stability every 3 iterations

datastruct.X=data;

fprintf('Computing the kernel...\n');
datastruct.K=calckernel(options,data,data);
datastruct.L=laplacian(options,data);
% datastruct.L=lap_gtruth(options,label);

%datastruct.L=lap_pickle(options,adjlist);

[data_num,dim] = size(data);

%% experiment
label_of_interest = 7;

% gtruth = label(:,label_of_interest)*2 - 1;   %for thumos data (multilabel)
gtruth = (label == label_of_interest)*2 - 1;           %for UCF data (multiclass)
%gtruth = gtruth (random_permutation+1);                   % for permuting embeddings

ind_pos = find (gtruth==1);        
ind_neg = find (gtruth==-1);        

APP= [];

for k_1 = 1:10
    num_pos_label = k_1;
    num_neg_label = k_1;

    AP_ = [];

    for seed_ = 1:20;

        rand('seed', 10*seed_);

        ind_ind_pos = randperm (length (ind_pos), num_pos_label);
        ind_ind_neg = randperm (length (ind_neg), num_neg_label);
        labeled_ind = [ind_pos(ind_ind_pos); ind_neg(ind_ind_neg)]';
        unlabeled_ind=setdiff(1:data_num, labeled_ind);                    

        datastruct.Y=zeros(size(gtruth));
        datastruct.Y (labeled_ind) = gtruth (labeled_ind);

        fprintf('Training SVM in the primal with Newton''s method...\n');
        
        classifier= lapsvmp(options,datastruct);
                
        fprintf('It took %f seconds.\n',classifier.traintime);

        %%%% Out1 is for MAP;
        out1 = datastruct.K(:,classifier.svs)*classifier.alpha+classifier.b;
        Selection = [gtruth(unlabeled_ind), out1(unlabeled_ind)];
        desc_sort = sortrows(Selection,-2);
        C_ = desc_sort(:,1);
        C_ = (C_ + 1)/2;
        CC = find(C_ == 1);
        tot_sum = 0;

        for i = 1:length(CC)
            tot_sum = tot_sum + i/CC(i);
        end

        AP = tot_sum/length(CC);
        AP_ = [AP_, AP];

    end
    APP = cat(1, APP, AP_);
end


%save('thumos_validation_result', 'APP');

