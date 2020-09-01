%This script is meant to analyze a data.  Previously, I fit all of
%the data using the ML algorithm.  This grabs both the data and those fits
%and then does a couple things.  First, it makes a map of the fit onto the
%pathogen's proteome.  Then it uses both the original data and proteome
%projection to generate ROCs.

% *** This program was developed by Neal and Robayet.***

clear
%clc
close all
fclose all;

tic

% ***data with median WNVization***
% Use  samples that have CV >=0.3 for a case and control
%file1 = 'data_NIBIB_HCV_ML_mod_CV317-Jul-2020-16-50.csv';
file1 = 'data_NIBIB_Dengue_ML_CTSSeraCare_mod_CV317-Jul-2020-00-08.csv';
%file2 = 'data_NIBIB_Chagas_ML_mod_CV316-Jul-2020-00-02.csv';
%file1= 'data_NIBIB_HBV_ML_mod_CV316-Jul-2020-00-01.csv';
%file1 = 'data_NIBIB_WNV_ML_mod_CV315-Jul-2020-23-57.csv';
file2 = 'data_NIBIB_Normal_ML_mod_CV315-Jul-2020-23-50.csv';
%includes all WNV samples



%file1='data_NIBIB_DENV_corr8_ML_mod_18-Nov-2018-19-59.csv';
%file2 = 'data_NIBIB_WNV_corr8_ML_mod_20-May-2019-18-12.csv';
%file1 = 'data_NIBIB_HCV_corr8_ML_mod_08-Nov-2018-23-07.csv';
%file1='data_NIBIB_HBV_corr8_ML_mod_08-Nov-2018-23-05.csv';
%file1='data_NIBIB_Chagas_corr8_ML_mod_08-Nov-2018-17-32.csv';
%file2='data_NIBIB_WNV_corr8_ML_mod_05-Jul-2019-17-33.csv';
%file2='data_WNV_only_NIBIB_ML_mod_16-Jul-2018-15-37.csv'; % this file
%file1 = 'data_NIBIB_WNV_ML_mod_01-Jun-2020-23-39.csv';

% Use all samples for a case and control
%file1 = 'data_NIBIB_HCV_ML_mod_21-Jun-2020-00-56.csv';
%file1 = 'data_NIBIB_Dengue_ML_mod_11-Apr-2020-13-59.csv';
%file2 = 'data_NIBIB_Chagas_ML_mod_22-Jun-2020-14-55.csv';
%file1= 'data_NIBIB_HBV_ML_mod_21-Jun-2020-00-57.csv';
%file1 = 'data_NIBIB_WNV_ML_mod_01-Jun-2020-23-39.csv';
%file2 = 'data_NIBIB_WNV_ML_mod_21-Jun-2020-00-59.csv';
%includes all WNV samples




% ***data with no WNVization***
%file1='data_NIBIB_Chagas_ML_noMed_CV315-Jul-2020-23-28.csv';
%file1= 'data_NIBIB_Dengue_ML_noMed_CV315-Jul-2020-23-11.csv';
%file = 'data_NIBIB_HCV_ML_noMed_CV3_17-Jul-2020-16-56.csv';
% file = 'data_NIBIB_HBV_ML_noMed_CV315-Jul-2020-23-25.csv';
% file = 'data_NIBIB_Chagas_ML_noMed_CV315-Jul-2020-23-28.csv';
% file = 'data_NIBIB_WNV_ML_noMed_CV315-Jul-2020-23-16.csv';
%file2 = 'data_NIBIB_WNV_ML_noMed_CV315-Jul-2020-23-35.csv';

SVMoptimize = false(1); % turn it true if you want to optimize the parameters associated with SVM model
c = [0 1;2.2 0]; % put weights on classes to avoid sample imbalance
%draw_boundary = false(1);
Neal_svm =true(1);  % turn it true to run Neal's code on SVM
Prot_SVM = false(1);  % turn it true to run SVM on proteome
Array_SVM = true(1);  % turn it true to run SVM on array
meadianData = false(1); % turn it true if you use median WNVized data on SVM on array
poolsize=36;
aminos='ADEFGHKLNPQRSVWY';
numrep=1000;
Nsize=1000000; % this is the num of peptides at a time the program can calculate binding for
sample_trainfraction=0.20; % percent of samples to be used in training the model
pep_num=10; % number of peptides in training set


fignum = 0;
if Prot_SVM
    fprintf('reading fastA file\n');
    [fastaFile,Fdir]=uigetfile('*.fasta','MultiSelect','off'); %gives the user a gui to open the file
%Read in the proteome
[Prot_Header, Prot_Sequence] = fastaread(fastaFile);
if ~iscellstr(Prot_Header)
    Prot_Header=cellstr(Prot_Header);
    Prot_Sequence=cellstr(Prot_Sequence);
end

%Strip out the names of the proteins
numprot=length(Prot_Header);
Prot_Name=cell(numprot,1);
for iprot=1:numprot
    a=Prot_Header{iprot};
    Prot_Name(iprot)=cellstr(a);
end

%tic
for type_dataset=1:2
disp('Select the parameter file associated with the set of runs you want to use')
[Wfile,pathname]=uigetfile('*.txt','MultiSelect','off'); %gives the user a gui to open the file
% read the parameter
if type_dataset==1
    case_param=readtable([pathname,Wfile]);
    case_param=table2cell(case_param);
    aminoEigen_case=str2double(case_param(find(strcmp(case_param,'aminoEigen')),2));
    % extract dataShift or noise from the parameter
    dataShift_case = case_param(find(strcmp(case_param,'dataShift')),2);
    if strcmp(dataShift_case{1},cellstr('True'))
        dataShift_case = true(1);
    elseif strcmp(dataShift_case{1},cellstr('False'))
        dataShift_case = false(1); 
    else
        dataShift_case=str2double(dataShift_case);
    end
    HiddenLayer_case=str2double(case_param(find(strcmp(case_param,'hiddenLayers')),2));
    fprintf('AminoEigen used in the NN program is %.f\n',aminoEigen_case)
    fprintf('No of Hidden layers is %d\n',HiddenLayer_case)
else
    control_param=readtable([pathname,Wfile]);
    control_param=table2cell(control_param);
    aminoEigen_control=str2double(control_param(find(strcmp(control_param,'aminoEigen')),2));
    % extract dataShift or noise from the parameter
    dataShift_control=control_param(find(strcmp(control_param,'dataShift')),2);
    if strcmp(dataShift_control{1},cellstr('True'))
        dataShift_control = true(1);
    elseif strcmp(dataShift_control{1},cellstr('False'))
        dataShift_control = false(1); 
    else
        dataShift_control=str2double(dataShift_control);
    end
    HiddenLayer_control=str2double(control_param(find(strcmp(control_param,'hiddenLayers')),2));
    fprintf('AminoEigen used in the NN program is %.f\n',aminoEigen_control)
    fprintf('No of Hidden layers is %d\n',HiddenLayer_control)
end
% print the filename that was used in the NN program
filename_index=find(pathname=='\');
filename=pathname(filename_index(8)+1:filename_index(9)-1);
fprintf('reading folder %s\n',filename);

foldername='Sample';
numsamples=0;


disp('Reading files');
%determine how many folders there are
while exist([pathname,foldername,num2str(numsamples+1)],'dir')==7 %as long as the next folder is there
    numsamples=numsamples+1;
end


%read the fit matrices
for isample=1:numsamples
    path=[pathname,foldername,num2str(isample),'\'];
    W(isample)=read_weights_w_bias(path);
end



%Generate an array of sequences to test
[M,K]=size(W(1).W1);
[RK,H]=size(W(1).W2); % it calculates the length of the peptides from the proteome
[sequence,seqindex]=prot2peparray(Prot_Sequence,RK/K);
numseq=size(sequence,1);

% we need to replace 4 AA's with the ones from aminos otherwise they will be ignored during the 
%calculation of binding of a proteome sequence

%Therefore,substitute equivalent amino acids that are read in fitting data (I=V, T=S,
%M=L, C is ignored and not given binding significance).  You have to put
%something in for these amino acids in a real sequence.  These
%substitutions are well supported by similarity matrices except for C to A,
%which shows only weak similarity
real_sequence=sequence; %store the real sequence
sequence(sequence=='I')='V';
sequence(sequence=='T')='S';
sequence(sequence=='M')='L';
sequence(sequence=='C')='A';
    
%project the fits onto the proteome
if numseq>Nsize
        disp('projection on a big proteome (Chagas)')
        if type_dataset==1
            disp('projecting onto the proteome for all case samples...');
        else
            disp('projecting onto the proteome for all control samples...');
        end
            
    F_calc=zeros(numseq,numsamples);
    Ntot=numseq;
    seqtot=sequence;
   
    % set up the parallel pool with a defined number of workers
    if poolsize>0
        if isempty(gcp('nocreate'))
            MyCluster=parcluster('local');
            MyCluster.NumWorkers=poolsize;
            parpool(MyCluster);
        end
    end
    % Now, run parallel pool with the current set up
     %tic
     ticBytes(gcp);
    parfor isample=1:numsamples
    num_sections=ceil(Ntot/Nsize);
    F_calc_sections=zeros(Nsize,num_sections);
    for nn=1:num_sections
           n1=(nn-1)*Nsize+1;
           
        if Ntot-n1>Nsize
            N=Nsize;
        else
            N=Ntot-n1+1;
        end
        seq_chunk=seqtot(n1:n1+N-1,:);
        %fcalc_chunk=zeros(Nsize,1);
        fcalc_chunk=MLevaluate_N_layer_w_bias(seq_chunk,aminos,W(isample));
        %tem=MLevaluate_N_layer_w_bias(seq_chunk,aminos,W(isample));
        %fcalc_chunk=tem;
        if length(fcalc_chunk)<Nsize
            tem=fcalc_chunk;
            fcalc_chunk=zeros(Nsize,1);
            fcalc_chunk(1:length(tem))=tem;
            tem = []; % clear tem to free up memory 
        end
        
        F_calc_sections(:,nn)=fcalc_chunk;
        fcalc_chunk = []; % clear it to free up memory 
    end
    F_calc(:,isample)= F_calc_sections(1:Ntot)';
    F_calc_sections = []; % clear it to free up memory 
    end
    tocBytes(gcp);
    %toc
    
    delete(gcp('nocreate')) % shut down parallel pool
     
    if type_dataset==1
        F_calc_case=F_calc;
        numsample_case = numsamples;
        case_sequence = real_sequence;
        clear F_calc
    else
        F_calc_control=F_calc;
        numsample_control = numsamples;
        control_sequence = real_sequence;
        clear F_calc
    end
else
    
    disp('projection on  a small proteome')
    if type_dataset==1
        disp('projecting onto the proteome for all case samples...');
        F_calc_case=zeros(numseq,numsamples);
        parfor isample=1:numsamples
            F_calc_case(:,isample)=MLevaluate_N_layer_w_bias(sequence,aminos,W(isample));
        end
        delete(gcp('nocreate')) % shut down parallel pool
        numsample_case = numsamples;
        case_sequence = real_sequence;
    else
        disp('projecting onto the proteome for all control samples...');
        F_calc_control=zeros(numseq,numsamples);
        control_sequence = real_sequence;
        parfor isample=1:numsamples
            F_calc_control(:,isample)=MLevaluate_N_layer_w_bias(sequence,aminos,W(isample));
        end
        delete(gcp('nocreate')) % shut down parallel pool
        numsample_control = numsamples;
        control_sequence = real_sequence;
    end   
end

fprintf('\n');
end
%toc
rng('shuffle'); % To genearate different random numbers on rand_sample (line 271)

proteome_seq =linspace(1,numseq,numseq);
% take the average data for each class
Fmean_case = mean(F_calc_case,2);
Fmean_control = mean(F_calc_control,2);
% merge two datasets
X1 = [Fmean_case,Fmean_control];
% calculate the difference between case and control
Mean_diff = Fmean_case-Fmean_control ;
[MeanD_sort,mean_ind]=sort(Mean_diff,'descend');
% label the each sample from the merged data
Y1=false(numseq,1); % vector of label [0,1]
for i=1:numseq
%     if X1(i,1)>=0 && Mean_diff(i)>0
   if Mean_diff(i)>0
        Y1(i)=true(1);
    end
end

%plot the average binding data and color them based on the differences
fignum=fignum+1;                       
figure(fignum)
scatter(proteome_seq,X1(:,1),'b','.')
hold on
scatter(proteome_seq,X1(:,2),'r','.')
legend('WNV','HBV')
xlabel('Proteome sequence')
ylabel('log10(Average predicted binding)')
title('Predicted binding from WNV and HBV samples')
hold off

fignum =fignum+1;
figure(fignum)
gscatter(X1(:,1),X1(:,2),Y1,'rb','.',7)  % scatter plot by group
xlabel('log10(Average predicted binding from WNV)')
ylabel('log10(Average predicted binding from HBV)')
legend({'HBV','WNV'},'Location','northwest')
title('Predicted binding from WNV and HBV samples-HL1')
%% prepare data to train a SVM model
data_CaseControl= [F_calc_case,F_calc_control];
[~,pval]=ttest2(F_calc_case',F_calc_control');
[N_prot,tot_sample]=size(data_CaseControl);

X2=data_CaseControl;
% label the samples with '1' for case and '0' for control
Y2=false(tot_sample,1);
Y2(1:numsample_case)=true(1);
% randomize the sample's data and their corresponding labels
rand_sample=randperm(tot_sample,tot_sample);
% % use the same sample indices for array data
X2=X2(:,rand_sample);
Y2=Y2(rand_sample);

%Plot the p-values
Inv_pval = 1./pval;
%Plot the p-values
if numseq<Nsize
fignum=fignum+1;
figure(fignum);
Inv_pval=1./pval;
proteome_seq=1:length(Inv_pval);
plot(Inv_pval,proteome_seq);
high_index=Inv_pval>1e10;
ypos=Inv_pval(high_index);
xpos=proteome_seq(high_index);
for i=1:sum(high_index)
    h=text(ypos(i),xpos(i),Prot_Name{seqindex(xpos(i))});
%     set(h,'Rotation',90);
end
end

% Run SVM on calculated data from proteome
% process the calculated data if it is needed to
if ~dataShift_case
     disp('take log of the calculated data')
     X2=log10(X2);
end
 



% run SVM and plot the results
plot_title_prot='WNV Vs. HBV (proteome)-HL1';
numpick_prot=zeros(tot_sample,1);
totscore_prot=zeros(tot_sample,1);
ntrain_prot=round(tot_sample*sample_trainfraction);
% avoid sample imbalance in the SVM
CaseProt_ind = find(Y2==1);  % save indices for the samples from case
ControlProt_ind = find(Y2==0); % save indices for the samples from control
ProtPeptrain_freq = zeros(N_prot,1); % keep track of indices of peptides that are being selected for training
disp('classifying two datasets from proteome using SVM');
if Neal_svm
     for irep=1:numrep
     %pick equal no of samples for train set from both of case and control
     %to avoid sampling imbalance
     CaseTrainindex_prot = sort(randsample(CaseProt_ind, round(ntrain_prot/2))); 
     ControlTrainindex_prot = sort(randsample(ControlProt_ind, round(ntrain_prot/2))); 
     % combine the indices for train and test sets from case and control
     trainindex_prot = sort([CaseTrainindex_prot;ControlTrainindex_prot]);
     temp = false(1,tot_sample);
     temp(trainindex_prot) = true(1);
     trainindex_prot = temp;
     testindex_prot =~ trainindex_prot;
     clear temp
     %testindex_prot = setdiff((1:1:tot_sample),trainindex_prot);
     [~,pvalue_prot]=ttest2(X2(:,Y2&trainindex_prot')',X2(:,~Y2&trainindex_prot')');
    [sortp_prot,ind_prot]=sort(pvalue_prot,'ascend');
    pepindex_prot=false(N_prot,1);
    pepindex_prot(ind_prot(1:pep_num))=true(1);
    ProtPeptrain_freq(ind_prot(1:pep_num)) = ProtPeptrain_freq(ind_prot(1:pep_num))+1;
    %train a SVM model with peptides and sampels
    %SVMmodel_prot=fitcsvm(X2(pepindex_prot,trainindex_prot)',Y2(trainindex_prot),'ClassNames',[0,1],'KernelFunction','linear','Cost',c);
    SVMmodel_prot=fitcsvm(X2(pepindex_prot,trainindex_prot)',Y2(trainindex_prot),'ClassNames',[0,1]);    
    [label_prot,score_prot]=predict(SVMmodel_prot,X2(pepindex_prot,testindex_prot)');
    numpick_prot(testindex_prot)=numpick_prot(testindex_prot)+1;
    totscore_prot(testindex_prot)=totscore_prot(testindex_prot)+score_prot(:,2);
    end

    finscore_prot=totscore_prot./numpick_prot;
    minfinscore_prot=min(finscore_prot);
    maxfinscore_prot=max(finscore_prot);

    %make the ROC
    fractdisease_prot=zeros(101,1);
    fractcontrol_prot=zeros(101,1);
    k=1;
    for threshold_prot=minfinscore_prot:(maxfinscore_prot-minfinscore_prot)/100:maxfinscore_prot
    %number of disease correct
       fractdisease_prot(k)=sum((Y2==1)&(finscore_prot>threshold_prot))/sum(Y2==1);
    %number of control correct
       fractcontrol_prot(k)=sum((Y2==0)&(finscore_prot<threshold_prot))/sum(Y2==0);
       k=k+1;
    end

senst_prot = fractdisease_prot;
specft_prot = fractcontrol_prot;
% senst_prot1 = senst_prot;
% senst_prot1(1)=[];
% senst_prot2=senst_prot;
% senst_prot2(end)=[];
% % Calculate AUC using the formula for area of a trapezoid
% height_prot = (senst_prot1+senst_prot2)/2;
% width_prot = diff(specft_prot);
% AUC_prot = sum(height_prot'*width_prot);   
AUC_prot = sum(0.5*(senst_prot(2:end)+senst_prot(1:end-1)).*(specft_array(2:end)-specft_array(1:end-1)));
    
    
fignum=fignum+1;
figure(fignum);
h=axes;
plot(fractcontrol_prot,fractdisease_prot,'-bo','LineWidth',2,'MarkerFaceColor','b');
set(h,'Xdir','reverse')  
xlabel('Specificity');
ylabel('Sensitivity');
%xlabel('Sensitivity');
%ylabel('Specificity');
legend({sprintf('AUC = %.3g',AUC_prot)}, 'Location', 'southeast')
title(plot_title_prot);
%disp('....')
else
    %pick samples for train set for both of case and control
     CaseTrainindex_prot = sort(randsample(CaseProt_ind, round(ntrain_prot/2))); 
     ControlTrainindex_prot = sort(randsample(ControlProt_ind, round(ntrain_prot/2))); 
     % combine the indices for train and test sets from case and control
     trainindex_prot = sort([CaseTrainindex_prot;ControlTrainindex_prot]);
     temp = false(1,tot_sample);
     temp(trainindex_prot) = true(1);
     trainindex_prot = temp;
     testindex_prot =~ trainindex_prot;
     clear temp
    [~,pvalue_prot]=ttest2(X2(:,Y2&trainindex_prot')',X2(:,~Y2&trainindex_prot')');
    [sortp,ind_prot]=sort(pvalue_prot,'ascend');
    pepindex_prot=false(N_prot,1);
    %rand_pepProt=sort(randperm(N_prot,pep_num));
    pepindex_prot(ind_prot(1:pep_num))=true(1);
    ProtPeptrain_freq(ind_prot(1:pep_num)) = ProtPeptrain_freq(ind_prot(1:pep_num))+1;
    %pepindex_prot(ind_prot(1:round(N_prot*pep_trainfrac)))=true(1);
    if SVMoptimize
    %rng default
    SVMmodel_prot = fitcsvm(X,Y,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'));
    else
    SVMmodel_prot=fitcsvm(X2(pepindex_prot,trainindex_prot)',Y2(trainindex_prot),'ClassNames',[0,1]);
    end
    [label_prot,score_prot]=predict(SVMmodel_prot,X2(pepindex_prot,testindex_prot)');
    [Xsvm_prot,Ysvm_prot,Tsvm_prot,AUC_prot]=perfcurve(Y2(testindex_prot),score_prot(:,logical(SVMmodel_prot.ClassNames)),'true');
    %make the ROC
    fignum=fignum+1;
    figure(fignum);
    plot(Xsvm_prot,Ysvm_prot,'-bo','LineWidth',0.05,'MarkerFaceColor','b'); 
    xlabel('False positive rate (1-Specificity)');
    ylabel('True positive rate (Sensitivity)');
    legend(sprintf('ROC Curve\nAUC = %.3g', AUC_prot))
    title(plot_title_prot);
    %disp('....')
end
[ProtPeptrain_freq_val,ProtPeptrain_freq_ind]=sort(ProtPeptrain_freq,'descend');
fprintf('\n*******Peptides that were most frequently selected to fit into SVM classifier*******\n');
for i=1:pep_num
    fprintf('%s %d %d\n',real_sequence(ProtPeptrain_freq_ind(i),:),ProtPeptrain_freq_ind(i),(ProtPeptrain_freq_val(i)/numrep)*100) 
end
end

%% Run SVM on measured data from array
% open the sequence containging file 
if Array_SVM 
    disp('Select the matlab file that contains peptide sequences on the array')
    [sequence,pathname_array]=uigetfile('*.mat','MultiSelect','off'); %gives the user a gui to open the file
sequence = load([pathname_array,sequence],'sequence');
array_seq = sequence.sequence;
meas_data1=dlmread(file1);
meas_data2=dlmread(file2);
[~,num_samCase]=size(meas_data1);
[~,num_samControl]=size(meas_data2);
    
% plot the measured data with differnt colors
mean_array1=mean(log10(meas_data1),2);
mean_array2=mean(log10(meas_data2),2);
Diff_meanArray=mean_array1-mean_array2;
Y_array = false(length(Diff_meanArray),1);
for i = 1:length(Diff_meanArray)
% if mean_array1(i)>0 && Diff_meanArray(i)>0
   if Diff_meanArray(i)>0
       Y_array(i)=true(1);
   end
end

fignum=fignum+1;
figure(fignum);
gscatter(mean_array1,mean_array2,Y_array,'rb','.',7);
xlabel('log10(Average array binding from WNV)')
ylabel('log10(Average array binding from HBV)')
if ~meadianData
    xlim([min(mean_array1)-0.5,5])
    ylim([min(mean_array2)-0.5,5])
end
legend({'HBV','WNV'},'Location','northwest')
title('Average array binding from WNV and HBV samples')

% Now prepare the data to fit into the SVM model
if Prot_SVM
    if isa(dataShift_case,'double') && isequal(dataShift_case,dataShift_control)
    % get dataShift from the case data and add it as noise 
    disp('adding noise to the measured binding')
    meas_data1=log10(meas_data1+dataShift_case);
    meas_data2=log10(meas_data2+dataShift_case);
    elseif isa(dataShift_case,'logical') && isa(dataShift_control,'logical')
    % subtract minimum value from each sample and add '1' afterwards
    disp('subtracting minimum binding value and adding 1 to the measured bindings')
    for i = 1:num_samCase
        meas_data1(:,i) = log10(meas_data1(:,i) - (min(meas_data1(:,i))-1));
    end

    for j = 1:num_samControl
        meas_data2(:,j) = log10(meas_data2(:,j) - (min(meas_data2(:,j))-1));
    end
    end
else
    
    for i = 1:num_samCase
        meas_data1(:,i) = log10(meas_data1(:,i) - (min(meas_data1(:,i))-1));
    end

    for j = 1:num_samControl
        meas_data2(:,j) = log10(meas_data2(:,j) - (min(meas_data2(:,j))-1));
    end
end



%calculate p-values from measured data
meas_data=[meas_data1,meas_data2];
[N_array,num_sam_array]=size(meas_data);
[~,P_meas]=ttest2(meas_data1',meas_data2');
% label the samples with '1' and '0'
Y3=false(num_sam_array,1);
Y3(1:num_samCase)=true(1);
X3=meas_data;
% randomize the sample's data and their corresponding labels
rand_sample_array=randperm(num_sam_array,num_sam_array);
X3=X3(:,rand_sample_array);
Y3=Y3(rand_sample_array);
plot_title='WNV Vs. HBV ';


% run SVM and plot the results
numpick_array=zeros(num_sam_array,1);
totscore_array=zeros(num_sam_array,1);
% if num_samCase <  num_samControl
%     ntrain_array=round(num_samCase*sample_trainfraction);
% else
%     ntrain_array=round(num_samControl*sample_trainfraction);
   
ntrain_array=round(num_sam_array*sample_trainfraction);

% avoid sample imbalance in the SVM
CaseArray_ind = find(Y3==1);  % save indices for the samples from case
ControlArray_ind = find(Y3==0); % save indices for the samples from case
ArrayPeptrain_freq = zeros(N_array,1); % keep track of indices of peptides that are being selected for training
disp('classifying two datasets from array using SVM');


if Neal_svm
for irep=1:numrep
     %pick equal no of samples for train set from both of case and control
     %to avoid sampling imbalance
     CaseTrainindex_array = sort(randsample(CaseArray_ind, round(ntrain_array/2))); 
     ControlTrainindex_array = sort(randsample(ControlArray_ind, round(ntrain_array/2))); 
     trainindex_array = sort([CaseTrainindex_array;ControlTrainindex_array]);
     temp = false(1,num_sam_array);
     temp(trainindex_array) = true(1);
     trainindex_array = temp;
     testindex_array =~ trainindex_array;
     clear temp
     % select peptides and use them with samples from train set to train SVM model
    [~,pvalue_array]=ttest2(X3(:,Y3&trainindex_array')',X3(:,~Y3&trainindex_array')');
    [sortp_array,ind_array]=sort(pvalue_array,'ascend');
    pepindex_array=false(N_array,1);
    %rand_peparray=sort(randperm(N_array,pep_num));
    pepindex_array(ind_array(1:pep_num))=true(1);
    ArrayPeptrain_freq(ind_array(1:pep_num))=ArrayPeptrain_freq(ind_array(1:pep_num))+1;
    %pepindex_array(ind_array(1:round(N_array*pep_trainfrac)))=true(1);
    %SVMmodel_array=fitcsvm(X3(pepindex_array,trainindex_array)',Y3(trainindex_array),'ClassNames',[0,1],'KernelFunction','linear','Cost',c););
    SVMmodel_array=fitcsvm(X3(pepindex_array,trainindex_array)',Y3(trainindex_array),'ClassNames',[0,1]);
    [label_array,score_array]=predict(SVMmodel_array,X3(pepindex_array,testindex_array)');
    numpick_array(testindex_array)=numpick_array(testindex_array)+1;
    totscore_array(testindex_array)=totscore_array(testindex_array)+score_array(:,2);
end

finscore_array=totscore_array./numpick_array;
minfinscore_array=min(finscore_array);
maxfinscore_array=max(finscore_array);

%make the ROC
fractdisease_array=zeros(101,1);
fractcontrol_array=zeros(101,1);
Accuracy_array = zeros(101,1);
k=1;
for threshold=minfinscore_array:(maxfinscore_array-minfinscore_array)/100:maxfinscore_array
    %number of disease correct (sensitivity)
    %fractdisease_array(k)=sum((Y3==1)&(finscore_array<threshold))/sum(Y3==1);
    fractdisease_array(k)=sum((Y3==1)&(finscore_array>threshold))/sum(Y3==1);
    %number of control correct (specificity)
    %fractcontrol_array(k)=sum((Y3==0)&(finscore_array>threshold))/sum(Y3==0);
    fractcontrol_array(k)=sum((Y3==0)&(finscore_array<threshold))/sum(Y3==0);
    Accuracy_array(k) = (sum((Y3==1)&(finscore_array>threshold)) + sum((Y3==0)&(finscore_array<threshold)))/num_sam_array;
    k=k+1;
end

senst_array = fractdisease_array;
specft_array = fractcontrol_array;
% % Calculate AUC using the formula for area of a trapezoid
% height_array = senst_array(2:end)+senst_array(1:end-1);
% width_array = diff(specft_array);
% AUC_array = sum((1/2)*(height_array).*width_array);
AUC_array = sum(0.5*(senst_array(2:end)+senst_array(1:end-1)).*(specft_array(2:end)-specft_array(1:end-1)));


fignum=fignum+1;
figure(fignum);
h=axes;
plot(fractcontrol_array,fractdisease_array,'-bo','LineWidth',2,'MarkerFaceColor','b');
set(h,'Xdir','reverse')  
%xlabel('Sensitivity');
%ylabel('Specificity');
xlabel('Specificity');
ylabel('Sensitivity');
legend({sprintf('AUC = %.3g', AUC_array)}, 'Location', 'southeast')  
title(plot_title);
else
         %pick samples for train set for both of case and control
     CaseTrainindex_array = sort(randsample(CaseArray_ind, round(ntrain_array/2))); 
     %CaseTestindex_prot = setdiff(CaseProt_ind,CaseTrainindex_prot);
     ControlTrainindex_array = sort(randsample(ControlArray_ind, round(ntrain_array/2))); 
     % combine the indices for train and test sets from case and control
     trainindex_array = sort([CaseTrainindex_array;ControlTrainindex_array]);
     temp = false(1,num_sam_array);
     temp(trainindex_array) = true(1);
     trainindex_array = temp;
     testindex_array =~ trainindex_array;
     clear temp
     % select peptides and use them with samples from train set to train SVM model
    [~,pvalue_array]=ttest2(X3(:,Y3&trainindex_array')',X3(:,~Y3&trainindex_array')');
    [sortp_array,ind_array]=sort(pvalue_array,'ascend');
    %select random peptides to train SVM
    pepindex_array=false(N_array,1);
    %rand_peparray=sort(randperm(N_array,pep_num));
    %pepindex_array(rand_peparray)=true(1);
    pepindex_array(ind_array(1:pep_num))=true(1);
    ArrayPeptrain_freq(ind_array(1:pep_num))=ArrayPeptrain_freq(ind_array(1:pep_num))+1;
    %pepindex_array(ind_array(1:round(N_array*pep_trainfrac)))=true(1);
    if SVMoptimize
    %rng default
    SVMmodel_array = fitcsvm(X,Y,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'));
    else
    SVMmodel_array=fitcsvm(X3(pepindex_array,trainindex_array)',Y3(trainindex_array),'ClassNames',[0,1]);
    end
    [label_array,score_array]=predict(SVMmodel_array,X3(pepindex_array,testindex_array)');
    [Xsvm_array,Ysvm_array,Tsvm_array,Accuracy_array]=perfcurve(Y3(testindex_array),score_array(:,logical(SVMmodel_array.ClassNames)),'true');
    %make the ROC
    fignum=fignum+1;
    figure(fignum);
    plot(Xsvm_array,Ysvm_array,'-bo','LineWidth',2,'MarkerFaceColor','b'); 
    xlabel('False positive rate (1-Specificity)');
    ylabel('True positive rate (Sensitivity)');
    legend(sprintf('ROC Curve\nAUC = %.3g', Accuracy_array))
    title(plot_title);
    %disp('....')
end
[ArrayPeptrain_freq_val,ArrayPeptrain_freq_ind]=sort(ArrayPeptrain_freq,'descend');
fprintf('\n*******Array Peptides that were most frequently selected to fit into SVM classifier*******\n');
% for i=1:20
%     fprintf('%s %d %d\n',array_seq(ArrayPeptrain_freq_ind(i),:),ArrayPeptrain_freq_ind(i),(ArrayPeptrain_freq_val(i)/numrep)*100) 
% end
for i=1:20
    fprintf('%s %d\n',array_seq(ArrayPeptrain_freq_ind(i),:),(ArrayPeptrain_freq_val(i)/numrep)*100) 
end
end
toc


