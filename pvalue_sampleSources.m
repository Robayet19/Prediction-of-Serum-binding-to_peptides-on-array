% The program was written by Robayet
%% This script calculates the pvalues between two sets of samples for Dengue,and between case samples and normal samples

clear
clc
pval_Dengue = false(1); % turn it true for Dengue infected samples
median_normalize=true(1);%if you median normalize, I assume you want to resave
bad_sample_tracking = true(1); % print out bad sample ID and its source
save_DENV_diffSample = true(1);
tnow = datestr(now, 'dd-mmm-yyyy-HH-MM');

%read in the sequence data
fprintf('Enter the matlab load file with the sequence data\n');
[loadfile,loadpath]=uigetfile('*.mat','MultiSelect','off'); %gives the user a gui to open the file
fprintf('Enter file with data\n');
load([loadpath,loadfile]);

if exist('gprfile_Zika')
    gprfile = gprfile_Zika;
elseif exist('gprfile_Chagas')
    gprfile = gprfile_Chagas;
elseif exist('gprfile_WNV')
    gprfile = gprfile_WNV;
elseif exist('gprfile_HCV')
    gprfile = gprfile_HCV;
end

underscr_ind= cell2mat(strfind(gprfile(:,:),'_'));
% extract sample ID from the gprfile_Zika ( actually it should have been
% named as gprfile_Dengue since it was generaed from Dengue4 dataset)
gprfile=char(gprfile);
tem = char(zeros(length(gprfile),10));
array_sample = char(zeros(length(gprfile),50));
for i =1:length(gprfile)
sample = gprfile(i,underscr_ind(i,3)+1:underscr_ind(i,4)-1);
tem(i,1:length(sample))=sample;
array =  gprfile(i,1:underscr_ind(i,4)-1);
array_sample(i,1:length(array)) = array;
end
sample_name=cellstr(tem);
array_sample =cellstr(array_sample);

if pval_Dengue
    % import the data file
fprintf('import the data file\n');
[dlmfile,dlmpath]=uigetfile('*.*','MultiSelect','off');
data1=dlmread([dlmpath,dlmfile]);

fprintf('import the sample source data file\n');
[dlmfile2,dlmpath2]=uigetfile('*.*','MultiSelect','off');
sample_source = readtable([dlmpath2,dlmfile2]);
sampleID=sample_source.AssignedID;
sourceName=sample_source.REGION_SUPPLIER; % source name for each sample

    
% now arrange samples in the sample source according to  order of the samples in the binding data (data1)  
% file
new_ind=zeros(1,length(sample_name));
for i =1:length(sample_name)
new_ind(i)= find(strcmp(sample_name,sampleID(i)));
end

[~,new_ind_sort]= sort(new_ind);
sampleID_sort= sampleID(new_ind_sort);
sourceName_sort=sourceName(new_ind_sort);

%data1_sort= data1(:,new_ind);

source_type=unique(sourceName_sort); % name of the sources 
% seperate the sample data based on the source
%CTS_data= data1_sort(:,strcmp(sourceName,source_type(1)));
CTS_data= data1(:,strcmp(sourceName_sort,source_type(1)));
[~,CTS_sample]= size(CTS_data);
%ElPaso_data= data1_sort(:,strcmp(sourceName,source_type(2)));
ElPaso_data= data1(:,strcmp(sourceName_sort,source_type(2)));
[~,ElPaso_sample]=size(ElPaso_data);
%Peru_data= data1_sort(:,strcmp(sourceName,source_type(3)));
Peru_data= data1(:,strcmp(sourceName_sort,source_type(3)));
[~,Peru_sample]=size(Peru_data);
%SeraCare_data= data1_sort(:,strcmp(sourceName,source_type(4)));
SeraCare_data= data1(:,strcmp(sourceName_sort,source_type(4)));
[~,SeraCare_sample]=size(SeraCare_data);

% track the samples if you get rid of bad samples
if bad_sample_tracking
    corr_limit=0.8;
    corr_data1 = corr(data1);
    meancorr1=(sum(corr_data1)-1)/(length(sampleID)-1);
    sample_accept_ind= meancorr1>corr_limit;
    sample_reject_ind= ~(meancorr1>corr_limit);
    sample_reject= sampleID_sort(~(meancorr1>corr_limit));
    sample_accept= sampleID_sort(meancorr1>corr_limit);
    source_accept=sourceName_sort(meancorr1>corr_limit);
    source_reject=sourceName_sort(~(meancorr1>corr_limit));
    disp('indices of rejected samples')
    disp(find(sample_reject_ind))
    disp('sources of rejected samples')
    disp(source_reject)
end




%%Median normalize each array
if median_normalize
    disp('Median normalizing all case samples')
    CTS_data=median_norm(CTS_data);
    ElPaso_data=median_norm(ElPaso_data);
    Peru_data=median_norm(Peru_data);
    SeraCare_data=median_norm(SeraCare_data);
    %data1_sort_med=median_norm(data1_sort);
    data1_med=median_norm(data1);
end

% calculate p value between two sources 
[~,pval_CTSvsElPaso] = ttest2(CTS_data',ElPaso_data');
x1= sum(pval_CTSvsElPaso<10e-5);
[~,pval_CTSvsPeru] = ttest2(CTS_data',Peru_data');
x2= sum(pval_CTSvsPeru<10e-5);
[~,pval_CTSvsSeraCare] = ttest2(CTS_data',SeraCare_data');
x3= sum(pval_CTSvsSeraCare<10e-5);
[~,pval_ElPasovsPeru] = ttest2(ElPaso_data',Peru_data');
x4= sum(pval_ElPasovsPeru<10e-5);
[~,pval_ElPasovsSeraCare] = ttest2(ElPaso_data',SeraCare_data');
x5= sum(pval_ElPasovsSeraCare<10e-5);
[~,pval_PeruvsSeraCare] = ttest2(Peru_data',SeraCare_data');
x6= sum(pval_PeruvsSeraCare<10e-5);

fignum = 1;
figure(fignum)
pvallog_CTSvsElPaso=-log10(pval_CTSvsElPaso);
plot(pvallog_CTSvsElPaso);
title('difference between CTS and ElPaso')
xlabel('array peptides')
ylabel('-log10(pvalue)')

fignum =fignum +1;
figure(fignum)
pvallog_CTSvsPeru=-log10(pval_CTSvsPeru);
plot(pvallog_CTSvsPeru);
title('difference between CTS and Peru')
xlabel('array peptides')
ylabel('-log10(pvalue)')

fignum =fignum +1;
figure(fignum)
pvallog_CTSvsSeraCare=-log10(pval_CTSvsSeraCare);
plot(pvallog_CTSvsSeraCare);
title('difference between CTS and SeraCare')
xlabel('array peptides')
ylabel('-log10(pvalue)')

fignum =fignum +1;
figure(fignum)
pvallog_ElPasovsPeru=-log10(pval_ElPasovsPeru);
plot(pvallog_ElPasovsPeru);
title('difference between ElPaso and Peru')
xlabel('array peptides')
ylabel('-log10(pvalue)')

fignum =fignum +1;
figure(fignum)
pvallog_ElPasovsSeraCare=-log10(pval_ElPasovsSeraCare);
plot(pvallog_ElPasovsSeraCare);
title('difference between ElPaso and SeraCare')
xlabel('array peptides')
ylabel('-log10(pvalue)')

fignum = fignum +1;
figure(fignum)
pvallog_PeruvsSeraCare=-log10(pval_PeruvsSeraCare);
plot(pvallog_PeruvsSeraCare);
title('difference between Peru and SeraCare')
xlabel('array peptides')
ylabel('-log10(pvalue)')

% save samples from different sources seperately
if save_DENV_diffSample
    [N,~]=size(sequence);
    disp('Saving sequences and CTS data in one file for machine learning')
    savename1='NIBIB_DengueCTS_data_ML_mod_';
    fid1=fopen(['sequence_data_',savename1,tnow,'.csv'],'w');
    for i=1:N
        seq=sequence(i,:);
        fprintf(fid1,'%s',seq);
        fprintf(fid1,',%12.6f',CTS_data(i,:));
        fprintf(fid1,'\n');
    end
    fclose(fid1);
    
    disp('Saving sequences and El Paso data in one file for machine learning')
    savename1='NIBIB_DengueElPaso_data_ML_mod_';
    fid2=fopen(['sequence_data_',savename1,tnow,'.csv'],'w');
    for i=1:N
        seq=sequence(i,:);
        fprintf(fid2,'%s',seq);
        fprintf(fid1,',%12.6f',ElPaso_data(i,:));
        fprintf(fid2,'\n');
    end
    fclose(fid1);
    
    
    disp('Saving sequences and Peru data in one file for machine learning')
    savename1='NIBIB_DenguePeru_data_ML_mod_';
    fid3=fopen(['sequence_data_',savename1,tnow,'.csv'],'w');
    for i=1:N
        seq=sequence(i,:);
        fprintf(fid3,'%s',seq);
        fprintf(fid3,',%12.6f',Peru_data(i,:));
        fprintf(fid3,'\n');
    end
    fclose(fid3);
    
    disp('Saving sequences and SeraCare data in one file for machine learning')
    savename1='NIBIB_DengueSeraCare_data_ML_mod_';
    fid4=fopen(['sequence_data_',savename1,tnow,'.csv'],'w');
    for i=1:N
        seq=sequence(i,:);
        fprintf(fid4,'%s',seq);
        fprintf(fid4,',%12.6f',SeraCare_data(i,:));
        fprintf(fid4,'\n');
    end
    fclose(fid4);
end


%% import the normal sample data file
fprintf('import the data file\n');
[dlmfile3,dlmpath3]=uigetfile('*.*','MultiSelect','off');
data3=dlmread([dlmpath3,dlmfile3]);

%%Median normalize each array
if median_normalize
    disp('Median normalizing all normal samples')
    data3=median_norm(data3);
end

% select equal no normal samples randomly as a function of no of samples of each case
[~,Normal_sample]= size(data3);
Normal_CTS=data3(:,sort(randperm(Normal_sample,CTS_sample)));
Normal_ElPaso=data3(:,sort(randperm(Normal_sample,ElPaso_sample)));
Normal_Peru=data3(:,sort(randperm(Normal_sample,Peru_sample)));
Normal_SeraCare=data3(:,sort(randperm(Normal_sample,SeraCare_sample)));

[~,pval_CTSvsNormal] = ttest2(CTS_data',Normal_CTS');
[~,pval_ElPasovsNormal] = ttest2(ElPaso_data',Normal_ElPaso');
[~,pval_PeruvsNormal] = ttest2(Peru_data',Normal_Peru');
[~,pval_SeraCarevsNormal] = ttest2(SeraCare_data',Normal_SeraCare');

fignum =fignum +1;
figure(fignum)
pvallog_CTSvsNormal=-log10(pval_CTSvsNormal);
plot(pvallog_CTSvsNormal);
title('difference between CTS Dengue and Normal')
xlabel('array peptides')
ylabel('-log10(pvalue)')

fignum =fignum +1;
figure(fignum)
pvallog_ElPasovsNormal=-log10(pval_ElPasovsNormal);
plot(pvallog_ElPasovsNormal);
title('difference between ElPaso Dengue and Normal')
xlabel('array peptides')
ylabel('-log10(pvalue)')

fignum =fignum +1;
figure(fignum)
pvallog_PeruvsNormal=-log10(pval_PeruvsNormal);
plot(pvallog_PeruvsNormal);
title('difference between Peru Dengue and Normal')
xlabel('array peptides')
ylabel('-log10(pvalue)')

fignum =fignum +1;
figure(fignum)
pvallog_SeraCarevsNormal=-log10(pval_SeraCarevsNormal);
plot(pvallog_SeraCarevsNormal);
title('difference between SeraCare Dengue and Normal')
xlabel('array peptides')
ylabel('-log10(pvalue)')

end
