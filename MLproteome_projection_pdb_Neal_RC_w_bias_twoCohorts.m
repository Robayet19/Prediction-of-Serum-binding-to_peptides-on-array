% This is jointly written in MATLAB by Robayet chowdhury
%This takes a fit from the machine learning algorithm and maps predicted
%binding onto protein sequences and structures from the PDB.  It then compares this to a scrambled set
%of sequences if desired

clear 
clc
tic
rng('shuffle');%randomize the random number generator

%USER PARAMETERS
R_prot=10; %this is the length of sequences that the proteome is cut up into
read_singleSample=true(1); % turn it true when Zscores will be used
sampleNo=1; % name of the sample folder containing Z score fits
Nsize=100000; % number of peptides to be generated from a protein 
poolsize=36; % no of workers to be used in the parallel programming
num_scramble=0;%compare each sequence to this many scrambled versions
Plot_Title='Projected on DENV1-E'; %The name on th e plots

%To project your equation on a sequence that
%is part of a structure in the PDB you must enter the PDB file identifier and
%enter the chain in that file you want to use (a, b, c...).
%----------Crystal structure of DENV with Antibody----------------------%
%pdb_file='4UTA';% Human MAb EDE1 with DENV E protein. This is the pdf file to download.  It contains the structure of the protein you are projecting onto
%pdb_file='4UT9'; % Human MAb EDE1 C10 with DENV E protein.
%pdb_file='4UTB'; %Human MAb EDE2 with DENV E protein
%pdb_file='4BZ1'; %Human MAb  with DENV E protein
%pdb_file='3J6U'; % CryoEM of DENV3 with Human MAb 5J7 
%pdb_file='4UIF'; % Human Monoclonal Ab 2D22 with DENV2
%pdb_file='5A1Z'; %DENV2 with human Ab 2D22 Fab at 37 degree celcius
%pdb_file='3UC0'; %DENV4 E protein with Fab of Chimpanzee MAb 5H2 
%----------Crystal structure of 4 serotypes of DENV----------------------%
pdb_file = '4CCT'; %Cryo_EM of DENV1
%pdb_file = '3J35'; %Cryo_EM of DENV2 at 37 C
%pdb_file = '3J6T'; %Cryo_EM of DENV3 at 37 C
%pdb_file = '4CBF'; %Cryo_EM of DENV4
%----------Crystal structure of WNV----------------------%
%pdb_file = '3J0B'; %Cryo_EM of WNV
%----------Crystal structure of HCV----------------------%
%pdb_file= '1QGT'; %Crystal structure of HBV capsid protein
% pdb_fie = '6VJT; %Co-crystals of broadly neutralizing antibody with the linear epitope from Hepatitis B surface antigen

%pdb_file='6CWT'; % Hepatitis B core-antigen in complex with Fab e21
%pdb_file='3IYW';  % WNV in complex with mAb CR4354
%pdb_file='1ZTX'; % WNV E-DIII with E16 Ab Fab
%pdb_file='3IXX';  % West Nile immature virus in complex with Fab fragments of the anti-fusion loop antibody E53
%pdb_file='3I50';
%pdb_file='4MWF'; %Hepatitis C Virus Envelope Glycoprotein E2 core bound to broadly neutralizing antibody AR3C
%pdb_file='4WEB'; %Hepatitis C Virus Envelope Glycoprotein E2 core bound to non neutralizing antibody 2A12
%pdb_file='3OPZ'; %complex of mouse mAb 13G9 and Trans-sialidase enzyme from T. cruzi
%pdb_file='1MS3'; % crystal structure of Trans-sialidase enzyme from T. cruzi

pdb_chain='A';%this signifies which chain of the pdf file you want to project onto
numproject=10;%this is the number of best peptides to identify in the structure
spin_struct=false(1); % this will enable pdb structure to rotate 
un_normalized_projection =false(1); % turn it true if you want to project predicted peptides from the proteome on the structure directly

aminos='ADEFGHKLNPQRSVWY';
Logfit=true(1);
trunc=false(1); % turn it to true if the pdb structure has any truncated portion

%TEST CODE*****************************************
%If you set TEST_CODE to true, it will run a test and print out
TEST_CODE=false(1);
if TEST_CODE
    num_scramble=5; %for this, we will keep if from scrambling but make sure it averages
end
fclose('all');

% Robayet partly wrote these lines of code
%******************************************************
if ~read_singleSample
    for type_dataset=1:2
        disp('Select the parameter file associated with the set of runs you want to use')
        [Wfile,pathname]=uigetfile('*.txt','MultiSelect','off'); %gives the user a gui to open the file
        param=readtable([pathname,Wfile]);
        param=table2cell(param);
        aminoEigen=str2double(param(find(strcmp(param,'aminoEigen')),2));
        dataShift=str2double(param(find(strcmp(param,'dataShift')),2));
        HiddenLayer=str2double(param(find(strcmp(param,'hiddenLayers')),2));
        aminoEigen=str2double(param(strcmp(param,'aminoEigen'),2));
        % extract dataShift or noise from the parameter
       dataShift= param(find(strcmp(param,'dataShift')),2);
       if strcmp(dataShift{1},cellstr('True'))
          dataShift= true(1);
       elseif strcmp(dataShift{1},cellstr('False'))
           dataShift = false(1); 
       else
           dataShift=str2double(dataShift);
       end
       if type_dataset==1
            HiddenLayer_case=HiddenLayer;
            aminoEigen_case = aminoEigen;
            dataShift_case = dataShift;
            fprintf('AminoEigen used in the NN program is %.f\n',aminoEigen_case)
            fprintf('No of Hidden layers is %d\n',HiddenLayer_case)
       else
            HiddenLayer_control=HiddenLayer;
            aminoEigen_control = aminoEigen;
            dataShift_control = dataShift;
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
       disp('reading weights of all samples')
       for isample=1:numsamples
           path=[pathname,foldername,num2str(isample),'\'];
           W(isample)=read_weights_w_bias(path);
       end
       if type_dataset==1
          W_case = W;
          numsample_case = numsamples;
       else
          W_control = W;
          numsample_control = numsamples;
       end
       %Generate an array of sequences to test
       % read data
       fprintf('reading fit data\n');
       [M,K]=size(W(1).W1);
       [RR,H]=size(W(1).W2); % it calculates the length of the peptides from the proteome
       fitR=RR/K;
       % *******Robayet's code ends here.***************
       %read in the fastA protein or proteome.  You can download these files very
       %simply from uniprot.org.  .

       if ~TEST_CODE
           pdbstruct = getpdb(pdb_file);
           %pdbstruct = getpdb(pdb_file,'ToFile','DENV.pdb');
           [~,num_chain]=size(pdbstruct.Sequence);
           Header=cell(num_chain,1);
           Prot_name=pdbstruct.Compound;
           Prot_Sequence=cell(num_chain,1);
           first_aa=zeros(num_chain,1);
           for i=1:num_chain
                 Header(i)=cellstr(pdbstruct.Sequence(i).ChainID);
                 Prot_Sequence(i)=cellstr(pdbstruct.Sequence(i).Sequence);
                 %first_aa(i)=pdbstruct.DBReferences(i).dbseqBegin;
                 first_aa(i)=pdbstruct.DBReferences(i).seqBegin;
           end
    
           chain_num = find(pdb_chain==(char(Header)));
           %chain_num=pdb_chain-'a'+1;
           Proteins_first_aa=first_aa;
           first_aa=first_aa(chain_num);
    
    
           %convert the protein or proteome to a set of sequences.  The index is a series of
           %increasing values, one for each peptide, such that all peptides from
           %the vcvcb
           %first protein have a '1' in the index, all the peptides for the second
           %have a'2', etc. so you can immediately reference back to the protein in
           %the proteome that the peptide came from.  Note that it needs to set this
           %up as peptides that are the right length for your analysis so it uses the
           %length you used in doing your analysis (R).
           [sequence,seqindex]=prot2peparray(Prot_Sequence(chain_num),R_prot);
       end

       %***************************************************
       [num_seq,~]=size(sequence);
       seq=char(zeros(num_seq,fitR));
       seq(:,1:R_prot)=sequence;
       sequence=seq;
       [N,R]=size(sequence);
       Ntot=N;
       %substitute equivalent amino acids %read in fitting data (I=V, T=S,
       %M=L, C is ignored and not given binding significance).  You have to put
       %something in for these amino acids in a real sequence.  These
       %substitutions are well supported by similarity matrices except for C to A,
       %which shows only weak similarity
       real_sequence=sequence; %store the real sequence
       sequence(sequence=='I')='V';
       sequence(sequence=='T')='S';
       sequence(sequence=='M')='L';
       sequence(sequence=='C')='A';

       seqtot=sequence;
       %project the fits onto a protein
       fprintf('number of peptides used in fit = %d\n',N);
       disp('projecting onto a protein');
  
       F_calc=zeros(N,numsamples);
       parfor isample=1:numsamples
            F_calc(:,isample)=MLevaluate_N_layer_w_bias(sequence,aminos,W(isample));
       end
       delete(gcp('nocreate')) % shut down parallel pool

       if Logfit
           F_calc =10.^(F_calc);
       end
 
        
       if type_dataset==1
           F_calc_case = F_calc;
       else
           F_calc_control = F_calc;
       end
       F_calc = [];
    end
else
    disp('Select the parameter file associated with the set of runs you want to use')
    [Wfile,pathname]=uigetfile('*.txt','MultiSelect','off'); %gives the user a gui to open the file
    param=readtable([pathname,Wfile]);
    param=table2cell(param);
    aminoEigen=str2double(param(find(strcmp(param,'aminoEigen')),2));
    dataShift=str2double(param(find(strcmp(param,'dataShift')),2));
    HiddenLayer=str2double(param(find(strcmp(param,'hiddenLayers')),2));
    fprintf('AminoEigen used in the NN program is %.f\n',aminoEigen)
    fprintf('No of Hidden layers is %d\n',HiddenLayer)
    aminoEigen=str2double(param(find(strcmp(param,'aminoEigen')),2));
    % extract dataShift or noise from the parameter
    dataShift= param(find(strcmp(param,'dataShift')),2);
    if strcmp(dataShift{1},cellstr('True'))
        dataShift= true(1);
    elseif strcmp(dataShift{1},cellstr('False'))
        dataShift = false(1); 
    else
        dataShift=str2double(dataShift);
    end
    fprintf('AminoEigen used in the NN program is %.f\n',aminoEigen)
    fprintf('No of Hidden layers is %d\n',HiddenLayer)

    % print the filename that was used in the NN program
    filename_index=find(pathname=='\');
    filename=pathname(filename_index(8)+1:filename_index(9)-1);
    fprintf('reading folder %s\n',filename);

    foldername='Sample';
    numsamples=1;
    disp('Reading files');
    %read the fit matrices
    fprintf('reding weights of sample no %d\n',sampleNo)
    path=[pathname,foldername,num2str(sampleNo),'\'];
    W=read_weights_w_bias(path);
   %Generate an array of sequences to test
   % read data
   fprintf('reading fit data\n');
   [M,K]=size(W(1).W1);
   [RR,H]=size(W(1).W2); % it calculates the length of the peptides from the proteome
   fitR=RR/K;
   % *******Robayet's code ends here.***************
   %read in the fastA protein or proteome.  You can download these files very
   %simply from uniprot.org.  If you go to a proteome, you can just look for a
   %specific protein, download a whole proteome or select parts.  Note that if
   %you download the whole human proteome and try to calculate C we will run
   %out of memory.  So we will have to break the large proteomes into pieces.

   if ~TEST_CODE
       pdbstruct = getpdb(pdb_file);
       %pdbstruct = getpdb(pdb_file,'ToFile','DENV.pdb');
    
       [~,num_chain]=size(pdbstruct.Sequence);
       Header=cell(num_chain,1);
       Prot_name=pdbstruct.Compound;
       Prot_Sequence=cell(num_chain,1);
       first_aa=zeros(num_chain,1);
       for i=1:num_chain
           Header(i)=cellstr(pdbstruct.Sequence(i).ChainID);
           Prot_Sequence(i)=cellstr(pdbstruct.Sequence(i).Sequence);
           %first_aa(i)=pdbstruct.DBReferences(i).dbseqBegin;
           first_aa(i)=pdbstruct.DBReferences(i).seqBegin;
       end
    
       chain_num = find(pdb_chain==(char(Header)));
       %chain_num=pdb_chain-'a'+1;
       Proteins_first_aa=first_aa;
       first_aa=first_aa(chain_num);
    
    
       %convert the protein or proteome to a set of sequences.  The index is a series of
       %increasing values, one for each peptide, such that all peptides from the
       %first protein have a '1' in the index, all the peptides for the second
       %have a'2', etc. so you can immediately reference back to the protein in
       %the proteome that the peptide came from.  Note that it needs to set this
       %up as peptides that are the right length for your analysis so it uses the
       %length you used in doing your analysis (R).
       [sequence,seqindex]=prot2peparray(Prot_Sequence(chain_num),R_prot);
   end
       %***************************************************


       [num_seq,~]=size(sequence);
       seq=char(zeros(num_seq,fitR));
       seq(:,1:R_prot)=sequence;
       sequence=seq;
       [N,R]=size(sequence);
       Ntot=N;
       %substitute equivalent amino acids %read in fitting data (I=V, T=S, 
       %M=L, C is ignored and not given binding significance).  You have to put
       %something in for these amino acids in a real sequence.  These
       %substitutions are well supported by similarity matrices except for C to A,
       %which shows only weak similarity
       real_sequence=sequence; %store the real sequence
       sequence(sequence=='I')='V';
       sequence(sequence=='T')='S';
       sequence(sequence=='M')='L';
       sequence(sequence=='C')='A';

       seqtot=sequence;
       %project the fits onto a protein
       fprintf('number of peptides used in fit = %d\n',N);
       disp('projecting onto a protein');
       Z_calc=MLevaluate_N_layer_w_bias(sequence,aminos,W);
end
    

% calculate Z scores from two chorts
if ~read_singleSample
    Z_calc=(mean(F_calc_case,2)-mean(F_calc_control,2))./(sqrt(std(F_calc_case,0,2).^2+std(F_calc_control,0,2).^2));
end


% Robayet wrote these lines of code
%****************************************
N=Ntot;
%Now scramble sequences and calculate Z scores
disp('performing the scrambling')
if num_scramble>0
    num_sections=ceil(Ntot/Nsize);
    if ~read_singleSample
        F_calc_sections_case=zeros(Nsize,num_sections);
        F_calc_sections_case_std=zeros(Nsize,num_sections); 
        F_calc_sections_control=zeros(Nsize,num_sections);
        F_calc_sections_control_std=zeros(Nsize,num_sections); 
    else
        F_calc_sections = zeros(Nsize,num_sections);
    end
    for nn=1:num_sections
           n1=(nn-1)*Nsize+1;
           
        if Ntot-n1>Nsize
            N=Nsize;
        else
            N=Ntot-n1+1;
        end
        
        seq_scr=seqtot(n1:n1+N-1,:);
        
        if ~read_singleSample
                                   
            for s=1:num_scramble
                if ~TEST_CODE %for the test we would not scramble
                    seq_scr(:,1:R_prot)=seq_scr(:,randperm(R_prot,R_prot)); %randomize order
                end
                
                for type_dataset = 1:2
                    % now calculate the value for samples in each cohort from using the machine learning equation
                     if type_dataset==1
                         numsamples = numsample_case;
                         W = W_case;
                     else
                         numsamples = numsample_control;
                         W = W_control;
                     end
                     %now calculate the value from using the machine learning equation
                     f_calc_sample=zeros(length(seq_scr),numsamples);
                     parfor isample=1:numsamples
                            f_calc_sample(:,isample)=MLevaluate_N_layer_w_bias(seq_scr,aminos,W(isample));
                     end
                     f_calc=mean(f_calc_sample,2);
                     f_calc_std=std(f_calc_sample,0,2);

                     if Logfit%generally we want to do subsequent manipulations in linear space
                     f_calc=10.^f_calc;
                     end
            
                    if length(f_calc)<Nsize 
                       tem=f_calc;
                       f_calc=zeros(Nsize,1);
                       f_calc(1:length(tem))=tem;
                       clear tem % clear tem to free up memory 
                       tem=f_calc_std;
                       f_calc_std=zeros(Nsize,1);
                       f_calc_std(1:length(tem))=tem;
                       clear tem % clear tem to free up memory 
                    end
            
                    if type_dataset==1
                        F_calc_sections_case(:,nn)= F_calc_sections_case(:,nn)+ f_calc;
                        F_calc_sections_case_std(:,nn)= F_calc_sections_case_std(:,nn)+ f_calc_std;
                    else
                        F_calc_sections_control(:,nn)= F_calc_sections_control(:,nn)+ f_calc;
                        F_calc_sections_control_std(:,nn)= F_calc_sections_control_std(:,nn)+ f_calc_std;
                    end
                    clear f_calc
                    clear f_calc_std
                end
            end
            F_calc_sections_case(:,nn) = F_calc_sections_case(:,nn)/num_scramble;
            F_calc_sections_case_std(:,nn) = F_calc_sections_case_std(:,nn)/num_scramble;
            F_calc_sections_control(:,nn) = F_calc_sections_control(:,nn)/num_scramble;
            F_calc_sections_control_std(:,nn) = F_calc_sections_control_std(:,nn)/num_scramble;         

        else
             for s=1:num_scramble
                 if ~TEST_CODE %for the test we would not scramble
                    seq_scr(:,1:R_prot)=seq_scr(:,randperm(R_prot,R_prot)); %randomize order
                 end
            %now calculate the value from using the machine learning equation
                 f_calc=MLevaluate_N_layer_w_bias(seq_scr,aminos,W);
                 if Logfit%generally we want to do subsequent manipulations in linear space
                    f_calc=10.^f_calc;
                 end
           
                if length(f_calc)<Nsize 
                    tem=f_calc;
                    f_calc=zeros(Nsize,1);
                    f_calc(1:length(tem))=tem;
                    clear tem % clear tem to free up memory 
                end
            
                F_calc_sections(:,nn)= F_calc_sections(:,nn)+ f_calc;
                clear f_calc
             end
             F_calc_sections(:,nn) = F_calc_sections(:,nn)/num_scramble;
        end   
    end
        % now normalize Z scores
    if ~read_singleSample
        % case and control
        F_calc_sections_case = F_calc_sections_case(1:Ntot);
        F_calc_sections_case_std = F_calc_sections_case_std(1:Ntot);
        F_calc_sections_control = F_calc_sections_control(1:Ntot);
        F_calc_sections_control_std = F_calc_sections_control_std(1:Ntot);
        % calculate Z-scores for peptides between two cohorts
        Z_scramble = (F_calc_sections_case - F_calc_sections_control)./(sqrt(F_calc_sections_case_std.^2+ F_calc_sections_control_std.^2));
    else
        Z_scramble = F_calc_sections(1:Ntot);
    end
    Z_calc_norm=Z_calc./Z_scramble;%normalize to the scrambled value
    
else
    Z_calc_norm=Z_calc;
end

%save the sequences and binding data
real_sequence_save = real_sequence;
Z_calc_save = Z_calc;
Z_calc_norm_save=Z_calc_norm;

%****************************************

if trunc
    real_sequence=real_sequence_save(15:end,:);
    Z_calc=Z_calc_save(15:end);
    Z_calc_norm= Z_calc_norm_save(15:end);
end

%map the raw protein signal

%print out the best binders
[Z_calc_sort,Zindex2]=sort(Z_calc,'descend');
top_seq=real_sequence(Zindex2,:);
top_index=seqindex(Zindex2);

if first_aa>1
    Zindex2=Zindex2+first_aa-1;
end

fprintf('\n*******The highest differential binding peptides from the protein*******\n');
for i=1:10
    %fprintf('%s %d  %4.3f  %s\n',top_seq(i,:),Findex2(i),Fcalc_sort(i),Header{top_index(i)});
    %fprintf('%s %d  %4.3f  %s\n',top_seq(i,:),Zindex2(i),Z_calc_sort(i),char(Header(chain_num)));
     fprintf('%s %d  %4.3f  \n',top_seq(i,:),Zindex2(i),Z_calc_sort(i));
end

figure(1);
if first_aa>1
    plot(sort(Zindex2),Z_calc)
else
    plot(Z_calc_save);
end
    ylabel('Calculated Z scores');
    xlabel('Position in sequence');
    title([pdb_file,':',Plot_Title,' (Unnormalized)']);
    drawnow;


%map the normalized proteome signal
[Z_calc_norm_sort,Zindex3]=sort(Z_calc_norm,'descend');
top_seq_norm=real_sequence(Zindex3,:);
top_index_norm=seqindex(Zindex3);

if first_aa>1
    Zindex3=Zindex3+first_aa-1;
end
%print out the best binders
fprintf('\n*******The highest normalized binding peptides from the protein*******\n');
for i=1:10
    %fprintf('%s %d  %4.3f  %s\n',top_seq_norm(i,:),Findex3(i),Fcalc_sort_norm(i),Header{top_index_norm(i)});
    fprintf('%s %d  %4.3f  %s\n',top_seq_norm(i,:),Zindex3(i),Z_calc_norm_sort(i),char(Header(chain_num)));
end

figure(2);
if first_aa>1
    plot(sort(Zindex3),Z_calc_norm)
else
    plot(Z_calc_norm_save);
end
    ylabel('Calculated Z scores');
    xlabel('Position in sequence');
    title([pdb_file,':',Plot_Title,' (Normalized)',]);
    drawnow;
    
    



%colors for structure plots % antigen= red and green, heavy chain = blue,
%light chain = gray
mycolor=cell(num_chain+numproject,1);
mycolor{1}='red';
%mycolor{1}='blue';
%mycolor{2}='Gray';
%mycolor{2}='blue';
mycolor{2}='green';
%mycolor{2}='red';
%mycolor{2}='HotPink';
%mycolor{3}='yellow';
%mycolor{3}='gray';
mycolor{3}='purple';
mycolor{4}='green';
%mycolor{4}='yellow';
mycolor{5}='yellow';
%mycolor{5}='magenta';
%mycolor{5}='green';
%mycolor{6}='blue';
%mycolor{6}='HotPink';
%mycolor{6}='Orchid';
%mycolor{6}='HotPink';
mycolor{6}='green';
mycolor{7}='red';
%mycolor{7}='Orange';
%mycolor{8}='greenblue';
%mycolor{8}='DarkGray';
mycolor{8}='green';
mycolor{9}='gray';
%mycolor{9}='SkyBlue';
%mycolor{10}='purple';
mycolor{10}='Gray';
%mycolor{11}='Blue';
mycolor{11}='GreenYellow';
%mycolor{12}='DarkGoldenrod';
mycolor{12}='Gray';
mycolor{13}='Cyan';
mycolor{13}='Turquoise';
mycolor{14}='Gold';
mycolor{15}='Aqua';
mycolor{16}='Teal';



for k=num_chain+1:length(mycolor)
    mycolor{k}='white';
end

if ~TEST_CODE
%generate the structure with the region highlighted by unnormalized
%peptides
disp('mapping top peptides (unnormalized) on the protein structure')
mystruct = getpdb(pdb_file);
h1=molviewer(mystruct);

for i=1:num_chain %make each chain a different color
    evalrasmolscript(h1,['select :',lower(Header{i})]);
    evalrasmolscript(h1,['color ',mycolor{i}]);
    X=sprintf('Color of chain %s : %s',Header{i},mycolor{i});
    disp(X)
end

% now rotate the structure around Y axis
if spin_struct
    h1.HandleVisibility='on';
    evalrasmolscript(h1,'background black; spin')
end


%now highlight each of the top picked peptides before normalization
%struct_res_unnorm=Findex2+first_aa-1; %often, the structure is a partial structure and does not start at 1
    

struct_res_unnorm=Zindex2;

for j=1:numproject
    if strcmp(cellstr(pdb_file),cellstr('3J35'))
        select_com =['select ',num2str(struct_res_unnorm(j)+502),'-',num2str(struct_res_unnorm(j)+502+R_prot-1),':',pdb_chain];
    else
        select_com =['select ',num2str(struct_res_unnorm(j)),'-',num2str(struct_res_unnorm(j)+R_prot-1),':',pdb_chain];
    end
    color_com=['color ',mycolor{j+i}];
    evalrasmolscript(h1,select_com);
    evalrasmolscript(h1,color_com);
    disp(mycolor{j+i})
end


if num_scramble>0
% generate the structure with the region highlighted by normalized
% (binding)peptides
disp('mapping top peptides (normalized by average binding after scrambling) on the protein structure')
h2=molviewer(mystruct);

for i=1:num_chain %make each chain a different color
    evalrasmolscript(h2,['select :',lower(Header{i})]);
    evalrasmolscript(h2,['color ',mycolor{i}]);
    X=sprintf('Color of chain %s : %s',Header{i},mycolor{i});
    disp(X)
end

% now rotate the structure around Y axis
if spin_struct
    h2.HandleVisibility='on';
    evalrasmolscript(h2,'background black; spin')
end

%now highlight each of the top picked peptides before normalization
%struct_res_unnorm=Findex2+first_aa-1; %often, the structure is a partial structure and does not start at 1
struct_res_norm=Zindex3;

for j=1:numproject
    select_com=['select ',num2str(struct_res_norm(j)),'-',num2str(struct_res_norm(j)+R_prot-1),':',pdb_chain];
    color_com=['color ',mycolor{j+i}];
    evalrasmolscript(h2,select_com);
    evalrasmolscript(h2,color_com);
    %disp(mycolor{j+i})
end
end

else %print scatter plots for test
disp('Do not make Test_code to true') 
end

%%change the title of the structure
disp('modifying title for structure')
h1_title= cat(2,h1.Name,' (Unnormalized)'); 
h1.Name=h1_title;

if num_scramble>0
    h2_title= cat(2,h2.Name,' (Normalized)'); 
    h2.Name=h2_title;
end


toc
