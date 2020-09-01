clear all
clc

% You have to keep the two files in a same folder
disp('loading Prediction file')
[Projection,path1]=uigetfile('*','Multiselect','off');
fprintf('reading Prediction file %s\n',Projection);


Projection_data=readtable([path1,Projection]);
Prot_seq=Projection_data.Sequence;
Z=Projection_data.Zscore;
[Zsort,ind]=sort(Z,'descend');
top_seq=Prot_seq(ind);
top_100=top_seq(1:30);
%% load IEDB data
disp('loading IEDB file')
[IEDB,path2]=uigetfile('*','Multiselect','off');
fprintf('reading IEDB file %s\n',IEDB);
IEDB_data=readtable([path2,IEDB]);
% IEDB_data=table2cell(IEDB_data);
[matched_pep_rank,matched_epitope,matched_Antigen,matched_pep]=find_sequence(top_100,IEDB_data);


% Now create a list of predicted peptides with the matched epitope or
% % epitopes
% num_pep=length(top_100);
% num_epitope=length(epitope_seq);
% Pred_epit=cell(num_pep,num_epitope);
% i=1;
% while i<=num_pep
%     epitope_indx=pep_logical(i,:);
%     find_match=find(epitope_indx);
%     Pred_epit(i,~epitope_indx)=cellstr('No match');
% 
%     for j=1:length(find_match)
%         Pred_epit(i,find_match(j))=epitope_seq(find_match(j));
%     end
%     i=i+1;
% end

    
 
 
