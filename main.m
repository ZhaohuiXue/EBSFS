%% Setup
clear;clc;close all;

%% Load test data for regression from Matlab R2022a
load accidents % 
X = hwydata; 
X(:,4) = []; %Use all other 16 variables as the input feature (more details can be found in hwyheaders)
Y = hwydata(:,4); %Use traffic fatalities as the output response

%   X: Input data of size n x d
%   Y: Output/target/observation of size n x do
%   n: number of samples/examples/patterns (in rows)
%   d: input data dimensionality/features (in columns)
%   do: output data dimensionality (variables, observations).
%% Main loop
%% parameters

opts = [];
opts.kernel_type = 'Cityblock_knn';% 'RBF_full','RBF_knn','Cosine_full','CR_full','CR_knn','CRT_full'，'CRT_knn','Cityblock_knn'
opts.knn_size = 5;
opts.CR_lambda = [];
opts.GRNN_spread = [];
opts.str_distance = 'cityblock';

fun = @(XT,yT,Xt,yt)mycritfun(XT,yT,Xt,yt,opts);
%%
ccc = cvpartition(Y,'KFold',10);

%% EBSFS训练
opts_2 = statset('display','iter');
inmodel = sequentialfs(fun,X,Y,...
    'cv',ccc,...
    'MCReps',1,...%Monte-Carlo repetitions for cross-validation default 1
    'Direction','backward',...%default forward;backward
    'options',opts_2);

ranking = find(inmodel==1);
imp=[];
[b,idx]=sort(ranking);





