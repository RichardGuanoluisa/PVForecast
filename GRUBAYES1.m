%% Bayesian optimization for GRU hyperparameters
% Gated Recurrent Unit
% Richard Guanoluisa
clear all; clc;
%% Load dataset
% Dir archivo xlsx
% filename = 'C:\Users\Richard\Documents\DOCS RICHARD\docs\Tesis\Datos\1_BALTRA.xlsx';
% Columna Potencia de salida de modulo PV segú modelo del panel 2018 - 2020
PV = xlsread(filename,'O11:O26314');                    % Arreglo de PV
datey=xlsread(filename,'Q11:Q26314');
datem=xlsread(filename,'R11:R26314');
dated=xlsread(filename,'S11:S26314');
dateh=xlsread(filename,'T11:T26314');
datemin=xlsread(filename,'U11:U26314');
dates=xlsread(filename,'V11:V26314');
Xdate=datetime(datey,datem,dated,dateh,datemin,dates);  % Arreglo de t-series
data=PV';
%% Processing data
% Partition the training and test data 80% 20%
trec=numel(data);                       % numel -> Num elementos en array
trrec=0.8*trec;                         % valor 80% de elementos del array
NTST=floor(trrec);                      % redondea el valor
dataTrain=data(1:NTST+1);               % obtener vector hasta el NTST (80% datos -> entrenamiento)
dataTest=data(NTST+1:end);              % obtener vector desde el NTST (20% datos -> test)
% Normalize data
valminDataTrain=min(dataTrain);         % Valor mínimo de dataTrain
valmaxDataTrain=max(dataTrain);         % Valor máximo de dataTrain
dataTrainNorm=(dataTrain-valminDataTrain)/(valmaxDataTrain-valminDataTrain);        % Normalización de datosTrain
dataTestNorm=(dataTest-valminDataTrain)/(valmaxDataTrain-valminDataTrain);          % Normalización de datosTest
%% Hyperparameters to optimize
optimVars=[
    optimizableVariable('NoHU',[125 275],'Type','integer')                          % Number of Hidden Units
    optimizableVariable('MaxEpochs',[150 225],'Type','integer')                     % Max epochs
    optimizableVariable('learnrate',[1e-4 1e-2],'Type','real',"Transform","log")    % Learning rate
    ];
%% Objective function
ObjFcnCR = makeObjFcnCR(dataTrainNorm,dataTestNorm);
%% Call optimizer
BayesObject = bayesopt(ObjFcnCR,optimVars, ...
    'MaxTime',10*60*60, ...
    'Verbose',0,...
    'IsObjectiveDeterministic',false, ...
    'MaxObjectiveEvaluations',15,...
    'AcquisitionFunctionName', 'expected-improvement-per-second-plus');
% Get the best point of the Bayesian Optimizaion
optVars = bestPoint(BayesObject)
%%
function ObjFcn = makeObjFcnCR(dataTrainNorm,dataTestNorm)
ObjFcn = @valErrorFunCR;
    function [valError,emp, net] = valErrorFunCR(optVars)
        emp = [];
        cv=cvpartition(numel(dataTrainNorm),'HoldOut',0.2);
        dataTrain=dataTrainNorm(cv.training);
        dataValida=dataTrainNorm(cv.test);
        XTrain=dataTrain(1:end-1);           % Vector datosTrainStd desplazado en 1
        YTrain=dataTrain(2:end);             % Vector datosTrainStd desplazado en 2
        XVal=dataValida(1:end-1);
        YVal=dataValida(2:end);
        XVal=num2cell(XVal);
        YVal=num2cell(YVal);
        NoF = 1;
        NoR = 1;
        % Define GRU Network Architecture
        layers = [...
            sequenceInputLayer(NoF,'Name',"SEQUENCE INPUT ")
            gruLayer(optVars.NoHU,'Name',"NHU")
            fullyConnectedLayer(NoR,'Name',"FULLY CONNECTED LAYER ")
            regressionLayer('Name','REGRESSION LAYER')
            ];
        % Specify Options for training deep learning neural network
        options = trainingOptions('adam', ...
            'MaxEpochs',optVars.MaxEpochs, ...
            'GradientThreshold',1, ...
            'InitialLearnRate',optVars.learnrate, ...
            'LearnRateSchedule',"piecewise", ...
            'LearnRateDropPeriod',32, ...
            'LearnRateDropFactor',0.8, ...
            'Verbose',false, ...
            'ValidationData',{XVal,YVal},...
            'Shuffle',"every-epoch");
        rng(0);
        net=trainNetwork(XTrain,YTrain,layers,options);
        XTest=dataTestNorm(1:end-1);             % Vector dataTestStd desplazado en 1
        YTest=dataTestNorm(2:end);               % Vector dataTestStd desplazado en 2
        Ys = predict(net,XTest);                 %
        e=YTest-Ys;                              % Error
        valError = mae(e);                       % Performance evaluation
    end
end