%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         task
% Description:  Run different tasks such as classification, regression, etc
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      July 02, 2020
% Updated:      December 02, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2020 Luong-Ha Nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef task
    methods (Static)                
        % LSTM
        function [ml, Sl, theta, Mem, mlPri, SlPri, Mem0_infer, Sq_infer, zOsm, SzOsm, rnnMemory] = runLSTM(net, lstm, x, y)
            [net, states, maxIdx, netInfo] = network.initialization(net);
            normStat = tagi.createInitNormStat(net);
            theta         = lstm.theta;
            % sv
            if isfield(lstm, 'sv')
                net.sv = lstm.sv;
            end
            % sequence length
            if ~isfield(lstm, 'Sq')
                Sq = [];
            else
                Sq = lstm.Sq;
            end

            if ~isfield(lstm, 'Mem')
                Mem = [];
            else
                Mem = lstm.Mem;
            end
            if isempty(theta)
                theta    = tagi.initializeWeightBias(net); % Initalize weights and bias at only 1st epoch
            end
            % Running
            [yPd, SyPd, lstm, yPd_pri, SyPd_pri, rnnSmooth] = network.LSTM(net, theta, normStat, states, maxIdx, Mem, x, y, Sq);

            % Smoothing for LSTM
            % rnnSmooth: contains all quantities needed to do smoothing in
            % LSTM: i.e., prior, posterior, and covariances for h, c, z^{O}
            if net.LSTMsmoothing == 1  % Do smoothing in LSTM
                if net.trainMode == 1
                    [Mem0_infer, zOsm, SzOsm, Sq_infer, rnnMemory] = tagi.rnnSmoother(net, rnnSmooth);   % smoother initial values of cell states c_0 and hidden states h0
                end
            else                        % Do not do smoothing
                Mem0_infer = Mem;
            end

            ml    = yPd;
            Sl    = SyPd + net.sv^2;
            mlPri = yPd_pri;
            SlPri = SyPd_pri + net.sv^2;
            theta = lstm.theta;
            Mem = lstm.Mem;
        end

        function [yPd, SyPd, theta, Mem, xBu, SxBu, xBp, SxBp, zl, Szl, Czz] = runHydrid_AGVI(net, lstm, bdlm, x, y)
            % exponential smoothing (ESM)
            [net, states, maxIdx, netInfo] = network.initialization(net);
            normStat = tagi.createInitNormStat(net);
            theta         = lstm.theta;
            if isfield(lstm, 'sv')
                net.sv = lstm.sv;
            end
            if ~isfield(lstm, 'Sq')
                Sq = [];
            else
                Sq = lstm.Sq;
            end
            if ~isfield(lstm, 'Mem')
                Mem = [];
            else
                Mem = lstm.Mem;
            end
            if isempty(Mem)
                [Mem] = rnn.initializeRnnMemory (net.layer, net.nodes, net.batchSize, 0);
            end
            if isempty(theta)
                theta    = tagi.initializeWeightBias(net); % Initalize weights and bias at only 1st epoch
            end
            % Running
            [yPd, SyPd, zl, Szl, xBp, SxBp, xBu, SxBu, lstm, Czz] = network.hydrid_AGVI(net, theta, normStat, states, maxIdx, Mem, x, y, Sq, bdlm);
            theta = lstm.theta;
            Mem = lstm.Mem;
        end
    end
end