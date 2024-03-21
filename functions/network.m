%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         network
% Description:  Build networks relating to each task
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      July 02, 2020
% Updated:      December 02, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2020 Luong-Ha Nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef network
    methods (Static)      
        % LSTM
        function [zl, Szl, lstm, zlpri, Szlpri,rnnSmooth] = LSTM(net, theta, normStat, states, maxIdx, Mem, x, y, Sq)
            sql       = net.sql;
            batchSize = net.batchSize;

            % Sequence length
            if isempty(Sq)
                ySq   = y(1:sql,:);
%                 ySq   = zeros(sql,batchSize);
%                 ySq   = rand(sql,1);
%                 ySq   = ones(sql,batchSize);
%                 SySq  = 1E0.*ones([sql batchSize],'like',x);
                SySq  = 0.*ones(sql,batchSize,'like',x);
                % for revision
%                 y(1:sql,:) = [];
%                 x(1:sql,:,:) = [];
%                 ySq = randn(sql, batchSize,'single').*sqrt(SySq);
            else
                ySq  = Sq{1};
                SySq = Sq{2};
            end
            numObs   = size(y,1);
            nblayer = length(net.layer);
            % Save for smoothing
            Chh    = cell(nblayer,numObs);  % cov(h_t,h_{t-1}): covariance between hiddens states of t and t-1
            Ccc    = cell(nblayer,numObs);  % cov(c_t,c_{t-1}): covariance between cell states of t and t-1
            Cxh    = cell(nblayer,numObs);
            Czz    = nan(numObs,1);         % cov(z^{O}_{t-1},z^{O}_t)
            cPrior  = cell(nblayer,numObs); % Prior of cell states
            ScPrior = cell(nblayer,numObs);
            hPrior  = cell(nblayer,numObs); % Prior of hidden states
            ShPrior = cell(nblayer,numObs);
            cPos  = cell(nblayer,numObs);   % Posterior of cell states
            ScPos = cell(nblayer,numObs);
            hPos  = cell(nblayer,numObs);   % Posterior of hidden states
            ShPos = cell(nblayer,numObs);

            % Prepare input for LSTM
            % xSqloop = [covariates ; sequence length (history of target time series)] 
            if ~isempty(x)
                xSq  = x(1,:,:);
                SxSq = zeros(size(xSq)); 
                xSqloop  = [reshape(permute(xSq,[1,3,2]),[],batchSize);ySq];
                xSqloop(isnan(xSqloop)) = 0;
                SxSqloop  = [reshape(permute(SxSq,[1,3,2]),[],batchSize);SySq];
            else
                xSqloop  = ySq;
                SxSqloop = SySq;
            end
            
            mem0 = Mem;
            %
            numDataPerBatch = net.repBatchSize*net.batchSize;
            zl      = zeros(numObs, net.ny*net.batchSize, 'like',x);
            Szl     = zeros(numObs, net.ny*net.batchSize, 'like',x);
            zlpri   = zl;
            Szlpri  = Szl;

            for i = 1:numObs
                idxBatch = i;
                
                % Prepare output
                ySqloop = y(idxBatch, :)';
                
                % Training and testing
                if strcmp(net.RNNtype, 'LSTM_lookback') w = 1; else w = sql; end
                for k = 1:1:w
                    % Loading memory
                    if strcmp(net.RNNtype, 'LSTM_lookback')
                        xloop   = xSqloop(:);
                        Sxloop  = SxSqloop(:);
                        yloop   = ySqloop(:);
                        mem     = Mem(1:4,1);
                    end
                    states  = tagi.lstmInitializeInputs(states, xloop, Sxloop, [], [], [], [], [], [], [], net.xsc, mem);
                    % Training
                    if net.trainMode == 1
                        [states, normStat, maxIdx] = tagi.feedForwardPass(net, theta, normStat, states, maxIdx);
                        if k == w
                            % Estimate priors
                            [~, ~, ma, Sa]   = tagi.extractStates(states);
                            [~, ~, ~, cPrior(:,i) , ScPrior(:,i), ~, ~] = tagi.lstmExtractStates(states); % extract the prior for cell states
                            hPrior(:,i)  = ma; % extract the prior for hidden states
                            ShPrior(:,i) = Sa;
                            zlpri(idxBatch, :)  = reshape(ma{end,1}, [net.ny, numDataPerBatch])'; % extract the prior for z^{O}
                            Szlpri(idxBatch, :) = reshape(Sa{end,1}, [net.ny, numDataPerBatch])';
                            if any(~isnan(yloop)) % when having observations
                                [deltaM, deltaS, deltaMx, deltaSx, ~, ~, deltaHm, deltaHs, Chh(:,i), Ccc(:,i), Cxh(:,i)] = tagi.hiddenStateBackwardPass(net, theta, normStat, states, yloop, [], [], maxIdx);
                                Czz(i) = tagi.covZZlstm(net, theta, ma, mem, Chh{end-1,i});
                                dtheta = tagi.parameterBackwardPass(net, theta, normStat, states, deltaM, deltaS, deltaMx, deltaSx);
                                theta  = tagi.globalParameterUpdate(theta, dtheta, net.gpu);
                                % Estimate posteriors:
                                % deltaHm and deltaHs: are the delta for means and variances to update the activation
                                % units; they are needed to estimate the
                                % posteriors for states
                                states = tagi.lstmPosterior(states, deltaM, deltaS, deltaHm, deltaHs, states{16}{5});
                                [~, ~, ma, Sa]   = tagi.extractStates(states);
                                [~, ~, ~, cPos(:,i) , ScPos(:,i), ~, ~] = tagi.lstmExtractStates(states); % extract the posterior for cell states
                                hPos(:,i)  = ma; % extract the posterior for hidden states
                                ShPos(:,i) = Sa;
                            else  % when missing data
                                % posteriors = priors
                                [~, ~, ~, ~, ~, ~, ~, ~, Chh(:,i), Ccc(:,i), Cxh(:,i)] = tagi.hiddenStateBackwardPass(net, theta, normStat, states, yloop, [], [], maxIdx);
                                Czz(i) = tagi.covZZlstm(net, theta, ma, mem, Chh{end-1,i});
                                cPos(:,i)  = cPrior(:,i); % cell states: posterior = prior
                                ScPos(:,i) = ScPrior(:,i);
                                hPos(:,i)  = hPrior(:,i); % hidden states: posterior = prior
                                ShPos(:,i) = ShPrior(:,i);
                            end

                            %  Posterior for z^{O}
                            zl(idxBatch, :)  = reshape(ma{end,1}, [net.ny, numDataPerBatch])';
                            Szl(idxBatch, :) = reshape(Sa{end,1}, [net.ny, numDataPerBatch])';

                            % Prepare input for the next time step:
                            if strcmp(net.teacherForcing,'zO')
                                ySq      = cat(1, ySq(2:end,:), zl(idxBatch, :));
                                SySq     = cat(1,SySq(2:end,:), Szl(idxBatch, :));
                            elseif strcmp(net.teacherForcing,'obs')
                                ySq      = cat(1, ySq(2:end,:), y(idxBatch, :));
                            end
                            if i<numObs
                                if ~isempty(x)
                                    xSq      = cat(1, xSq(2:end,:,:), x(i+1,:,:));
                                    xSqloop  = [reshape(permute(xSq,[1,3,2]),[],batchSize);ySq];
                                    SxSqloop = [reshape(permute(SxSq,[1,3,2]),[],batchSize);SySq];
                                else
                                    xSqloop  = ySq;
                                    SxSqloop = SySq;
                                end
                            end
                            xSqloop(isnan(xSqloop)) = 0;
                            %
                        end


                    % Testing: not used for smoothing
                    else
                        [states, normStat, maxIdx] = tagi.feedForwardPass(net, theta, normStat, states, maxIdx);
                        if k == w
                            [~, ~, ma, Sa]   = tagi.extractStates(states);
                            if any(~isnan(yloop))
                                [deltaM, deltaS, ~, ~, ~, ~, deltaHm, deltaHs] = tagi.hiddenStateBackwardPass(net, theta, normStat, states, yloop, [], [], maxIdx);
                                states = tagi.lstmPosterior(states, deltaM, deltaS, deltaHm, deltaHs, states{16}{5});
                                [~, ~, ma, Sa]   = tagi.extractStates(states);
                            end
                            zl(idxBatch, :)  = reshape(ma{end,1}, [net.ny, numDataPerBatch])';
                            Szl(idxBatch, :) = reshape(Sa{end,1}, [net.ny, numDataPerBatch])';

                            % Prepare input for the next time step:
                            ySq      = cat(1, ySq(2:end,:), zl(idxBatch, :));
                            SySq     = cat(1,SySq(2:end,:), Szl(idxBatch, :));
                            if i<numObs
                                if ~isempty(x)
                                    xSq      = cat(1, xSq(2:end,:,:), x(i+1,:,:));
                                    xSqloop  = [reshape(permute(xSq,[1,3,2]),[],batchSize);ySq];
                                    SxSqloop = [reshape(permute(SxSq,[1,3,2]),[],batchSize);SySq];
                                else
                                    xSqloop  = ySq;
                                    SxSqloop = SySq;
                                end
                            end
                            xSqloop(isnan(xSqloop)) = 0;
                            SxSqloop(isnan(SxSqloop)) = 0;
                            %
                        end
                    end
                    % Update lstm memory after each timestamp
                    Mem = tagi.updateRnnMemory(net.RNNtype, states);
                end
                if i == 1 % save the input at t=1 for smoothing the sequence
                    x0  = xloop;
                    Sx0 = Sxloop;
                end
            end
            % update lstm memory after each epoch
            lstm.theta = theta;
            lstm.Mem   = Mem;
            
            % save for smoother
            if net.trainMode == 1
                % for cell states
                rnnSmooth.cPrior  = [mem0{3},cPrior];
                rnnSmooth.ScPrior = [mem0{4},ScPrior];
                rnnSmooth.cPos    = [mem0{3},cPos];
                rnnSmooth.ScPos   = [mem0{4},ScPos];
                % for hidden states
                rnnSmooth.hPrior  = [mem0{1},hPrior];
                rnnSmooth.ShPrior = [mem0{2},ShPrior];
                rnnSmooth.hPos    = [mem0{1},hPos];
                rnnSmooth.ShPos   = [mem0{2},ShPos];
                % for z^{O}
                rnnSmooth.zOPos  = zl;
                rnnSmooth.SzOPos = Szl;
                rnnSmooth.zOPrior   = zlpri;
                rnnSmooth.SzOPrior  = Szlpri;
                % for initial LSTM's memory
                rnnSmooth.initMem  = mem0;
                % for covariances cov(h_t,h_{t-1}),  cov(c_t,c_{t-1})
                % cov(x_t,h^{1st LSTM layer}_{t})
                rnnSmooth.Chh = [Chh(:,1), Chh];
                rnnSmooth.Ccc = [Ccc(:,1), Ccc];
                rnnSmooth.Cxh = Cxh;
                rnnSmooth.Czz = Czz;
                % the input x at t=0
                rnnSmooth.x0 = x0;
                rnnSmooth.Sx0 = Sx0;
            else
                rnnSmooth = [];
            end
        end
        % Initialization
        function [net, states, maxIdx, netInfo] = initialization(net)
            % Build indices
            net = indices.initialization(net);
            net = indices.layerEncoder(net);
            net = indices.parameters(net);
            net = indices.covariance(net);
            netInfo = indices.savedInfo(net);
            % States
            states = tagi.initializeStates(net.nodes, net.batchSize, net.repBatchSize, net.xsc, net.dtype, net.gpu);
            if strcmp (net.RNNtype, 'LSTM_stateful') || strcmp (net.RNNtype,'LSTM_stateless') || strcmp (net.RNNtype, 'LSTM_lookback')
                states = tagi.lstmInitializeStates(states);
            elseif strcmp (net.RNNtype, 'GRU_stateful') || strcmp (net.RNNtype,'GRU_stateless') || strcmp (net.RNNtype, 'GRU_lookback')
                states = tagi.gruInitializeStates(states);
            end
            maxIdx = tagi.initializeMaxPoolingIndices(net.nodes, net.layer, net.layerEncoder, net.batchSize, net.repBatchSize, net.dtype, net.gpu);   
        end
         
    end
end