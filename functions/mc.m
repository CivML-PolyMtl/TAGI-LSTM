classdef mc
    methods (Static)
        % Main Function for model construction
        function [data, model, option]          = initSetup(data, model, option)
            % Data initialization
            [labelFARobs, idxFAR4R] = mc.getLabelFARobs(model);
            compIC                  = cell2mat(model.components.ic);
            mainObs                 = setdiff(1:model.nb_obs, compIC);            
            if ~isempty(cell2mat(labelFARobs))
                % Identify the existence of the FAR block in order to add
                % the correspoding observations
                model.labelFARobs   = labelFARobs;
                model.idxFAR4R      = idxFAR4R;
            end            
            [data, option] = mc.dataInitialization(data, model, option);
            
            
            % Model initialization
            Nclass      = model.nb_class;
            Nobs        = size(data.values,2);
            if ~isfield(model,'init')
                model.init  = defaultSetting.getInit(model, data); 
            end
            if isfield(model,'initRef')
                model.init.x = model.initRef.x; 
                model.init.V = model.initRef.V; 
                model.init.S = model.initRef.S; 
                model.init.U = model.initRef.U;
                model.init.D = model.initRef.D; 
            end
            Nhs = size(model.init.x(:,:,1),1);
            if ~isfield(model,'param_properties')
                model.param_properties = defaultSetting.getParamProp(model, data);
            end           
            
            % Identification of different time steps
            unique_dtSteps       = unique(data.dt_steps, 'stable');
            idx_udt              = mc.idx4dtStepsRef(data.dt_steps, unique_dtSteps);
            d                    = length(unique_dtSteps);
            model.unique_dtSteps = unique_dtSteps;
            model.idx_udt        = idx_udt;
            model.dt_ref         = data.dt_ref;
            
            % Time-independant model matrices
            [Cf, blkParamCf] = matrixC.getC(model);
            Rf               = matrixR.getR(model);
            Zf               = matrixZ.getZ(model);
            
            % Evaluation of  model matrices A, Q with respect to time steps
            Audt         = zeros(Nhs, Nhs, Nclass, d);
            Cudt         = zeros(Nobs, Nhs, Nclass, d);
            Qudt         = zeros(Nhs, Nhs, Nclass, Nclass, d);
            Rudt         = zeros(Nobs, Nobs, Nclass, d);
            Zudt         = zeros(Nclass, Nclass, d);
            blkParamAudt = zeros(Nhs, Nhs, Nclass, d);
            blkParamCudt = zeros(Nobs, Nhs, Nclass, d);
            blkParamQudt = zeros(Nhs, Nhs, Nclass, Nclass, d);
            for i = 1:d
                [Audt(:,:,:,i), blkParamAudt(:,:,:,i)] = matrixA.getA(model,unique_dtSteps(i));                
                [Qudt(:,:,:,:,i), blkParamQudt(:,:,:,:,i), labelHS, obsIdxHS] = matrixQ.getQ(model,unique_dtSteps(i));
                blkParamCudt(:,:,:,i) = blkParamCf;
                Cudt(:,:,:,i)         = Cf;
                Rudt(:,:,:,i)         = Rf;
                Zudt(:,:,i)           = Zf;               
            end
            blkParamRudt = Rudt;
           
            % Indexes for parameter
            idxParamAudt = blkParamAudt~=0;
            idxParamCudt = blkParamCudt~=0;
            idxParamRudt = blkParamRudt~=0;
            idxParamQudt = blkParamQudt~=0;
             
            % Indexes for varying part
            Avar = double(Audt~=0);
            Cvar = double(Cudt~=0);
            Qvar = double(Qudt~=0);
            Rvar = double(Rudt~=0);

            % Parameter indexes
            data_udt          = data;
            data_udt.dt_steps = model.unique_dtSteps;
            Cprop             = mp.matrixC(model, data_udt, Cudt, Cvar, idxParamCudt);
            Aprop             = mp.matrixA(model, data_udt, Audt, Avar, idxParamAudt);
            Qprop             = mp.matrixQ(model, data_udt, Qudt, Qvar, idxParamQudt);
            Rprop             = mp.matrixR(model, data_udt, Rudt, Rvar, idxParamRudt);
            Zprop             = mp.matrixZ(model, data_udt);
            parameter         = mc.getOptimParam(model, Aprop.paramAprop, Cprop.paramCprop, Qprop.paramQprop, Rprop.paramRprop, Zprop.paramZprop);

            % Regroup indexies of parameter and observation for each model 
            idxParam4model           = mc.regroupIdx4model(Aprop.idx, Cprop.idx, Qprop.idx, Rprop.idx, Zprop.idx);
            idxObs4model             = mc.regroupIdx4model(Aprop.idxObs, Cprop.idxObs, Qprop.idxObs, Rprop.idxObs, Zprop.idxObs);
            parameter.idxParam4model = idxParam4model;
            parameter.idxObs4model   = idxObs4model;
            model.parameter          = parameter;
            
            % If there exists the FAR block (42), the modification of R
            % matrix is required because of the presence of the fictive
            % observations 
            if ~isempty(cell2mat(labelFARobs))
                [paramValues_phiAR, paramValues_SigAR, idxFictiveObs,~] = mc.getParamValues4FAR(model, [], []);
                dtOverDtRef   = model.unique_dtSteps/model.dt_ref;
                Rnew          = mc.evalR4FAR(Rprop.Rconst, paramValues_phiAR, paramValues_SigAR, dtOverDtRef, idxFictiveObs);
                Rprop.Rconst  = Rnew;
                model.nb_obsF = length(cell2mat(labelFARobs));
            else
                model.nb_obsF = model.nb_obs;
            end
            
            % Save model matrices to model
            model.A.Acudt_pp       = Aprop.Aconst;
            model.C.Ccudt_pp       = Cprop.Cconst; 
            model.C.CcudtReg_pp    = Cprop.CconstReg;
            model.Q.Qcudt_pp       = Qprop.Qconst;
            model.R.Rcudt_pp       = Rprop.Rconst;
            model.Z.Zcudt_pp       = Zprop.Zconst;
                                 
            MM.eval  = {model.A.Acudt_pp;model.C.CcudtReg_pp;model.Q.Qcudt_pp;model.R.Rcudt_pp;model.Z.Zcudt_pp};
            MM.const = {model.A.Acudt_pp;model.C.Ccudt_pp;model.Q.Qcudt_pp;model.R.Rcudt_pp;model.Z.Zcudt_pp};
            MM.timestep = {Audt, Cudt, Qudt, Rudt, Zudt, Cudt};
            if isfield(model.components,'PCA')
                model.C.CcudtPCA_pp = Cprop.CconstPCA;
                MM.eval{model.C.PCAlocation,1} = model.C.CcudtPCA_pp;
            end
            
            % supInfo
            idxDK = labelHS(:,:,1)'==model.block.DK;
            if any(idxDK)
                idxDK_temp    = cell(2,1);
                idx_1         = cumsum(idxDK)==1;
                idx_2         = cumsum(idxDK)==(model.components.NcpDK(1)+1);
                idx_prod      = cumsum(idxDK)==(model.components.NcpDK(1)+model.components.NcpDK(2)+1);
                idx_prod      = cumsum(idx_prod)==1;
                idxDK_temp{1} = idx_1|idx_2;
                idxDK_temp{2} = idx_prod;
                model.supInfo.idxDK = idxDK_temp;
            else
                idxDK_temp    = cell(2,1);
                model.supInfo.idxDK = idxDK_temp;
            end
            
            idxTK = labelHS(:,:,1)'==model.block.TK;
            if any(idxTK)
                idxTK_temp    = cell(2,1);
                idx_1         = cumsum(idxTK)==1&idxTK;
                idx_2         = cumsum(idxTK)==(model.components.NcpTK+1)&idxTK;
                idx_prod      = cumsum(idxTK)==(model.components.NcpTK+2+1)&idxTK;
                idxTK_temp{1} = idx_1|idx_2;
                idxTK_temp{2} = idx_prod;
                model.supInfo.idxTK = idxTK_temp;
            else
                idxTK_temp    = cell(2,1);
                model.supInfo.idxTK = idxTK_temp;
            end
            
            idxARP = labelHS(:,:,1)'==model.block.ARP;
            if any(idxARP)
                idxARP_temp    = cell(2,1); 
                idxARP_temp{1} = cumsum(idxARP)<=2&idxARP;
                idxARP_temp{2} = cumsum(idxARP)==1&idxARP;
                model.supInfo.idxARP = idxARP_temp;
            else
                idxARP_temp          = cell(2,1);
                model.supInfo.idxARP = idxARP_temp;
            end
                      
            % Output
            model.MM.eval     = MM.eval;     % Save model matrices {A,C,Q,R,Z,PCA}
            model.MM.timestep = MM.timestep; % Save the model matrices relating to time steps such as the baseline block
            model.labelHS     = labelHS;     % Save the block label for HS 
            model.nb_HS       = size(labelHS(:,:,1),2); % Save the number of hidden states
            model.obsIdxHS    = obsIdxHS;    % Save the observation label with resptect to the hidden states          
            model.MM.const    = MM.const;    % It is employed for displaying model matrices
            model.mainObs     = mainObs;              
        end
        
        % Identify the parameters to be optimized
        function [parameter]                    = getOptimParam(model, pA, pC, pQ, pR, pZ)
            if isnan(pZ{1,model.up.cParamIdx})
                paramProp = [pA;pC;pQ;pR];
            else
                paramProp = [pA;pC;pQ;pR;pZ];
            end
            Nparam          = size(paramProp,1);
            idxTypeParam    = cell(1, Nparam);
            for n = 1:Nparam
                idxTypeParam{n} = n*ones(1,length(paramProp{n, model.up.cParamValue}));
            end
            idxTypeParam    = cell2mat(idxTypeParam)';
            idxParamValues  = cell2mat(paramProp(:,model.up.cParamIdx));
            [idxParamValues,idxu,~]  = unique(idxParamValues,'stable');
            idxTypeParam    = idxTypeParam(idxu);
            idxFixedParam   = mp.sepParam(model,idxParamValues); 
            idxUkParamValue = idxParamValues(~idxFixedParam);
            idxUkTypeParam  = idxTypeParam(~idxFixedParam);
            ukParamValue    = [model.param_properties{idxUkParamValue,model.cParamValue}]';
            
            % Transformation function
            transfun = [];
            for i = 1:size(paramProp,1)
                transfun = [transfun; paramProp{i,model.up.cTransFun}];
            end
            transFun4ukParam = transfun(~idxFixedParam,:);
            
            ukParamValueTR  = zeros(size(ukParamValue));
            for i = 1:length(idxUkTypeParam)
                ukParamValueTR(i)   = transFun4ukParam{i,1}(ukParamValue(i));
            end 
            
            % Output
            parameter.prop              = paramProp;         % properties of unknown parameters
            parameter.idxUkParamValue   = idxUkParamValue;   % indexes of unknown parameters in model.param_properties
            parameter.ukParamValue      = ukParamValue;      % unknown parameters in original space
            parameter.ukParamValueTR    = ukParamValueTR;    % unknown parameters in transformed space
            parameter.idxUkParamProp    = idxUkTypeParam;    % indexes of unknown parameters in paramProp
            parameter.transFun4ukParam  = transFun4ukParam;  % transformation function for unknown parameters           
        end
        function idx_dt                         = idx4dtStepsRef(dtSteps, unique_dtSteps)            
            d               = length(unique_dtSteps);
            [~,idx]         = ismember(dtSteps, unique_dtSteps);
            idx_dt          = cell(d, 1);
            for i = 1:d
                idx_dt{i} = idx==i;
            end
        end
        function idx4model                      = regroupIdx4model(idxFromA, idxFromC, idxFromQ, idxFromR, idxFromZ)
            Nclass    = size(idxFromA,1);
            idx4model = cell(Nclass, 1);
            for j = 1:Nclass
                idx4model{j} = [idxFromA{j},idxFromC{j},idxFromQ{j},idxFromR{j},idxFromZ{j}];
            end
        end
        
        % Get model matrix for the entire dataset
        function MM_new                         = getFullModelMatrix(MM_old, MMu_new, Aloc, Cloc, Qloc, Rloc, Zloc, idx_dt)
            A = MM_old{Aloc};
            C = MM_old{Cloc};
            Q = MM_old{Qloc};
            R = MM_old{Rloc};
            Z = MM_old{Zloc};
            
            Audt = MMu_new{Aloc};
            Cudt = MMu_new{Cloc};
            Qudt = MMu_new{Qloc};
            Rudt = MMu_new{Rloc};
            Zudt = MMu_new{Zloc};
            
            d  = size(idx_dt, 1); 
            for i = 1:d
                A(:,:,:,idx_dt{i})      = repmat(Audt(:,:,:,i),[1,1,1,sum(idx_dt{i})]);
                C(:,:,:,idx_dt{i})      = repmat(Cudt(:,:,:,i),[1,1,1,sum(idx_dt{i})]);
                Q(:,:,:,:,idx_dt{i})    = repmat(Qudt(:,:,:,:,i),[1,1,1,1,sum(idx_dt{i})]);
                R(:,:,:,idx_dt{i})      = repmat(Rudt(:,:,:,i),[1,1,1,sum(idx_dt{i})]);
                Z(:,:,idx_dt{i})        = repmat(Zudt(:,:,i),[1,1,sum(idx_dt{i})]);
            end
            MM_new = {A;C;Q;R;Z};
        end 
        
        % Main function for Parameter update 
        function [MMnew, MMuEvalNew, CPnew]     = updateModelMatrix(model, paramUpdate, idxParamUpdate, idxTypeParam, MMold,  MMuEval, idx_udt,...
                unique_dtSteps)
            % MMold: Model matrices for the entire dataset
            % MMuEval: model matrices for unique_dt_steps. In case the PCA
            % is employed, MMuEval separates the Creg(2nd postion) and Cpca
            % (6th position)
            
            % Initialization 
            CPnew     = [];
            paramProp = mc.updateNewParamValues(model, paramUpdate, idxParamUpdate, idxTypeParam);
            model.parameter.prop    = paramProp;
            [idxSK, idxDK, idxDH, idxNK, idxDNK, idxTK] = mc.idenIdxParam4DRC(model, paramProp);
            
            % Indexes of DRC relating to the updating parameters
            idxSKp      = find(idxSK(idxTypeParam));
            idxDKp      = find(idxDK(idxTypeParam));
            idxDHp      = find(idxDH(idxTypeParam));
            idxNKp      = find(idxNK(idxTypeParam));
            idxDNKp     = find(idxDNK(idxTypeParam));
            idxTKp      = find(idxTK(idxTypeParam));
%             idxDRCp    = [idxDHp,idxSKp,idxDKp,idxNKp,idxDNKp];
            
            % Indexes of DRC exist in paramProp that might also include the
            % updating parameter above            
            idxDRCtp    = find(idxDH|idxSK|idxDK|idxNK|idxDNK|idxTK);
            
            idxNotDRC   = ~idxSK(idxTypeParam)&~idxDK(idxTypeParam)&~idxDH(idxTypeParam)&~idxNK(idxTypeParam)&~idxDNK(idxTypeParam)&~idxTK(idxTypeParam);  
            paramUpdateNormal = paramUpdate(idxNotDRC);
           
            MMts   = model.MM.timestep;
            Aloc   = model.A.location;
            Cloc   = model.C.location;
            Qloc   = model.Q.location;
            Rloc   = model.R.location;
            Zloc   = model.Z.location;
            
            dcTF   = model.up.cTempFun;         
            dcPV   = model.up.cParamValue;
            dcTS   = model.up.cTimestep;
            dcI4M  = model.up.cIdx4matrix; 
            dcID   = model.up.cIdxDist;
            dcIML  = model.up.cIdxMatrixLoc;
            dcSF   = model.up.cSupFun;
            
            % Fictive autoregressive components
            if isfield(model,'labelFARobs')&&any([model.param_properties{idxParamUpdate,model.cEncoder}]==model.block.AR)
                [paramValues_phiAR, paramValues_SigAR, idxFictiveObs,~] = mc.getParamValues4FAR(model, paramProp, idxParamUpdate);
                dtOverDtRef                = model.unique_dtSteps/model.dt_ref;
%                 MMeval_udt_temp            = mc.getMMuEval(MMConstTempo, model.unique_dtSteps, unique_dtSteps, Aloc, Cloc, Qloc, Rloc, Zloc);
                Rnew                       = mc.evalR4FAR(MMuEval{model.R.location}, paramValues_phiAR, paramValues_SigAR, dtOverDtRef, idxFictiveObs);
                MMuEval{model.R.location}  = Rnew ;
            end
            
            % Normal parameters
            if ~isempty(paramUpdateNormal)
                idxTypeParamNormal      = idxTypeParam(idxNotDRC);
                [Mc, idxc, idxLocation] = mc.updateTempModelMatrix(model, idxTypeParamNormal);
                % Only model matrices corresponding the parameter updates
                % is evaluted in mc.MconstTimesMvar so that we only take
                % MMuEval(idxLocation)
                MM                      = mc.MconstTimesMvar(MMuEval(idxLocation), Mc, MMts(idxLocation), idxc, idxLocation);
                MMuEval(idxLocation,1)  = MM;                                          
            end
            
            % In the case where the PCA method is deployed, an addtional
            % operation needs to be completed to get the final model
            % matrices i.e. Creg*Cpca because the scaling factor along with
            % the elements in P matrix is added on top of regression
            % coefficient for each hidden state such as periodic and AR
            % components.  If the PCA method is employed, the size of
            % model.MM.eval is greater than the last location e.g. Zlocation
            MMeEvalTempo = MMuEval;
            if size(model.MM.eval, 1) > model.Z.location
                MMeEvalTempo{model.C.location,1} = bsxfun(@times, MMeEvalTempo{model.C.location}, MMeEvalTempo{model.C.PCAlocation});
            end
            MMConstTempo    = MMeEvalTempo(model.A.location:model.Z.location,1);
            MMuEvalNew      = MMuEval;
            % Get model matrices correponding to unique_dtSteps at a
            % specific dataset. Because it depends on the dataset that
            % we employ so that we can  chose the corresponding model
            % matrices. This task is done by identifying the location
            % of the unique_dtSteps's positions inside the
            % model.unique_dtSteps that is defined for the entire
            % dataset
            MMconst_udt = mc.getMMuEval(MMConstTempo, model.unique_dtSteps, unique_dtSteps, Aloc, Cloc, Qloc, Rloc, Zloc);
            MMnewTemp   = mc.getFullModelMatrix(MMold, MMconst_udt, Aloc, Cloc, Qloc, Rloc, Zloc, idx_udt);
            
            % Dynamic regression component a.k.a blocks {51, 52, 53, 54}
            % Update full model matrices for regression Kernel component
            % because their values have been removed in the
            % mc.getFullModelMatrix.  
            if ~isempty(idxDRCtp)
                for i = idxDRCtp(:)'
                    MMnewTemp{paramProp{i,dcIML}}(paramProp{i,dcI4M}) = MMold{paramProp{i,dcIML}}(paramProp{i,dcI4M});
                end
            end
            
            % The DRC parameters are changed
            % SK 52
            if ~isempty(idxSKp)
                NcompDRC          = model.components.NcpSK-1; 
                idxTypeParam4SKp  = idxTypeParam(idxSKp);
                for i = idxTypeParam4SKp(:)'                    
                    Mnew = mc.evalModelMatrixDRC(paramProp(i,:), MMnewTemp{paramProp{i,dcIML}}, dcTF, dcPV, dcTS, dcI4M, dcID, NcompDRC);
                    MMnewTemp{paramProp{i,dcIML}} = Mnew;
                end   
            end  
                % DH 51
            if ~isempty(idxDHp) 
                NcompDRC         = 1;
                idxTypeParam4DHp = idxTypeParam(idxDHp);
                for i = idxTypeParam4DHp(:)'
                    Mnew = mc.evalModelMatrixDRC(paramProp(i,:), MMnewTemp{paramProp{i,dcIML}}, dcTF, dcPV, dcTS, dcI4M, dcID, NcompDRC);
                    MMnewTemp{paramProp{i,dcIML}} = Mnew;
                end   
            end   
                % DK 53
            if ~isempty(idxDKp)
                NcompDRC         = sum(model.components.NcpDK)-2;
                idxTypeParam4DKp = idxTypeParam(idxDKp);
                for i = idxTypeParam4DKp(:)'
                    Mnew = mc.evalModelMatrixDRC(paramProp(i,:), MMnewTemp{paramProp{i,dcIML}}, dcTF, dcPV, dcTS, dcI4M, dcID, NcompDRC);
                    MMnewTemp{paramProp{i,dcIML}} = Mnew;
                end
            end   
                % NK 54
            if ~isempty(idxNKp)
                NcompNK          = model.components.NcpNK;
                idxTypeParam4NKp = idxTypeParam(idxNKp);
                for i = idxTypeParam4NKp(:)'
                    [Mnew, CPnew] = mc.evalModelMatrixNK(paramProp(i,:), MMnewTemp{paramProp{i,dcIML}}, dcTF, dcPV, dcTS, dcI4M, dcID, dcSF, NcompNK);
                    MMnewTemp{paramProp{i,dcIML}} = Mnew;
                end
            end    
                % DNK 55
            if ~isempty(idxDNKp)
                NcompDNK          = model.components.NcpDNK-1;
                idxTypeParam4DNKp = idxTypeParam(idxDNKp);
                for i = idxTypeParam4DNKp(:)'
                    [Mnew, CPnew] = mc.evalModelMatrixNK(paramProp(i,:), MMnewTemp{paramProp{i,dcIML}}, dcTF, dcPV, dcTS, dcI4M, dcID, dcSF, NcompDNK);
                    MMnewTemp{paramProp{i,dcIML}} = Mnew;
                end
            end
                % TK 56
            if ~isempty(idxTKp)
                NcompDRC         = model.components.NcpTK-1;
                idxTypeParam4TKp = idxTypeParam(idxTKp);
                for i = idxTypeParam4TKp(:)'
                    Mnew = mc.evalModelMatrixDRC(paramProp(i,:), MMnewTemp{paramProp{i,dcIML}}, dcTF, dcPV, dcTS, dcI4M, dcID, NcompDRC);
                    MMnewTemp{paramProp{i,dcIML}} = Mnew;
                end
            end  
            MMnew = MMnewTemp;
        end
        function [MMnew, paramProp]             = updateModelMatrix4RBPF(model, paramProp, MMuEvalSamples, MMtsSamples, idxIdenMMloc,...
                uniqueMatLoc, GPU)  
            % Initialization 
            NparamType  = size(paramProp,1);            
            
            % Matrix location
            Aloc     = model.A.location;
            Cloc     = model.C.location;
            Qloc     = model.Q.location;
            Rloc     = model.R.location;
            Zloc     = model.Z.location;
            PCAloc   = model.C.PCAlocation;           
            
            % Column number in model.parameter.prop
            up_cTF   = model.up.cTempFun;         
            up_cPV   = model.up.cParamValue;
            up_cPC   = model.up.cParamClass;
            up_cTS   = model.up.cTimestep;
            up_cIM   = model.up.cIdx4matrix; 
            up_cID   = model.up.cIdxDist;
            up_cIML  = model.up.cIdxMatrixLoc;
            up_cSF   = model.up.cSupFun;
            up_cSI   = model.up.cSupInfo;
            
            % Block for hidden states
            blockDH  = model.block.DH;
            blockSK  = model.block.SK;
            blockDK  = model.block.DK;
            blockNK  = model.block.NK;
            blockDNK = model.block.DNK; 
            
            supInfo  = [paramProp{:,up_cSI}]';
            SI_PCA   = model.supInfo.PCA;
            SI_Z     = model.supInfo.Z;
            
            % Preallocation
            Mvar_pt                   = cell(1, NparamType);
            idx_pt                    = cell(1, NparamType);
            NcompK                    = zeros(NparamType, 1);
            NcompK(supInfo==blockDH)  = model.components.NcpDH;
            NcompK(supInfo==blockSK)  = model.components.NcpSK-1;
            NcompK(supInfo==blockDK)  = model.components.NcpDK-1;
            NcompK(supInfo==blockNK)  = model.components.NcpNK;
            NcompK(supInfo==blockDNK) = model.components.NcpDNK-1;
            
            % Fictive autoregressive components
            if isfield(model,'labelFARobs')
                [paramValues_phiAR, paramValues_SigAR, idxFictiveObs, dtOverDtRef] = mc.getParamValues4FAR(model, paramProp,...
                    model.parameter.idxUkParamValue);
                FARexistence  = 1;
            else
                 FARexistence = 0;
            end
            
            % Evaluate the model matrices as a function of time
            for i = 1:NparamType           
                [Mvar_pt{i}, paramProp(i,:), idx_pt{i}] = mc.evalTempModelMatrix4RBPF(paramProp(i,:), up_cTF,...
                    up_cPV, up_cPC, up_cTS, up_cIM, up_cSI, up_cSF, up_cID, up_cIML, Qloc, SI_PCA, SI_Z, NcompK(i),...
                    blockDH, blockSK, blockDK, blockNK, blockDNK, GPU);
            end 
            
            if GPU
                 uniqueMatLoc       = gpuArray(uniqueMatLoc);
            end
            MMuEvalNew              = cell(size(MMuEvalSamples));
            Niden                   = size(idxIdenMMloc, 2);
            for i = 1:Niden
                S_loop              = max(size(idxIdenMMloc{i}));
                if S_loop>1
                    if GPU
                        MMrem_loop  = gpuArray(0);
                        idxRem_loop = gpuArray(0);
                    else
                        MMrem_loop  = 0;
                        idxRem_loop = 0;
                    end
                    for l = 1:length(idxIdenMMloc{i})
                        MMrem_loop                  = bsxfun(@plus, MMrem_loop, Mvar_pt{idxIdenMMloc{i}(l)});
                        idxRem_loop                 = bsxfun(@plus, idxRem_loop, idx_pt{idxIdenMMloc{i}(l)});
                        Mvar_pt{idxIdenMMloc{i}(l)} = [];
                        idx_pt{idxIdenMMloc{i}(l)}  = [];
                    end  
                else
                    MMrem_loop                  = Mvar_pt{idxIdenMMloc{i}};
                    idxRem_loop                 = idx_pt{idxIdenMMloc{i}};
                    Mvar_pt{idxIdenMMloc{i}}    = [];
                    idx_pt{idxIdenMMloc{i}}     = [];
                end
                if isempty(idxRem_loop)
                     MMuEvalNew{uniqueMatLoc(i)} = MMrem_loop;    
                     clear MMrem_loop
                else
                    
                    idxRemHat_loop              = ~logical(idxRem_loop);
                    clear idxRem_loop
                    MMuEvalSamplesM_loop        = MMuEvalSamples{uniqueMatLoc(i)};
                    Mtrans                      = bsxfun(@times, MMuEvalSamplesM_loop,idxRemHat_loop);
                    clear MMuEvalSamplesM_loop idxRemHat_loop
                    MMtsSamplesM_loop           = MMtsSamples{uniqueMatLoc(i)};
                    Mvarts                      = bsxfun(@times, MMtsSamplesM_loop, MMrem_loop);
                    MMuEvalNew{uniqueMatLoc(i)} = bsxfun(@plus, Mtrans, Mvarts);
                    clear MMtsSamplesM_loop Mvarts Mtrans MMrem_loop idxRem_loop
                end
            end  
            
            MMnew               = MMuEvalSamples(Aloc:Zloc);
            MMnew(uniqueMatLoc) = MMuEvalNew(uniqueMatLoc);
            if max(size(MMuEvalNew))>Zloc
                MMnew{Cloc}     = bsxfun(@times, MMnew{Cloc}, MMnew{PCAloc});
            end
            
            % FAR block existence ?
            if FARexistence
                R_CPU           = gather(MMnew{Rloc});
                MMnew{Rloc}     = mc.evalR4FAR(R_CPU, paramValues_phiAR, paramValues_SigAR, dtOverDtRef, idxFictiveObs);
            end
        end
        
        % Evaluate model matrix with respect to time steps
        function [Mc, idxc, idxLocation]        = updateTempModelMatrix(model, idxTypeParam)
            % idxParamUpdate: parameter's indexes in model.param_properties
            % idxTypeParam : parameter's indexes in paramProp
            
            % Initialization  
            paramProp           = model.parameter.prop;
            dcTempFun           = model.up.cTempFun;         
            dcParamValue        = model.up.cParamValue;
            dcTimestep          = model.up.cTimestep;
            dcIdx4matrix        = model.up.cIdx4matrix;
            dcIdxMatrixLoc      = model.up.cIdxMatrixLoc;
            dcSupInfo           = model.up.cSupInfo;
            dcSupFun            = model.up.cSupFun;
            dcIdxDist           = model.up.cIdxDist;
            supInfoPCA          = model.supInfo.PCA;
            supInfoZ            = model.supInfo.Z;
            PCAlocation         = model.C.PCAlocation;
            
            % Indexes of the entire unknown parameters in paramProp 
            idxTypeParamUpdate  = unique(idxTypeParam,'stable');
            idxLocationDist     = [paramProp{idxTypeParamUpdate, dcIdxMatrixLoc}]';
            idxLocation         = unique(idxLocationDist,'sorted');         
         
            NtypeParamUpdate    = length(idxTypeParamUpdate);
            MMc                 = cell(1, NtypeParamUpdate);
            idx4MMc             = cell(1, NtypeParamUpdate);
            for u = 1:NtypeParamUpdate
                [MMc{u}, idx4MMc{u}] = mc.evalTempModelMatrix(paramProp, idxTypeParamUpdate(u), dcTempFun, dcParamValue, dcTimestep,...
                    dcIdx4matrix, dcSupInfo, dcSupFun, dcIdxDist, supInfoPCA, supInfoZ);
            end 
            [Mc, idxc] = mc.classifyModelMatrixUpdate(MMc, idx4MMc, idxLocationDist, idxLocation, PCAlocation);                
        end               
        function [Mvar, idx]                    = evalTempModelMatrix(paramProp, idxTypeParam, dcTempFun, dcParamValue, dcTimestep,...
                dcIdx4matrix, dcSupInfo, dcSupFun, dcIdxDist, supInfoPCA, supInfoZ)
            % paramValues = entire parameter values in
            % model.param_properties.
            idx         = paramProp{idxTypeParam,dcIdx4matrix};
            Mvar        = double(idx);
                       
            % Evaluate the paramters with respect to time step in which the
            % parameters related to either PCA or transition probability Z
            % employs different temporal function. See mp.m for
            % further details
            if paramProp{idxTypeParam,dcSupInfo}==supInfoPCA
                Mtemp = paramProp{idxTypeParam,dcTempFun}(paramProp{idxTypeParam,dcParamValue}, paramProp{idxTypeParam,dcTimestep}); 
            elseif paramProp{idxTypeParam,dcSupInfo}==supInfoZ
                Zdt   = paramProp{idxTypeParam,dcSupFun}(paramProp{idxTypeParam,dcParamValue},paramProp{idxTypeParam,dcTimestep}); 
                Mtemp = mp.evalZtemp(Zdt, paramProp{idxTypeParam,dcTimestep});
            else
                Mtemp = bsxfun(paramProp{idxTypeParam,dcTempFun}, paramProp{idxTypeParam,dcParamValue}, paramProp{idxTypeParam,dcTimestep});   
            end
            
            % Assign the new parameter values to its correspoding location
            % in the model matrices
            if paramProp{idxTypeParam,dcSupInfo}==supInfoPCA
                Mvar(idx) = reshape(Mtemp', size(Mtemp,1)*size(Mtemp,2),1);
            elseif paramProp{idxTypeParam,dcSupInfo}==supInfoZ                
                Mvar = Mtemp;
            else
                Mtemp      = repmat(Mtemp,[1,size(paramProp{idxTypeParam,dcIdxDist},2)/size(Mtemp,2)]);
                Mtemp      = reshape(Mtemp', size(Mtemp,1)*size(Mtemp,2),1);
                Mvar(idx)  = Mtemp;
            end  
        end
        function [MvarSaved, paramProp, idx]    = evalTempModelMatrix4RBPF(paramProp, up_cTF, up_cPV, up_cPC, up_cTS, up_cIM,...
                up_cSI, up_cSF, up_cID, up_cIML, Qloc, SI_PCA, SI_Z, NcompK, blockDH, blockSK, blockDK, blockNK, blockDNK, GPU)
            
            % Initialization
            idx          = paramProp{up_cIM};            
            paramSamples = paramProp{up_cPV};
            Nsamples     = size(paramSamples,2);
            Nparam       = size(paramSamples,1);
            supInfo      = paramProp{up_cSI};
            tempValues   = paramProp{up_cTS};
            IML          = paramProp{up_cIML};
            NparamDist   = length(paramProp{up_cID});
            tempFun      = paramProp{up_cTF};
            paramClass   = paramProp{up_cPC};               
            
            % Control points for Non-periodic Kernel
            if supInfo == blockDNK||supInfo == blockNK
                CP           = paramProp{up_cSF}{1};
                PM           = paramProp{up_cSF}{2};
                K_prod       = paramProp{up_cSF}{3};
                cpOpt        = paramProp{up_cSF}{4};
                y            = paramProp{up_cTS}(:,2:end);
                ts_t         = paramProp{up_cTS}(end,1);
                ts           = paramProp{up_cTS}(:,1);
                
                CPsaved      = zeros(size(CP));
                PMsaved      = zeros(size(PM));
                K_prodSaved  = zeros(size(K_prod));
                cpOptSaved   = zeros(size(cpOpt));
            else
                CP           = nan(1,1,Nsamples);
                PM           = nan(1,1,Nsamples);
                K_prod       = nan(Nsamples, 1);
                cpOpt        = nan(Nsamples, 1);
                y            = NaN;
                ts_t         = NaN;
                ts           = NaN;
                
                CPsaved      = zeros(1,1,Nsamples);
                PMsaved      = zeros(1,1,Nsamples);
                K_prodSaved  = zeros(Nsamples, 1);
                cpOptSaved   = zeros(Nsamples, 1);
                
            end
            if GPU~=1||paramClass==6||paramClass==7%||(paramClass==1&&IML==Qloc)
                % Because of the difference dimension between the model matrix
                % so that the highest dimensiton is selected as the reference
                % dimension. Such matrix A i.e. 4-D matrix need to be permuted
                % to become 5-D matrix corresponding the highest dimension. The
                % task done is enable the parallel setup.
                Mvar         = idx*1;
                MvarSavedCell = cell(1,Nsamples);
                MvarPerm      = permute(Mvar,[1 2 3 5 4]);
                idxPerm       = permute(idx,[1 2 3 5 4]);
                if supInfo == SI_Z
                    tempFun    = paramProp{up_cSF};
                    MvarPerm   = nan(1,1,1,1,Nsamples);
                    idxPerm    = nan(1,1,1,1,Nsamples);
                elseif IML==Qloc
                    MvarPerm  = Mvar;
                    idxPerm   = idx;
                end
                
                % Evaluate the paramters with respect to time step in which the
                % parameters related to either PCA or transition probability Z
                % employs different temporal function. See mp.m for
                % further details                
                for s = 1:Nsamples
                    [MvarSavedCell{s}, CPsaved(:,:,s), PMsaved(:,:,s), K_prodSaved(s,:), cpOptSaved(s)] = mc.paramWithTempFun(tempFun, paramSamples(:,s),...
                        tempValues, MvarPerm(:,:,:,:,s), idxPerm(:,:,:,:,s), CP(:,:,s), PM(:,:,s), K_prod(s,:), cpOpt(s), y, ts_t, ts, NparamDist,...
                        SI_PCA, SI_Z, NcompK, supInfo, blockDH, blockSK, blockDK, blockNK, blockDNK);
                end
                if supInfo == SI_Z
                    MvarSaved = cat(3, MvarSavedCell{:});
                elseif IML==Qloc
                    MvarSaved = cat(5, MvarSavedCell{:});
                else
                    MvarSaved = cat(4, MvarSavedCell{:});
                end
                if supInfo==blockNK||supInfo==blockDNK
                    paramProp{up_cSF} = {CPsaved, PMsaved, K_prodSaved, cpOptSaved};
                end
                
                % Transfer to GPU
                if GPU
                    MvarSaved = gpuArray(MvarSaved);
                end
            else
                % GPU computation
%                 idx             = gpuArray(idx);
                Mvar            = bsxfun(@times,idx,1);
                paramSamples    = gpuArray(paramSamples);
                tempValues      = gpuArray(tempValues);
                NparamDist      = gpuArray(NparamDist);
                Nsamples        = gpuArray(Nsamples);
                Nparam          = gpuArray(Nparam);
                paramClass      = gpuArray(paramClass);
                
                if paramClass==1% p*(dt/dref)
                    Mtemp       = bsxfun(@times,paramSamples,tempValues);                                     
                    Mtemp       = repmat(Mtemp,[NparamDist,1]);
                    Mtemp       = reshape(Mtemp,size(Mtemp,1)*size(Mtemp,2),1);
                    Mvar(idx)   = Mtemp;
                    MvarSaved   = Mvar; 
                    
                elseif paramClass==2 % p^dt/dref
                    Mtemp       = bsxfun(@power,paramSamples,tempValues);                                     
                    Mtemp       = repmat(Mtemp,[NparamDist,1]);
                    Mtemp       = reshape(Mtemp,size(Mtemp,1)*size(Mtemp,2),1);
                    Mvar(idx)   = Mtemp;
                    MvarSaved   = Mvar; 
                elseif paramClass==3 % [cos -sin sin cos]
                    
                    fcos        = arrayfun(@(x,y) cos(y./x),paramSamples,tempValues);
                    fsin        = arrayfun(@(x,y) sin(y./x),paramSamples,tempValues);
                    Mtemp       = [fcos;-fsin;fsin;fcos];
                    Mtemp       = repmat(Mtemp,[NparamDist,1]);
                    Mtemp       = reshape(Mtemp,size(Mtemp,1)*size(Mtemp,2),1);                    
                    Mvar(idx)   = Mtemp;
                    MvarSaved   = Mvar; 
                                        
                elseif paramClass==4 % Kernel Regression
                    timestamp_ref = paramProp{up_cSF};
                    Mtemp       = mp.dynacmicKernelComponent4GPU(paramSamples,tempValues,timestamp_ref,NcompK);
                    Mtemp       = pagefun(@transpose,Mtemp );
                    Mtemp       = repmat(Mtemp,[NparamDist/NcompK 1]);
                    Mtemp       = reshape(Mtemp, size(Mtemp,1)*size(Mtemp,2),1);                    
                    Mvar(idx)   = Mtemp;
                    MvarSaved   = Mvar;
                    
                elseif paramClass==5 % Transition probabilities
                    countM_ref                      = [1:Nparam]';
                    countM                          = repmat(countM_ref,[1 Nparam]);
                    countM                          = countM-diag(countM_ref);
                    countM(countM==0)               = [];
                    countM                          = gpuArray(countM);
                    MvarSaved                       = ones(Nparam,Nparam,Nsamples,'gpuArray');
                    onev                            = ones(size(paramSamples),'gpuArray');
                    MvarSavedDiag                   = eye(Nparam,'gpuArray');
                    MvarSavedOffDiag                = ones(Nparam,'gpuArray')-eye(Nparam,'gpuArray');
                    MvarSavedDiag                   = repmat(MvarSavedDiag, [1 1 Nsamples]);
                    MvarSavedOffDiag                = repmat(MvarSavedOffDiag,[1 1 Nsamples]);
                    Mtemp                           = arrayfun(@(x,y,z) (x-y)./z,onev, paramSamples, Nparam-1);
                    Mtemp                           = Mtemp(countM,:);
                    paramSamples                    = reshape(paramSamples,Nparam*Nsamples,1);
                    Mtemp                           = reshape(Mtemp,size(Mtemp,1)*size(Mtemp,2),1);
                    MvarSaved(MvarSavedDiag==1)     = paramSamples;
                    MvarSaved(MvarSavedOffDiag==1)  = Mtemp;
                end
            end            
        end
        function [MvarSaved, CPsaved, PMsaved,...
                K_prodSaved, cpOptSaved]        = paramWithTempFun(tempFun, param, tempValues, Mvar, idx, CP, PM,...
                K_prod, cpOpt, y, ts_t, ts, NparamDist, SI_PCA, SI_Z, NcompK, supInfo, blockDH, blockSK, blockDK, blockNK, blockDNK)
            
            % Initialization
            CPsaved            = NaN;
            PMsaved            = NaN;
            K_prodSaved        = NaN;
            cpOptSaved         = NaN;
            if and(size(Mvar,4)<1,~isempty(Mvar))
                Mvar           = permute(Mvar,[1 2 3 5 4]);
                idx            = permute(idx, [1 2 3 5 4]);
            end
                 
             % PCA method
             if supInfo == SI_PCA
                 Mvar_loop          = Mvar;
                 Mtemp              = tempFun(param, tempValues);
                 Mvar_loop(idx)     = reshape(Mtemp', size(Mtemp,1)*size(Mtemp,2),1);
                 MvarSaved          = Mvar_loop;
                 
                 % Matrix Z
             elseif supInfo == SI_Z
                 Zdt                = tempFun(param, tempValues);
                 Mtemp              = mp.evalZtemp(Zdt, tempValues);
                 Mvar_loop          = Mtemp;
                 MvarSaved          = Mvar_loop;
                 
                 % Dynamic Regression (51), Periodic Kernel (52, 53)
             elseif supInfo==blockSK||supInfo==blockDK||supInfo==blockDH
                 Mvar_loop          = Mvar;
                 Mtemp              = repmat(tempFun(param, tempValues),[1, NparamDist/NcompK]);
                 Mtemp              = reshape(Mtemp', size(Mtemp,1)*size(Mtemp,2),1);
                 Mvar_loop(idx)     = Mtemp;
                 MvarSaved          = Mvar_loop;
                 
                 % Non-periodic Kernel (54, 55)
             elseif supInfo==blockNK||supInfo==blockDNK
                 Mvar_loop          = Mvar;
                 NKout              = tempFun(y, ts_t, ts, param, CP, PM, K_prod, cpOpt);
                 Mtemp              = repmat(NKout{1},[1, NparamDist/NcompK]);
                 Mvar_loop(idx)     = Mtemp;
                 CPsaved            = NKout{2};
                 PMsaved            = NKout{3};
                 K_prodSaved        = NKout{4};
                 cpOptSaved         = NKout{5};
                 MvarSaved          = Mvar_loop;
             else
                 Mtemp              = bsxfun(tempFun, param, tempValues);
                 Mtemp              = repmat(Mtemp,[1, NparamDist/size(Mtemp,2)]);
                 Mtemp              = reshape(Mtemp', size(Mtemp,1)*size(Mtemp,2),1);
                 Mvar_loop          = Mvar;
                 Mvar_loop(idx)     = Mtemp;
                 MvarSaved          = Mvar_loop;               
             end
        end
        function Mnew                           = evalModelMatrixDRC(paramProp, Mold, dcTF, dcPV, dcTS, dcI4M, dcID, NcompDRC)
            Nsteps      = length(paramProp{dcTS});
            NparamDist  = length(paramProp{dcID});
            Mtemp       = zeros(Nsteps, NparamDist);
            for t = 1:Nsteps
                Mtemp(t,:) = repmat(paramProp{dcTF}(paramProp{dcPV},paramProp{dcTS}(t)),[1, NparamDist/NcompDRC]);
            end
            Mtemp = reshape(Mtemp', size(Mtemp,1)*size(Mtemp,2),1);
            Mold(paramProp{dcI4M}) = Mtemp;
            Mnew  = Mold;
        end
        function [Mnew, CPnew]                  = evalModelMatrixNK(paramProp, Mold, dcTF, dcPV, dcTS, dcI4M, dcID, dcSF, NcompNK)
            Nsteps      = size(paramProp{dcTS},1);
            NparamDist  = length(paramProp{dcID});
            Mtemp       = zeros(Nsteps, NparamDist);
            for t = 1:Nsteps
                NKout_loop = paramProp{dcTF}(paramProp{dcTS}(:,2:end),paramProp{dcTS}(t,1),paramProp{dcTS}(:,1), paramProp{dcPV}, paramProp{dcSF}{1},...
                    paramProp{dcSF}{2},paramProp{dcSF}{3}, paramProp{dcSF}{4});
                Mtemp(t,:) = repmat(NKout_loop{1},[1, NparamDist/NcompNK]);
                paramProp{dcSF} = NKout_loop(2:5);
            end
            Mtemp = reshape(Mtemp', size(Mtemp,1)*size(Mtemp,2),1);
            Mold(paramProp{dcI4M}) = Mtemp;
            Mnew  = Mold;
            CPnew = paramProp{dcSF};
        end
        function [Mc, idxc]                     = classifyModelMatrixUpdate(MMc, idx4MMc, idxLocationDist, idxLocation, PCAlocation)
            % classify the updated parameters corresponding to its own 
            % model matrix. Example \sigma_LL and \sigma_AR are updated at
            % the same time so that they are classified into matrix Q that
            % belongs to location No 3.
            Nlocation   = length(idxLocation);
            Mc          = cell(1, Nlocation);
            idxc        = cell(1, Nlocation);
            for l = 1:Nlocation
                if idxLocation(l)==PCAlocation
                    idx_loop  = find(idxLocationDist==idxLocation(l),1,'first');
                    Mc{l}     = MMc(idx_loop);
                    idxc{l}   = idx4MMc(idx_loop);
                else
                    idx_loop  = idxLocationDist==idxLocation(l);
                    Mc{l}     = MMc(idx_loop);
                    idxc{l}   = idx4MMc(idx_loop);
                end
            end
        end
        function [M, idx, idxHat]               = regroupeModelMatrix(Mc, idxc)
            % Regroup together the parameters belonging to the same matrix
            % in order to update the model matrix at once
            Npu = size(Mc,2);
            if Npu==1
                M       = Mc{Npu};
                idx     = logical(idxc{Npu});
                idxHat  = ~idx;                
            else
                M   = 0;
                idx = 0;
                for u = 1:Npu
                    M   = bsxfun(@plus, M, Mc{u});
                    idx = bsxfun(@plus, idx, idxc{u});
                end
                idx     = logical(idx);
                idxHat  = ~idx;
            end
        end       
        function MM                             = MconstTimesMvar(Meval, Mc, Mtsc, idxc, idxLocation)
            Nlocation           = length(idxLocation);
            MMcfinal            = cell(1, Nlocation);
            idx4MMcfinal        = cell(1, Nlocation);
            idxHat4MMcfinal     = cell(1, Nlocation);
            MM                  = cell(1, Nlocation);
            for l = 1:Nlocation
                [MMcfinal{l}, idx4MMcfinal{l}, idxHat4MMcfinal{l}] = mc.regroupeModelMatrix(Mc{l}, idxc{l});
                MM{l} = mc.evalModelMatrix(Meval{l}, MMcfinal{l}, Mtsc{l},idxHat4MMcfinal{l});
            end
        end 
        function M                              = evalModelMatrix(Meval, Mvar, Mts, idxHat)
            if isempty(idxHat)
                M = Mvar;
            else
                Mtrans = bsxfun(@times, Meval, idxHat);
                Mvarts = bsxfun(@times, Mts, Mvar);
                M = bsxfun(@plus, Mtrans, Mvarts);
            end
        end
        
        % Preallocate the model matrices for all dataset and update new
        % paramerer values for paramProp
        function MMpreAlloc                     = modelMatrixPreAlloc(Nsteps, Nclass, Nobs, Nhs)
            A           = zeros(Nhs, Nhs, Nclass, Nsteps);
            C           = zeros(Nobs, Nhs, Nclass, Nsteps);
            R           = zeros(Nobs, Nobs, Nclass, Nsteps);
            Q           = zeros(Nhs, Nhs, Nclass, Nclass, Nsteps);
            Z           = zeros(Nclass, Nclass, Nsteps);
            MMpreAlloc  = {A;C;Q;R;Z};
        end                
        function paramProp                      = updateNewParamValues(model, paramUpdate, idxParamUpdate, idxTypeParam)
            % idxParamUpdate: parameter's indexes in model.param_properties
            % idxTypeParam : parameter's indexes in paramProp
            
            % Initialization
            paramProp   = model.parameter.prop;        
            paramValues = [model.param_properties{:,model.cParamValue}]';
            idxParamRef = [model.param_properties{:,model.cParamRef}]';
            paramValues = paramValues(idxParamRef);
            paramValues(idxParamUpdate) = paramUpdate;
            % Identify the standard deviation parameter type. They need to
            % be squared to assign into the covariance matrix
            idxSig      = strcmp(model.param_properties(:,model.cMatrix),'Q')...
                |strcmp(model.param_properties(:,model.cMatrix),'R');
            paramValues(idxSig) = paramValues(idxSig).^2;
            % Update parameter values in paramProp according to the
            % parameter type ex. PCA, DK, DH, SK parameters. Because each
            % parameter type might contain a vector of parameters
            idxTypeParam_unique = unique(idxTypeParam,'stable');
            for i = 1:length(idxTypeParam_unique)
                paramProp{idxTypeParam_unique(i),model.up.cParamValue} = paramValues(paramProp{idxTypeParam_unique(i),model.up.cParamIdx});
            end          
        end
        function paramProp                      = updateNewParamValues_V2(paramProp, paramUpdate, idxParamUpdate, idxTypeParam, paramProperties,...
                cPV, cPR, cMat,up_cPV, up_cPI)
            
            paramValues = [paramProperties{:, cPV}]';
            idxParamRef = [paramProperties{:, cPR}]';
            paramValues = paramValues(idxParamRef);
            paramValues(idxParamUpdate) = paramUpdate;
            
            % Identify the standard deviation parameter type. They need to
            % be squared to assign into the covariance matrix
            idxSig      = strcmp(paramProperties(:, cMat), 'Q')...
                |strcmp(paramProperties(:, cMat), 'R');
            paramValues(idxSig) = paramValues(idxSig).^2;
            
            % Update parameter values in paramProp according to the
            % parameter type ex. PCA, DK, DH, SK parameters. Because each
            % parameter type might contain a vector of parameters
            idxTypeParam_unique = unique(idxTypeParam, 'stable');
            for i = idxTypeParam_unique(:)'
                paramProp{i, up_cPV} = paramValues(paramProp{i, up_cPI});
            end
        end
        function [idxSK, idxDK, idxDH, idxNK,...
                idxDNK, idxTK]                  = idenIdxParam4DRC(model, paramProp)
            idxSK  = [paramProp{:,model.up.cSupInfo}]'==model.block.SK;
            idxDK  = [paramProp{:,model.up.cSupInfo}]'==model.block.DK;
            idxDH  = [paramProp{:,model.up.cSupInfo}]'==model.block.DH;
            idxNK  = [paramProp{:,model.up.cSupInfo}]'==model.block.NK;
            idxDNK = [paramProp{:,model.up.cSupInfo}]'==model.block.DNK;
            idxTK  = [paramProp{:,model.up.cSupInfo}]'==model.block.TK;
        end
        function MMuEval                        = getMMuEval(MMuEvalRef, unique_dtSteps_ref, unique_dtSteps, Aloc, Cloc, Qloc, Rloc, Zloc)
            [~,idx] = ismember(unique_dtSteps, unique_dtSteps_ref);
            MMuEval{Aloc} = MMuEvalRef{Aloc}(:,:,:,idx);
            MMuEval{Cloc} = MMuEvalRef{Cloc}(:,:,:,idx);
            MMuEval{Qloc} = MMuEvalRef{Qloc}(:,:,:,:,idx);
            MMuEval{Rloc} = MMuEvalRef{Rloc}(:,:,:,idx);
            MMuEval{Zloc} = MMuEvalRef{Zloc}(:,:,idx);
        end
            
        % Data initilization
        function [data, option]                 = dataInitialization(data, model, option)           
            
            %Identify time step index for the start and end of the training period
            option.training_start_idx   = find(abs(data.timestamps-data.timestamps(1)-option.training_period(1)+1)...
                ==min(abs(data.timestamps-data.timestamps(1)-option.training_period(1)+1)),1,'first');
            option.training_end_idx     = find(abs(data.timestamps-data.timestamps(1)-option.training_period(2)+1)...
                ==min(abs(data.timestamps-data.timestamps(1)-option.training_period(2)+1)),1,'first');
            option.nb_data_training     = option.training_end_idx-option.training_start_idx; % Define the number of data point in training period
            
            data.dt_mean    = mean(diff(data.timestamps)); % Mean time step length
            data.dt_steps   = diff(data.timestamps);
            
            unique_dt_steps = unique(data.dt_steps(option.training_start_idx:option.training_end_idx-1));
            counts_dt_steps = [unique_dt_steps,histc(data.dt_steps(option.training_start_idx:option.training_end_idx-1),unique_dt_steps)];
            data.dt_ref     = counts_dt_steps(find(counts_dt_steps(:,2)==max(counts_dt_steps(:,2)),1,'first'),1);  %Define the reference time step as the most frequent
            data.dt_steps   = [data.dt_ref;data.dt_steps]; % Define time step vector
            data.nb_steps   = length(data.timestamps);     % Store the number of time steps
            
            % If there exists the FAR block, the fictive observations will
            % be added to observation matrix. These fictive observations
            % are not used for log-likelihood-computation purpose
            if isfield(model,'labelFARobs')              
                labelFARobs            = cell2mat(model.labelFARobs);
                obsLabel               = cell(1,length(labelFARobs));
                for i = 1:length(labelFARobs)
                    obsLabel{i} = char(['y_{FAR,', '', num2str(labelFARobs(i)),'}']);
                end
                values                 = zeros(length(data.timestamps),length(labelFARobs));
                [~,idxObsUnique]       = unique(labelFARobs, 'stable');
                values(:,idxObsUnique) = data.values;
                obsLabel(idxObsUnique) = data.labels;
                data.labels            = obsLabel;
                data.values            = values;
            end
            
            % Data Forecast
            if isfield(model,'forecast')
                data = mc.mergeDataForecast(data, model);
            end                                 
        end
        function data                           = mergeDataForecast(data, model)
            if isfield(model.forecast,'dt')
                dtSteps = model.forecast.dt;
                Ndiv            = dtSteps*(1:round(model.forecast.duration/dtSteps));
                tsForecast      = data.timestamps(end)+ Ndiv;
                tsForecast      = tsForecast';
                dataForecast    = nan(length(tsForecast),size(data.values,2));
                data.values     = [data.values;dataForecast];
                data.timestamps = [data.timestamps;tsForecast];
                data.dt_steps   = [data.dt_ref; diff(data.timestamps)];
                data.nb_steps   = length(data.dt_steps);
            else
%                 dtSteps = data.dt_ref;
                fc_end  = find(data.timestampsRef==data.timestamps(end))+1;
                data.timestamps = data.timestampsRef;
                data.dt_steps   = [data.dt_ref; diff(data.timestamps)];
                data.nb_steps   = length(data.dt_steps);
                values          = data.valuesRef;
                values(fc_end:end,:)=nan;
                data.values     = values;
            end                        
            
            if isfield(data, 'ref')
                nanDataRef      = nan(size(data.ref,1),length(tsForecast));
                nanParamRef     = nan(size(data.paramRef,1),length(tsForecast));
                nanDataSref      = nan(length(tsForecast), size(data.S,2));
                
                data.ref        = [data.ref, nanDataRef];
                data.paramRef   = [data.paramRef,nanParamRef];
                data.paramTRref = [data.paramTRref, nanParamRef];
                data.S          = [data.S;nanDataSref ];
            end
            if isfield(data,'valuesRef')&&length(data.timestampsRef)<length(data.timestamps)
                valuesRef                               = data.values;
                valuesRef(1:length(data.timestampsRef)) = data.valuesRef;
                data.timestampRef                       = data.timestamps;
                data.valuesRef                          = valuesRef;
            elseif isfield(data,'valuesRef')&&length(data.timestampsRef)>length(data.timestamps)
                data.values(find(data.timestampsRef==data.timestamps(end))+1,:) = NaN;
                data.timestamps = data.timestampsRef(1:length(data.values));  
                data.dt_steps   = [data.dt_ref; diff(data.timestamps)];
                data.nb_steps   = length(data.dt_steps);
            end
            
            check=1;
        end
        function data                           = getDataAverage(data, varargin)
            
            % Initialization 
            dv    = datevec(data.timestamps);
            args  = varargin;
            nargs = length(args);
            uniTS = 0;
            for n = 1:2:nargs
                switch args{n}
                    case 'timeStep',  dt = args{n+1};
                    case 'hourly',    dt = 1/24;
                    case 'daily',     dt = 1;
                    case 'weekly',    dt = 7;
                    case 'biweekly',  dt = 14;
                    case 'quarterly', dt = 120;
                    case 'monthly',   dt = 30;
                    case 'yearly',    dt = 365;
                    case 'uniTS',     uniTS = args{n+1};
                    otherwise, error(['unrecognized argument' args{n}]);
                end
            end
           if dt == 1/24
               col = 4;
               [~,idx4timestamps,idx4values] = unique(dv(:,1:col),'rows');
           elseif dt == 1
               col = 3;
               [~,idx4timestamps,idx4values] = unique(dv(:,1:col),'rows');
           elseif dt == 7
               % [monday to sunday]
               wd               = rem(weekday(data.timestamps')+5, dt) + 1;
               idx4timestamps   = [true, diff(wd == 1) == 1]';
               idx4values       = cumsum(idx4timestamps);
           elseif dt == 30
               col = 2;
               [~,idx4timestamps,idx4values] = unique(dv(:,1:col),'rows');
           elseif dt == 120
               col              = 2;
               nm               = dt/30;
               qrt              = rem(dv(:,col)'+nm-1,nm) + 1;
               idx4timestamps   = [true diff(qrt == 1) == 1]';
               idx4values       = cumsum(idx4timestamps);
           elseif dt == 365
               col = 1;
               [~,idx4timestamps,idx4values] = unique(dv(:,1:col),'rows');
           else
               timeStamps    = data.timestamps;
               idx4values    = size(length(data.timestamps),1);    
               search        = 1;
               loop          = 1;
               idx           = true;
               while search
                   idx_loopRef   = find(abs(timeStamps-timeStamps(loop)-dt+1)==min(abs(timeStamps-timeStamps(loop)-dt+1)),1,'first');
                   if idx_loopRef == loop&&loop+1<length(data.timestamps)||idx_loopRef
                       idx_loopRef  = find(abs(timeStamps-timeStamps(loop+1)-dt+1)==min(abs(timeStamps-timeStamps(loop+1)-dt+1)),1,'first');                     
                   end
                   idx_loop      = false(1,idx_loopRef-loop);
                   idx_loop(end) = true;
                   loop          = idx_loopRef; 
                   idx           = [idx, idx_loop];
                   if length(idx)==length(data.timestamps)||loop>=length(timeStamps)
                       idx4values       = cumsum(idx)';
                       idx4timestamps   = idx';
                       break
                   end
               end
           end
           data.timestamps = data.timestamps(idx4timestamps);
           values          = zeros(length(data.timestamps),size(data.values,2));
           for obs = 1:size(data.values,2)
               values(:,obs) = accumarray(idx4values, data.values(:,obs),[],@nanmean);
           end
           data.values = values;
           if uniTS
               timstampsRef    = data.timestamps;
               data.timestamps = [data.timestamps(1):dt:data.timestamps(end)]';
               [idx_1,idx_2]   = ismember(timstampsRef,data.timestamps);
               values          = nan(length(data.timestamps),size(data.values,2));
               idx_2(idx_2==0) = [];
               values(idx_2,:)   = data.values(idx_1,:);
               data.values     = values;
           end
        end
        function data                           = intervalSelection4data(data, period)
            idxYearStart    = period(1);
            idxYearEnd      = period(2);
            dtvec           = datevec(data.timestamps);
            idxStart        = find(dtvec(:,1)==idxYearStart,1,'first');
            idxEnd          = find(dtvec(:,1)==idxYearEnd,1,'last');
            if isempty(idxStart)||isempty(idxEnd)
                error('The selected period is not valid')
            end
            data.values     = data.values(idxStart:idxEnd,:);
            data.timestamps = data.timestamps(idxStart:idxEnd);
        end
        function data                           = mergeDataset(dataRef, dataM)
            [idx_1, idx_2]  = ismember(dataM.timestamps, dataRef.timestamps);
            idx_2(idx_2==0) = [];
            values          = nan(length(dataRef.timestamps),size(dataRef.values,2)+1);
            values(:,1:size(dataRef.values,2))     = dataRef.values;
            values(idx_2,size(dataRef.values,2)+1) = dataM.values(idx_1);
            data.values     = values;
            data.timestamps = dataRef.timestamps;
        end
        function data                           = getDataAverage_V2(data, varargin)
            
            % Initialization 
            dv    = datevec(data.timestamps);
            args  = varargin;
            nargs = length(args);
            uniTS = 0;
            for n = 1:2:nargs
                switch args{n}
                    case 'timeStep',  dt = args{n+1};
                    case 'hourly',    dt = 1/24;
                    case 'daily',     dt = 1;
                    case 'weekly',    dt = 7;
                    case 'biweekly',  dt = 14;
                    case 'quarterly', dt = 120;
                    case 'monthly',   dt = 30;
                    case 'yearly',    dt = 365;
                    case 'uniTS',     uniTS = args{n+1};
                    otherwise, error(['unrecognized argument' args{n}]);
                end
            end
           if dt == 1/24
               col = 4;
               [~,idx4timestamps,~] = unique(dv(:,1:col),'rows');
               idx4values       = idx4timestamps;
           elseif dt == 1
               col = 3;
               [~,idx4timestamps,~] = unique(dv(:,1:col),'rows');
               idx4values       = idx4timestamps;
           elseif dt == 7
               % [monday to sunday]
               wd               = rem(weekday(data.timestamps')+5, dt) + 1;
               idx4timestamps   = [true, diff(wd == 1) == 1]';
               idx4values       = idx4timestamps;
           elseif dt == 30
               col = 2;
               [~,idx4timestamps,~] = unique(dv(:,1:col),'rows');
               idx4values = idx4timestamps;
           elseif dt == 120
               col              = 2;
               nm               = dt/30;
               qrt              = rem(dv(:,col)'+nm-1,nm) + 1;
               idx4timestamps   = [true diff(qrt == 1) == 1]';
               idx4values       = idx4timestamps;
           elseif dt == 365
               col = 1;
               [~,idx4timestamps,~] = unique(dv(:,1:col),'rows');
               idx4values       = idx4timestamps;
           else
               timeStamps    = data.timestamps;
               idx4values    = size(length(data.timestamps),1);    
               search        = 1;
               loop          = 1;
               idx           = true;
               while search
                   idx_loopRef   = find(abs(timeStamps-timeStamps(loop)-dt)==min(abs(timeStamps-timeStamps(loop)-dt)),1,'first');
                   if idx_loopRef == loop&&loop+1<length(data.timestamps)||idx_loopRef
                       idx_loopRef  = find(abs(timeStamps-timeStamps(loop+1)-dt)==min(abs(timeStamps-timeStamps(loop+1)-dt)),1,'first');                     
                   end
                   idx_loop      = false(1,idx_loopRef-loop);
                   idx_loop(end) = true;
                   loop          = idx_loopRef; 
                   idx           = [idx, idx_loop];
                   if length(idx)==length(data.timestamps)||loop>=length(timeStamps)                      
                       idx4timestamps   = idx';
                       idx4values       = idx4timestamps;
                       break
                   end
               end
           end
           data.timestamps = data.timestamps(idx4timestamps);
           values          = zeros(length(data.timestamps),size(data.values,2));
           for obs = 1:size(data.values,2)
               values(:,obs) = data.values(idx4values,:);
%                values(:,obs) = accumarray(idx4values, data.values(:,obs),[],@nanmean);
           end
           data.values = values;
           if uniTS
               timstampsRef    = data.timestamps;
               data.timestamps = [data.timestamps(1):dt:data.timestamps(end)]';
               [idx_1,idx_2]   = ismember(timstampsRef,data.timestamps);
               values          = nan(length(data.timestamps),size(data.values,2));
               idx_2(idx_2==0) = [];
               values(idx_2,:)   = data.values(idx_1,:);
               data.values     = values;
           end
        end
        
        % Simulated data
        function [data,option]                  = simData(data, model, option)
            % Initialization
            option.GPU = 0;
            Nsteps   = length(data.timestamps);           
            Nobs     = model.nb_obs;
            Nhs      = model.nb_HS;
            Nclass   = model.nb_class;
            Nsamples = 1;
            
            initX    = model.init.x;
            
            % Location
            cPM      = model.cPriorMu;
            cPS      = model.cPriorSig;          
            up_cSI   = model.up.cSupInfo;           
            up_cIML  = model.up.cIdxMatrixLoc;
            
            % Block for hidden states
            blockNK  = model.block.NK;
            blockDNK = model.block.DNK;
            
            Aloc     = model.A.location;
            Cloc     = model.C.location;
            Qloc     = model.Q.location;
            Rloc     = model.R.location;
            Zloc     = model.Z.location;
            
            % Parameter properties
            idxParamInit     = model.parameter.idxUkParamValue; % unknown paramerer's indexes in model.param_properties 
            idxParamPropInit = model.parameter.idxUkParamProp;  % type parameter's indexes in model.parameter.prop
            paramInit        = model.parameter.ukParamValue;
            paramTRinit      = model.parameter.ukParamValueTR;           
            muTR             = [model.param_properties{idxParamInit, cPM}]';
            sigTR            = [model.param_properties{idxParamInit, cPS}]';
            Nparam           = length(paramInit);
            matLoc           = [model.parameter.prop{:, up_cIML}]';
            uniqueMatLoc     = unique(matLoc,'stable');
            idxIdenMMloc     = cell(1, length(uniqueMatLoc));
            for i = 1:length(uniqueMatLoc)
                idxIdenMMloc{i} = find(matLoc==uniqueMatLoc(i))';
            end
            supInfo          = [model.parameter.prop{idxParamPropInit,up_cSI}]';
            
            % Transformation function
            funOR2TR         = cell(Nparam, 1);
            funTR2OR         = cell(Nparam, 1);            
            funGradTR2OR     = cell(Nparam, 1);
            clear i
            for i = 1:Nparam
                funOR2TR{i}     = model.parameter.transFun4ukParam{i,1};
                funTR2OR{i}     = model.parameter.transFun4ukParam{i,2};
                funGradTR2OR{i} = model.parameter.transFun4ukParam{i,3};
            end
            
            % Preallocation            
            A         = zeros(Nhs, Nhs, Nclass, Nsteps);
            C         = zeros(Nobs, Nhs, Nclass, Nsteps);
            Q         = zeros(Nhs, Nhs, Nclass, Nclass);
            R         = zeros(Nobs, Nobs, Nclass, Nsteps);
            Z         = zeros(Nclass, Nclass, Nsteps);
            x         = zeros(Nhs, Nsteps);
            y         = zeros(Nsteps, Nobs);
                      
                                  
            S         = zeros(Nsteps, Nclass);
            
            paramSamples   = zeros(Nparam, Nsamples, Nsteps);
            paramTRsamples = zeros(Nparam, Nsamples, Nsteps);            
            
            % Initialize the state probablities
            initIdxS = 1; 
            
            for t = 1:Nsteps
                if t>1
                    initX        = x(:,t-1);
                    prevX        = repmat(initX, [1 1 1 1 Nsamples]);
                    idxSprev     = idxS;                  
                else
                    idxSprev     = initIdxS;
                    paramPropNew = model.parameter.prop;
                    prevX        = initX(:,:,idxSprev,1);                   
                end
                % Model parameters

                if t == 1
                    paramTRsamples(:,:,t) = paramTRinit + mvnrnd(zeros(Nparam,1), diag(sigTR.^2))';
                else
                    paramTRsamples(:,:,t) = paramTRsamples(:,:,t-1) + mvnrnd(zeros(Nparam,1), diag(sigTR.^2))'; 
                end
                
                for n = 1:Nparam
                    paramSamples(n,:,t)   = funTR2OR{n}(paramTRsamples(n,t));
                end
%                 if t<4000||t>4500
%                     paramSamples([3;4],1,t)=1E-6;
%                 end
                
                % Update non-periodic-kernel-regression properties for
                % time step t
                if ~isempty(supInfo)&&(any(supInfo==blockNK)||any(supInfo==blockDNK))
                    CP      = paramPropNew{up_cSF}{1}; % CP
                    PM      = paramPropNew{up_cSF}{2}; % PM
                    K_prod  = paramPropNew{up_cSF}{3}; % K_prop
                    cpOpt   = paramPropNew{up_cSF}{4}; % cpOpt
                else
                    CP      = [];
                    PM      = [];
                    K_prod  = [];
                    cpOpt   = 0;
                end
                [paramProp, MMuEvalSamples, MMtsSamples] = opt.getParamProp41ts(t, paramSamples(:,:,t), data, model, CP, PM, K_prod, cpOpt);                              
                
                % Update model matrices for time step t
                [MMnew, ~, paramPropNew] = mc.updateModelMatrix4RBPF(model, paramProp, MMuEvalSamples, MMtsSamples, idxIdenMMloc,...
                uniqueMatLoc, option.GPU);
            
                A(:,:,:,t)   = MMnew{Aloc};
                C(:,:,:,t)   = MMnew{Cloc};
                Q(:,:,:,:,t) = MMnew{Qloc};
                R(:,:,:,t)   = MMnew{Rloc};
                Z(:,:,t)     = MMnew{Zloc};
                
                % Hidden states & observations
                idxS         = length(find(cumsum(Z(idxSprev,:,t)')<rand))+1;
                S(t,idxS)    = 1 ;
                x(:,t)       = A(:,:,idxS,t)*prevX + mvnrnd(zeros(Nhs,1),Q(:,:,idxSprev,idxS,t))';
                y(t,:)       = C(:,:,idxS,t)*x(:,t) + mvnrnd(zeros(Nobs,1),R(:,:,idxS,t))';    
            end
            data.values     = y;
            data.ref        = x';
            data.S          = S;
            data.paramRef   = paramSamples;
            data.paramTRref = paramTRsamples;
            
            % Data Initialization
            [data, option] = mc.dataInitialization(data, model, option);
            plot(data.timestamps, data.values);
        end
%         function data                           = artificialAnomaly(data, model, option)
%             N_a                     = 2;
%             N_1                     = 2*30;
%             N_2                     = 2*30;
%             anomalySlope            = [0.01,0.01];
%             timeStart               = [round(length(data.timestamps)/3); round(length(data.timestamps)/3)+N_1+5*365];
%             timeEnd                 = [round(length(data.timestamps)/3)+N_1; round(length(data.timestamps)/3)+N_1+5*365+N_2];
%             estim                   = estimation.state_estimation(data, model, option,'smooth',1);
%             baseline                = estim.x(1,:)';
%             ynew                    = data.values(:,1)-baseline;
%             data.S                  = [ones(length(data.timestamps),1) zeros(length(data.timestamps),1)];
%             for i=1:N_a
%                 baseline                          = mc.artificialAnomalyType(baseline, data.timestamps, timeStart(i), timeEnd(i), anomalySlope(i), 1);
%                 data.S(timeStart(i):timeEnd(i),1) = 0;
%                 data.S(timeStart(i):timeEnd(i),2) = 1;
%             end
%             data.values             = ynew + baseline; 
%             anomaly.slope           = anomalySlope;
%             anomaly.timeStart       = timeStart;
%             anomaly.timeEnd         = timeEnd;
%             data.anomaly            = anomaly;
%         end
        function data                           = artificialAnomaly(data, model, option)
            N_a                     = 2;
            N_1                     = 2*30;
            N_2                     = 2*30;
            anomalySlope            = [0.015,0.015];
            timeStart               = [round(length(data.timestamps)/3); round(length(data.timestamps)/3)+N_1+5*365];
            timeEnd                 = [round(length(data.timestamps)/3)+N_1; round(length(data.timestamps)/3)+N_1+5*365+N_2];
%             estim                   = estimation.state_estimation(data, model, option,'smooth',1);
            baseline                = zeros(length(data.timestamps),1);
            data.S                  = [ones(length(data.timestamps),1) zeros(length(data.timestamps),1)];
            for i=1:N_a
                baseline                          = mc.artificialAnomalyType(baseline, data.timestamps, timeStart(i), timeEnd(i), anomalySlope(i), 1);
                data.S(timeStart(i):timeEnd(i),1) = 0;
                data.S(timeStart(i):timeEnd(i),2) = 1;
            end
            data.values             = data.values + baseline; 
            anomaly.slope           = anomalySlope;
            anomaly.timeStart       = timeStart;
            anomaly.timeEnd         = timeEnd;
            data.anomaly            = anomaly;
        end
        function x                              = artificialAnomalyType(x, timestamps, ts, te, anomalySlope, label_AA)
            if label_AA==1
                x = [x(1:ts);x(ts+1:te)-anomalySlope*(timestamps(ts+1:te)-timestamps(ts));x(te+1:end)-anomalySlope*(te-ts)];
            end
        end
        
        % Identify the fictive AR-components
        function [labelFARobs_c, idxFAR4R]      = getLabelFARobs(model)
            Nobs          = model.nb_obs;
            labelFARobs_c = cell(1, Nobs);
            for obs = 1:Nobs
                blockComp        = cell2mat(model.components.block{1}(1,obs));
                N_FAR            = sum(blockComp==model.block.FAR);
                labelFARobs_c{obs} = obs*ones(1, N_FAR+1);
            end           
            labelFARobs_a         = cell2mat(labelFARobs_c);           
            if length(labelFARobs_a)==Nobs
                labelFARobs_c = [];
                idxFAR4R    = [];
            else
                [~,idxObsUnique]    = unique(labelFARobs_a,'stable');
                Rvec                = ones(1, length(labelFARobs_a));
                Rvec(idxObsUnique)  = 0;
                idxFAR4R            = diag(Rvec);
                idxFAR4R            = logical(idxFAR4R);
            end
        end 
        function [paramValues_phiAR,...
                paramValues_SigAR,...
                idxFictiveObs, dtOverDtRef]     = getParamValues4FAR(model, paramProp, idxParamUpdate)           
            labelFARobs               = model.labelFARobs;
            idxParamRef               = [model.param_properties{:, model.cParamRef}]';
            idxParamRef_u             = unique(idxParamRef,'stable');
            eBlock                    = [model.param_properties{:,model.cEncoder}]';
            eParamName                = model.param_properties(:, model.cParamName);
            eObservation              = model.param_properties(:, model.cObservation);           
            idxParam4model            = model.parameter.idxParam4model;
            labelFARobs_a             = cell2mat(labelFARobs);
            [~,idxRealObs]            = unique(labelFARobs_a, 'stable');
            idxFictiveObs             = ones(size(labelFARobs_a));
            idxFictiveObs(idxRealObs) = 0;
            idxFictiveObs             = cumsum(idxFictiveObs);
            
            
            % Get the parameter values for \phi_AR et \sigma_AR
            [eIdxParamPhiAR, ~]        = mc.getIdx4specificParam('\phi', model.block.AR, idxParamRef_u, eBlock, eParamName, eObservation);
            [eIdxParamSigAR, ~]        = mc.getIdx4specificParam('\sigma_w', model.block.AR, idxParamRef_u, eBlock, eParamName, eObservation);
            if isempty(idxParamUpdate)
                idxParamUpdate         = idxParamRef_u;
            end
            [~,idxOverlap_phiAR]       = intersect(eIdxParamPhiAR,idxParamUpdate);
            [~,idxOverlap_sigAR]       = intersect(eIdxParamSigAR,idxParamUpdate);
            idx4paramValues            = union(idxOverlap_phiAR, idxOverlap_sigAR);
            idxParamPhiAR              = eIdxParamPhiAR(idx4paramValues);
            idxParamSigAR              = eIdxParamSigAR(idx4paramValues);
            idxFictiveObs_temp         = ismember(idxFictiveObs, idx4paramValues);
            idxFictiveObs(idxFictiveObs_temp)  = 1;
            idxFictiveObs(~idxFictiveObs_temp) = 0;
            idxFictiveObs              = logical(diag(idxFictiveObs));
                      
            if ~isempty(paramProp)
                eParamValues          = mc.buildParamValueMatrix(model, paramProp);
                idxParamProp4AR_phi   = model.parameter.idxUkParamProp(model.parameter.idxUkParamValue==idxParamPhiAR(1));
                idxParamProp4AR_sig   = model.parameter.idxUkParamProp(model.parameter.idxUkParamValue==idxParamSigAR(1));
                idxParamProp4AR       = union(idxParamProp4AR_phi,idxParamProp4AR_sig);
                dtOverDtRef           = paramProp{idxParamProp4AR(1),model.up.cTimestep};
            else
                eParamValues          = [model.param_properties{:, model.cParamValue}]';
                eParamValues          = eParamValues(idxParamRef);
                dtOverDtRef           = [];               
            end
            paramValues_phiAR          = mc.getParamValues(idxParamPhiAR, idxParamRef, idxParam4model, eParamValues);
            paramValues_SigAR          = mc.getParamValues(idxParamSigAR, idxParamRef, idxParam4model, eParamValues);                               
        end
        function Rnew                           = evalR4FAR(Rold, phiAR, sigAR, dtOverDtRef, idxFicR)
            dimR                = size(Rold); 
            idxFAR4R            = logical(idxFicR);
            idxParamFAR_R       = repmat(idxFAR4R,[1 1 dimR(3:end)]);
            varAR               = sigAR.^2;
            phiARsq             = phiAR.^2;
            varR                = ((varAR.*(dtOverDtRef'))./(1-phiARsq.^(dtOverDtRef')));
            Rold(idxParamFAR_R) = varR;
            Rnew                = Rold;
        end
        function [idxParam, idxObs]             = getIdx4specificParam(paramName, block, idxParamUpdate, eBlock, eParamName, eObservation)
            Nparam            = size(eBlock,1);
            idxPUV            = zeros(Nparam,1);
            idxPUV(idxParamUpdate) = idxParamUpdate;
            idxSearch         = eBlock==block&strcmp(eParamName, paramName);
            idxParam          = idxPUV(idxSearch);
            idxObs            = str2double(eObservation(idxParam));
        end
        function paramValues                    = getParamValues(idxParam, idxParamRef, idxParam4model, eParamValues)
            Nclass          = size(idxParam4model,1);
            paramValues     = cell(Nclass,1);                       
            for j = 1:Nclass
                idxParam_loop     = idxParam(ismember(idxParam, idxParam4model{j}));
                idxParamFromRef   = idxParamRef(ismember(idxParamRef,idxParam_loop));
                idxParamFromOrder = find(ismember(idxParamRef,idxParam_loop));
                [~,idxS]          = sort(idxParamFromOrder,'ascend');
                paramValues{j}    = eParamValues(idxParamFromRef(idxS),:);                
            end            
            paramValues = vertcat(paramValues{:,1});
        end
        function paramValueMatrix               = buildParamValueMatrix(model, paramProp)
            Nsamples                           = size(paramProp{1,model.up.cParamValue},2);
            paramValueMatrixNew                = vertcat(paramProp{:,model.up.cParamValue});
            paramValuesIdx                     = vertcat(paramProp{:,model.up.cParamIdx});
            idxSig                             = strcmp(model.param_properties(paramValuesIdx, model.cMatrix), 'Q')...
                |strcmp(model.param_properties(paramValuesIdx, model.cMatrix), 'R');
            paramValueMatrixNew(idxSig,:)      = paramValueMatrixNew(idxSig,:).^(1/2);
            
            paramValues                        = [model.param_properties{:, model.cParamValue}]';
            idxParamRef                        = [model.param_properties{:, model.cParamRef}]';
            paramValues                        = paramValues(idxParamRef);
            paramValueMatrix                   = repmat(paramValues,[1 Nsamples]);
            paramValueMatrix(paramValuesIdx,:) = paramValueMatrixNew;
        end
    end
end