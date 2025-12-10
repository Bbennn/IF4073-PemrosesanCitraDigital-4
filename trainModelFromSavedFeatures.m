function trainModelFromSavedFeatures(featuresFile, modelOutFile)
    if nargin < 2 || isempty(modelOutFile), modelOutFile = 'fruitModel_fromFeatures.mat'; end
    
    % load features
    fprintf('=== Load features from: %s ===\n', featuresFile);
    t0 = tic;
    S = load(featuresFile);
    if ~isfield(S,'X') || ~isfield(S,'Y') || ~isfield(S,'classNames')
        error('File fitur harus berisi variabel X, Y, dan classNames.');
    end
    X = S.X; Y = S.Y; classNames = S.classNames;
    fprintf('Loaded %d samples in %.2f s\n', size(X,1), toc(t0));
    
    % check
    nSamples = size(X,1);
    if nSamples ~= numel(Y)
        error('Jumlah baris X (%d) berbeda dengan panjang Y (%d).', nSamples, numel(Y));
    end
    nClasses = numel(classNames);
    if nClasses < 2
        error('Dibutuhkan minimal 2 kelas (classNames length = %d).', nClasses);
    end
    
    % If Y values are not 1:nClasses, attempt to remap to 1:nClasses
    uniqueY = unique(Y);
    if ~isequal(sort(uniqueY(:))', 1:nClasses)
        fprintf('Info: Y label tidak berupa 1..N. Melakukan remapping label sesuai classNames.\n');
        % try mapping: if classNames are strings and Y are strings, should have used index mapping earlier.
        % map uniqueY to 1..numel(uniqueY) (to preserve original ordering of uniqueY)
        mapVals = uniqueY(:);
        newY = zeros(size(Y));
        for i = 1:numel(mapVals)
            newY(Y == mapVals(i)) = i;
        end
        Y = newY;
        if numel(uniqueY) ~= nClasses
            warning('Jumlah label unik di Y (%d) berbeda dengan classNames (%d). Pastikan classNames sesuai.', numel(uniqueY), nClasses);
        end
    else
        % do nothing
        % Y = 1..nClasses
    end
    
    % Normalize
    fprintf('\n=== Normalizing features ===\n');
    t1 = tic;
    mu = mean(X,1);
    sigma = std(X,[],1);
    sigma(sigma==0) = 1;
    Xn = (X - mu) ./ sigma;
    fprintf('Normalization done in %.2f s\n', toc(t1));
    
    % Train ECOC model
    fprintf('\n=== Training SVM ECOC (RBF) ===\n');
    t2 = tic;
    t = templateSVM('KernelFunction','rbf','KernelScale','auto','Standardize',false);
    
    % compute expected number of binary learners for one-vs-one
    numLearners = nchoosek(nClasses,2);
    fprintf('ECOC one-vs-one with %d classes -> %d binary learners\n', nClasses, numLearners);
    
    try
        Mdl = fitcecoc(Xn, Y, 'Learners', t, 'Coding', 'onevsone', 'ClassNames', 1:nClasses);
    catch ME
        error('fitcecoc gagal: %s', ME.message);
    end
    fprintf('ECOC training finished in %.2f s\n', toc(t2));
    
    % Cross-validation
    fprintf('\n=== Cross-validation (5-fold) ===\n');
    t3 = tic;
    try
        CVMdl = crossval(Mdl,'KFold',5);
        loss = kfoldLoss(CVMdl);
        fprintf('5-fold CV loss = %.4f (accuracy = %.2f%%) — computed in %.2f s\n', loss, (1-loss)*100, toc(t3));
    catch cvErr
        warning(cvErr.identifier,'Cross-val gagal: %s', cvErr.message);
    end
    
    % save model + normalization + classNames
    fprintf('\n=== Saving model to %s ===\n', modelOutFile);
    t4 = tic;
    save(modelOutFile, 'Mdl', 'mu', 'sigma', 'classNames', '-v7.3');
    fprintf('Saved in %.2f s\n', toc(t4));
    fprintf('Model saved. Training complete.\n');
end
