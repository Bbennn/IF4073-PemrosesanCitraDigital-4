function extractAndSaveFeatures(datasetDir, outFile, doSegmentation, forceRecompute, percentage)
    if nargin < 3 || isempty(doSegmentation), doSegmentation = true; end
    if nargin < 4 || isempty(forceRecompute), forceRecompute = false; end
    if nargin < 5 || isempty(percentage), percentage = 1; end
    
    % handle percentage input (if given 80 -> 0.8)
    if percentage > 1
        if percentage > 100
            error('percentage > 100 tidak valid.');
        end
        percentage = percentage / 100;
    end
    
    if exist(outFile,'file') && ~forceRecompute
        fprintf('File %s sudah ada — skip ekstraksi.\n', outFile);
        return;
    end
    
    % read classes
    d = dir(datasetDir);
    d = d([d.isdir]);
    d = d(~ismember({d.name},{'.','..'}));
    classNames = {d.name};
    nClasses = numel(classNames);
    
    X_cells = {};
    Y = [];
    filePaths = {};
    
    fprintf('Mulai ekstraksi "%s"\n', datasetDir);
    fprintf('Segmentation: %d | percentage: %.2f%%\n', doSegmentation, percentage*100);
    
    rng('shuffle'); % shuffle randomness
    
    for c = 1:nClasses
        folder = fullfile(datasetDir, classNames{c});
        imgs = [dir(fullfile(folder,'*.jpg')); dir(fullfile(folder,'*.jpeg')); dir(fullfile(folder,'*.png')); dir(fullfile(folder,'*.bmp'))];
    
        nImgs = numel(imgs);
        if nImgs == 0
            fprintf('Class %s tidak ada gambar, skip.\n', classNames{c});
            continue;
        end
    
        % shuffle gambar
        imgs = imgs(randperm(nImgs));
    
        % jumlah file yang digunakan
        nUse = max(1, round(nImgs * percentage));
    
        fprintf('\nClass %d/%d: %s — ambil %d dari %d file\n', ...
            c, nClasses, classNames{c}, nUse, nImgs);
    
        % progress checkpoints (0,20,40,60,80,100%)
        checkpoints = round(linspace(1, nUse, 6));
        nextCP = 1;
    
        for k = 1:nUse
            imgPath = fullfile(folder, imgs(k).name);
    
            try
                I = imread(imgPath);
    
                if doSegmentation
                    try
                        [cropImg, ~] = segmentFruit(I);
                        if isempty(cropImg), cropImg = I; end
                    catch
                        cropImg = I;
                    end
                    feat = extractFeatures(cropImg);
                else
                    feat = extractFeatures(I);
                end
    
                X_cells{end+1,1} = feat;
                Y(end+1,1) = c;
                filePaths{end+1,1} = imgPath;
    
            catch ME
                warning('Gagal memproses %s: %s', imgPath, ME.message);
            end
    
            % show progress
            if k == checkpoints(nextCP)
                percentDone = (nextCP-1) * 20;
                fprintf('   Progress %s: %d%% (%d/%d)\n', classNames{c}, percentDone, k, nUse);
                nextCP = min(nextCP + 1, numel(checkpoints));
            end
        end
    
        fprintf('   Progress %s: 100%% selesai.\n', classNames{c});
    end
    
    % Convert cell to matrix
    len0 = numel(X_cells{1});
    X = zeros(numel(X_cells), len0);
    for i = 1:numel(X_cells)
        X(i,:) = X_cells{i};
    end
    
    % global shuffle
    perm = randperm(size(X,1));
    X = X(perm,:);  Y = Y(perm);  filePaths = filePaths(perm);
    
    save(outFile, 'X', 'Y', 'filePaths', 'classNames', '-v7.3');
    fprintf('\nSelesai. Total %d sampel disimpan ke "%s".\n', size(X,1), outFile);
end
