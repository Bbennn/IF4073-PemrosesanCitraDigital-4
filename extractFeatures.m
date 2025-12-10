function feat = extractFeatures(I)
    % Extract features from image

    % check
    if isempty(I)
        error('Input image is empty.');
    end
    
    if size(I,3) == 1
        I = repmat(I, [1 1 3]);
    end
    
    if ~isa(I,'double')
        I = im2double(I);
    end
    
    % normalize size for consistency
    I = imresize(I, [200 200]);
    
    % denoise
    I2 = medfilt3(I,[3 3 1]);
    
    % Color moments (mean & std)
    R = I2(:,:,1); G = I2(:,:,2); B = I2(:,:,3);
    meanRGB = [mean(R(:)), mean(G(:)), mean(B(:))];
    stdRGB  = [std(R(:)), std(G(:)), std(B(:))];
    
    % HSV histograms (16 bins)
    HSV = rgb2hsv(I2);
    H = HSV(:,:,1); S = HSV(:,:,2); V = HSV(:,:,3);
    nbins = 16;
    hHist = imhist(H, nbins); hHist = hHist / sum(hHist + eps);
    sHist = imhist(S, nbins); sHist = sHist / sum(sHist + eps);
    vHist = imhist(V, nbins); vHist = vHist / sum(vHist + eps);
    
    % GLCM texture features on grayscale
    Igray = im2uint8(rgb2gray(I2));
    offsets = [0 1; -1 1; -1 0; -1 -1];
    glcms = graycomatrix(Igray,'Offset',offsets,'Symmetric',true);
    stats = graycoprops(glcms,{'Contrast','Correlation','Energy','Homogeneity'});
    contrast = mean(stats.Contrast);
    correlation = mean(stats.Correlation);
    energy = mean(stats.Energy);
    homogeneity = mean(stats.Homogeneity);
    
    % HOG features
    try
        cellSize = [16 16];
        hog = extractHOGFeatures(Igray,'CellSize',cellSize);
    catch
        hog = []; % fallback if function unavailable
    end
    
    % Region shape features (area, perimeter, eccentricity) 
    bw = imbinarize(Igray, 'adaptive', 'Sensitivity', 0.4);
    bw = imfill(bw,'holes');
    bw = imopen(bw, strel('disk',3));
    cc = bwconncomp(bw);
    if cc.NumObjects > 0
        props = regionprops(cc, 'Area','Perimeter','Eccentricity','MajorAxisLength','MinorAxisLength');
        % choose largest
        areas = [props.Area];
        [~, idx] = max(areas);
        sfeat = [props(idx).Area, props(idx).Perimeter, props(idx).Eccentricity, ...
                 props(idx).MajorAxisLength, props(idx).MinorAxisLength];
    else
        sfeat = zeros(1,5);
    end
    
    % Combine features
    feat = [meanRGB, stdRGB, hHist', sHist', vHist', contrast, correlation, energy, homogeneity, sfeat, hog];
    
    % Ensure row double
    feat = double(feat(:))';
end
