function [cropImg, mask] = segmentFruit(I)
    % Segment fruit object using edge detection + morphology
    
    if size(I,3)==1
        I = cat(3,I,I,I);
    end
    
    gray = rgb2gray(I);
    gray = im2uint8(im2double(gray));
    
    % Detect edges
    edges = edge(gray,'Canny');
    
    % Morphology clean-up
    se = strel('disk',8);
    mask = imclose(imfill(imopen(imdilate(edges,se),se),'holes'), strel('disk',10));
    
    % keep largest connected component
    cc = bwconncomp(mask);
    if cc.NumObjects == 0
        cropImg = I;  % fallback
        return;
    end
    
    stats = regionprops(cc,'Area','BoundingBox');
    [~, idx] = max([stats.Area]);
    bb = round(stats(idx).BoundingBox);
    
    % only largest object
    mask = ismember(labelmatrix(cc), idx);
    
    % crop image 
    cropImg = imcrop(I, bb);
end

