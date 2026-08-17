import numpy as np

def compute_iou(boxA, boxB):
    """
    Computes Intersection over Minimum Area (IoM) instead of standard IoU.
    This is extremely effective for Text Detection to remove small duplicate boxes 
    that are fully contained inside larger boxes.
    Boxes are in format [xmin, ymin, xmax, ymax].
    """
    xA = max(float(boxA[0]), float(boxB[0]))
    yA = max(float(boxA[1]), float(boxB[1]))
    xB = min(float(boxA[2]), float(boxB[2]))
    yB = min(float(boxA[3]), float(boxB[3]))

    # Compute intersection area
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    if interArea == 0.0:
        return 0.0

    # Compute both areas
    boxAArea = float(boxA[2] - boxA[0]) * float(boxA[3] - boxA[1])
    boxBArea = float(boxB[2] - boxB[0]) * float(boxB[3] - boxB[1])

    # Compute IoM (Intersection over Minimum Area)
    minArea = min(boxAArea, boxBArea)
    if minArea <= 0:
        return 0.0
        
    iom = interArea / minArea
    return iom

def poly_to_bbox(poly):
    """
    Converts a 4-point polygon to a bounding box [xmin, ymin, xmax, ymax] as floats.
    """
    arr = np.array(poly, dtype=np.float64)
    xmin = float(np.min(arr[:, 0]))
    ymin = float(np.min(arr[:, 1]))
    xmax = float(np.max(arr[:, 0]))
    ymax = float(np.max(arr[:, 1]))
    return [xmin, ymin, xmax, ymax]

def poly_nms(polys, iou_threshold=0.3):
    """
    Applies Non-Maximum Suppression to a list of polygons to remove duplicates.
    Since we don't have prediction confidence scores from PaddleOCR dt_polys,
    we'll just sort them by area (preferring larger bounding boxes) and suppress smaller overlaps.
    
    Args:
        polys: List of 4-point polygons.
        iou_threshold: Threshold for IoU overlap.
        
    Returns:
        List of filtered polygons.
    """
    if len(polys) == 0:
        return []

    # Convert all polys to bboxes
    bboxes = [poly_to_bbox(p) for p in polys]
    
    # Compute areas
    areas = [float(box[2] - box[0]) * float(box[3] - box[1]) for box in bboxes]
    
    # Sort indices by area (descending)
    idxs = np.argsort(areas)[::-1]
    
    keep_indices = []
    
    while len(idxs) > 0:
        # Pick the largest box
        current_idx = idxs[0]
        keep_indices.append(current_idx)
        
        # Compare with the rest
        rest_idxs = idxs[1:]
        
        ious = []
        for r_idx in rest_idxs:
            iou = compute_iou(bboxes[current_idx], bboxes[r_idx])
            ious.append(iou)
            
        ious = np.array(ious)
        
        # Keep only the boxes that have IoU < threshold with the current box
        keep_mask = ious < iou_threshold
        idxs = rest_idxs[keep_mask]
        
    return [np.array(polys[i]).tolist() for i in keep_indices]
