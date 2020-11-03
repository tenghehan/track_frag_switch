
def tlwh_to_xyxy(bbox):
    top = bbox[0]
    left = bbox[1]
    width = bbox[2]
    height = bbox[3]

    x1 = max(bbox[0], 0)
    y1 = max(bbox[1], 0)
    x2 = bbox[0] + bbox[2]
    y2 = bbox[1] + bbox[3]

    return [x1, y1, x2, y2]


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def bbox_iou(track_bbox, gt_bbox):
    track_bbox = tlwh_to_xyxy(track_bbox)
    gt_bbox = tlwh_to_xyxy(gt_bbox)

    iou = bb_intersection_over_union(track_bbox, gt_bbox)

    return iou
