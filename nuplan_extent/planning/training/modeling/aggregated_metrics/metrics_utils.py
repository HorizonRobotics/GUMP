import numpy as np
from numpy.typing import NDArray
from shapely.geometry import LineString
from typing import List, Tuple, Union
from scipy.spatial import distance
import torch
from numba import jit



def chamfer_distance(line1: NDArray, line2: NDArray) -> float:
    ''' Calculate chamfer distance between two lines. Make sure the 
    lines are interpolated.

    Args:
        line1 (array): coordinates of line1
        line2 (array): coordinates of line2
    
    Returns:
        distance (float): chamfer distance
    '''
    
    dist_matrix = distance.cdist(line1, line2, 'euclidean')
    dist12 = dist_matrix.min(-1).sum() / len(line1)
    dist21 = dist_matrix.min(-2).sum() / len(line2)

    return (dist12 + dist21) / 2

def frechet_distance(line1: NDArray, line2: NDArray) -> float:
    ''' Calculate frechet distance between two lines. Make sure the 
    lines are interpolated.

    Args:
        line1 (array): coordinates of line1
        line2 (array): coordinates of line2
    
    Returns:
        distance (float): frechet distance
    '''
    
    raise NotImplementedError

def chamfer_distance_batch(pred_lines, gt_lines):
    ''' Calculate chamfer distance between two group of lines. Make sure the 
    lines are interpolated.

    Args:
        pred_lines (array or tensor): shape (m, num_pts, 2 or 3)
        gt_lines (array or tensor): shape (n, num_pts, 2 or 3)
    
    Returns:
        distance (array): chamfer distance
    '''
    _, num_pts, coord_dims = pred_lines.shape
    
    if not isinstance(pred_lines, torch.Tensor):
        pred_lines = torch.tensor(pred_lines)
    if not isinstance(gt_lines, torch.Tensor):
        gt_lines = torch.tensor(gt_lines)
    dist_mat = torch.cdist(pred_lines.view(-1, coord_dims), 
                    gt_lines.view(-1, coord_dims), p=2) 
    # (num_query*num_points, num_gt*num_points)
    dist_mat = torch.stack(torch.split(dist_mat, num_pts)) 
    # (num_query, num_points, num_gt*num_points)
    dist_mat = torch.stack(torch.split(dist_mat, num_pts, dim=-1)) 
    # (num_gt, num_q, num_pts, num_pts)

    dist1 = dist_mat.min(-1)[0].sum(-1)
    dist2 = dist_mat.min(-2)[0].sum(-1)

    dist_matrix = (dist1 + dist2).transpose(0, 1) / (2 * num_pts)
    
    return dist_matrix.numpy()

def instance_match(pred_lines: NDArray, 
                   scores: NDArray, 
                   gt_lines: NDArray, 
                   thresholds: Union[Tuple, List], 
                   metric: str='chamfer') -> List:
    """Compute whether detected lines are true positive or false positive.

    Args:
        pred_lines (array): Detected lines of a sample, of shape (M, INTERP_NUM, 2 or 3).
        scores (array): Confidence score of each line, of shape (M, ).
        gt_lines (array): GT lines of a sample, of shape (N, INTERP_NUM, 2 or 3).
        thresholds (list of tuple): List of thresholds.
        metric (str): Distance function for lines matching. Default: 'chamfer'.

    Returns:
        list_of_tp_fp (list): tp-fp matching result at all thresholds
    """

    if metric == 'chamfer':
        distance_fn = chamfer_distance

    elif metric == 'frechet':
        distance_fn = frechet_distance
    
    else:
        raise ValueError(f'unknown distance function {metric}')

    num_preds = pred_lines.shape[0]
    num_gts = gt_lines.shape[0]

    # tp and fp
    tp_fp_list = []
    tp = np.zeros((num_preds), dtype=np.float32)
    fp = np.zeros((num_preds), dtype=np.float32)

    # if there is no gt lines in this sample, then all pred lines are false positives
    if num_gts == 0:
        fp[...] = 1
        for thr in thresholds:
            tp_fp_list.append((tp.copy(), fp.copy()))
        return tp_fp_list
    
    if num_preds == 0:
        for thr in thresholds:
            tp_fp_list.append((tp.copy(), fp.copy()))
        return tp_fp_list

    assert pred_lines.shape[1] == gt_lines.shape[1], \
        "sample points num should be the same"

    # distance matrix: M x N
    matrix = np.zeros((num_preds, num_gts))

    # for i in range(num_preds):
    #     for j in range(num_gts):
    #         matrix[i, j] = distance_fn(pred_lines[i], gt_lines[j])
    
    matrix = chamfer_distance_batch(pred_lines, gt_lines)
    # for each det, the min distance with all gts
    matrix_min = matrix.min(axis=1)

    # for each det, which gt is the closest to it
    matrix_argmin = matrix.argmin(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-scores)

    # match under different thresholds
    for thr in thresholds:
        tp = np.zeros((num_preds), dtype=np.float32)
        fp = np.zeros((num_preds), dtype=np.float32)

        gt_covered = np.zeros(num_gts, dtype=bool)
        for i in sort_inds:
            if matrix_min[i] <= thr:
                matched_gt = matrix_argmin[i]
                if not gt_covered[matched_gt]:
                    gt_covered[matched_gt] = True
                    tp[i] = 1
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        
        tp_fp_list.append((tp, fp))

    return tp_fp_list

def interp_fixed_num(vector: NDArray, 
                        num_pts: int) -> NDArray:
    ''' Interpolate a polyline.
    
    Args:
        vector (array): line coordinates, shape (M, 2)
        num_pts (int): 
    
    Returns:
        sampled_points (array): interpolated coordinates
    '''
    line = LineString(vector)
    distances = np.linspace(0, line.length, num_pts)
    sampled_points = np.array([list(line.interpolate(distance).coords) 
        for distance in distances]).squeeze()
    
    return sampled_points

def interp_fixed_dist(vector: NDArray,
                        sample_dist: float) -> NDArray:
    ''' Interpolate a line at fixed interval.
    
    Args:
        vector (LineString): vector
        sample_dist (float): sample interval
    
    Returns:
        points (array): interpolated points, shape (N, 2)
    '''
    line = LineString(vector)
    distances = list(np.arange(sample_dist, line.length, sample_dist))
    # make sure to sample at least two points when sample_dist > line.length
    distances = [0,] + distances + [line.length,] 
    
    sampled_points = np.array([list(line.interpolate(distance).coords)
                            for distance in distances]).squeeze()
    
    return sampled_points


@jit(nopython=True)
def rotate_box(corners, angle, center):
    angle = np.float32(angle)
    center = center.astype(np.float32)
    corners = corners.astype(np.float32)
    
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    rotation_matrix = np.array([[cos_a, -sin_a],
                                [sin_a, cos_a]], dtype=np.float32)
    
    rotated_corners = np.empty_like(corners, dtype=np.float32)
    for i in range(corners.shape[0]):
        rotated_corners[i, 0:2] = np.dot(rotation_matrix, (corners[i, 0:2] - center[0:2])) + center[0:2]
        rotated_corners[i, 2] = corners[i, 2]
    
    return rotated_corners

@jit(nopython=True)
def get_box_corners(box):
    x, y, z, l, w, h, yaw = box.astype(np.float32)
    l /= 2
    w /= 2
    h /= 2
    corners = np.array([
        [x - l, y - w, z - h],
        [x - l, y - w, z + h],
        [x - l, y + w, z - h],
        [x - l, y + w, z + h],
        [x + l, y - w, z - h],
        [x + l, y - w, z + h],
        [x + l, y + w, z - h],
        [x + l, y + w, z + h]
    ], dtype=np.float32)
    return corners

@jit(nopython=True)
def manual_min(arr):
    min_val = arr[0].copy()
    for i in range(1, arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] < min_val[j]:
                min_val[j] = arr[i, j]
    return min_val

@jit(nopython=True)
def manual_max(arr):
    max_val = arr[0].copy()
    for i in range(1, arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] > max_val[j]:
                max_val[j] = arr[i, j]
    return max_val

@jit(nopython=True)
def get_intersection_corners(box1_corners, box2_corners):
    min_corner = np.maximum(manual_min(box1_corners), manual_min(box2_corners))
    max_corner = np.minimum(manual_max(box1_corners), manual_max(box2_corners))
    if np.any(min_corner > max_corner):
        return np.zeros((8, 3), dtype=np.float32)
    return np.array([
        [min_corner[0], min_corner[1], min_corner[2]],
        [min_corner[0], min_corner[1], max_corner[2]],
        [min_corner[0], max_corner[1], min_corner[2]],
        [min_corner[0], max_corner[1], max_corner[2]],
        [max_corner[0], min_corner[1], min_corner[2]],
        [max_corner[0], min_corner[1], max_corner[2]],
        [max_corner[0], max_corner[1], min_corner[2]],
        [max_corner[0], max_corner[1], max_corner[2]]], dtype=np.float32)

@jit(nopython=True)
def compute_volume(corners):
    if np.any(corners == 0):
        return 0.0
    return np.prod(manual_max(corners) - manual_min(corners))

@jit(nopython=True)
def compute_intersection_volume(box1, box2):
    box1_corners = get_box_corners(box1)
    box2_corners = get_box_corners(box2)

    box1_corners = rotate_box(box1_corners, box1[6], box1[:3])
    box2_corners = rotate_box(box2_corners, box2[6], box2[:3])

    intersection_corners = get_intersection_corners(box1_corners, box2_corners)
    intersection_volume = compute_volume(intersection_corners)
    return intersection_volume

@jit(nopython=True)
def compute_box_volume(box):
    return box[3] * box[4] * box[5]

@jit(nopython=True)
def calculate_3d_iou_matrix(gt_bboxes, pred_bboxes):
    if len(gt_bboxes) == 0 or len(pred_bboxes) == 0:
        return np.zeros((len(gt_bboxes), len(pred_bboxes)), dtype=np.float32)

    iou_matrix = np.zeros((gt_bboxes.shape[0], pred_bboxes.shape[0]), dtype=np.float32)
    
    for i in range(gt_bboxes.shape[0]):
        for j in range(pred_bboxes.shape[0]):
            intersection_volume = compute_intersection_volume(gt_bboxes[i], pred_bboxes[j])
            union_volume = compute_box_volume(gt_bboxes[i]) + compute_box_volume(pred_bboxes[j]) - intersection_volume
            iou_matrix[i, j] = intersection_volume / union_volume if union_volume != 0 else 0.0
    
    return iou_matrix
