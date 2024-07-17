import numpy as np
import numpy.typing as npt


def box_to_corners(boxes: npt.NDArray) -> npt.NDArray:
    """Calculate the corners of the boxes.

    :param boxes: boxes of shape [..., 5], each box is represented by 5 numbers:
        center_x, center_y, heading, half_length, half_width
    :return: corners of the boxes of shape [..., 4, 2]. The four corners are in
        the order of front-left, front-right, back-right and back-left.
    """
    o = boxes[..., :2]
    heading = boxes[..., 2]
    half_length = boxes[..., 3]
    half_width = boxes[..., 4]
    c = np.cos(heading)
    s = np.sin(heading)
    half_length_vec = half_length[..., None] * np.stack([c, s], axis=-1)
    half_width_vec = half_width[..., None] * np.stack([-s, c], axis=-1)
    fl = half_length_vec + half_width_vec
    fr = half_length_vec - half_width_vec
    corners = np.stack([fl, fr, -fl, -fr], axis=-2)  # shape [..., 4, 2]
    return corners + o[..., None, :]


def line_segment_intersected(a1: npt.NDArray, a2: npt.NDArray, b1: npt.NDArray,
                             b2: npt.NDArray) -> npt.NDArray[np.bool_]:
    """Whether two line segments intersect.

    Whether line segment a1--a2 intersects with line segment b1--b2.

    `a1`, `a2`, `b1`, `b2` should be broadcastable.
    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line_segment

    :param a1: [..., 2]
    :param a2: [..., 2]
    :param b1: [..., 2]
    :param b2: [..., 2]
    :return: array of bool indicating whether each pair of line segments
        intersect
    """
    err = np.geterr()
    np.seterr(all='ignore')

    da = a1 - a2
    dab = a1 - b1
    db = b1 - b2
    d = da[..., 0] * db[..., 1] - da[..., 1] * db[..., 0]
    t = (dab[..., 0] * db[..., 1] - dab[..., 1] * db[..., 0]) / d
    u = (dab[..., 0] * da[..., 1] - dab[..., 1] * da[..., 0]) / d

    np.seterr(**err)
    return (0 <= t) & (t <= 1) & (0 <= u) & (u <= 1)


def to_box_frame(p: npt.NDArray, box: npt.NDArray) -> npt.NDArray:
    """Transform points to the frame of the box.

    The box frame is defined as follows:
    - the origin is the center of the box
    - the x axis is along the heading of the box
    - the y axis is perpendicular to the x axis and pointing to the left

    `p.shape[:-1]` and `box.shape[:-1]` should be broadcastable.

    :param p: a batch of points of shape [..., 2]
    :param box: a batch of boxes of shape [..., 5], each box is represented by 5 numbers:
        center_x, center_y, heading, half_length, half_width
    :return: x, y coordinates of the points in the frame of the box
    """
    o = box[..., :2]
    heading = box[..., 2]
    c = np.cos(heading)
    s = np.sin(heading)
    q = p - o
    x = q[..., 0] * c + q[..., 1] * s
    y = q[..., 1] * c - q[..., 0] * s

    return x, y


def point_box_edge_dist(p: npt.NDArray, box: npt.NDArray) -> npt.NDArray:
    """Calculate the distance between a point and the edges of a box.

    `p.shape[:-1]` and `box.shape[:-1]` should be broadcastable.

    :param p: [..., 2]
    :param box: box of shape [..., 5], each box is represented by 5 numbers:
        center_x, center_y, heading, half_length, half_width
    :return: distance between the point and the box
    """
    x, y = to_box_frame(p, box)

    half_length = box[..., 3]
    half_width = box[..., 4]

    dx = (x - np.clip(x, -half_length, half_length))**2
    dy = (y - np.clip(y, -half_width, half_width))**2

    d1 = (x - half_length)**2 + dy
    d2 = (x + half_length)**2 + dy
    d3 = dx + (y - half_width)**2
    d4 = dx + (y + half_width)**2
    return np.sqrt(np.stack([d1, d2, d3, d4], axis=-1).min(axis=-1))


def point_inside_box(p: npt.NDArray,
                     box: npt.NDArray) -> npt.NDArray[np.bool_]:
    """Whether a point is inside a box.

    `p.shape[:-1]` and `box.shape[:-1]` should be broadcastable.

    :param p: a batch of points of shape [..., 2]
    :param box: a batch of boxes of shape [..., 5], each box is represented by
        5 numbers: center_x, center_y, heading, half_length, half_width
    :return: array of bool indicating whether each point is inside the
        corresponding box
    """
    x, y = to_box_frame(p, box)
    half_length = box[..., 3]
    half_width = box[..., 4]
    return (np.abs(x) <= half_length) & (np.abs(y) <= half_width)


def box_box_dist(box1: npt.NDArray, box2: npt.NDArray) -> npt.NDArray:
    """Calculate the distance between two boxes.

    The distance is defined as the minimum distance between any two points on
    the two boxes.

    `box1.shape[:-1]` and `box2.shape[:-1]` should be broadcastable.

    :param box1: a batch of boxes of shape [..., 5], each box is represented by
        5 numbers: center_x, center_y, heading, half_length, half_width
    :param box2: a batch of boxes of shape [..., 5], each box is represented by
        5 numbers: center_x, center_y, heading, half_length, half_width
    :return: distances between corresponding boxes
    """
    corners1 = box_to_corners(box1)
    corners2 = box_to_corners(box2)
    d1 = point_box_edge_dist(corners1, box2[..., None, :])
    d2 = point_box_edge_dist(corners2, box1[..., None, :])
    d = np.minimum(d1, d2).min(axis=-1)

    inside = point_inside_box(box1[..., :2], box2)
    inside = inside | point_inside_box(box2[..., :2], box1)

    a1 = corners1
    a2 = a1[..., [1, 2, 3, 0], :]
    b1 = corners2
    b2 = b1[..., [1, 2, 3, 0], :]
    intersected = line_segment_intersected(
        a1[..., None, :, :], a2[..., None, :, :], b1[..., :, None, :],
        b2[..., :, None, :])
    intersected = inside | intersected.any(axis=(-2, -1))
    d[intersected] = 0

    return d
