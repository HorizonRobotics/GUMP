import math

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def subtract(self, v):
        return Vector(self.x - v.x, self.y - v.y)

    def dot(self, v):
        return self.x * v.x + self.y * v.y

    def perpendicular(self):
        return Vector(-self.y, self.x)

    def normalize(self):
        length = math.sqrt(self.x ** 2 + self.y ** 2)
        if length > 0:
            return Vector(self.x / length, self.y / length)
        return Vector(self.x, self.y)

def get_corners(x, y, width, length, heading):
    # Convert heading to radians
    angle = math.radians(heading)
    
    # Calculate corner offsets
    dx = width / 2
    dy = length / 2
    
    # Calculate corners
    corners = []
    for dx_i in [dx, -dx]:
        for dy_i in [dy, -dy]:
            corner_x = x + (dx_i * math.cos(angle) - dy_i * math.sin(angle))
            corner_y = y + (dx_i * math.sin(angle) + dy_i * math.cos(angle))
            corners.append(Vector(corner_x, corner_y))
    
    return corners

def project_polygon(axis, corners):
    min_proj = max_proj = axis.dot(corners[0])
    for corner in corners[1:]:
        projection = axis.dot(corner)
        min_proj = min(min_proj, projection)
        max_proj = max(max_proj, projection)
    return min_proj, max_proj

def overlap_along_axis(axis, corners1, corners2):
    min1, max1 = project_polygon(axis, corners1)
    min2, max2 = project_polygon(axis, corners2)
    return max(min1, min2) <= min(max1, max2)

def boxes_overlap(x1, y1, w1, l1, h1, x2, y2, w2, l2, h2):
    corners1 = get_corners(x1, y1, w1, l1, h1)
    corners2 = get_corners(x2, y2, w2, l2, h2)

    axes = [corners1[i].subtract(corners1[i-1]).perpendicular().normalize() for i in range(len(corners1))]
    axes += [corners2[i].subtract(corners2[i-1]).perpendicular().normalize() for i in range(len(corners2))]

    for axis in axes:
        if not overlap_along_axis(axis, corners1, corners2):
            return False
    return True

def boxes_overlap_axis_align(x1, y1, w1, l1, x2, y2, w2, l2):
    # Calculate the half dimensions of each box
    half_w1, half_l1 = w1 / 2, l1 / 2
    half_w2, half_l2 = w2 / 2, l2 / 2

    # Check for overlap in the x-axis
    if abs(x1 - x2) >= half_w1 + half_w2:
        return False

    # Check for overlap in the y-axis
    if abs(y1 - y2) >= half_l1 + half_l2:
        return False

    # If both checks pass, boxes overlap
    return True