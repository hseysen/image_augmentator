def yolotocv(x1, y1, x2, y2, h, w):
    """
    Converts YOLO format bounding box dimensions to OpenCV format
    :param x1: X-coordinate of the center of the bounding box
    :param y1: Y-coordinate of the center of the bounding box
    :param x2: Width of the bounding box
    :param y2: Height of the bounding box
    :param h: Height of the image
    :param w: Width of the image
    :return: OpenCV format coordinates of the center and width/height
    """
    bbox_width = x2 * w
    bbox_height = y2 * h
    center_x = x1 * w
    center_y = y1 * h

    ret = [center_x - (bbox_width / 2), center_y - (bbox_height / 2),
           center_x + (bbox_width / 2), center_y + (bbox_height / 2)]

    return [int(v) for v in ret]


def cvtoyolo(x1, y1, x2, y2, h, w):
    """
    Converts OpenCV format bounding box dimensions to YOLO format
    :param x1: X-coordinate of the first corner
    :param y1: Y-coordinate of the first corner
    :param x2: X-coordinate of the second corner
    :param y2: Y-coordinate of the second corner
    :param h: Height of the image
    :param w: Width of the image
    :return: YOLO format center coordinates and width/height
    """
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    center_bbox_x = (x1 + x2) / 2
    center_bbox_y = (y1 + y2) / 2

    return [round(center_bbox_x / w, 10), round(center_bbox_y / h, 10), round(bbox_w / w, 10), round(bbox_h / h, 10)]


def clamp_value(val, minclamp, maxclamp):
    return max(minclamp, min(maxclamp, val))
