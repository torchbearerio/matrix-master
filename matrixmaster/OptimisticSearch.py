import MaskMaker
from pythoncore import Constants


def get_salient_area_at_degrees(img, sm, deg):
    salient_boxes = MaskMaker.make_bounding_boxes(sm)
    img_height = img.shape[0]
    img_width = img.shape[1]
    longitude = deg / Constants.STREETVIEW_FOV * img_width
    intersected_boxes = [box for box in salient_boxes if (box['x1'] < longitude < box['x2'])]

    def box_comparator(b):
        center = b['y2'] - b['y1']
        return abs(center - img_height / 2.0)

    # If we have more than one salient region lying on the vertical, choose the one closest to the horizontal center
    intersected_boxes.sort(key=box_comparator)

    return intersected_boxes[0] if len(intersected_boxes) > 0 else None
