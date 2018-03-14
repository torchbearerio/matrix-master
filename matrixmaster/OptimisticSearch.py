import MaskMaker
from pythoncore import Constants

'''
Finds salient areas intersecting a vertical degree.
Center of image is at 0 degrees, with positive degrees to right and negative to left.
'''


def get_salient_area_at_degrees(img, sm, deg):
    salient_boxes = MaskMaker.make_bounding_boxes(sm)
    img_height = img.shape[0]
    img_width = img.shape[1]
    pixels_per_degree = img_width / Constants.STREETVIEW_FOV

    # Since `deg` is (-180, 180), must adjust so that 0deg => FOV/2deg (center of image)
    adjusted_deg = int(deg + Constants.STREETVIEW_FOV / 2)

    # Clamp between (0, img_width - 1)
    longitude = max(min(int(adjusted_deg * pixels_per_degree), img_width - 1), 0)
    intersected_boxes = [box for box in salient_boxes if (box['x1'] < longitude < box['x2'])]

    def box_comparator(b):
        center = b['y2'] - b['y1']
        return abs(center - img_height / 2.0)

    # If we have more than one salient region lying on the vertical, choose the one closest to the horizontal center
    intersected_boxes.sort(key=box_comparator)

    return intersected_boxes[0] if len(intersected_boxes) > 0 else None
