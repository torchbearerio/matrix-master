from scipy import ndimage
import cv2
import numpy as np
from skimage.feature import peak_local_max
import os


def make_bounding_boxes(sm):

    sm[300:400, 20:200] = 255

    # Apply Otsu's thresholding to saliency matrix
    # Otsu' finds optimal value for threshold--values > than thresh get 255, < get 0
    # This gives a binary segmentation of salient/non-salient
    thresh = cv2.threshold(sm, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    if os.environ.get('debug'):
        cv2.imshow("Thresh", thresh)
        cv2.waitKey()

    # For each "salient" pixel in binary threshold, compute distance to nearest non-salient pixel
    # This will leave us with the most intense values being at the center of salient regions,
    # as they are farthest from boundary of salient region

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    #cv2.imshow("cleaned", cleaned)
    #cv2.waitKey()

    # sure background (non-salient) area
    sure_bg = cv2.dilate(cleaned, kernel, iterations=3)

    #cv2.imshow("sure_bg", sure_bg)
    #cv2.waitKey()

    # Finding sure foreground (salient) area.
    dist_transform = cv2.distanceTransform(cleaned, cv2.cv.CV_DIST_L2, 5)

    #cv2.imshow("distance_transform", dist_transform / 255)
    #cv2.waitKey()

    sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)[1]

    #cv2.imshow("sure_fg", sure_fg)
    #cv2.waitKey()

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    #cv2.imshow("unknown", unknown)
    #cv2.waitKey()

    # Marker labelling - 8-connectivity connected component analysis
    markers = ndimage.label(sure_fg, np.ones((3, 3), dtype=int))[0]

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    #cv2.imshow("markers", 255 / markers)
    #cv2.waitKey()

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    #cv2.imshow("markers", 255 / markers)
    #cv2.waitKey()

    # Saves final labels into markers
    cv2.watershed(cv2.cvtColor(sm, cv2.COLOR_GRAY2BGR), markers)

    bounding_boxes = []

    # loop over the unique regions returned by the Watershed
    # algorithm
    for region in np.unique(markers):
        # if the region is 1, we are examining the 'background'
        # if the region is -1, this is the boundary between regions
        if region == 1 or region == -1:
            continue

        # otherwise, allocate memory for the region and draw it on the mask
        mask = np.zeros(sm.shape, dtype="uint8")
        mask[markers == region] = 1

        if os.environ.get('debug'):
            cv2.imshow("Output", mask * sm)
            cv2.waitKey(0)

        # If this region is smaller than 2% of image, ignore it
        if np.count_nonzero(mask) < 0.02 * sm.size:
            continue

        # Otherwise, create a landmark based on the bounding box around this region
        # Find bounding box based on original saliency mask
        y1, y2, x1, x2 = __bbox2(mask)

        bounding_boxes.append({
            'y1': y1,
            'y2': y2,
            'x1': x1,
            'x2': x2
        })

    return bounding_boxes


def __bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax