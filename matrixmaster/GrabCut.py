import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import os


def __bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def crop_image_with_saliency_mask(img, saliency_mask):
    # First, run GrabCut to segment image based on Saliency Mask
    mask_copy = np.copy(saliency_mask).astype('uint8')
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    # mask[saliency_mask == 1] = 3
    cv2.grabCut(img, mask_copy, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    # Compute binary mask based on grabcut output (perfect "cut out" of object)
    segment_mask = np.where((mask_copy==cv2.GC_PR_FGD)|(mask_copy==cv2.GC_FGD), 1, 0).astype('uint8')

    # Find bounding box based on original saliency mask
    rmin, rmax, cmin, cmax = __bbox2(saliency_mask)

    """
    # Create matrix of 1's based on bounding box--this gives a rectangle around segmented region
    region = np.ones((rmax-rmin, cmax-cmin), dtype=np.uint8)
    segment_mask[rmin:rmax, cmin:cmax] = region
    """

    # Create transparent image
    img_transparent_bg = np.copy(img)

    # Project transparent image into RGBA space
    img_transparent_bg = cv2.cvtColor(img_transparent_bg, cv2.COLOR_BGR2BGRA)

    # Set alpha based on segment_mask. Set red channel to 100% for viewers that don't support alpha channel.
    img_transparent_bg[segment_mask == 0] = [255, 0, 0, 0]

    # Crop images based on region, convert from BGR to RGB
    image = img[rmin:rmax, cmin:cmax, :]
    image_transparent = img_transparent_bg[rmin:rmax, cmin:cmax, :]

    if os.environ.get('debug'):
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),plt.show()

    return {
        "image":                    Image.fromarray(image, 'RGB'),
        "image_transparent":        Image.fromarray(image_transparent, 'RGBA'),
        "rect":                     {"x1": cmin, "x2": cmax, "y1": rmin, "y2": rmax},
    }
