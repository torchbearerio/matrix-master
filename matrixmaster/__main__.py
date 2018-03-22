import matplotlib

# Disable X server for matplotlib
matplotlib.use('Agg')

from pythoncore import Constants, WorkerService
from MaskTask import MaskTask
from CropTask import CropTask
from ScoreTask import ScoreTask
from LandmarkMarker import LandmarkMarker
import cv2


# disable multithreading in OpenCV for main thread to avoid problems after fork
# This is likely only needed on OSX, and multithreading could be re-enabled in production
cv2.setNumThreads(0)


def handle_mask_task(task_input, task_token):
    ep_id = task_input["epId"]
    hit_id = task_input["hitId"]
    mt = MaskTask(ep_id, hit_id, task_token)
    mt.run()


def handle_score_task(task_input, task_token):
    ep_id = task_input["epId"]
    hit_id = task_input["hitId"]
    st = ScoreTask(ep_id, hit_id, task_token)
    st.run()


def handle_mark_task(task_input, task_token):
    ep_id = task_input["epId"]
    hit_id = task_input["hitId"]
    lm = LandmarkMarker(ep_id, hit_id, task_token)
    lm.run()


def handle_crop_task(task_input, task_token):
    ep_id = task_input["epId"]
    hit_id = task_input["hitId"]
    ct = CropTask(ep_id, hit_id, task_token)
    ct.run()

if __name__ == '__main__':
    # handle_mask_task({"epId": 355, "hitId": 773}, "asdfadsf")
    # handle_score_task({"epId": 355, "hitId": 779}, "asdfadsf")
    maskTask = Constants.TASK_ARNS['DERIVE_RECTS_FROM_MASK']
    scoreTask = Constants.TASK_ARNS['SCORE_VISUAL_SALIENCY']
    markTask = Constants.TASK_ARNS['LANDMARK_MARKER']
    # cropTask = Constants.TASK_ARNS['CROP_LANDMARKS']

    WorkerService.start(
        (scoreTask, handle_score_task, 2),
        (markTask, handle_mark_task, 2),
        (maskTask, handle_mask_task, 2)
    )
