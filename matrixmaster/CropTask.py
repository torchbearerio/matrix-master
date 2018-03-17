from pythoncore import Task, Constants
from pythoncore.AWS import AWSClient
from pythoncore.Model import TorchbearerDB
from pythoncore.Model.Landmark import Landmark
from PIL import Image
from io import StringIO
import traceback
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


class CropTask(Task.Task):
    def __init__(self, ep_id, hit_id, task_token):
        super(CropTask, self).__init__(ep_id, hit_id, task_token)

    def run(self):
        # Create DB session
        session = TorchbearerDB.Session()

        try:
            # Load from S3, across all positions available for corresponding ExecutionPoint
            for position in Constants.LANDMARK_POSITIONS.values():
                if AWSClient.s3_key_exists(Constants.S3_BUCKETS['STREETVIEW_IMAGES'],
                                           "{}_{}.jpg".format(self.ep_id, position)):

                    img = self._get_streetview_image(position)

                    # Load all Landmarks for this hit, position
                    for landmark in session.query(Landmark).filter_by(hit_id=self.hit_id, position=position).all():
                        # Crop image according to Landmark rect
                        rect = landmark.get_rect()
                        rect = (rect['x1'], rect['y1'], rect['x2'], rect['y2'])

                        # Perform image segmentation to extract foreground object
                        arr = np.asarray(img)
                        cv_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

                        mask = np.zeros(cv_img.shape[:2], np.uint8)
                        bgdModel = np.zeros((1, 65), np.float64)
                        fgdModel = np.zeros((1, 65), np.float64)

                        cv2.grabCut(cv_img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

                        # If mask==2 or mask== 1, mask2 get 0, other wise it gets 1 as 'uint8' type.
                        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

                        # Project image into RGBA space
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2BGRA)

                        # Set alpha based on segment_mask.
                        # Set red channel to 100% for viewers that don't support alpha channel.
                        cv_img[mask2 == 0] = [255, 0, 0, 0]

                        # Convert cv2 img back to PIL img
                        img = Image.fromarray(cv_img)

                        # Crop cut image down to landmark rect
                        img = img.crop(rect)

                        if os.environ.get('debug'):
                            plt.imshow(img)
                            plt.show()

                        # Put cropped images into S3
                        self._put_cropped_image(img, landmark.landmark_id)

            # Commit DB inserts, if any
            session.commit()

            # Send success!
            self.send_success()

        except Exception as e:
            traceback.print_exc()
            session.rollback()
            self.send_failure('CROP_ERROR', e.message)

        finally:
            session.close()

    def _get_streetview_image(self, position):
        client = AWSClient.get_client('s3')
        response = client.get_object(
            Bucket=Constants.S3_BUCKETS['STREETVIEW_IMAGES'],
            Key="{}_{}.jpg".format(self.ep_id, position)
        )
        img = Image.open(response['Body'])
        # img.show()
        return img

    @staticmethod
    def _put_cropped_image(img, landmark_id):
        img_file = StringIO()
        img.save(img_file, 'PNG')

        client = AWSClient.get_client('s3')

        # Put cropped image
        client.put_object(
            Body=img_file,
            Bucket=Constants.S3_BUCKETS['TRANSPARENT_CROPPED_IMAGES'],
            Key="{0}.png".format(landmark_id)
        )


if __name__ == '__main__':
    # Test
    x = CropTask(21, 6, '123456gh')
    x.run()
