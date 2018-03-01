import json
import uuid
from io import BytesIO
import traceback
import numpy as np
from PIL import Image
from pythoncore import Task, Constants
from pythoncore.AWS import AWSClient
from pythoncore.Model import TorchbearerDB, Landmark
from matplotlib import pyplot as plt
import os
import cv2
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import GrabCut
import MaskMaker


class MaskTask(Task.Task):
    def __init__(self, ep_id, hit_id, task_token):
        super(MaskTask, self).__init__(ep_id, hit_id, task_token)

    def run(self):
        print("Starting mask task for ep {}, hit {}".format(self.ep_id, self.hit_id))

        # Create DB session
        session = TorchbearerDB.Session()

        try:
            for position in Constants.LANDMARK_POSITIONS.values():
                if AWSClient.s3_key_exists(Constants.S3_BUCKETS['SALIENCY_MAPS'],
                                           "{}_{}.json".format(self.hit_id, position)):
                    # Load saliency mask and image from S3
                    sm = self._get_saliency_matrix(position)

                    bounding_boxes = MaskMaker.make_bounding_boxes(sm)

                    for bb in bounding_boxes:
                        x1, x2, y1, y2 = [bb[k] for k in ('x1', 'x2', 'y1', 'y2')]
                        if os.environ.get('debug'):
                            cv2.imshow("Output", sm[y1:y2, x1:x2])
                            cv2.waitKey(0)

                        landmark = {
                            'id': uuid.uuid1(),
                            'rect': {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2},
                            'position': position
                        }

                        # Insert candidate landmark into DB
                        self._insert_candidate_landmark(landmark, session)

            # Commit DB inserts
            session.commit()

            # Send success!
            self.send_success()
            print("Completed mask task for ep {}, hit {}".format(self.ep_id, self.hit_id))

        except Exception as e:
            traceback.print_exc()
            session.rollback()
            self.send_failure('MATRIX MASTER ERROR', e.message)

    def _get_saliency_matrix(self, position):
        client = AWSClient.get_client('s3')
        response = client.get_object(
            Bucket=Constants.S3_BUCKETS['SALIENCY_MAPS'],
            Key="{}_{}.json".format(self.hit_id, position)
        )
        data = json.load(response['Body'])
        return np.array(data["saliencyMatrix"], np.uint8)

    def _get_streetview_image(self, position):
        client = AWSClient.get_client('s3')
        response = client.get_object(
            Bucket=Constants.S3_BUCKETS['STREETVIEW_IMAGES'],
            Key="{}_{}.jpg".format(self.ep_id, position)
        )
        img = Image.open(response['Body'])
        # img.show()
        return np.array(img)

    @staticmethod
    def _put_cropped_images(candidate):
        img_file = BytesIO()
        transparent_img_file = BytesIO()
        candidate['image'].save(img_file, 'PNG')
        candidate['image_transparent'].save(transparent_img_file, 'PNG')
        img_file.seek(0)
        transparent_img_file.seek(0)

        client = AWSClient.get_client('s3')

        # Put cropped image
        client.put_object(
            Body=img_file,
            ContentType='image/png',
            Bucket=Constants.S3_BUCKETS['CROPPED_IMAGES'],
            Key="{0}.png".format(candidate['id'])
        )

        # Put transparent cropped image
        client.put_object(
            Body=transparent_img_file,
            ContentType='image/png',
            Bucket=Constants.S3_BUCKETS['TRANSPARENT_CROPPED_IMAGES'],
            Key="{0}.png".format(candidate['id'])
        )

    def _insert_candidate_landmark(self, candidate, session):
        landmark = Landmark.Landmark(
            landmark_id=candidate['id'],
            hit_id=self.hit_id,
            position=candidate['position']
        )
        landmark.set_rect(candidate['rect'])

        session.add(landmark)


if __name__ == '__main__':
    # Test
    x = MaskTask(21, '123456gh')
    x.run()
