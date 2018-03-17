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
import OptimisticSearch

import GrabCut
import MaskMaker


class ScoreTask(Task.Task):
    def __init__(self, ep_id, hit_id, task_token):
        super(ScoreTask, self).__init__(ep_id, hit_id, task_token)

    def run(self):
        print("Starting score task for ep {}, hit {}".format(self.ep_id, self.hit_id))

        # Create DB session
        session = TorchbearerDB.Session()

        try:
            # Load saliency mask and image from S3, across all positions available for this ExecutionPoint
            for position in Constants.LANDMARK_POSITIONS.values():
                if AWSClient.s3_key_exists(Constants.S3_BUCKETS['STREETVIEW_IMAGES'],
                                           "{}_{}.jpg".format(self.ep_id, position)):
                    sm = self._get_saliency_matrix(position)
                    img = self._get_streetview_image(position)

                    if os.environ.get('debug'):
                        plt.imshow(img, alpha=1)
                        plt.imshow(sm, alpha=0.6)
                        plt.show()

                    # Compute sum of saliency mask
                    image_saliency_sum = sm.sum()

                    # Retrieve landmarks for this hit and position
                    for landmark in session.query(Landmark.Landmark) \
                            .filter_by(hit_id=self.hit_id, position=position).all():

                        # Compute visual saliency score for this landmark
                        r = landmark.get_rect()

                        # If rectangle was not already defined by another service, do optimistic saliency search
                        # Note that this can still return None
                        # Save this rect with landmark back to db
                        if r is None:
                            r = OptimisticSearch.get_salient_area_at_degrees(img, sm, landmark.relative_bearing)
                            landmark.set_rect(r)

                        visual_saliency_score = sm[r['y1']:r['y2'], r['x1']:r['x2']].sum() / float(image_saliency_sum) \
                            if r is not None else 0

                        landmark.visual_saliency_score = visual_saliency_score

            # Commit DB inserts
            session.commit()

            # Send success!
            self.send_success()
            print("Completed score task for ep {}, hit {}".format(self.ep_id, self.hit_id))

        except Exception as e:
            traceback.print_exc()
            session.rollback()
            self.send_failure('MATRIX MASTER ERROR', e.message)

        finally:
            session.close()

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

if __name__ == '__main__':
    # Test
    x = ScoreTask(21, 12, '123456gh')
    x.run()
