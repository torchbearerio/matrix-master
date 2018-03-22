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
import GrabCut
import MaskMaker


class CropFromSaliencyTask(Task.Task):
    def __init__(self, ep_id, hit_id, task_token):
        super(CropFromSaliencyTask, self).__init__(ep_id, hit_id, task_token)

    def run(self):
        # Create DB session
        session = TorchbearerDB.Session()

        try:
            for position in Constants.LANDMARK_POSITIONS.values():
                if AWSClient.s3_key_exists(Constants.S3_BUCKETS['STREETVIEW_IMAGES'],
                                           "{}_{}.jpg".format(self.hit_id, position)):
                    # Load saliency mask and image from S3
                    sm = self._get_saliency_matrix()
                    img = self._get_streetview_image(position)

                    if os.environ.get('debug'):
                        plt.imshow(img, alpha=1)
                        plt.imshow(sm, alpha=0.6)
                        plt.show()

                    # Compute sum of saliency mask
                    image_saliency_sum = sm.sum()

                    # Get list of individual salient regions
                    masks = MaskMaker.get_masks_from_saliency_map(sm)

                    for mask in masks:
                        candidate = GrabCut.crop_image_with_saliency_mask(img, mask)

                        # Tack a GUID onto candidate dict
                        candidate['id'] = uuid.uuid1()

                        candidate['position'] = position

                        # Compute visual saliency score
                        r = candidate['rect']
                        region_saliency_sum = sm[r['y1']:r['y2'], r['x1']:r['x2']].sum()
                        candidate['visual_saliency_score'] = region_saliency_sum / float(image_saliency_sum)

                        # candidate["image"].show()
                        # candidate["image_transparent"].show()

                        # Put cropped images into S3
                        self._put_cropped_images(candidate)

                        # Insert candidate landmark into DB
                        self._insert_candidate_landmark(candidate, session)

            # Commit DB inserts
            session.commit()

            # Send success!
            self.send_success()

        except Exception as e:
            traceback.print_exc()
            session.rollback()
            self.send_failure('MATRIX MASTER ERROR', e.message)

    def _get_saliency_matrix(self):
        client = AWSClient.get_client('s3')
        response = client.get_object(
            Bucket=Constants.S3_BUCKETS['SALIENCY_MAPS'],
            Key="{0}.json".format(self.hit_id)
        )
        data = json.load(response['Body'])
        return np.array(data["saliencyMatrix"], np.uint8)

    def _get_streetview_image(self, position):
        client = AWSClient.get_client('s3')
        response = client.get_object(
            Bucket=Constants.S3_BUCKETS['STREETVIEW_IMAGES'],
            Key="{}_{}.jpg".format(self.hit_id, position)
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
            visual_saliency_score=candidate['visual_saliency_score'],
            position=candidate['position']
        )
        landmark.set_rect(candidate['rect'])

        session.add(landmark)


if __name__ == '__main__':
    # Test
    x = MaskTask(21, '123456gh')
    x.run()
