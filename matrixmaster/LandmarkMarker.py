import traceback
from io import BytesIO
from pythoncore import Task, Constants
from pythoncore.AWS import AWSClient
from pythoncore.Model.Landmark import Landmark
from pythoncore.Model.Hit import Hit
from pythoncore.Model import TorchbearerDB
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class LandmarkMarker (Task.Task):

    def __init__(self, ep_id, hit_id, task_token):
        super(LandmarkMarker, self).__init__(ep_id, hit_id, task_token)

    def _run_landmark_marker(self):
        print("Starting mark task for ep {}, hit {}".format(self.ep_id, self.hit_id))

        # Create DB session
        session = TorchbearerDB.Session()

        try:
            # Load from S3, across all positions available for corresponding ExecutionPoint
            for position in Constants.LANDMARK_POSITIONS.values():
                if AWSClient.s3_key_exists(Constants.S3_BUCKETS['STREETVIEW_IMAGES'],
                                           "{}_{}.jpg".format(self.ep_id, position)):

                    # Get streetview image for this Hit's ExecutionPoint
                    img_array = self._get_streetview_image(position)

                    # Get all landmarks for this hit for given position
                    for landmark in session.query(Landmark).filter_by(hit_id=self.hit_id, position=position).all():
                        # Create figure and axes
                        fig, ax = plt.subplots(1)

                        # Remove axes
                        ax.axis('off')
                        fig.axes[0].get_xaxis().set_visible(False)
                        fig.axes[0].get_yaxis().set_visible(False)

                        # Add the image
                        ax.imshow(img_array)

                        # Create a Rectangle patch
                        x1 = landmark.get_rect()["x1"]
                        x2 = landmark.get_rect()["x2"]
                        y1 = landmark.get_rect()["y1"]
                        y2 = landmark.get_rect()["y2"]

                        width = x2 - x1
                        height = y2 - y1

                        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')

                        # Add the patch to the Axes
                        ax.add_patch(rect)

                        # Save image
                        img_file = BytesIO()
                        fig.savefig(img_file, format='png', bbox_inches='tight', pad_inches=0)
                        self._put_marked_streetview_image(img_file, landmark.landmark_id)
                        # fig.show()

            # Commit DB inserts
            session.commit()

            self.send_success()
            print("Completed mark task for ep {}, hit {}".format(self.ep_id, self.hit_id))

        except Exception, e:
            traceback.print_exc()
            self.send_failure('LANDMARK_MARK_ERROR', e.message)

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
        return np.array(img, dtype=np.uint8)

    @staticmethod
    def _put_marked_streetview_image(img_file, landmark_id):
        client = AWSClient.get_client('s3')

        # Make sure file is "rewound"
        img_file.seek(0)

        # Put cropped image
        client.put_object(
            Body=img_file,
            Bucket=Constants.S3_BUCKETS['MARKED_LANDMARK_IMAGES'],
            Key="{0}.png".format(landmark_id)
        )

    def run(self):
        self._run_landmark_marker()

if __name__ == '__main__':
    sn = LandmarkMarker(21, 6, "qwd")
    sn.run()
