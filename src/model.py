import os
import torch
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1


workers = 0 if os.name == 'nt' else 4


class FaceDetector(object):
    def __init__(self, classifier=False, img_size=160, device=None):
        self.classifier = classifier
        self.img_size = img_size
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self, image):
        print("start detection:")
        if type(image) == list:
            batch = []
            for img in image:
                img_to_draw = self.mtcnn(img)
                img_aligned = self.post_process(img_to_draw)
                batch.append(img_aligned)
            batch = torch.stack(batch).to(self.device)
        else:
            batch = []
            img_to_draw = self.mtcnn(image)
            img_aligned = self.post_process(img_to_draw)
            batch.append(img_aligned)
            batch = torch.stack(batch).to(self.device)

        print("creating embeddings:")
        embeddings = self.resnet(batch)

        with open("pickles/normalizer.pickle", "rb") as handle:
            normalizer = pickle.load(handle)
        embeddings_norm = normalizer.transform(embeddings)

        print("done!")

        return img_to_draw.cpu().numpy(), embeddings_norm

    @staticmethod
    def post_process(image_tensor):
        image_tensor = (image_tensor - 127.5) / 128.0
        return image_tensor

    def mtcnn(self, image):
        mtcnn = MTCNN(
            image_size=self.img_size, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False,
            device=self.device, select_largest=True
        )

        aligned, prob = mtcnn(image, return_prob=True)
        if aligned is not None and len(aligned) == 1:
            print('Face detected with probability: {:8f}'.format(prob))
        return aligned

    def resnet(self, aligned):
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        # aligned = aligned[None, :, :, :].to(self.device)
        embeddings = resnet(aligned).detach().cpu()
        return embeddings


