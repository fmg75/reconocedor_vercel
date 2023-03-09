import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle

class FaceNetModels:
    def __init__(self):
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        #self.model = torch.jit.load('model_resnet.pt')
        self.mtcnn = MTCNN(min_face_size = 50, keep_all = False)
        with open('caracteristica_16K.pkl', "rb") as f:
            self.caracteristicas = pickle.load(f)

    def embedding(self,img_tensor):
        img_embedding = self.model(img_tensor.unsqueeze(0))
        return img_embedding

    def Distancia (self,img_embedding):
        distances = []
        for label, emb in self.caracteristicas.items():
            distance = torch.dist(emb, img_embedding)
            distances.append((label, distance))
        # ordena la lista distancia de menor a mayor
        sorted_distances = sorted(distances, key=lambda x: x[1])
        # obtener los untimos valores de E y D
        E = [sorted_distances[i][0] for i in range(0,1,1)]
        D = [sorted_distances[i][1].item() for i in range(0,1,1)]

        return E, D
