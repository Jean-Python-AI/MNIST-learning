from pathlib import Path
import numpy as np
from PIL import Image


class DATA():
    def __init__(self, train=True):
        self.X = []
        self.Y = []

        nom_dossier = "training" if train else "testing"
        racine = Path(__file__).resolve().parent / nom_dossier
        for chiffre in range(10):
            dossier = racine / str(chiffre)
            images = sorted(
                dossier.glob("*.png"),
                key=lambda chemin: int(chemin.stem)
            )
            for chemin_image in images:
                vecteur = DATA.imgVectorisate(chemin_image)
                self.X.append(vecteur)
                y = [0 for i in range(10)]
                y[chiffre] = 1
                self.Y.append(y)

        self.X = np.array(self.X, dtype=np.float32)
        self.Y = np.array(self.Y, dtype=np.float32)

    @staticmethod
    def imgVectorisate(path):
        image = Image.open(path).convert("L").resize((28, 28))
        tableau = np.asarray(image, dtype=np.float32) / 255.0
        vecteur = tableau.reshape(-1)

        return vecteur

        