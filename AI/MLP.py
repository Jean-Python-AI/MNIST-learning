import numpy as np


class MLP():
    #========== INIT MODEL ==========#
    def __init__(self, dimensions, Niew, nameAI):
        # AI paramètres
        self.NAME_AI = nameAI
        self.DIM = dimensions # = [nbInputs, nbLayer1, nbLayer2, ..., nbOutputLayer]
        self.PARAMETRES = {} # réseau avec tous les W et b
        self.ACTIVATIONS = {} # ce qu'a retourner chaque perceptron
        self.C = len(self.DIM) # nombre de couches (en contant les X inputs)
        # Pour l'entrainement
        self.ALPHA = 0.01

        # Train
        self.X = None # Inputs
        self.Y = None # Outputs attendu

        if Niew:
            self.createNiewMLP()
        else:
            self.load()

    # CREATE A NIEW MLP #
    def createNiewMLP(self):
        for c in range(1, self.C):
            self.PARAMETRES["W" + str(c)] = np.random.randn(self.DIM[c], self.DIM[c - 1])
            self.PARAMETRES["b" + str(c)] = np.random.randn(self.DIM[c], 1)
        self.save()


    #========== LOAD and SAVE MODEL ==========#
    def load(self):
        try:
            data = np.load(f"./AI/models/{self.NAME_AI}.npz")
        except:
            print("ERROR: nom du modèle de l'IA surement incorect === ligne 131 dans ./Agent/mlp.py")
            return

        with np.load(f"./AI/models/{self.NAME_AI}.npz") as data:
            self.PARAMETRES = {key: data[key] for key in data.files}

    def save(self):
        np.savez(f"./AI/models/{self.NAME_AI}.npz", **self.PARAMETRES)


    #========== USE MODEL ==========#
    def forwardPropagation(self, X):
        # On check qu'il y ai le bon nombre de X
        if len(X[0]) != self.DIM[0]:
            print("\nERROR: le nombre d'inputs est mauvais, le réseaux en a besoins de ",
                  self.DIM[0], " mais en reçoit ", len(X))
            return

        self.ACTIVATIONS = {} # réinitialisation des activations
        self.X = X
        for i, x in enumerate(X):
            self.ACTIVATIONS[f"{i}A0"] = np.asarray(x).reshape(-1, 1)
            self.ACTIVATIONS[f"{i}Z0"] = np.asarray(x).reshape(-1, 1)
            for c in range(1, self.C):
                W = self.PARAMETRES["W" + str(c)]
                b = self.PARAMETRES["b" + str(c)]
                a = self.ACTIVATIONS[f"{i}A" + str(c-1)]

                Z = W.dot(a) + b
                A = 1 / (1 + np.exp(-Z))

                self.ACTIVATIONS[f"{i}A{c}"] = A
                self.ACTIVATIONS[f"{i}Z{c}"] = Z

        results = np.array([self.ACTIVATIONS[f"{i}A{self.C -1}"] for i in range(len(X))])

        return results


    #========== TRAIN MODEL ==========#
    def LogLoss(self, Y):
        m = len(self.X)
        nb_sorties = self.DIM[-1]
        Y = np.asarray(Y, dtype=float).reshape(m, nb_sorties, 1)
        self.Y = Y
        A = np.array([self.ACTIVATIONS[f"{i}A{self.C -1}"] for i in range(len(self.X))])
        A = np.clip(A, 1e-15, 1 - 1e-15) # pour éviter d'avoir 0 ou 1 pille
        toSUM = Y*np.log(A) + (1-Y)*np.log(1-A)
        L = -(1/len(Y)) * sum(toSUM)
        L_mean = sum(L)/len(L)
        return L, L_mean

    def BackProp(self):
        m = len(self.X)
        nb_sorties = self.DIM[-1]
        Y = self.Y.reshape(m, nb_sorties, 1)
        dZn = 1
        for c in range(self.C-1, 0, -1):
            pastA = np.stack([self.ACTIVATIONS[f"{i}A{c-1}"] for i in range(m)], axis=0)
            A = np.stack([self.ACTIVATIONS[f"{i}A{c}"] for i in range(m)], axis=0)
            # For the last Layer
            if c == self.C-1:
                dZn *= (A - Y) / m
            else:
                Z = np.array([self.ACTIVATIONS[f"{i}Z{c}"] for i in range(m)])
                futurW = self.PARAMETRES[f"W{c+1}"]
                dL_dA = futurW.T @ dZn
                dZn = dL_dA * A * (1 - A)
            # changement des paramètres
            dL_dW_par_test = dZn @ np.swapaxes(pastA, 1, 2)
            dL_dW = np.sum(dL_dW_par_test, axis=0)
            dL_db = sum(dZn)
            self.PARAMETRES["W" + str(c)] -= self.ALPHA * dL_dW
            self.PARAMETRES["b" + str(c)] -= self.ALPHA * dL_db

        #self.save() # on enregistre les changements
