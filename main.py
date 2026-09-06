import numpy as np
import math as m
import matplotlib.pyplot as plt
from AI.MLP import MLP
from Dataset.Dataset import DATA

##====== ASK QUESTIONS ======##
while True:
    trainORshow = int(input("\nTRAIN or SHOW ??? \n\nTrain = 0\nShow = 1 \n\n =>   "))
    if trainORshow == 0:
        TRAIN = True
        break
    elif trainORshow == 1:
        TRAIN = False
        break
    else:
        print("Met 0 ou 1 (un nombre valide)")


print("\nPREPARING DATA\n")
##========== INIT ============##
## INIT NERONAL STRUCTURE
DIM = [784, 100, 140, 80, 10] # 784 (28x28) inputs, 10 outputs

## INIT IMPORTS
mlp = MLP(dimensions=DIM, Niew=False, nameAI="AI")
data = DATA(train=TRAIN)

# Train set
X = data.X
Y = data.Y

rng = np.random.default_rng()


##======= TRAIN =======##
if TRAIN:
    print("Start TRAINING")
    batch_size = 32 # entrainement de 32 img à la fois
    nb_epochs = 7 # nombre de fois où l'on entraine sur tout le DATASET

    LOSS = []

    for epoch in range(nb_epochs):
        indices = rng.permutation(len(X))

        LOSS_epoch = []

        for debut in range(0, len(X), batch_size):
            batch_indices = indices[debut:debut + batch_size]

            X_batch = X[batch_indices]
            Y_batch = Y[batch_indices]

            mlp.forwardPropagation(X_batch)
            loss, loss_mean = mlp.LogLoss(Y_batch)
            LOSS_epoch.append(loss_mean[0])
            LOSS.append(loss_mean[0])
            mlp.BackProp()

        mlp.save() # save de niew model
        LOSS_epoch_mean = sum(LOSS_epoch) / len(LOSS_epoch)
        print(f"LOSS ({epoch+1}/{nb_epochs}) =>    ", LOSS_epoch_mean)

    # Lorsque tous les epochs sont fini
    print("=======  FINISH !!! =====")
    print("MEAN LOSS (du dernier epoch) =>   ", LOSS_epoch_mean)

    # Preparation des data pour l'affichage
    LOSS = np.array(LOSS, dtype=np.float32)
    x = np.linspace(0, nb_epochs, len(LOSS))
    y = LOSS

    plt.plot(x, y)

    plt.show()


##======= SHOW ========##
else:
    print("START PREDICTIONS")
    ranges, columns = 3, 4
    nb_images = ranges * columns

    indices = rng.choice(len(X), size=nb_images, replace=False) # sélections des images aléatoire

    X_show = X[indices]
    Y_show = Y[indices]

    predictions = mlp.forwardPropagation(X_show)
    predictions = predictions[:, :, 0]

    chiffres_predits = np.argmax(predictions, axis=1)
    chiffres_reels = np.argmax(Y_show, axis=1)

    fig, ax = plt.subplots(ranges, columns, figsize=(10, 8))

    for i, axe in enumerate(ax.flat):
        image = X_show[i].reshape(28, 28)

        axe.imshow(image, cmap="gray")
        axe.axis("off")

        prediction = chiffres_predits[i]
        reel = chiffres_reels[i]

        couleur = "green" if prediction == reel else "red"
        percent = predictions[i, prediction] * 100
        axe.set_title(
            f"Prédit : {prediction} ({percent:.2f}%)\nVrai : {reel}",
            color=couleur
        )

    plt.tight_layout()
    plt.show()