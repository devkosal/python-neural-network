import numpy as np

from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from analysis.confustion_matrix import plot_confusion_matrix
from analysis.one_hot_encoder import indices_to_one_hot
from network.network import MultiLayerNetwork
from network.preprocessor import Preprocessor
from network.trainer import Trainer




def main():

    dat = np.loadtxt(
        "dataset/iris/iris.data",
        delimiter=',',
    )

    x,y = load_boston(return_X_y=True)

    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42)

    scl_x = StandardScaler()
    x_train = scl_x.fit_transform(x_train)
    x_test = scl_x.transform(x_test)

    scl_y = StandardScaler()
    y_train = scl_y.fit_transform(y_train[:,None])
    y_test = scl_y.transform(y_test[:,None])

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_test)

    input_dim = len(x_train[0])
    neurons = [3, 1]
    activations = ["relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.007,
        loss_fun="mse",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_test))

    preds = net(x_train)
    targets = y_train
    print("Training mean_squared_error: {}".format( mean_squared_error(preds, targets)**.5))

    preds = net(x_test)
    targets = y_test
    print("Validation mean_squared_error: {}".format( mean_squared_error(preds, targets)**.5))


if __name__ == "__main__":
    main()
