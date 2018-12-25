import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten,Conv2D,MaxPooling2D,Dropout,Activation
from keras.utils import np_utils
from keras_sequential_ascii import sequential_model_to_ascii_printout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, image

def vizualizaResultados ( cnn ) :

    plt.figure(0)
    plt.plot(cnn.history['acc'],'r')
    plt.plot(cnn.history['val_acc'],'g')
    plt.xticks(np.arange(0, 13, 2.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.legend(['train','validation'])
    plt.grid(True)

    plt.show()

    plt.figure(0)
    plt.plot(cnn.history['loss'],'r')
    plt.plot(cnn.history['val_loss'],'g')
    plt.xticks(np.arange(0, 13, 2.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.legend(['train','validation'])
    plt.grid(True)

    plt.show()

    return

def montaCNN ( largImagem, alturaImagem ) :
    """

    Função que monta a cnn do objeto

    """

    print("---------- Inicio de Treinamento ------------\n")

    cNN = Sequential ( )

    # primeira camadas convolução
    cNN.add ( Conv2D(
        filters = 64,
        kernel_size = (3, 3),
        input_shape = ( largImagem, alturaImagem, 3 ),
        activation = "relu"
    ) )
    cNN.add(MaxPooling2D(
        pool_size = (2, 2)
    ))
    cNN.add ( BatchNormalization() )

    # segunda camadas convolução
    cNN.add ( Conv2D(
        filters = 32,
        kernel_size = (3, 3),
        input_shape = ( largImagem, alturaImagem, 3 ),
        activation = "relu"
    ) )
    cNN.add(MaxPooling2D(
        pool_size = (2, 2)
    ))
    cNN.add ( BatchNormalization() )

    # camada de Flatten
    cNN.add ( Flatten() )

    # rede neural densa
    cNN.add ( Dense(
        units = 128,
        activation = "relu"
    ) )
    cNN.add ( Dropout ( 0.5 ) )
    cNN.add(Dense(
        units = 128,
        activation = "relu"
    ))
    cNN.add(Dropout(0.3))
    cNN.add(Dense(
        units = 1
    ))
    cNN.add ( Activation (
        tf.nn.sigmoid
    ) )

    cNN.compile (
         loss = "binary_crossentropy",
         optimizer = "SGD",
         metrics = ["accuracy"]
     )

    return cNN

def main (  ) :

    geradorTreinamento = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 7,
        horizontal_flip = True,
        shear_range = 0.2,
        height_shift_range = 0.07,
        zoom_range = 0.2
    )

    geradorTeste = ImageDataGenerator(
        rescale = 1./255
    )

    baseTreinamento = geradorTreinamento.flow_from_directory(
        "dataset/dataset/training_set",
        target_size = ( 64, 64 ),
        batch_size = 64,
        class_mode = "binary"
    )

    baseTeste = geradorTreinamento.flow_from_directory(
        "dataset/dataset/test_set",
        target_size = ( 64, 64 ),
        batch_size = 64,
        class_mode = "binary"
    )

    cNN = montaCNN (
        alturaImagem = 64,
        largImagem = 64
    )

    sequential_model_to_ascii_printout ( cNN )

    avaliacao = cNN.fit_generator ( baseTreinamento,
              steps_per_epoch = 300,
              epochs = 1,
              validation_data = baseTeste,
              validation_steps = 34
    )

    print("Precisão : {}\n".format ( avaliacao ) )
    vizualizaResultados ( avaliacao )

    # classificando apenas uma imagem
    imagemTeste = image.load_img (
        "dataset/dataset/test_set/gato/cat.3500.jpg",
        target_size = ( 64, 64 )
    )

    imagemTeste = image.img_to_array(
        imagemTeste
    )

    imagemTeste /= 255
    imagemTeste = np.expand_dims (
        imagemTeste,
        axis = 0
    )

    previsao = cNN.predict ( imagemTeste )

    if ( previsao >= 0.6 ) :

        print("{0.2f} % de chance de ser um gato".format( previsao[0][0]*100 ) )

    else :
        porcentagem = 100 - ( previsao[0][0]*100 )
        print("{0.2f} % de chance de ser um cachorro".format( porcentagem ) )

    return

if __name__ == '__main__':
    main()
