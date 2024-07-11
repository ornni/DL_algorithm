# import library

import tensorflow as tf
from tensorflow.keras import Sequential, layers

 

# build vgg16 model
def vgg16():

    model = Sequential()
    
    model.add(layers.Conv2D(64, kernel_size = (3, 3), padding = 'same', activation = 'relu', input_shape=(224, 224, 3)))
    model.add(layers.Conv2D(64, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(layers.MaxPooling2D(2))
    
    model.add(layers.Conv2D(128, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(layers.Conv2D(128, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(layers.MaxPooling2D(2))
    
    model.add(layers.Conv2D(256, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(layers.Conv2D(256, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(layers.Conv2D(256, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(layers.MaxPooling2D(2))
    
    model.add(layers.Conv2D(512, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(layers.Conv2D(512, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(layers.Conv2D(512, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(layers.MaxPooling2D(2))
    
    model.add(layers.Conv2D(512, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(layers.Conv2D(512, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(layers.Conv2D(512, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(layers.MaxPooling2D(2))
    
    # Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dense(1000, activation="softmax"))
    
    return model

def main():
    model = vgg16()
    model.summary()
    
if __name__ == "__main__":
    main()
