import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import BatchNormalization, Dropout


def lrn(x, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75):
    return tf.nn.local_response_normalization(x,
                                              depth_radius=depth_radius,
                                              bias=bias,
                                              alpha=alpha,
                                              beta=beta)

def build_AlexNet():

    model = Sequential()
    
    # Convolutional Layer
    model.add(layers.Conv2D(96, kernel_size = 11, strides = 4, activation = 'relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((3, 3), strides = 2))
    model.add(layers.Lambda(lrn))
    # model.add(BatchNormalization())

    model.add(layers.Conv2D(256, kernel_size = 5, strides = 1, activation = 'relu', padding = 'same'))
    model.add(layers.MaxPooling2D((3, 3), strides = 2))
    model.add(layers.Lambda(lrn))
    # model.add(BatchNormalization())
    
    model.add(layers.Conv2D(384, kernel_size = 3, strides = 1, activation = 'relu', padding = 'same'))
    model.add(layers.Conv2D(384, kernel_size = 3, strides = 1, activation = 'relu', padding = 'same'))
    model.add(layers.Conv2D(256, kernel_size = 3, strides = 1, activation = 'relu', padding = 'same'))
    model.add(layers.MaxPooling2D((3, 3), strides = 2))

    # Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(layers.Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(layers.Dense(1000, activation="softmax"))
    
    return model

def main():
    model = build_AlexNet()
    model.summary()
    
if __name__ == "__main__":
    main()