from keras.layers import Input, Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from processor import DataProcessor

NUM_EPOCHS = 2


class CNN(object):

    def __init__(self):
        self.model = self.get_model()

    def fit(self,
            train_generator,
            train_size,
            valid_generator,
            valid_size,
            n_epochs):

        self.model.fit_generator(train_generator,
                                 samples_per_epoch=train_size,
                                 validation_data=valid_generator,
                                 nb_val_samples=n_epochs,
                                 nb_epoch=n_epochs)

    def save(self):
        self.model.save('model.h5')

    def get_model(self):

        input_shape = (DataProcessor.HEIGHT,
                       DataProcessor.WIDTH,
                       DataProcessor.NUM_CHANNELS)

        model = Sequential()

        model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='elu', input_shape=input_shape))
        model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='elu'))
        model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='elu'))
        model.add(Conv2D(64, 3, 3, subsample=(1, 1), activation='elu'))
        model.add(Conv2D(64, 3, 3, subsample=(1, 1), activation='elu'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(100))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        return model


def main():

    train_generator, valid_generator, n_train, n_valid = DataProcessor.load()
    model = CNN()
    model.fit(train_generator, n_train, valid_generator, n_valid, NUM_EPOCHS)
    model.save()

if __name__ == '__main__':
    main()
