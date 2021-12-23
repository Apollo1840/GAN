import numpy as np
from keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout


def Generator(latent_dim):
    model = Sequential()
    # foundation for 7x7 image
    n_nodes = 128 * 7 * 7
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))
    # upsample to 14x14
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 28x28
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (7, 7), activation='sigmoid', padding='same'))
    return model


def Discriminator(in_shape=(28, 28, 1)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3),
                     strides=(2, 2),
                     padding='same',
                     input_shape=in_shape))

    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3, 3),
                     strides=(2, 2),
                     padding='same'))

    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def GAN(g_model, d_model):
    # connect them
    model = Sequential()

    # add generator
    model.add(g_model)

    # add the discriminator
    d_model.trainable = False
    model.add(d_model)

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    return model


# load and prepare mnist training images
def load_real_samples():
    # load mnist dataset
    (trainX, _), (_, _) = load_data()
    # expand to 3d, e.g. add channels dimension
    X = np.expand_dims(trainX, axis=-1)
    # convert from unsigned ints to floats
    X = X.astype('float32')

    # scale from [0,255] to [0,1]
    X = X / 255.0
    return X


# select real samples
def generate_real_data(images, n_samples):
    """

    :param dataset:
    :param n_samples: batch_size
    :return:
    """

    # choose random instances
    ix = np.random.randint(0, images.shape[0], n_samples)
    X = images[ix]

    # generate 'real' class labels (1)
    y = np.ones((n_samples, 1))
    return X, y


# use the generator to generate n fake examples, with class labels
def generate_fake_data(generator: Model, latent_dim, n_samples):
    """

    :param dataset:
    :param n_samples: batch_size
    :return:
    """

    # generate points in latent space
    x_input = np.random.randn(n_samples, latent_dim)
    X = generator.predict(x_input)

    y = np.zeros((n_samples, 1))
    return X, y


def generate_dcm_batch(generator, images, batch_size, latent_dim):
    """
    generate training data for discriminator

    :param generator:
    :param dataset:
    :param batch_size:
    :param latent_dim:
    :return:
    """

    # get randomly selected 'real' samples
    X_real, y_real = generate_real_data(images, batch_size // 2)
    # generate 'fake' examples
    X_fake, y_fake = generate_fake_data(generator, latent_dim, batch_size // 2)

    # create training set for the discriminator
    X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))

    return X, y


def generate_gan_batch(latent_dim, batch_size):
    # here is a little bit different than Ivans paper.
    # we did not use the same fake image in task batch:

    x = np.random.randn(batch_size, latent_dim)  # latent space sample
    y = np.ones((batch_size, 1))  # all true

    return x, y


# create and save a plot of generated images (reversed grayscale)
def show_plot(examples, n, with_channel=True):
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        if with_channel:
            # shape = (n_sample, x_axis, y_axis, channel)
            plt.imshow(examples[i, :, :, 0], cmap='gray_r')
        else:
            # shape = (n_sample, x_axis, y_axis)
            plt.imshow(examples[i], cmap='gray_r')

    plt.show()


if __name__ == '__main__':
    # load image data
    images = load_real_samples()

    latent_dim = 100
    G = Generator(latent_dim)
    D = Discriminator()
    gan = GAN(G, D)

    # train model
    batch_size = 256
    n_epochs = 100
    n_batch = int(images.shape[0] / batch_size)

    for i in range(n_epochs):
        for j in tqdm(range(n_batch)):
            x_img, y_img = generate_dcm_batch(G, images, batch_size, latent_dim)
            d_loss, _ = D.train_on_batch(x_img, y_img)

            x_hid, y_hid = generate_gan_batch(latent_dim, batch_size)
            g_loss = gan.train_on_batch(x_hid, y_hid)  # the D is frozen in gan

        # summarize loss on this epoch
        print('>%d/%d, disc_loss=%.3f, gan_loss=%.3f' % (i + 1, n_epochs, d_loss, g_loss))
        x_img_make = G.predict(np.random.randn(batch_size, latent_dim))
        print("origin:")
        show_plot(x_img, 5)
        print("make:")
        show_plot(x_img_make, 5)
