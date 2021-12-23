import numpy as np
from keras.datasets.fashion_mnist import load_data
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
from keras.layers import (Dropout, Input, Embedding, Concatenate)


# define the standalone generator model
def Generator(latent_dim, n_classes=10):
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
    # linear multiplication
    n_nodes = 7 * 7
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((7, 7, 1))(li)

    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 7x7 image
    n_nodes = 128 * 7 * 7
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((7, 7, 128))(gen)

    # merge image gen and label input
    merge = Concatenate()([gen, li])
    # upsample to 14x14
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(merge)
    gen = LeakyReLU(alpha=0.2)(gen)
    # upsample to 28x28
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # output
    out_layer = Conv2D(1, (7, 7), activation='tanh', padding='same')(gen)

    # define model
    model = Model([in_lat, in_label], out_layer)
    return model


# define the standalone discriminator model
def Discriminator(in_shape=(28, 28, 1), n_classes=10):
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
    # scale up to image dimensions with linear activation
    n_nodes = in_shape[0] * in_shape[1]
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((in_shape[0], in_shape[1], 1))(li)

    # image input
    in_image = Input(shape=in_shape)

    # concat label as a channel
    merge = Concatenate()([in_image, li])
    # downsample
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(merge)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # flatten feature maps
    fe = Flatten()(fe)
    # dropout
    fe = Dropout(0.4)(fe)
    # output
    out_layer = Dense(1, activation='sigmoid')(fe)

    # define model
    model = Model([in_image, in_label], out_layer)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def GAN(g_model, d_model):
    # get noise and label inputs from generator model
    gen_noise, gen_label = g_model.input

    # connect image output and label input from generator as inputs to discriminator
    gan_output = d_model([g_model.output, gen_label])
    # make weights in the discriminator not trainable
    d_model.trainable = False

    # define gan model as taking noise and label and outputting a classification
    model = Model([gen_noise, gen_label], gan_output)

    # compile model
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    return model


# load fashion mnist images
def load_real_samples():
    # load dataset
    (trainX, trainy), (_, _) = load_data()
    # expand to 3d, e.g. add channels
    X = np.expand_dims(trainX, axis=-1)
    # convert from ints to floats
    X = X.astype('float32')

    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return [X, trainy]


# select real samples
def generate_real_samples(images, labels, n_samples):
    ix = np.random.randint(0, images.shape[0], n_samples)
    x, l = images[ix], labels[ix]

    # 1 -> true -> real
    y = np.ones((n_samples, 1))
    return [x, l], y


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples, n_classes):
    h = np.random.randn(n_samples, latent_dim)
    l = np.random.randint(0, n_classes, n_samples)
    x = generator.predict([h, l])

    # 0 -> false -> fake
    y = np.zeros((n_samples, 1))
    return [x, l], y


# select real samples
def generate_dcm_batch(generator, dataset, batch_size, latent_dim, n_classes):
    images, labels = dataset

    [x_real, l_real], y_real = generate_real_samples(images, labels, batch_size // 2)
    [x_fake, l_fake], y_fake = generate_fake_samples(generator, latent_dim, batch_size // 2, n_classes)

    x = np.concatenate([x_real, x_fake], axis=0)
    l = np.concatenate([l_real, l_fake], axis=0)
    y = np.concatenate([y_real, y_fake], axis=0)
    return [x, l], y


def generate_gan_batch(latent_dim, batch_size, n_classes):
    h = np.random.randn(batch_size, latent_dim)
    l = np.random.randint(0, n_classes, batch_size)
    y = np.ones((batch_size, 1))  # all true
    return [h, l], y


# create and save a plot of generated images (reversed grayscale)
def show_plot(examples, n, with_channel=True, titles=None):
    # plot images
    for i in range(n * n):

        if titles:
            plt.title(titles[i])

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
    dataset = load_real_samples()
    images, labels = dataset
    n_classes = max(labels)

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
            x_img, y_img = generate_dcm_batch(G, dataset, batch_size, latent_dim, n_classes)
            d_loss, _ = D.train_on_batch(x_img, y_img)

            x_hid, y_hid = generate_gan_batch(latent_dim, batch_size, n_classes)
            g_loss = gan.train_on_batch(x_hid, y_hid)  # the D is frozen in gan

        # summarize loss on this epoch
        print('>%d/%d, disc_loss=%.3f, gan_loss=%.3f' % (i + 1, n_epochs, d_loss, g_loss))
        h = np.random.randn(batch_size, latent_dim)
        l = np.random.randint(0, n_classes, batch_size)
        x_img_make = G.predict([h, l])
        print("origin:")
        show_plot(x_img[0], 5)  # x_img = x_img, x_label
        print("make:")
        show_plot(x_img_make, 5, titles=[str(d) for d in l])
