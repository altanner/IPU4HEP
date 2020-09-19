import sys
import time
import numpy as np
import argparse
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input,
    Flatten,
    Dense,
    Reshape,
    Dropout,
    Embedding, #! unused
    Multiply, #! unused
    Activation,
    Conv2D,
    ZeroPadding2D, #! unused
    LocallyConnected2D, #! unused
    Concatenate, #! unused
    GRU, #! unused
    Lambda, #! unused
    Conv2DTranspose,
    BatchNormalization,
    LeakyReLU
)
from tensorflow.keras.models import Model, Sequential #! Sequential is unused
from tensorflow.compat.v1.keras.layers import UpSampling2D #! pylint hates this, unused
# tf.disable_v2_behavior() #! can be discarded I guess?
print(f"Using Tensorflow version {tf.__version__}")


_EPSILON = K.epsilon() #todo should these be args?
batch_size = 128 #! will this always be the same as the GAN_noise_size?
samples = 60000
GAN_noise_size = 128
GAN_output_size = 7
#! regression problem, not classification problem


def assemble_training_dataset(batch_size, samples):

    """
    #todo docstring
    """

    #! train targets = normal dist mean = 0, variance = 1, 60000 samples, 7 dimensions
    train_images = np.random.normal(0, 1, (samples, 7))
    train_images = train_images.reshape(train_images.shape[0], 7).astype("float32")
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train_images)
        .batch(batch_size, drop_remainder=True)
        .repeat(1)
    )

    return train_dataset


def build_generator():

    """
    Build the network layers of the generator model.
    #todo docstring
    Returns the model as "Generator".
    """

    G_input = Input(shape=(GAN_noise_size,))
    G = Dense(128, kernel_initializer="glorot_uniform")(G_input)
    G = LeakyReLU(alpha=0.2)(G)
    G = BatchNormalization()(G)
    G = Reshape([8, 8, 2])(G)  # default: channel last
    G = Conv2DTranspose(32, kernel_size=2, strides=1, padding="same")(G)
    G = LeakyReLU(alpha=0.2)(G)
    G = BatchNormalization()(G)
    G = Conv2DTranspose(16, kernel_size=3, strides=1, padding="same")(G)
    G = LeakyReLU(alpha=0.2)(G)
    G = BatchNormalization()(G)
    G = Flatten()(G) #! justify this model
    G_output = Dense(GAN_output_size)(G)
    G_output = Activation("tanh")(G_output) #! hyperbolic tangent
    Generator = Model(G_input, G_output)

    return Generator


def build_discriminator():

    """
    Build the network layers of the discriminator model.
    #todo comprehensive docstring
    Returns the model as "Discriminator".
    """

    D_input = Input(shape=(GAN_output_size,))
    D = Dense(128)(D_input)
    D = Reshape((8, 8, 2))(D)
    D = Conv2D(64, kernel_size=3, strides=1, padding="same")(D)
    D = LeakyReLU(alpha=0.2)(D)
    D = Conv2D(32, kernel_size=3, strides=1, padding="same")(D)
    D = LeakyReLU(alpha=0.2)(D)
    D = Conv2D(16, kernel_size=3, strides=1, padding="same")(D)
    D = LeakyReLU(alpha=0.2)(D)
    D = Flatten()(D)
    D = LeakyReLU(alpha=0.2)(D)
    D = Dropout(0.2)(D) #! justify this model
    D_output = Dense(1, activation="sigmoid")(D)
    Discriminator = Model(D_input, D_output)

    return Discriminator


def assign_optimizers():

    """
    #todo docstring
    #? although is is this func redundant?
    """

    #! optimizer_stacked and optimizer_D (identical) are unused.
    # optimizer_stacked = tf.compat.v1.train.AdamOptimizer(
    #     learning_rate=0.001,
    #     beta1=0.9,
    #     beta2=0.999,
    #     epsilon=1e-08
    # )

#    optimizer_D = optimizer_stacked #! this was duplicated in OG

    gen_optimizer = tf.keras.optimizers.Adam(0.0002)

    disc_optimizer = gen_optimizer #! this was duplicated in OG

   # return optimizer_stacked, optimizer_D, gen_optimizer, disc_optimizer
    return gen_optimizer, disc_optimizer


def _loss_generator(y_true, y_pred):

    """
    #todo docstring
    """

    y_pred = K.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
    out = -(K.log(y_pred)) #! pylint hates this (why does it still work?)

    return K.mean(out, axis=-1)


@tf.function
def train_step(
    images,
    Generator,
    Discriminator,
    gen_optimizer,
    disc_optimizer):

    """
    #todo docstring
    """

    noise = tf.random.normal([batch_size, 128])
    noise_stacked = tf.random.normal((int(batch_size * 2), 128), 0, 1)

    with tf.GradientTape(persistent=True) as tape:
        generated_images = Generator(noise)

        in_values = tf.concat([generated_images, images], 0)
        labels_D_0 = tf.zeros((batch_size, 1)) #! tf tensor of 0s, falses
        labels_D_1 = tf.ones((batch_size, 1))  #! tf tensor of 1s, trues

        labels_D = tf.concat([labels_D_0, labels_D_1], 0)

        out_values = Discriminator(in_values)
        loss_D = tf.keras.losses.binary_crossentropy(labels_D, out_values)
        loss_D = tf.math.reduce_mean(loss_D)

        fake_images2 = Generator(noise_stacked)
        stacked_output = Discriminator(fake_images2)

        labels_stacked = tf.ones((int(batch_size * 2), 1))

        loss_stacked = _loss_generator(labels_stacked, stacked_output)
        loss_stacked = tf.math.reduce_mean(loss_stacked)

    grad_disc = tape.gradient(loss_D, Discriminator.trainable_variables)
    grad_gen = tape.gradient(loss_stacked, Generator.trainable_variables)

    disc_optimizer.apply_gradients(zip(grad_disc, Discriminator.trainable_variables))
    gen_optimizer.apply_gradients(zip(grad_gen, Generator.trainable_variables))

    return loss_stacked, loss_D


def main():

    # see docstrings :)
    train_dataset = assemble_training_dataset(batch_size, samples)

    Generator = build_generator()
    Generator.summary()

    Discriminator = build_discriminator()
    Discriminator.summary()

#    optimizer_stacked, optimizer_D, gen_optimizer, disc_optimizer = assign_optimizers()
    gen_optimizer, disc_optimizer = assign_optimizers()

    # run loop
    for epoch in range(10):

        print(f"Epoch {epoch}")

        with tqdm(total=int(samples / batch_size),
                  file=sys.stdout,
                  ncols=64,
                  unit="its") as pbar:

            for images in train_dataset:

                gen_loss, disc_loss = train_step(
                    images,
                    Generator,
                    Discriminator,
                    gen_optimizer,
                    disc_optimizer)
                #todo report losses etc
                pbar.update(1)
    #todo output model, benchmark, logging


if __name__ == "__main__":
    main()
