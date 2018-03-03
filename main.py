import glob as gb

import numpy as np
from PIL import Image

import createData
import data_utils
from losses import adversarial_loss, generator_loss
from model import generator_model, discriminator_model, generator_containing_discriminator


def train(batch_size, epoch_num):
    # Note the x(blur) in the second, the y(full) in the first
    y_train, x_train = createData.load_data()
    print(len(y_train), '\n', len(x_train))
    # GAN
    g = generator_model()
    d = discriminator_model()
    d_on_g = generator_containing_discriminator(g, d)

    # compile the models, use default optimizer parameters
    # generator use adversarial loss
    g.compile(optimizer='adam', loss=generator_loss)
    # discriminator use binary cross entropy loss
    d.compile(optimizer='adam', loss='binary_crossentropy')
    # adversarial net use adversarial loss
    d_on_g.compile(optimizer='adam', loss=adversarial_loss)

    for epoch in range(epoch_num):
        print('epoch: ', epoch + 1, '/', epoch_num)
        print('batches: ', int(x_train.shape[0] / batch_size))

        for index in range(int(x_train.shape[0] / batch_size)):
            # select a batch data
            image_blur_batch = x_train[index * batch_size:(index + 1) * batch_size]
            
            image_full_batch = y_train[index * batch_size:(index + 1) * batch_size]
            #print('full ',image_blur_batch)
            generated_images = g.predict(x=image_blur_batch, batch_size=batch_size)
            #print('generated_images ', generated_images)
            # output generated images for each 30 iters
            if (index % 3 == 0) and (index != 0):
                createData.generate_image(image_full_batch, image_blur_batch, generated_images,
                                          'result/interim/', epoch, index)

            # concatenate the full and generated images,
            # the full images at top, the generated images at bottom
            x = np.concatenate([image_full_batch, generated_images])
            #print('x ',x)
            # generate labels for the full and generated images
            #y = [1] * batch_size + [0] * batch_size
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            # train discriminator
            print('discriminator')
            d_loss = d.train_on_batch(x, y)
            #d_loss = d.train_on_batch(image_full_batch, [1] * batch_size)
            print('batch %d d_loss : %f' % (index + 1, d_loss))

            # let discriminator can't be trained
            d.trainable = False

            # train adversarial net
            z = np.ones([batch_size, 1])
            print('adversarial')
            d_on_g_loss = d_on_g.train_on_batch(image_blur_batch, z)
            #d_on_g_loss = d_on_g.train_on_batch(image_blur_batch, [1] * batch_size)
            print('batch %d d_on_g_loss : %f' % (index + 1, d_on_g_loss))

            # train generator
            print('generator')
            g_loss = g.train_on_batch(image_blur_batch, image_full_batch)
            print('batch %d g_loss : %f' % (index + 1, g_loss))

            # let discriminator can be trained
            d.trainable = True

            # output weights for generator and discriminator each 30 iters
            if (index % 3 == 0) and (index != 0):
                g.save_weights('weight/generator_weights_%d_%d.h5' % (epoch , index), True)
                d.save_weights('weight/discriminator_weights_%d_%d.h5' % (epoch , index), True)


def test(batch_size):
    # Note the x(blur) in the second, the y(full) in the first
    y_test, x_test = createData.load_data(data_type='test')
    g = generator_model()
    g.load_weights('weight/generator_weights.h5')
    generated_images = g.predict(x=x_test, batch_size=batch_size)
    createData.generate_image(y_test, x_test, generated_images, 'result/finally/')


def test_pictures(batch_size):
    data_path = 'data/test/*.jpeg'
    images_path = gb.glob(data_path)
    data_blur = []
    for image_path in images_path:
        image_blur = Image.open(image_path)
        data_blur.append(np.array(image_blur))

    data_blur = np.array(data_blur).astype(np.float32)
    data_blur = createData.normalization(data_blur)

    g = generator_model()
    g.load_weights('weight/generator_weights.h5')
    generated_images = g.predict(x=data_blur, batch_size=batch_size)
    generated = generated_images * 127.5 + 127.5
    for i in range(generated.shape[0]):
        image_generated = generated[i, :, :, :]
        Image.fromarray(image_generated.astype(np.uint8)).save('result/test/' + str(i) + '.png')


if __name__ == '__main__':
    train(batch_size=10, epoch_num=10)
    #test(4)
    #test_pictures(2)
