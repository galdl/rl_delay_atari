import tensorflow as tf
import collections
import argparse

EPS = 1e-12

# parser = argparse.ArgumentParser()
# # parser.add_argument("--input_dir", help="path to folder containing images")
# # parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
# # parser.add_argument("--output_dir", required=True, help="where to put output files")
# # parser.add_argument("--seed", type=int)
# # parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
# #
# # parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
# # parser.add_argument("--max_epochs", type=int, help="number of training epochs")
# # parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
# # parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
# # parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
# # parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
# # parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")
# #
# parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
# # parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
# # parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
# # parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
# # parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
# parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
# parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
# # parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
# # parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
# # parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
# # parser.set_defaults(flip=True)
# parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
# parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
# parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
# parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
# #
# # # export options
# # parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
# a = parser.parse_args()
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")



def gen_deconv(batch_input, out_channels, config, resize_shape=None):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if config.separable_conv:
        _b, h, w, _c = batch_input.shape
        resize_shape = [h * 2, w * 2] if resize_shape is None else resize_shape[1:3]
        resized_input = tf.image.resize_images(batch_input, resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer,
                                          reuse=tf.AUTO_REUSE)

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02),
                                         reuse=tf.AUTO_REUSE)

def gen_conv(batch_input, out_channels, config):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if config.separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer,
                                reuse=tf.AUTO_REUSE)

def create_generator(generator_inputs, generator_outputs_channels, config):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, config.ngf, config)
        layers.append(output)

    layer_specs = [
        config.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        config.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        config.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        config.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        config.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        config.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        config.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels, config)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (config.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (config.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (config.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (config.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (config.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (config.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (config.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels, config, layers[max(skip_layer - 1, 0)].shape)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels, config)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]

def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02),
                            reuse=tf.AUTO_REUSE)

def create_model(inputs, targets, ac_space=None, config=None):
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(input, config.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = config.ndf * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = discrim_conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels, config)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * config.gan_weight + gen_loss_L1 * config.l1_weight

    tf.init_scope()
    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if "discriminator" in var.name]
        discrim_optim = tf.train.AdamOptimizer(config.lr, config.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if "generator" in var.name]
            gen_optim = tf.train.AdamOptimizer(config.lr, config.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=1e-6) # originally: decay=0.99, but wandb already does the smoothing
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    # global_step = tf.train.get_or_create_global_step()
    # incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, gen_train),
    )
