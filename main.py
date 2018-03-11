import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    image_input = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    # TODO: Implement function
    #1x1 conv of layer 7
    conv_1x1_layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size= 1,
                                   strides= (1,1), padding= 'same',
                                   kernel_initializer= tf.truncated_normal_initializer(stddev=0.1),
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    #upsample by 2 with strides 2, kernel 4
    output_layer7 = tf.layers.conv2d_transpose(conv_1x1_layer7, num_classes, kernel_size= 4,
                                             strides= (2,2), padding= 'same',
                                             kernel_initializer= tf.truncated_normal_initializer(stddev=0.1),
                                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    #1x1 conv of layer 4
    conv_1x1_layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size= 1,
                                   strides= (1,1), padding= 'same',
                                   kernel_initializer= tf.truncated_normal_initializer(stddev=0.1),
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    #skip connection 1
    output7_plus_layer4conv1x1 = tf.add(output_layer7,conv_1x1_layer4)
    #upsample by 2 with stride 2, kernel 4
    output_layer4 = tf.layers.conv2d_transpose(output7_plus_layer4conv1x1, num_classes, kernel_size= 4,
                                             strides= (2,2), padding= 'same',
                                             kernel_initializer= tf.truncated_normal_initializer(stddev=0.1),
                                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    #1x1 conv of layer 3
    conv_1x1_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, kernel_size= 1,
                                   strides= (1,1), padding= 'same',
                                   kernel_initializer= tf.truncated_normal_initializer(stddev=0.1),
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    #skip connection 2
    output4_plus_layer3conv1x1 = tf.add(output_layer4,conv_1x1_layer3)
    #upsample by 8 with strides 8 and kernel 16
    nn_last_layer = tf.layers.conv2d_transpose(output4_plus_layer3conv1x1, num_classes, kernel_size= 16,
                                               strides= (8,8), padding= 'same',
                                               kernel_initializer= tf.truncated_normal_initializer(stddev=0.1),
                                               kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    return nn_last_layer
tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    # TODO: Implement function
    #flatten or reshape to 2D with rows as pixels and cols as class both logits and correct_label
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    #correct_label = tf.reshape(correct_label, (-1, num_classes))
    #loss Function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = correct_label))
    regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = cross_entropy_loss + regularization_loss
    #training function or optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    train_op = optimizer.minimize(loss)
    return logits, train_op, loss

tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate):
    # TODO: Implement function
    for i in range(epochs):
        print("EPOCH {} ...".format(i+1))
        for image, label in get_batches_fn(batch_size):
            #training
            _, loss = sess.run([train_op, cross_entropy_loss],
                                feed_dict={input_image: image, correct_label: label,
								keep_prob: 0.5, learning_rate: 0.001})
            print("Loss: = {:.4f}".format(loss))
            print()

tests.test_train_nn(train_nn)

#Hyper parameters for tuning
NUM_EPOCHS = 10
BATCH_SIZE = 5

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    epochs = NUM_EPOCHS
    batch_size = BATCH_SIZE
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        # TODO: Build NN using load_vgg, layers, and optimize function
        #TF placeholders
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss, input_image, correct_label, keep_prob, learning_rate)
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
