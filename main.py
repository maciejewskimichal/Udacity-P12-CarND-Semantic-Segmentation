#!/usr/bin/env python3
import os.path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import sys
from glob import glob
import time
import cv2

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    model = tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)

    image_input = tf.get_default_graph().get_tensor_by_name('image_input:0')
    keep_prob   = tf.get_default_graph().get_tensor_by_name('keep_prob:0'  )
    layer3_out  = tf.get_default_graph().get_tensor_by_name('layer3_out:0' )
    layer4_out  = tf.get_default_graph().get_tensor_by_name('layer4_out:0' )
    layer7_out  = tf.get_default_graph().get_tensor_by_name('layer7_out:0' )

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out 

tests.test_load_vgg(load_vgg, tf)

def kernel_initializer():
    return tf.truncated_normal_initializer(stddev=0.01)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, stddev=0.01):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :param stdev: truncated normal stddev
    :return: The Tensor for the last layer of output
    """
    layer7_conv      = tf.layers.conv2d(          vgg_layer7_out                   , num_classes, 1   , padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
    layer7_deconv    = tf.layers.conv2d_transpose(layer7_conv                      , num_classes, 4, 2, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
    layer4_conv      = tf.layers.conv2d(          vgg_layer4_out                   , num_classes, 1   , padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
    layer4_deconv    = tf.layers.conv2d_transpose(tf.add(layer7_deconv,layer4_conv), num_classes, 4, 2, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
    layer3_conv      = tf.layers.conv2d(          vgg_layer3_out                   , num_classes, 1   , padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
    layer_out_deconv = tf.layers.conv2d_transpose(tf.add(layer4_deconv,layer3_conv), num_classes,16, 8, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
    
    return layer_out_deconv

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits             = tf.reshape(nn_last_layer, (-1, num_classes))
    labels             = tf.reshape(correct_label, (-1, num_classes))
    
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op           = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, keep_prob_val, learning_rate_val, num_training_image, output_file):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param keep_prob_val: keep probability value
    :param learning_rate_val: learning rate value
    :param num_training_image: total number of training images
    :param output_file: file in which there are writen all data
    """
    
    # Training cycle
    
    f = open(output_file, "a")
    
    f.write(output_file+"\n")
    
    print("Training started:")
    print("Epoch = 1 / %d:" % (epochs))
    tot_loss = 0
    for epoch in range(epochs):
        tot_loss = 0
        tot_samples_length = 0
        for image, label in get_batches_fn(batch_size):
            tot_samples_length += len(image)
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={
                input_image:   image,
                correct_label: label,
                keep_prob:     keep_prob_val,
                learning_rate: learning_rate_val
            })
            tot_loss += loss
            string = "Loss: %g, Images: %d/%d %.2f%% done per epoch" % (loss, tot_samples_length, num_training_image, float(tot_samples_length)/float(num_training_image)*100.0)
            print(string)
            f.write(string+"\n")
      
        # Total training loss
        tot_loss /= tot_samples_length
        string = "Epoch = %d / %d Tot loss = %g" %(epoch+1,epochs,tot_loss)
        print(string)
        f.write(string+"\n")
    f.close()
    return tot_loss
   
tests.test_train_nn(train_nn)

def make_video(image_dir,fps,output_name) :
 
    video_name = os.path.join(image_dir,output_name)
 
    print("Creating video : %s"%video_name)

    images = [img for img in os.listdir(image_dir) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_dir, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_dir, image)))

    #cv2.destroyAllWindows()
    video.release()

def run():

    epochs            = 1
    batch_size        = 16
    keep_prob_val     = 0.5
    learning_rate_val = 0.0001
    truncated_normal_stddev=0.01
    
    if len(sys.argv)>1: epochs                  = int(  sys.argv[1])
    if len(sys.argv)>2: batch_size              = int(  sys.argv[2])
    if len(sys.argv)>3: keep_prob_val           = float(sys.argv[3])
    if len(sys.argv)>4: learning_rate_val       = float(sys.argv[4])
    if len(sys.argv)>5: truncated_normal_stddev = float(sys.argv[5])

    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    
    training_data_dir = 'data_road/training'
    tests.test_for_kitti_dataset(data_dir)
    
    num_training_image = 0
    for image_file in glob(os.path.join(data_dir,training_data_dir, 'image_2', '*.png')):
        num_training_image+=1
    print("Total training images = %d" % num_training_image)     
    
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, training_data_dir), image_shape)
        
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.int32  )
        learning_rate = tf.placeholder(tf.float32)
        
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_out_deconv                                           = layers(layer3_out, layer4_out, layer7_out, num_classes, stddev= truncated_normal_stddev)
        logits, train_op, cross_entropy_loss = optimize(layer_out_deconv, correct_label, learning_rate, num_classes)
        
        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver()
       
        output_dir_name = str(time.time())+ '_ep_%d_prob_%g_lrate_%g_stddev_%g'%(epochs, keep_prob_val, learning_rate_val,truncated_normal_stddev)
        
        output_file = os.path.join(runs_dir,"model_"+output_dir_name+".txt")
        
        loss = train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input, correct_label, keep_prob, learning_rate, keep_prob_val, learning_rate_val, num_training_image, output_file)
        
        output_dir_name += "_loss_%g"%(loss)
        output_file_with_loss = os.path.join(runs_dir,"model_"+output_dir_name+".txt")
                        
        os.rename(output_file,output_file_with_loss)
        
        print("Output model   : %s"%(output_dir_name))
        print("Saving file    : %s"%(output_file_with_loss))
        #save_path = saver.save(sess, runs_dir+"/model.ckpt")
        #print("Save model to file: %s"%save_path)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input, output_dir_name)

        # OPTIONAL: Apply the trained model to a video
        make_video(os.path.join(runs_dir,output_dir_name),2,"1_video.avi") 
        
if __name__ == '__main__':
    run()
