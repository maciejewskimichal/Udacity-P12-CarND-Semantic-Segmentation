# CarND-Semantic-Segmentation

[Self-Driving Car Engineer Nanodegree Program Term 3 Project 12](https://eu.udacity.com/course/self-driving-car-engineer-nanodegree--nd013)

The goal of this project is running a semantic segmanatation neuronal network to **find on car camera video stream pixels belonging to the road**. This is done by taking a [VGG-16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) network and extend it to to fully connected convolutional network ([FCN-VGG16](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)).

![Running demo FCN network](https://github.com/maciejewskimichal/Udacity-P12-CarND-Semantic-Segmentation/blob/master/runs/ep_50_prob_0.5_lrate_0.0001_stddev_0.01_loss_0.0049/output.gif)

This network has added skip connection to resolve better details and it looks like:

![FCN network with skip connections](https://github.com/maciejewskimichal/Udacity-P12-CarND-Semantic-Segmentation/blob/master/images/network_with_skip_conn.png)

In practice network is created by following code:

```python
model            = tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)

image_input      = tf.get_default_graph().get_tensor_by_name('image_input:0')
keep_prob        = tf.get_default_graph().get_tensor_by_name('keep_prob:0'  )
layer3_out       = tf.get_default_graph().get_tensor_by_name('layer3_out:0' )
layer4_out       = tf.get_default_graph().get_tensor_by_name('layer4_out:0' )
layer7_out       = tf.get_default_graph().get_tensor_by_name('layer7_out:0' )

    kern_init = tf.truncated_normal_initializer(stddev=0.01)
    # regularizer added in later version
    kern_reg  = tf.contrib.layers.l2_regularizer(1e-3)
    layer7_conv      = tf.layers.conv2d(          vgg_layer7_out                   , num_classes, 1   , padding='same', kernel_initializer=kern_init, kernel_regularizer=kern_reg)
    layer7_deconv    = tf.layers.conv2d_transpose(layer7_conv                      , num_classes, 4, 2, padding='same', kernel_initializer=kern_init, kernel_regularizer=kern_reg)
    layer4_conv      = tf.layers.conv2d(          vgg_layer4_out                   , num_classes, 1   , padding='same', kernel_initializer=kern_init, kernel_regularizer=kern_reg)
    layer4_deconv    = tf.layers.conv2d_transpose(tf.add(layer7_deconv,layer4_conv), num_classes, 4, 2, padding='same', kernel_initializer=kern_init, kernel_regularizer=kern_reg)
    layer3_conv      = tf.layers.conv2d(          vgg_layer3_out                   , num_classes, 1   , padding='same', kernel_initializer=kern_init, kernel_regularizer=kern_reg)
    layer_out        = tf.layers.conv2d_transpose(tf.add(layer4_deconv,layer3_conv), num_classes,16, 8, padding='same', kernel_initializer=kern_init, kernel_regularizer=kern_reg)
```

It was run with *AdamOptimizer*.

# Best parameters

Network was trained with [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) on Udacity VM with NVIDIA K80 GPU.

##First run

Best fit first run parameters were set to:
```python
epochs            = 20
batch_size        = 16
keep_prob_val     = 0.5
learning_rate_val = 0.0001
truncated_normal_stddev=0.01
regularizer not used

Epoch = 1 /20 Loss = 0.0313111
Epoch = 2 /20 Loss = 0.0239016
Epoch = 3 /20 Loss = 0.0212975
Epoch = 4 /20 Loss = 0.0165934
Epoch = 5 /20 Loss = 0.0113107
Epoch = 6 /20 Loss = 0.00814866
Epoch = 7 /20 Loss = 0.00651643
Epoch = 8 /20 Loss = 0.00672304
Epoch = 9 /20 Loss = 0.00592527
Epoch = 10/20 Loss = 0.00547108
Epoch = 11/20 Loss = 0.00471787
Epoch = 12/20 Loss = 0.00475233
Epoch = 13/20 Loss = 0.00479306
Epoch = 14/20 Loss = 0.00399876
Epoch = 15/20 Loss = 0.00388238
Epoch = 16/20 Loss = 0.00381149
Epoch = 17/20 Loss = 0.0042731
Epoch = 18/20 Loss = 0.00407425
Epoch = 19/20 Loss = 0.00355994
Epoch = 20/20 Loss = 0.00346907
```
Some examples of segmentation:

![Best1](https://github.com/maciejewskimichal/Udacity-P12-CarND-Semantic-Segmentation/tree/master/runs/ep_20_prob_0.5_lrate_0.0001_stddev_0.01_loss_0.0037/uu_000026.png)

![Best2](https://github.com/maciejewskimichal/Udacity-P12-CarND-Semantic-Segmentation/tree/master/runs/ep_20_prob_0.5_lrate_0.0001_stddev_0.01_loss_0.0037/uu_000093.png)

![Best3](https://github.com/maciejewskimichal/Udacity-P12-CarND-Semantic-Segmentation/tree/master/runs/ep_20_prob_0.5_lrate_0.0001_stddev_0.01_loss_0.0037/um_000081.png)

![Best4](https://github.com/maciejewskimichal/Udacity-P12-CarND-Semantic-Segmentation/tree/master/runs/ep_20_prob_0.5_lrate_0.0001_stddev_0.01_loss_0.0037/umm_000009.png)

![Best5](https://github.com/maciejewskimichal/Udacity-P12-CarND-Semantic-Segmentation/tree/master/runs/ep_20_prob_0.5_lrate_0.0001_stddev_0.01_loss_0.0037/umm_000025.png)

![Best6](https://github.com/maciejewskimichal/Udacity-P12-CarND-Semantic-Segmentation/tree/master/runs/ep_20_prob_0.5_lrate_0.0001_stddev_0.01_loss_0.0037/umm_000029.png)


##Seocnd optimized run

Best fit first run parameters were set to:
```python
epochs            = 50
batch_size        = 4
keep_prob_val     = 0.5
learning_rate_val = 0.0001
truncated_normal_stddev=0.01
regularizer used

Epoch = 1 /50 Loss = 0.126532
Epoch = 2 /50 Loss = 0.0387358
Epoch = 3 /50 Loss = 0.0336458
Epoch = 4 /50 Loss = 0.0266071
Epoch = 5 /50 Loss = 0.0225733
Epoch = 6 /50 Loss = 0.0208974
Epoch = 7 /50 Loss = 0.0181564
Epoch = 8 /50 Loss = 0.0153333
Epoch = 9 /50 Loss = 0.0161906
Epoch = 10/50 Loss = 0.0143521
Epoch = 15/50 Loss = 0.0106265
Epoch = 20/50 Loss = 0.010997
Epoch = 25/50 Loss = 0.00765995
Epoch = 30/50 Loss = 0.00681202
Epoch = 35/50 Loss = 0.00614292
Epoch = 40/50 Loss = 0.00591604
Epoch = 45/50 Loss = 0.00526914
Epoch = 50/50 Loss = 0.00486118
```
Few best examples of segmentation:

![Best1](https://github.com/maciejewskimichal/Udacity-P12-CarND-Semantic-Segmentation/tree/master/runs/ep_50_prob_0.5_lrate_0.0001_stddev_0.01_loss_0.0049/uu_000026.png)

![Best2](https://github.com/maciejewskimichal/Udacity-P12-CarND-Semantic-Segmentation/tree/master/runs/ep_50_prob_0.5_lrate_0.0001_stddev_0.01_loss_0.0049/uu_000093.png)

![Best3](https://github.com/maciejewskimichal/Udacity-P12-CarND-Semantic-Segmentation/tree/master/runs/ep_50_prob_0.5_lrate_0.0001_stddev_0.01_loss_0.0049/um_000081.png)

![Best4](https://github.com/maciejewskimichal/Udacity-P12-CarND-Semantic-Segmentation/tree/master/runs/ep_50_prob_0.5_lrate_0.0001_stddev_0.01_loss_0.0049/umm_000009.png)

![Best5](https://github.com/maciejewskimichal/Udacity-P12-CarND-Semantic-Segmentation/tree/master/runs/ep_50_prob_0.5_lrate_0.0001_stddev_0.01_loss_0.0049/umm_000025.png)

![Best6](https://github.com/maciejewskimichal/Udacity-P12-CarND-Semantic-Segmentation/tree/master/runs/ep_50_prob_0.5_lrate_0.0001_stddev_0.01_loss_0.0049/umm_000029.png)


# Project setup
### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

You may also need [Python Image Library (PIL)](https://pillow.readthedocs.io/) for SciPy's `imresize` function.

##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note:** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

#### Example Outputs
Here are examples of a sufficient vs. insufficient output from a trained network:

Sufficient Result          |  Insufficient Result
:-------------------------:|:-------------------------:
![Sufficient](./examples/sufficient_result.png)  |  ![Insufficient](./examples/insufficient_result.png)

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.

### Why Layer 3, 4 and 7?
In `main.py`, you'll notice that layers 3, 4 and 7 of VGG16 are utilized in creating skip layers for a fully convolutional network. The reasons for this are contained in the paper [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1605.06211.pdf).

In section 4.3, and further under header "Skip Architectures for Segmentation" and Figure 3, they note these provided for 8x, 16x and 32x upsampling, respectively. Using each of these in their FCN-8s was the most effective architecture they found. 

### Optional sections
Within `main.py`, there are a few optional sections you can also choose to implement, but are not required for the project.

1. Train and perform inference on the [Cityscapes Dataset](https://www.cityscapes-dataset.com/). Note that the `project_tests.py` is not currently set up to also unit test for this alternate dataset, and `helper.py` will also need alterations, along with changing `num_classes` and `input_shape` in `main.py`. Cityscapes is a much more extensive dataset, with segmentation of 30 different classes (compared to road vs. not road on KITTI) on either 5,000 finely annotated images or 20,000 coarsely annotated images.
2. Add image augmentation. You can use some of the augmentation techniques you may have used on Traffic Sign Classification or Behavioral Cloning, or look into additional methods for more robust training!
3. Apply the trained model to a video. This project only involves performing inference on a set of test images, but you can also try to utilize it on a full video.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
