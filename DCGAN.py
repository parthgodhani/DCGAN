import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")


def generator(inputs, training=False, reuse=False):

    with tf.variable_scope("Generator",reuse=reuse):
        with tf.variable_scope('reshape'):
            outputs = tf.layers.dense(inputs, 1024 * 4 * 4)

            outputs = tf.reshape(outputs, [-1, 4, 4, 1024])

            outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
        with tf.variable_scope("deconv1"):
            outputs = tf.layers.conv2d_transpose(outputs,512, [5,5] , strides=(2,2), padding="SAME")          #tf.layers.conv2d_transpose(inputs,filters,kernel_size,
            outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training,name="outputs"))
        with tf.variable_scope("deconv2"):
            outputs = tf.layers.conv2d_transpose(outputs,256,[5,5],strides=(2,2),padding="SAME")
            outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training,name="outputs"))
        with tf.variable_scope("deconv3"):
            outputs = tf.layers.conv2d_transpose(outputs,128,[5,5],strides=(2,2),padding="SAME")
            outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training,name="outputs"))
        with tf.variable_scope("deconv4"):
            outputs = tf.layers.conv2d_transpose(outputs,1,[5,5],strides=(2,2),padding="SAME")
            #output the image
        with tf.variable_scope("tanh"):
            outputs = tf.tanh(outputs,name="outputs")

    return outputs


def discriminator(inputs, training = False,reuse=False):

    with tf.variable_scope("Discriminator",reuse=reuse):
        with tf.variable_scope("conv1"):

            outputs = tf.layers.conv2d(inputs, 64,[5,5],strides=(2,2), padding="SAME")
            outputs = tf.nn.leaky_relu(tf.layers.batch_normalization(outputs, training=training),alpha=0.2)
        with tf.variable_scope("conv2"):
            outputs = tf.layers.conv2d(outputs, 128,[5,5],strides=(2,2), padding="SAME")
            outputs = tf.nn.leaky_relu(tf.layers.batch_normalization(outputs, training=training),alpha=0.2)
        with tf.variable_scope("conv3"):
            outputs = tf.layers.conv2d(outputs, 256,[5,5],strides=(2,2), padding="SAME")
            outputs = tf.nn.leaky_relu(tf.layers.batch_normalization(outputs, training=training),alpha=0.2)
        with tf.variable_scope("conv4"):
            outputs = tf.layers.conv2d(outputs, 512,[5,5],strides=(2,2), padding="SAME")
            outputs = tf.nn.leaky_relu(tf.layers.batch_normalization(outputs, training=training),alpha=0.2)

        with tf.variable_scope("classify"):
            # batch_size = outputs.get_shape()[0].value
            # print(batch_size)
            # print("batch size test")
            # reshape = tf.reshape(outputs,[100, -1])
            # outputs = tf.layers.dense(reshape,2,name="outputs")
            logits = tf.layers.conv2d(outputs, 1, [5, 5], strides=(5, 5), padding='SAME')
            outputs = tf.nn.sigmoid(logits)
                # output layer

    return outputs,logits


### DATA ### TODO BATCH SIZE 10 or 100
batch_size = 10

# variables : input
x = tf.placeholder(tf.float32, shape=(None, 64, 64, 1))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
isTrain = tf.placeholder(dtype=tf.bool)

# networks : generator
G_z = generator(z, isTrain) # shape=(?, 64, 64, 1)

# networks : discriminator on Real and Fake data
D_real, D_real_logits = discriminator(x, isTrain) # shape=(?, 1, 1, 1)
D_fake, D_fake_logits = discriminator(G_z, isTrain, reuse=True) # shape=(?, 1, 1, 1)


# loss for each network
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))

### training operations ###

# AdamOpt Parameters
learning_rate = 0.0002
beta1 = 0.5

g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Generator")
d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Discriminator")

g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)

g_opt_op = g_opt.minimize(G_loss,var_list=g_variables)
d_opt_op = d_opt.minimize(D_loss,var_list=d_variables)

# Loss containers
G_losses = []
D_losses = []


z_dim = 100
train_epochs = 20
max_steps = mnist.train.num_examples // batch_size # 550

# for visualization purpose of learned generator
fixed_z_ = np.random.uniform(-1, 1, size=(batch_size ,1,1, z_dim)).astype(np.float32)

# for checkpoints saving
saver = tf.train.Saver(max_to_keep=None)

# TODO for the issue of OOM on GPU : choose either CPU or smaller batch size
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

# if i have to start using cpu for more conv filter channels because gpu too small
# config = tf.ConfigProto(
#         device_count = {'GPU': 0}
#     )
# tf.Session(config=config)
print("started at: ",time.ctime())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for e in range(train_epochs): # train_epochs 20

        G_loss_epoch = []
        D_loss_epoch = []
        print("epoch: ",e)
        print(time.ctime())

        for step in range(max_steps): # max_steps 550
            if(step%100==0):
                print("step : ",step," / ",max_steps," | epoch : ",e," / ",train_epochs)

            # reshape training data
            real_image_batch = mnist.train.next_batch(batch_size)[0] # (100, 784)
            real_image_batch = np.reshape(real_image_batch,(-1,28,28,1)) # (100, 28, 28, 1)
            train_images_ = tf.image.resize_images(real_image_batch, [64,64]).eval() # (100, 64, 64, 1)

            # discriminator update
            z_ = np.random.uniform(-1, 1, size=(batch_size ,1,1, z_dim)).astype(np.float32)
            loss_d_, _ = sess.run([D_loss, d_opt_op], {x: train_images_, z: z_,isTrain: True})
            D_loss_epoch.append(loss_d_)

            # generator update
            z_ = np.random.uniform(-1, 1, size=(batch_size ,1,1, z_dim)).astype(np.float32)
            loss_g_, _ = sess.run([G_loss, g_opt_op], {x: train_images_, z: z_,isTrain: True})
            G_loss_epoch.append(loss_g_)

        # add to losses the mean of last epoch
        G_losses.append(np.mean(G_loss_epoch))
        D_losses.append(np.mean(D_loss_epoch))
        print("epoch : ",str(e))
        print("LOSSES")
        print("g : {} | d : {}".format(G_losses[-1],D_losses[-1]))

        # testing generator output  for last z value
        test_g = sess.run(G_z,{z: fixed_z_, isTrain: False}) # (100, 64, 64, 1)

        #print("saving 10 images of epoch: ",e)
        #for k in range(10): # save 10 of 100 images
        #    plt.imshow(np.reshape(test_g[k,:,:,:],(64,64)))
        #    plt.savefig("./results/"+"e"+str(e+1)+"i"+str(k)+".png")

        # saving checkpoint
        print("saving checkpoint"," of epoch: ",str(e))
        save_path = saver.save(sess, "./checkpoint_smallbatch/"+str(e)+"model.ckpt")


    # TODO save losses using tensorboard summary.scalar
    print("dumping losses")
    pickle.dump( G_losses, open( "G_losses.p", "wb" ) )

print("Training done")
