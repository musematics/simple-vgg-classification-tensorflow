import os
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage import io, transform
import glob
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
path = './easy/'          # for one dataset cross validation
train_path = './example/train/' # for train and test set
test_path = './example/test/'
w = 224
h = 224
c = 3
n_class = 4

def read_img(path):
    cate   = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs   = []
    labels = []
    label_list = np.eye(n_class)
    for idx, folder in enumerate(cate):            #search folder
        for im in glob.glob(folder + '/*.jpg'):    #change doc type if necessary
            img = io.imread(im)
            img = transform.resize(img, (w, h, c))
            imgs.append(img)                        # (sum,224,224,3)
            labels.append(label_list[idx])          # (sum,4)                         
    return np.asarray(imgs, np.float32), np.asarray(labels, np.float32)

#------------------train and test set------------
data, label = read_img(train_path)
   
num_example = data.shape[0]                        
arr = np.arange(num_example)                    
np.random.shuffle(arr)                           # random sequence
x_train = data[arr]
y_train = label[arr]

s = num_example

data_t, label_t = read_img(test_path)
   
s_test = data_t.shape[0]                        
arr = np.arange(s_test)                    
np.random.shuffle(arr)                           # random sequence
x_val = data_t[arr]
y_val = label_t[arr]

# ------------------one dataset cross validation ----------
#data, label = read_img(path)
   
#num_example = data.shape[0]                        
#arr = np.arange(num_example)                    
#np.random.shuffle(arr)                           # random sequence
#data = data[arr]
#label = label[arr]

#ratio = 0.8
#s = np.int(num_example * ratio)
#x_train = data[:s]                         # (sum_train,224,224,3)
#y_train = label[:s]                        # (sum_train,4)
#x_val   = data[s:]
#y_val   = label[s:]    

#------------------vgg16 structure----------------
 
x = tf.placeholder(tf.float32, shape=[None, h, w, c])
y = tf.placeholder(tf.float32, shape=[None, n_class])     

#----------------- conv1 ------------------------

w_conv1_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1))
b_conv1_1 = tf.Variable(tf.constant(0.1, shape=[64]))
L_conv1_1 = tf.nn.relu(tf.nn.conv2d(x, w_conv1_1,strides=[1, 1, 1, 1], padding='SAME') + b_conv1_1)

w_conv1_2 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
b_conv1_2 = tf.Variable(tf.constant(0.1, shape=[64]))
L_conv1_2 = tf.nn.relu(tf.nn.conv2d(L_conv1_1, w_conv1_2,strides=[1, 1, 1, 1], padding='SAME') + b_conv1_2)

L_pool1 = tf.nn.max_pool(L_conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#----------------- conv2 ------------------------

w_conv2_1 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
b_conv2_1 = tf.Variable(tf.constant(0.1, shape=[128]))
L_conv2_1 = tf.nn.relu(tf.nn.conv2d(L_pool1, w_conv2_1,strides=[1, 1, 1, 1], padding='SAME') + b_conv2_1)

w_conv2_2 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1))
b_conv2_2 = tf.Variable(tf.constant(0.1, shape=[128]))
L_conv2_2 = tf.nn.relu(tf.nn.conv2d(L_conv2_1, w_conv2_2,strides=[1, 1, 1, 1], padding='SAME') + b_conv2_2)

L_pool2 = tf.nn.max_pool(L_conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#-------------------conv3 -------------------------

w_conv3_1 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1))
b_conv3_1 = tf.Variable(tf.constant(0.1, shape=[256]))
L_conv3_1 = tf.nn.relu(tf.nn.conv2d(L_pool2, w_conv3_1,strides=[1, 1, 1, 1], padding='SAME') + b_conv3_1)


w_conv3_2 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1))
b_conv3_2 = tf.Variable(tf.constant(0.1, shape=[256]))
L_conv3_2 = tf.nn.relu(tf.nn.conv2d(L_conv3_1, w_conv3_2,strides=[1, 1, 1, 1], padding='SAME') + b_conv3_2)


w_conv3_3 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1))
b_conv3_3 = tf.Variable(tf.constant(0.1, shape=[256]))
L_conv3_3 = tf.nn.relu(tf.nn.conv2d(L_conv3_2, w_conv3_3,strides=[1, 1, 1, 1], padding='SAME') + b_conv3_3)

L_pool3 = tf.nn.max_pool(L_conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#---------------------- conv4 -----------------------------------

w_conv4_1 = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.1))
b_conv4_1 = tf.Variable(tf.constant(0.1, shape=[512]))
L_conv4_1 = tf.nn.relu(tf.nn.conv2d(L_pool3, w_conv4_1,strides=[1, 1, 1, 1], padding='SAME') + b_conv4_1)


w_conv4_2 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1))
b_conv4_2 = tf.Variable(tf.constant(0.1, shape=[512]))
L_conv4_2 = tf.nn.relu(tf.nn.conv2d(L_conv4_1, w_conv4_2,strides=[1, 1, 1, 1], padding='SAME') + b_conv4_2)


w_conv4_3 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1))
b_conv4_3 = tf.Variable(tf.constant(0.1, shape=[512]))
L_conv4_3 = tf.nn.relu(tf.nn.conv2d(L_conv4_2, w_conv4_3,strides=[1, 1, 1, 1], padding='SAME') + b_conv4_3)

L_pool4 = tf.nn.max_pool(L_conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#---------------------- conv5 -----------------------------------

w_conv5_1 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1))
b_conv5_1 = tf.Variable(tf.constant(0.1, shape=[512]))
L_conv5_1 = tf.nn.relu(tf.nn.conv2d(L_pool4, w_conv5_1,strides=[1, 1, 1, 1], padding='SAME') + b_conv5_1)


w_conv5_2 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1))
b_conv5_2 = tf.Variable(tf.constant(0.1, shape=[512]))
L_conv5_2 = tf.nn.relu(tf.nn.conv2d(L_conv5_1, w_conv5_2,strides=[1, 1, 1, 1], padding='SAME') + b_conv5_2)


w_conv5_3 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1))
b_conv5_3 = tf.Variable(tf.constant(0.1, shape=[512]))
L_conv5_3 = tf.nn.relu(tf.nn.conv2d(L_conv5_2, w_conv5_3,strides=[1, 1, 1, 1], padding='SAME') + b_conv5_3)

L_pool5 = tf.nn.max_pool(L_conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#------------------------ fully connected 6 -------------------------------------
shape6 = int(np.prod(L_pool5.get_shape()[1:]))   
w_fc6 = tf.Variable(tf.truncated_normal([shape6, 4096], stddev=0.1))
b_fc6 = tf.Variable(tf.constant(0.1, shape=[4096]))
f_fc6 = tf.reshape(L_pool5, [-1, shape6])
L_fc6 = tf.nn.relu(tf.matmul(f_fc6, w_fc6) + b_fc6)

#------------------------  fully connected 7  ----------------------------

# ---------------------- Drop --------------------------
keep_prob = tf.placeholder(tf.float32)
d_fc7 = tf.nn.dropout(L_fc6, keep_prob)

w_fc7 = tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.1))
b_fc7 = tf.Variable(tf.constant(0.1, shape=[4096]))
L_fc7 = tf.nn.relu(tf.matmul(d_fc7, w_fc7) + b_fc7)

#------------------------ fully connected 8 ------------------------------------

w_fc8 = tf.Variable(tf.truncated_normal([4096, n_class], stddev=0.1))
b_fc8 = tf.Variable(tf.constant(0.1, shape=[n_class]))
L_fc8 = tf.matmul(L_fc7, w_fc8) + b_fc8

#-------------------- final output -------------------------------

#y_conv = tf.nn.softmax(L_fc8)
y_conv = L_fc8
y_ = tf.nn.softmax(L_fc8)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --------------- save the model ----------------------------
saver = tf.train.Saver()

#------------------ run --------------------------------

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())

    print ("Input %s images, %s labels" % (s, s)) #training set

    # divide batches
    batch_size = 2
    epochs = 100
    batches_count = int(s / batch_size) # iterations per epoch
    remainder = s % batch_size
    print ("Dataset is divided into %s batches,  %s images per batch, %s images for last batch" % (batches_count+1, batch_size, remainder))
    prev_loss = 0     # determination of convergence 
    stable_epoch = 0  # determination of convergence 

    # training
    for ep in range(epochs):
        ep_acc = 0
        ep_loss = 0		
        # transfer input to np.array
        for n in range(batches_count):          
            train_step.run(feed_dict={x: x_train[n*batch_size:(n+1)*batch_size], y: y_train[n*batch_size:(n+1)*batch_size], keep_prob: 0.99})
            acc, loss = sess.run([accuracy, cross_entropy], feed_dict={x: x_train[n*batch_size:(n+1)*batch_size], y: y_train[n*batch_size:(n+1)*batch_size], keep_prob: 0.99})
            ep_acc = ep_acc + acc
            ep_loss = ep_loss + loss
        ep_acc = ep_acc * batch_size
        ep_loss = ep_loss * batch_size		
        if remainder > 0:
            start_index = batches_count * batch_size;
            train_step.run(feed_dict={x: x_train[start_index:s-1], y: y_train[start_index:s-1], keep_prob: 0.99})
            acc, loss = sess.run([accuracy, cross_entropy], feed_dict={x: x_train[start_index:s-1], y: y_train[start_index:s-1], keep_prob: 0.99})
            acc = acc * remainder
            loss = loss * remainder
            ep_acc = ep_acc + acc
            ep_loss = ep_loss + loss			
        ep_acc = ep_acc / s
        ep_loss = ep_loss / s        			
        if prev_loss == ep_loss:   # determination of convergence
            stable_epoch = stable_epoch + 1
        else:
            stable_epoch = 0
            prev_loss = ep_loss
        if ep%5 == 0:
            print ('epoch %d: training accuracy %s' % (ep, ep_acc))
            print ('epoch %d: loss %s' % (ep, ep_loss))
            val_acc, val_loss = sess.run([accuracy, cross_entropy],feed_dict={x: x_val, y: y_val, keep_prob: 1.0})
            print ('epoch %d: testing accuracy %s' % (ep, val_acc))			
        if stable_epoch > 20:       #converges after 20 epoches
            break
    save_path = saver.save(sess,"./vgg.ckpt")  
    print ('--------- training finished! --------------')
    print ('Total epochs %d: training accuracy %s' % (ep, ep_acc))
    print ('Total epochs %d: loss %s' % (ep, ep_loss))
    val_acc, val_loss = sess.run([accuracy, cross_entropy],feed_dict={x: x_val, y: y_val, keep_prob: 1.0})
    print ('Total epochs %d: testing accuracy %s' % (ep, val_acc))
