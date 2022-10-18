from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

#tf.logging.set_verbosity(tf.logging.INFO)


from scipy import io

import matplotlib.pyplot as plt
#%matplotlib notebook


# You've just going to input the data in the main function if something happens (errors)

# Training Parameters
learning_rate = 0.001 #!!!!
# num_steps = 20
# display_step = 5

# Network Parameters
num_input = 2048
num_classes = 7 

batch_size = tf.placeholder(tf.int64)

#Exectute "Importing data" only once - it takes between 30-60 sec

#Exectute "Importing data" only once - it takes between 30-60 sec


#Importing data without Noise
bpskMatFile = io.loadmat('Datasets/bpsk_2048_5k.mat', squeeze_me=True)
bpskInput = bpskMatFile['bpsk_I']

qpskMatFile = io.loadmat('Datasets/qpsk_2048_5k.mat', squeeze_me=True)
qpskInput = qpskMatFile['qpsk_I']

qam16MatFile = io.loadmat('Datasets/qam16_2048_5k.mat', squeeze_me=True)
qam16Input = qam16MatFile['qam16_I']

qam64MatFile = io.loadmat('Datasets/qam64_2048_5k.mat', squeeze_me=True)
qam64Input = qam64MatFile['qam64_I']

ask2MatFile = io.loadmat('Datasets/ask2_2048_5k.mat', squeeze_me=True)
ask2Input = ask2MatFile['ask2_I']

fsk4MatFile = io.loadmat('Datasets/fsk4_2048_5k.mat', squeeze_me=True)
fsk4Input = fsk4MatFile['fsk4_I']

ofdmMatFile = io.loadmat('Datasets/ofdm_2048_5k_IQ.mat', squeeze_me=True)
ofdmInput = ofdmMatFile['ofdm_I']

# The seventh

#MatFile = io.loadmat('Datasets/_2048_5k.mat', squeeze_me=True)
#Input = MatFile['_I']


##Importing data with Noise
#
bpskMatFileNo = io.loadmat('Datasets/bpsk/bpsk_2048_5k_GN_12dB.mat', squeeze_me=True)
bpskInputNo = bpskMatFileNo['bpsk_I_GN']

qpskMatFileNo = io.loadmat('Datasets/qpsk/qpsk_2048_5k_GN_12dB.mat', squeeze_me=True)
qpskInputNo = qpskMatFileNo['qpsk_I_GN']

qam16MatFileNo = io.loadmat('Datasets/qam16/qam16_2048_5k_GN_12dB.mat', squeeze_me=True)
qam16InputNo = qam16MatFileNo['qam16_I_GN']

qam64MatFileNo = io.loadmat('Datasets/qam64/qam64_2048_5k_GN_12dB.mat', squeeze_me=True)
qam64InputNo = qam64MatFileNo['qam64_I_GN']

ask2MatFileNo = io.loadmat('Datasets/ask2/ask2_2048_5k_GN_12dB.mat', squeeze_me=True)
ask2InputNo = ask2MatFileNo['ask2_I_GN']

fsk4MatFileNo = io.loadmat('Datasets/fsk4/fsk4_2048_5k_GN_12dB.mat', squeeze_me=True)
fsk4InputNo = fsk4MatFileNo['fsk4_I_GN']

ofdmMatFileNo = io.loadmat('Datasets/ofdm/ofdm_2048_5k_GN_12dB.mat', squeeze_me=True)
ofdmInputNo = ofdmMatFileNo['ofdm_I_GN']

# The seventh

#MatFile = io.loadmat('Datasets/_2048_5k.mat', squeeze_me=True)
#Input = MatFile['_I']

k = 2048
kNo = 512

bpsk_train = bpskInput[0:k, :]
#Rescale
bpsk_train = bpsk_train / np.amax(bpsk_train)
print('BPSK train shape', bpsk_train.shape)

#bpsk_test = bpskInputNo[k:512+k, :]
#bpsk_test = bpsk_test / np.amax(bpsk_test)

bpsk_test = bpskInputNo[0:kNo, :]
bpsk_test = bpsk_test / np.amax(bpsk_test)
print('BPSK test shape', bpsk_test.shape)

#bpsk_test_labels = 
bpsk_labels = np.zeros((bpskInput.shape[0], num_classes))
bpsk_labels[:, 0] = 1                                            # It's just not working that way
print(bpsk_labels.shape)
print()


#bpsk_labels = np.ones((bpskInput.shape[0], 1))*1
#bpsk_labels[:, 0] = 1
#print('Shape of BPSK labels', bpsk_labels.shape)
#print()


qpsk_train = qpskInput[0:k, :]
qpsk_train = qpsk_train / np.amax(qpsk_train)

qpsk_test = qpskInputNo[0:kNo, :]
qpsk_test = qpsk_test / np.amax(qpsk_test)

qpsk_labels = np.zeros((qpskInput.shape[0], num_classes))
qpsk_labels[:, 1] = 1


#qpsk_labels = np.ones((qpskInput.shape[0], 1))*2


qam16_train = qam16Input[0:k, :]
qam16_train = qam16_train / np.amax(qam16_train)

qam16_test = qam16InputNo[0:kNo, :]
qam16_test = qam16_test / np.amax(qam16_test)

qam16_labels = np.zeros((qam16Input.shape[0], num_classes))
qam16_labels[:, 2] = 1


#qam16_labels = np.ones((qam16Input.shape[0], 1))*3


qam64_train = qam64Input[0:k, :]
#Rescale
qam64_train = qam64_train / np.amax(qam64_train)

qam64_test = qam64InputNo[0:kNo, :]
qam64_test = qam64_test / np.amax(qam64_test)

qam64_labels = np.zeros((qam64Input.shape[0], num_classes))
qam64_labels[:, 3] = 1


#qam64_labels = np.ones((qam64Input.shape[0], 1))*4

###

ask2_train = ask2Input[0:k, :]
ask2_train = ask2_train / np.amax(ask2_train)

ask2_test = ask2InputNo[0:kNo, :]
ask2_test = ask2_test / np.amax(ask2_test)

ask2_labels = np.zeros((ask2Input.shape[0], num_classes))
ask2_labels[:, 4] = 1


fsk4_train = fsk4Input[0:k, :]
fsk4_train = fsk4_train / np.amax(fsk4_train)

fsk4_test = fsk4InputNo[0:kNo, :]
fsk4_test = fsk4_test / np.amax(fsk4_test)

fsk4_labels = np.zeros((fsk4Input.shape[0], num_classes))
fsk4_labels[:, 5] = 1


ofdm_train = ofdmInput[0:k, :]
ofdm_train = ofdm_train / np.amax(ofdm_train)

ofdm_test = ofdmInputNo[0:kNo, :]
ofdm_test = ofdm_test / np.amax(ofdm_test)

ofdm_labels = np.zeros((ofdmInput.shape[0], num_classes))
ofdm_labels[:, 6] = 1

# The seventh

#_train = qInput[0:k, :]
#_train = _train / np.amax(_train)

#_test = Input[k:512+k, :]
#_test = _test / np.amax(_test)

#_labels = np.zeros((Input.shape[0], num_classes))
#_labels[:, 7] = 1

###


train_data = np.append(bpsk_train, qpsk_train, 0)
train_data = np.append(train_data, qam16_train, 0)
train_data = np.append(train_data, qam64_train, 0)

train_data = np.append(train_data, ask2_train, 0)
train_data = np.append(train_data, fsk4_train, 0)
train_data = np.append(train_data, ofdm_train, 0)
#train_data = np.append(train_data, _train, 0)


test_data = np.append(bpsk_test, qpsk_test, 0)
test_data = np.append(test_data, qam16_test, 0)
test_data = np.append(test_data, qam64_test, 0)

test_data = np.append(test_data, ask2_test, 0)
test_data = np.append(test_data, fsk4_test, 0)
test_data = np.append(test_data, ofdm_test, 0)
#test_data = np.append(test_data, _test, 0)

train_labels = np.append(bpsk_labels[0:k, :], qpsk_labels[0:k, :], 0)
train_labels = np.append(train_labels, qam16_labels[0:k, :], 0)
train_labels = np.append(train_labels, qam64_labels[0:k, :], 0)

train_labels = np.append(train_labels, ask2_labels[0:k, :], 0)
train_labels = np.append(train_labels, fsk4_labels[0:k, :], 0)
train_labels = np.append(train_labels, ofdm_labels[0:k, :], 0)

#the eight
#train_labels = np.append(train_labels, _labels[0:k, :], 0)



#test_labels = np.append(bpsk_labels[k:512+k, :], qpsk_labels[k:512+k, :], 0)
#test_labels = np.append(test_labels, qam16_labels[k:512+k, :], 0)
#test_labels = np.append(test_labels, qam64_labels[k:512+k, :], 0)
#
#test_labels = np.append(test_labels, ask2_labels[k:512+k, :], 0)
#test_labels = np.append(test_labels, fsk4_labels[k:512+k, :], 0)
#test_labels = np.append(test_labels, ofdm_labels[k:512+k, :], 0)

# the eight
#test_labels = np.append(test_labels, _labels[k:512+k, :], 0)

test_labels = np.append(bpsk_labels[0:kNo, :], qpsk_labels[0:kNo, :], 0)
test_labels = np.append(test_labels, qam16_labels[0:kNo, :], 0)
test_labels = np.append(test_labels, qam64_labels[0:kNo, :], 0)

test_labels = np.append(test_labels, ask2_labels[0:kNo, :], 0)
test_labels = np.append(test_labels, fsk4_labels[0:kNo, :], 0)
test_labels = np.append(test_labels, ofdm_labels[0:kNo, :],None 0)

# the eight
#test_labels = np.append(test_labels, _labels[0:kNo, :], 0)

print('Train data shape',train_data.shape, 'Train data labels', train_labels.shape,
      'Test data shape', test_data.shape, 'Test data labels', test_labels.shape)



#
BATCH_SIZE = 1 

X = tf.placeholder(tf.float32, shape=[None, 2, 2048])
Y = tf.placeholder(tf.float32, shape=[None, 4])

dataset = tf.data.Dataset.from_tensor_slices((X, Y)).batch(BATCH_SIZE).repeat()
# print(dataset.get_shape)
#print(dataset)

train_data = (train_data, train_labels)
test_data = (test_data, test_labels)

#print(train_data)


#iter = dataset.make_initializable_iterator()
#features, labels = iter.get_next()

#print('Features')
#print(features)

#print('Labels')
#print(labels)



def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel

  # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])    #[batch_size, image_height, image_width, channels 
                                                                # (3 for rgb, 1 for grayscale)] in original code
                                                                # -1 for batch size - it will be computed

  input_layer = tf.reshape(features["x"], [-1, 2048, 1, 1])     # shape of the matrix

  #give a try if the things are not going well
  #input_layer = tf.reshape(features["x"], [-1, 2048, 1])       # 3dimension shape of the matrix you could multiply by 2
  
  #print('Input_layer shape:',input_layer.get_shape)

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]None
  # Output Tensor Shape: [batch_size, 28, 28, 32]
    
  conv1 = tf.layers.conv2d(                                     # of the original code
      inputs=input_layer,
      filters=32,                                               # number of filters
      kernel_size=[5, 5],                                       # it's just five
      padding="same",                                           # the outpit will be the same dimension as the input 32x32
      activation=tf.nn.relu)                                    # the output: [batch_size, 28, 28, 32]


  #print('Conv1 shape:', conv1.get_shape)                        #
    
  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
    
  #
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 1], strides=2) # [batch_size, 14, 14, 32]
  #pool1 = tf.nn.max_pool(value=conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  #pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2) # [batch_size, 14, 14, 32] 1d layer

  #print('Pool1 shape:', pool1.get_shape)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]

  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)                                    # [batch_size, 14, 14, 64]

  #print('Conv2 shape:', conv2.get_shape)
    
  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]

  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 1], strides=2)        # [batch_size, 7, 7, 64]
  #pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  #pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=[2, 1], strides=2)        # [batch_size, 7, 7, 64]
    
  #print('Pool2 shape:',pool2.get_shape)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    
  # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])                                  # the shape of every reshaped element
  
  
  #n_features_t1 = pool2.get_shape().as_list()[1]
  #n_features_t2 = pool2.get_shape().as_list()[2]
  #n_features_t3 = pool2.get_shape().as_list()[3]
    
  #print('N features 1:', n_features_t1)
  #print('N features 2:', n_features_t2)
  #print('N features 3:', n_features_t3)
  
    
  n_features_p2 = pool2.get_shape().as_list()[1] * pool2.get_shape().as_list()[2] * pool2.get_shape().as_list()[3]## 
  
  #print('N features after equation', n_features_p2)
    
  pool2_flat = tf.reshape(pool2, [-1, n_features_p2])                # the shape of every reshaped element
                                                                     # 8 numbers of layers at the end its a magic num
  #print('Pool2_flat shape:',pool2_flat.get_shape)
    
  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
    
  #dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)   # units are the number of neurons
                                                                                   # [batch_size, 3136] 

  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)    # units are the number of neurons
                                                                                   # [batch_size, 8192] 
 
  #print('Dense shape:',dense.get_shape)
    
  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)        # rate - percent of the dropout elements
   
  #print('Dropout shape:', dropout.get_shape)
    
  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  # logits = tf.layers.dense(inputs=dropout, units=10)                          # 10 is the nuber of digits (0-9)
  logits = tf.layers.dense(inputs=dropout, units=num_classes)                   # broi modulacii
                                                                                # output [batch_size 10] 
  #print('Logits shape:',logits.get_shape)
  #print('We are...', logits)
  #print('Labels shape:',labels.get_shape)
  #print('We are...', labels)
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.argmax(labels,1), logits=logits)
  
  loss1 = tf.losses.sparse_softmax_cross_entropy(labels=tf.argmax(labels,1), logits=logits)
  #predictionTW = tf.nn.softmax(logits)
  #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
  #  logits=predictionTW, labels=labels))

  #print('Shape of the loss variable', loss.get_shape())

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)          # learning rate and optimizer
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
 
    
  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=tf.argmax(labels,1), predictions=predictions["classes"])}
      
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss1, eval_metric_ops=eval_metric_ops)
 
    
def main(unused_argv):
  # Load training and eval data
  global train_data, train_labels, test_data, test_labels             #####
  #train_data = train_data
  #train_labels = train_labels
  #test_data = test_data
  #test_labels = test_labels

  #print('Train data shape',train_data.shape[0], 'Train data labels', train_labels.shape[0],
  #    'Test data shape', test_data.shape[0], 'Test data labels', test_labels.shapep[0])
    
  BATCH_SIZE = 512
  n_batches = train_data[0].shape[0] // BATCH_SIZE

  print(n_batches)
  #print(train_data[0].shape, train_data[1].shape)   

# Create the Estimator
  mod_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/localfiles/mod_convnet_model/All_mod_Noise2_SNR_-12dB")

# Set up logging for predictions
# Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
    
  logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=500)
  for i in range(0,20):
  # Train the model
      train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data[0]},
        y=train_data[1],
        batch_size=BATCH_SIZE,                                                         #100 
        num_epochs=1,    #                                                     #till the end and number of steps (20000)
        shuffle=True)
      mod_classifier.train(
        input_fn=train_input_fn,
        steps=None,                                                             # 20000
        hooks = [logging_hook] )

    # Evaluate the model and print results
      test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data[0]},
        y=test_data[1],
        num_epochs=1, #
        shuffle=True)
      eval_results = mod_classifier.evaluate(input_fn=test_input_fn)
      print("")
      print("eval_results", eval_results)
      print("")
  
if __name__ == "__main__":
        tf.app.run()
  

print("Optimization Finished!")

