
# coding: utf-8

# Deep Learning
# =============
# 
# Assignment 6
# ------------
# 
# After training a skip-gram model in `5_word2vec.ipynb`, the goal of this notebook is to train a LSTM character model over [Text8](http://mattmahoney.net/dc/textdata) data.

# In[112]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import os
import numpy as np
import random
import string
import tensorflow as tf
import urllib
import zipfile
import time
get_ipython().magic(u'matplotlib inline')


# In[2]:

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print 'Found and verified', filename
  else:
    print statinfo.st_size
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)


# In[9]:

def read_data(filename):
  f = zipfile.ZipFile(filename)
  print f.namelist()
  for name in f.namelist():
    return f.read(name)
  f.close()
  
text = read_data(filename)
print "Data size", len(text)


# In[10]:

# look at some text
print text[:100]
print text[-100:]


# ### Create a small validation set.

# In[17]:

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print train_size, train_text[:64]
print valid_size, valid_text[:64]


# ### Utility functions to map characters to vocabulary IDs and back.

# In[11]:

string.ascii_lowercase


# In[15]:

vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])  # ordinal of letter 'a'

def char2id(char):
  if char in string.ascii_lowercase:  # if detect a lowercase letter
    return ord(char) - first_letter + 1  # return value 1-26
  elif char == ' ':
    return 0
  else:
    print 'Unexpected character:', char
    return 0
  
def id2char(dictid):
  if dictid > 0:
    return chr(dictid + first_letter - 1)  # return [a-z]
  else:
    return ' '

print char2id('a'), char2id('z'), char2id(' '), char2id('ï')
print('-'*16)
print id2char(1), id2char(26), id2char(0)


# ### Function to generate a training batch for the LSTM model.

# In[250]:

batch_size=64
num_unrollings=10

class BatchGenerator(object):
  def __init__(self, text, batch_size, num_unrollings):
    self._text = text
    self._text_size = len(text)
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings
    segment = self._text_size / batch_size
    # list of offsets within batch
    self._cursor = [ offset * segment for offset in xrange(batch_size)]
    self._last_batch = self._next_batch()
  
  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
    for b in xrange(self._batch_size):
      batch[b, char2id(self._text[self._cursor[b]])] = 1.0  # get id of a char
      self._cursor[b] = (self._cursor[b] + 1) % self._text_size  # move cursor
    return batch
  
  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    for step in xrange(self._num_unrollings):
      batches.append(self._next_batch())  # add id of char for 1 to num_unrollings
    self._last_batch = batches[-1]
    return batches

def characters(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (most likely) character representation.
  """
  return [id2char(c) for c in np.argmax(probabilities, 1)]  # get char of an id

def batches2string(batches):
  """Convert a sequence of batches back into their (most likely) string
  representation.
  """
  #print 'Batch shape: {}'.format(batches[0].shape)
  #print 'First batch, First char: {}'.format(batches[0][0])
  s = [''] * batches[0].shape[0]
  for b in batches:
    s = [''.join(x) for x in zip(s, characters(b))]
  return s


# ### Generate batches

# In[270]:

# training and validation batches
train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1) # returns batch size 1, +1 unrolling 

# look at the text from various segments
segment_look = 0
num_char = 64
show = segment_look * len(train_text)/batch_size
print "index {} to {}:\n{}".format(show, show+num_char, train_text[show:show+num_char])
print('-'*16)

print batches2string(train_batches.next())
print batches2string(train_batches.next())
print('-'*16)
print batches2string(valid_batches.next())
print batches2string(valid_batches.next())


# ### Functions for predictions

# In[75]:

def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in xrange(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p

def random_distribution():
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
  return b/np.sum(b, axis=1)[:,None]


# ### Simple LSTM Model
# http://colah.github.io/posts/2015-08-Understanding-LSTMs/  
# 
# <img width="60%" src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png">

# In[173]:

num_nodes = 64

graph = tf.Graph()
with graph.as_default():
  
  ## Parameters:
  # Input (Write) gate: input, previous output, and bias.
  ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  ib = tf.Variable(tf.zeros([1, num_nodes]))
    
  # Forget (Keep) gate: input, previous output, and bias.
  fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  fb = tf.Variable(tf.zeros([1, num_nodes]))
    
  # Memory cell: input, state and bias.        
  cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  cb = tf.Variable(tf.zeros([1, num_nodes]))
    
  # Output (Read) gate: input, previous output, and bias.
  ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  ob = tf.Variable(tf.zeros([1, num_nodes]))
    
  # Variables saving state across unrollings.
  saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)

  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
  b = tf.Variable(tf.zeros([vocabulary_size]))
  
  # Definition of the cell computation.
  def lstm_cell(i, o, state):
    """
    Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
    Note that in this formulation, we omit the various connections between the
    previous state and the gates.
    """
    # what to keep (1) or forget (0) from cell state
    forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
    
    # new info to store in cell state
    # values to update
    input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
    # new candidate values to add to state
    update = tf.tanh(tf.matmul(i, cx) + tf.matmul(o, cm) + cb)
    
    # update old cell state C[t-1] into new cell state C[t]
    state = (forget_gate * state) + (input_gate * update)
    
    # decide the output
    output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
    h = output_gate * tf.tanh(state)
    return h, state

  # Input data.
  train_data = list()
  for _ in xrange(num_unrollings + 1):
    train_data.append(
      tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
  train_inputs = train_data[:num_unrollings]
  train_labels = train_data[1:]  # labels are inputs shifted by one time step.

  # Unrolled LSTM loop.
  outputs = list()
  output = saved_output
  state = saved_state
  for i in train_inputs:
    output, state = lstm_cell(i, output, state)
    outputs.append(output)

  # State saving across unrollings.
  with tf.control_dependencies([saved_output.assign(output),
                                saved_state.assign(state)]):
    # Classifier.
    logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(
        logits, tf.concat(0, train_labels)))

  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(
    10.0, global_step, 5000, 0.1, staircase=False)  ## orig 10.0, 5000, 0.1, True
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(
    zip(gradients, v), global_step=global_step)

  # Predictions.
  train_prediction = tf.nn.softmax(logits)
  
  # Sampling and validation eval: batch 1, no unrolling.
  sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
  saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
  reset_sample_state = tf.group(
    saved_sample_output.assign(tf.zeros([1, num_nodes])),
    saved_sample_state.assign(tf.zeros([1, num_nodes])))
  sample_output, sample_state = lstm_cell(
    sample_input, saved_sample_output, saved_sample_state)
  with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                saved_sample_state.assign(sample_state)]):
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))


# ### Run the LSTM

# In[174]:

num_steps = 7001  ## orig 7001
summary_frequency = 100

t0 = time.time()
with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print 'Initialized\n=========='
  mean_loss = 0
  for step in xrange(num_steps):
    batches = train_batches.next()
    feed_dict = dict()
    for i in xrange(num_unrollings + 1):
      feed_dict[train_data[i]] = batches[i]
    _, l, predictions, lr = session.run(
      [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % (5.*summary_frequency) == 0:  ## orig 2.5*summary_frequency
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print 'Average loss at step ', step, '=', mean_loss, '\nlearning rate:', lr
      mean_loss = 0
      labels = np.concatenate(list(batches)[1:])
      print 'Minibatch perplexity: %.2f' % float(
        np.exp(logprob(predictions, labels)))
      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print '=' * 80
        for _ in xrange(5):
          feed = sample(random_distribution())
          sentence = characters(feed)[0]
          reset_sample_state.run()
          for _ in xrange(79):
            prediction = sample_prediction.eval({sample_input: feed})
            feed = sample(prediction)
            sentence += characters(feed)[0]
          print sentence
        print '=' * 80
      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      for _ in xrange(valid_size):
        b = valid_batches.next()
        predictions = sample_prediction.eval({sample_input: b[0]})
        valid_logprob = valid_logprob + logprob(predictions, b[1])
      print 'Validation set perplexity: %.2f' % float(np.exp(
        valid_logprob / valid_size))
      print '-' * 30
# show how much time elapsed
print (time.time()-t0)/60., 'minutes elapsed'


# ---
# Problem 1
# ---------
# 
# You might have noticed that the definition of the LSTM cell involves 4 matrix multiplications with the input, and 4 matrix multiplications with the output. Simplify the expression by using a single matrix multiply for each, and variables that are 4 times larger.
# 

# Cell state (akin to a conveyor belt):  
# http://colah.github.io/posts/2015-08-Understanding-LSTMs/  
# <img width="70%" src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-C-line.png"></a>
# 
# 
# 1. Forget gate:  
# <img width="70%"src='http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png'>
# 
# 2. New info to store in cell state (Input gate * New candidate values:    
# <img width="70%"src='http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png'>
# 
# 3. Update the old cell state, C[t−1], into the new cell state C[t]:  
# <img width="70%"src='http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-C.png'>
# 
# 4. Decide output (Output gate * cell state C[t]):  
# <img width="70%"src='http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-o.png'>
# 
# ---

# ### Use single matrix multiplication for LSTM cell

# In[203]:

num_nodes = 64

graph = tf.Graph()
with graph.as_default():
  
  ## Parameters:
  '''original gates, cell
  # Input (Write) gate: input, previous output, and bias.
  ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  ib = tf.Variable(tf.zeros([1, num_nodes]))
    
  # Forget (Keep) gate: input, previous output, and bias.
  fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  print fx.get_shape().as_list()
  fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  fb = tf.Variable(tf.zeros([1, num_nodes]))
    
  # Memory cell: input, state and bias.        
  cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  cb = tf.Variable(tf.zeros([1, num_nodes]))
    
  # Output (Read) gate: input, previous output, and bias.
  ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  ob = tf.Variable(tf.zeros([1, num_nodes]))
  '''

  # combined f,i,c,o
  fico_x = tf.Variable(tf.truncated_normal([4, vocabulary_size, num_nodes], -0.1, 0.1))
  print fico_x.get_shape().as_list()
  fico_m = tf.Variable(tf.truncated_normal([4, num_nodes, num_nodes], -0.1, 0.1))
  fico_b = tf.Variable(tf.zeros([4, 1, num_nodes]))
    
  # Variables saving state across unrollings.
  saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)

  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
  b = tf.Variable(tf.zeros([vocabulary_size]))
  
  # Definition of the cell computation.
  def lstm_cell(i, o, state):
    """
    Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
    Note that in this formulation, we omit the various connections between the
    previous state and the gates.
    """                   
    i_list = tf.pack([i, i, i, i])
    #print i_list.get_shape().as_list()
    o_list = tf.pack([o, o, o, o])
                          
    ins = tf.batch_matmul(i_list, fico_x)
    outs = tf.batch_matmul(o_list, fico_m)
    
    h_x = ins + outs + fico_b
    #print h_x.get_shape().as_list()
    
    #forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
    forget_gate = tf.sigmoid(h_x[0,:,:])
    
    #input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
    input_gate = tf.sigmoid(h_x[1,:,:])
    
    #update = tf.tanh(tf.matmul(i, cx) + tf.matmul(o, cm) + cb)
    update = tf.tanh(h_x[2,:,:])
    
    state = forget_gate*state + input_gate*update
    
    #output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
    output_gate = tf.sigmoid(h_x[3,:,:])
    
    h = output_gate * tf.tanh(state)
    #print 'h', h.get_shape().as_list()
    return h, state

  # Input data.
  train_data = list()
  for _ in xrange(num_unrollings + 1):
    train_data.append(
      tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
  train_inputs = train_data[:num_unrollings]
  train_labels = train_data[1:]  # labels are inputs shifted by one time step.

  # Unrolled LSTM loop.
  outputs = list()
  output = saved_output
  state = saved_state
  for i in train_inputs:
    output, state = lstm_cell(i, output, state)
    outputs.append(output)

  # State saving across unrollings.
  with tf.control_dependencies([saved_output.assign(output),
                                saved_state.assign(state)]):
    # Classifier.
    logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(
        logits, tf.concat(0, train_labels)))

  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(
    10.0, global_step, 5000, 0.1, staircase=False)  ## orig 10.0, 5000, 0.1, True
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(
    zip(gradients, v), global_step=global_step)

  # Predictions.
  train_prediction = tf.nn.softmax(logits)
  
  # Sampling and validation eval: batch 1, no unrolling.
  sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
  saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
  reset_sample_state = tf.group(
    saved_sample_output.assign(tf.zeros([1, num_nodes])),
    saved_sample_state.assign(tf.zeros([1, num_nodes])))
  sample_output, sample_state = lstm_cell(
    sample_input, saved_sample_output, saved_sample_state)
  with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                saved_sample_state.assign(sample_state)]):
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))


# ### Run it with single matrix

# In[172]:

num_steps = 7001  ## orig 7001
summary_frequency = 100

t0 = time.time()
with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print 'Initialized\n=========='
  mean_loss = 0
  for step in xrange(num_steps):
    batches = train_batches.next()
    feed_dict = dict()
    for i in xrange(num_unrollings + 1):
      feed_dict[train_data[i]] = batches[i]
    _, l, predictions, lr = session.run(
      [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % (5.*summary_frequency) == 0:  ## orig 2.5*summary_frequency
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print 'Average loss at step ', step, '=', mean_loss, '\nlearning rate:', lr
      mean_loss = 0
      labels = np.concatenate(list(batches)[1:])
      print 'Minibatch perplexity: %.2f' % float(
        np.exp(logprob(predictions, labels)))
      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print '=' * 80
        for _ in xrange(5):
          feed = sample(random_distribution())
          sentence = characters(feed)[0]
          reset_sample_state.run()
          for _ in xrange(79):
            prediction = sample_prediction.eval({sample_input: feed})
            feed = sample(prediction)
            sentence += characters(feed)[0]
          print sentence
        print '=' * 80
      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      for _ in xrange(valid_size):
        b = valid_batches.next()
        predictions = sample_prediction.eval({sample_input: b[0]})
        valid_logprob = valid_logprob + logprob(predictions, b[1])
      print 'Validation set perplexity: %.2f' % float(np.exp(
        valid_logprob / valid_size))
      print '-' * 30
# show how much time elapsed
print (time.time()-t0)/60., 'minutes elapsed'


# ## Variants on LSTM 
# 
# ### Peephole connections:  
# ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf  
# <img width='80%' src='http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-peepholes.png'>
# 
# ### Gated Recurrent Unit (GRU):  
# http://arxiv.org/pdf/1406.1078v3.pdf  
# - combine forget and input gates
# - merge cell and hidden states  
# <img width='80%' src='http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png'>
# 

# ### Attention
# http://arxiv.org/pdf/1502.03044v2.pdf
# >Let every step of an RNN pick information to look at from some larger collection of information. For example, if you are using an RNN to create a caption describing an image, it might pick a part of the image to look at for every word it outputs
# 
# 

# ---
# Problem 2
# ---------
# 
# We want to train a LSTM over bigrams, that is pairs of consecutive characters like 'ab' instead of single characters like 'a'. Since the number of possible bigrams is large, feeding them directly to the LSTM using 1-hot encodings will lead to a very sparse representation that is very wasteful computationally.
# 
# a- Introduce an embedding lookup on the inputs, and feed the embeddings to the LSTM cell instead of the inputs themselves.
# 
# b- Write a bigram-based LSTM, modeled on the character LSTM above.
# 
# c- Introduce Dropout. For best practices on how to use Dropout in LSTMs, refer to this [article](http://arxiv.org/abs/1409.2329).
# 
# ---

# ### Function to generate a training batch for embedded bigrams

# In[315]:

batch_size=64
num_unrollings=10

class BatchGeneratorBigram(object):
  def __init__(self, text, batch_size, num_unrollings):
    self._text = text
    self._text_size = len(text)
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings
    segment = self._text_size / batch_size
    # list of offsets within batch
    self._cursor = [ offset * segment for offset in xrange(batch_size)]
    self._last_batch = self._next_batch()
  
  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    batch = np.zeros(shape=(self._batch_size), dtype=np.int)  # id of char to be embedded
    for b in xrange(self._batch_size):
      batch[b] = char2id(self._text[self._cursor[b]]) # get id of a char
      self._cursor[b] = (self._cursor[b] + 1) % self._text_size  # move cursor
    return batch
  
  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    for step in xrange(self._num_unrollings):
      batches.append(self._next_batch())  # add id of char for 1 to num_unrollings
    self._last_batch = batches[-1]
    return batches

def bigrambatches2string(batches):
  """Convert a sequence of batches back into string
  representation.
  """
  s = [''] * batches[0].shape[0]
  for b in batches:
    s = [''.join(x) for x in zip(s, [id2char(c) for c in b])]
  return s


# ### Generate training, validation batches for embedded bigrams

# In[324]:

# training and validation batches
train_batches = BatchGeneratorBigram(train_text, batch_size, num_unrollings)
valid_batches = BatchGeneratorBigram(valid_text, 1, 1) # returns batch size 1, +1 unrolling
train_labels = BatchGenerator(train_text, batch_size, num_unrollings)
valid_labels = BatchGenerator(valid_text, 1, 1) # returns batch size 1, +1 unrolling 

# look at the text from various segments
segment_look = 0
show = segment_look * len(train_text)/batch_size
print "index {} to {}:\n{}".format(show, show+80, train_text[show:show+64])
print('-'*16)

print bigrambatches2string(train_batches.next())
print bigrambatches2string(train_batches.next())
print('-'*16)
print valid_batches.next()
print valid_labels.next()
print bigrambatches2string(valid_batches.next())


# ### Functions to predict embedded bigrams

# In[325]:

def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in xrange(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p

def random_distribution():
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
  return b/np.sum(b, axis=1)[:,None]


# ### Build the bigram graph with embeddings

# In[589]:

num_nodes = 64
#vocabulary_size = (len(string.ascii_lowercase) + 1)**2
embedding_size = 128 # Dimension of the embedding vector.
batch_size=64
num_unrollings=10

graph = tf.Graph()
with graph.as_default():
  
  ## Parameters:
  fico_x = tf.Variable(tf.truncated_normal([4, embedding_size, num_nodes], -0.1, 0.1))
  print fico_x.get_shape().as_list()
  fico_m = tf.Variable(tf.truncated_normal([4, num_nodes, num_nodes], -0.1, 0.1))
  fico_b = tf.Variable(tf.zeros([4, 1, num_nodes]))
    
  # Variables saving state across unrollings.
  saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)

  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
  b = tf.Variable(tf.zeros([vocabulary_size]))
    
  # Embedding Variables.
  embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), trainable=False)
  
  # Definition of the cell computation.
  def lstm_cell(i, o, state):
    """
    Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
    Note that in this formulation, we omit the various connections between the
    previous state and the gates.
    """                   
    i_list = tf.pack([i, i, i, i])
    o_list = tf.pack([o, o, o, o])
                          
    ins = tf.batch_matmul(i_list, fico_x)
    outs = tf.batch_matmul(o_list, fico_m)
    
    h_x = ins + outs + fico_b

    forget_gate = tf.sigmoid(h_x[0,:,:])

    input_gate = tf.sigmoid(h_x[1,:,:])
    update = tf.tanh(h_x[2,:,:])
    state = forget_gate*state + input_gate*update
    
    output_gate = tf.sigmoid(h_x[3,:,:])
    
    h = output_gate * tf.tanh(state)
    return h, state

  # Input data.
  train_data = list()
  train_data_y = list()
  for _ in xrange(num_unrollings + 1):
    train_data.append(
      tf.placeholder(tf.int32, shape=[batch_size]))  # removed ohe of char
    train_data_y.append(
      tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))  # uses ohe of char
  train_labels = train_data_y[1:]
  
  # Embedded input data
  encoded_inputs = list()
  for bigram in train_data:
    embed = tf.nn.embedding_lookup(embeddings, bigram)
    encoded_inputs.append(embed)
  train_inputs = encoded_inputs[:num_unrollings]

  # Unrolled LSTM loop.
  outputs = list()
  output = saved_output
  state = saved_state
  for i in train_inputs:
    output, state = lstm_cell(i, output, state)
    outputs.append(output)

  # State saving across unrollings.
  with tf.control_dependencies([saved_output.assign(output),
                                saved_state.assign(state)]):
    # Classifier.
    logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
    print 'logits', logits.get_shape().as_list()
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, train_labels)))
    print 'labels', tf.concat(0, train_labels).get_shape().as_list()

  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(
    10.0, global_step, 5000, 0.1, staircase=False)  ## orig 10.0, 5000, 0.1, True
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(
    zip(gradients, v), global_step=global_step)

  # Predictions.
  train_prediction = tf.nn.softmax(logits)
  
  # Sampling and validation eval: batch 1, no unrolling.
  sample_input = tf.placeholder(tf.int32, shape=[1]) # removed ohe of char
  sample_input_emb = tf.nn.embedding_lookup(embeddings, sample_input)
  saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
  reset_sample_state = tf.group(
    saved_sample_output.assign(tf.zeros([1, num_nodes])),
    saved_sample_state.assign(tf.zeros([1, num_nodes])))
  sample_output, sample_state = lstm_cell(
    sample_input_emb, saved_sample_output, saved_sample_state)
  with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                saved_sample_state.assign(sample_state)]):
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))


# ### Run it with bigrams

# In[590]:

# training and validation batches
train_batches = BatchGeneratorBigram(train_text, batch_size, num_unrollings)
valid_batches = BatchGeneratorBigram(valid_text, 1, 1) # returns batch size 1, +1 unrolling
train_batches_y = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches_y = BatchGenerator(valid_text, 1, 1) # returns batch size 1, +1 unrolling 

num_steps = 7001  ## orig 7001
summary_frequency = 100

t0 = time.time()
with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print 'Initialized\n=========='
  mean_loss = 0
  for step in xrange(num_steps):
    batches = train_batches.next()
    batches_y = train_batches_y.next()
    
    feed_dict = dict()
    for i in xrange(num_unrollings + 1):
      feed_dict[train_data[i]] = batches[i]
      feed_dict[train_data_y[i]] = batches_y[i]
    
    _, l, predictions, lr = session.run(
      [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % (5.*summary_frequency) == 0:  ## orig 2.5*summary_frequency
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print 'Average loss at step', step, '=', mean_loss, '\nlearning rate:', lr
      mean_loss = 0
      labels = np.concatenate(list(batches_y)[1:])
      print 'Minibatch perplexity: %.2f' % float(
        np.exp(logprob(predictions, labels)))
      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print '=' * 80
        for _ in xrange(5):
          #feed = sample(random_distribution())  # random vector
          feed = np.random.randint(27, size=[1])#.astype('int32')
          #sentence = characters(feed)[0]
          sentence = id2char(feed)
          reset_sample_state.run()
          for _ in xrange(79):
            prediction = sample_prediction.eval({sample_input: feed})
            feed = sample(prediction)  # get ohe of predicted proba
            feed = np.array([np.argmax(feed)])  # get id of predicted char
            sentence += id2char(feed)  # add predicted char
          print sentence
        print '=' * 80
      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      for _ in xrange(valid_size):
        b = valid_batches.next()
        b_y = valid_batches_y.next()
        predictions = sample_prediction.eval({sample_input: b[0]})
        valid_logprob = valid_logprob + logprob(predictions, b_y[1])
      print 'Validation set perplexity: %.2f' % float(np.exp(
        valid_logprob / valid_size))
      print '-' * 30
# show how much time elapsed
print (time.time()-t0)/60., 'minutes elapsed'


# In[ ]:




# ### Embeddings example

# In[75]:


batch_size = 128
embedding_size = 128 # Dimension of the embedding vector.
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. 
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(100,valid_window+100), valid_size))
num_sampled = 64 # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)  ## 16 random from top 100
  
  # Variables.
  embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
  softmax_weights = tf.Variable(
    tf.truncated_normal([vocabulary_size, embedding_size],
                         stddev=1.0 / math.sqrt(embedding_size)))
  softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
  
  # Model.
  # Look up embeddings for inputs.
  embed = tf.nn.embedding_lookup(embeddings, train_dataset)
  print embed.get_shape().as_list()
  # Compute the softmax loss, using a sample of the negative labels each time.
  # https://www.tensorflow.org/versions/0.6.0/api_docs/python/nn.html#sampled_softmax_loss
  loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                               train_labels, num_sampled, vocabulary_size))
  # Optimizer.
  optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
  
  # Compute the similarity between minibatch examples and all embeddings.
  # We use the cosine distance:
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
    normalized_embeddings, valid_dataset)
  similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


# In[ ]:


num_steps = 100001  # 100001

t0 = time.time()
with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print "Initialized in {} secs\n=========".format(time.time()-t0)
  average_loss = 0
  for step in xrange(num_steps):
    batch_data, batch_labels = generate_batch(
      batch_size, num_skips, skip_window)
    feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
    _, lz = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += lz
    if step % 5000 == 0:
      if step > 0:
        average_loss = average_loss / 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print "Average loss at step", step, ":", average_loss
      print "----------------------"
      average_loss = 0
    # note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 20000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 4 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log = "%s %s," % (log, close_word)
        print log
  final_embeddings = normalized_embeddings.eval()
# show how much time elapsed
print (time.time()-t0)/60., 'minutes elapsed'


# ---
# Problem 3
# ---------
# 
# (difficult!)
# 
# Write a sequence-to-sequence LSTM which mirrors all the words in a sentence. For example, if your input is:
# 
#     the quick brown fox
#     
# the model should attempt to output:
# 
#     eht kciuq nworb xof
#     
# Refer to the lecture on how to put together a sequence-to-sequence model, as well as [this article](http://arxiv.org/abs/1409.3215) for best practices.
# 
# ---

# In[ ]:



