# Databricks notebook source exported at Mon, 6 Jun 2016 15:10:06 UTC
# MAGIC %md Deep Learning
# MAGIC =============
# MAGIC 
# MAGIC Assignment 1
# MAGIC ------------
# MAGIC 
# MAGIC The objective of this assignment is to learn about simple data curation practices, and familiarize you with some of the data we'll be reusing later.
# MAGIC 
# MAGIC This notebook uses the [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) dataset to be used with python experiments. This dataset is designed to look like the classic [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, while looking a little more like real data: it's a harder task, and the data is a lot less 'clean' than MNIST.

# COMMAND ----------

# These are all the modules we'll be using later. Make sure you can import them before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import Image # display, 
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle 
# pickling is a fundamental and powerful algorithm for serializing and de-serialzing a Python object structure
# It's the process whereby a Python object hierarchy is converted into a byte stream, and "unpickling" is the inverse operation, whereby a byte stream is converted back into an object hierarchy 

# Config the matlotlib backend as plotting inline in IPython, but this line doesn't work in Databricks
# %matplotlib inline 

# COMMAND ----------

# MAGIC %md First, we'll download the dataset to our local machine. The data consists of characters rendered in a variety of fonts on a 28x28 image. The labels are limited to 'A' through 'J' (10 classes). The training set has about 500k and the testset 19000 labelled examples. Given these sizes, it should be possible to train models quickly on any machine.

# COMMAND ----------

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 1% change in download progress.
  """
  global last_percent_reported # this makes sure that last_percent_reported can be updated and called outside the function
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

# COMMAND ----------

# MAGIC %md Extract the dataset from the compressed .tar.gz file.
# MAGIC This should give you a set of directories, labelled A through J.

# COMMAND ----------

num_classes = 10
np.random.seed(133)

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception( # Use raise and Exception more often in my own script
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders
  
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

# COMMAND ----------

# MAGIC %md ---
# MAGIC Problem 1
# MAGIC ---------
# MAGIC 
# MAGIC Let's take a peek at some of the data to make sure it looks sensible. Each exemplar should be an image of a character A through J rendered in a different font. Display a sample of the images that we just downloaded. Hint: you can use the package IPython.display.
# MAGIC 
# MAGIC ---

# COMMAND ----------

import matplotlib.image as mpimg
def display_samples(folders, color = None):
    fig = plt.figure() # this creates a fig
    fig.set_figwidth(10); fig.set_figheight(4) # this changes the default figure size of Databricks
    for n, i in enumerate(folders): # enumerate() returns the index and the item
        sample = np.random.choice(os.listdir(i), 1)[0] # this randomly selects a .png file in the given folder
        random = os.path.join(i, sample) # this is the full path to the .png file
        img = mpimg.imread(random) # this produces a numpy array of the .png, a 28 by 28 numpy array of dtype float32
        ax = fig.add_subplot(2, 5, n + 1) # there are 10 subplots (A-J), a (rows) by b(columns) by c(figure number)
        ax = plt.imshow(img, cmap = color) # this produces <matplotlib.image.AxesImage> in black and white
    display(fig)

# COMMAND ----------

display_samples(train_folders, 'gray')

# COMMAND ----------

display_samples(test_folders, 'gray')

# COMMAND ----------

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# random = os.path.join(train_folders[0], os.listdir(train_folders[0])[0])
# img = mpimg.imread(random) # this produces a numpy array for the image (28 by 28) of dtype float32
# fig, ax = plt.subplots(figsize=(3, 3))
# ax = plt.imshow(img) # this produces <matplotlib.image.AxesImage at 0x7f6597627dd0>
# display(fig)

# random = os.path.join(train_folders[0], os.listdir(train_folders[0])[0])
# img = mpimg.imread(random) # this produces a numpy array for the image (28 by 28) of dtype float32
# fig = plt.figure()
# ax = fig.add_subplot(121)
# ax = plt.imshow(img, cmap = 'gray') # this produces <matplotlib.image.AxesImage at 0x7f6597627dd0>
# ax = fig.add_subplot(122)
# ax = plt.imshow(img, cmap = 'gray') # this produces <matplotlib.image.AxesImage at 0x7f6597627dd0>
# display(fig)

# COMMAND ----------

# MAGIC %md Now let's load the data in a more manageable format. Since, depending on your computer setup you might not be able to fit it all in memory, we'll load each class into a separate dataset, store them on disk and curate them independently. Later we'll merge them into a single dataset of manageable size.
# MAGIC 
# MAGIC We'll convert the entire dataset into a 3D array (image index, x, y) of floating point values, normalized to have approximately zero mean and standard deviation ~0.5 to make training easier down the road. 
# MAGIC 
# MAGIC A few images might not be readable, we'll just skip them.

# COMMAND ----------

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  for image_index, image in enumerate(image_files):
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[image_index, :, :] = image_data
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  num_images = image_index + 1
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

# COMMAND ----------

# MAGIC %md ---
# MAGIC Problem 2
# MAGIC ---------
# MAGIC 
# MAGIC Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray. Hint: you can use matplotlib.pyplot.
# MAGIC 
# MAGIC ---

# COMMAND ----------

print("Taking a look at train_datasets[0]: %s"%(train_datasets[0]))
x = open(train_datasets[0], 'rb') # open connection in read binary mode
y = pickle.load(x) # extract the content of the opened file in binary format
x.close() # close the connection
print("The shape and content of %s: %s" %(train_datasets[0], y.shape,))
print("The first row of the first letter A: \n%s"%(y[0][0]))

# COMMAND ----------

def display_pickled_samples(folder, color = None):
  fig = plt.figure() # this creates a fig 
  fig.set_figheight(20); fig.set_figwidth(10) # this sets the figure height and width
  for i, f in enumerate(folder): 
    x = open(f, 'rb') # this opens notMNIST_large/A.pickle, for instance
    y = pickle.load(x) # this gets the content in binary format
    x.close() # this closes the connection
    
    random_5 = y[np.random.choice(len(y), 5)] # this generates five random numpy arrays that correspond to five random images of a particular letter
    
    for k in range(1, 6):
      ax = fig.add_subplot(10, 5, i*5+k)
      ax = plt.imshow(random_5[k-1])
  display(fig)

# COMMAND ----------

display_pickled_samples(train_datasets)

# COMMAND ----------

display_pickled_samples(test_datasets)

# COMMAND ----------

# MAGIC %md ---
# MAGIC Problem 3
# MAGIC ---------
# MAGIC Another check: we expect the data to be balanced across classes. Verify that.
# MAGIC 
# MAGIC ---

# COMMAND ----------

# By balanced, they mean that the size of each class is roughly the same, and inside each class, the data are roughly equally varied

def check_balances(folder):
  for f in folder:
    with open(f, 'rb') as x: 
      datasets = pickle.load(x)
      x.close()
    print(f, ", ", len(datasets), ", ", np.std(datasets))
check_balances(train_datasets)
check_balances(test_datasets)

# COMMAND ----------

# MAGIC %md Merge and prune the training data as needed. Depending on your computer setup, you might not be able to fit it all in memory, and you can tune `train_size` as needed. The labels will be stored into a separate array of integers 0 through 9.
# MAGIC 
# MAGIC Also create a validation dataset for hyperparameter tuning.

# COMMAND ----------

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32) # square images
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size) # image_size = 28 is given
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set # cool,np.random.shuffle() does that!
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels
            
            
train_size = 200000
valid_size = 10000
test_size = 10000

# get datasets and labels from train and test datasets
valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

# COMMAND ----------

# MAGIC %md Next, we'll randomize the data. It's important to have the labels well shuffled for the training and test distributions to match.

# COMMAND ----------

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

# COMMAND ----------

# MAGIC %md ---
# MAGIC Problem 4
# MAGIC ---------
# MAGIC Convince yourself that the data is still good after shuffling!
# MAGIC 
# MAGIC ---

# COMMAND ----------

def display_pickled_data(data, color = None):
  fig = plt.figure() # this creates a fig 
  fig.set_figheight(10); fig.set_figwidth(10) 
  random_25 = data[np.random.choice(len(data), 25)] 
  for k in range(1, 26):
    ax = fig.add_subplot(5, 5, k)
    ax = plt.imshow(random_25[k-1])
  display(fig)
display_pickled_data(train_dataset)

# COMMAND ----------

display_pickled_data(test_dataset)

# COMMAND ----------

# MAGIC %md Finally, let's save the data for later reuse:

# COMMAND ----------

pickle_file = 'notMNIST.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    } # a dictionary
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL) # this seems like the standard code pickle.dump(), write a pickled representation of obj to the open file object file
  # to understand better, see: https://docs.python.org/3/library/pickle.html#usage
  # there have been several ways to save pickle files, but not all of them are compatible with all versions of Python, some are not good at compressing data
  # pickle.HIGHEST_PROTOCOL makes sure that you use the most recent version of compressing
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

# COMMAND ----------

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

# COMMAND ----------

# MAGIC %md ---
# MAGIC Problem 5
# MAGIC ---------
# MAGIC 
# MAGIC By construction, this dataset might contain a lot of overlapping samples, including training data that's also contained in the validation and test set! Overlap between training and test can skew the results if you expect to use your model in an environment where there is never an overlap, but are actually ok if you expect to see training samples recur when you use it.
# MAGIC Measure how much overlap there is between training, validation and test samples.
# MAGIC 
# MAGIC Optional questions:
# MAGIC - What about near duplicates between datasets? (images that are almost identical)
# MAGIC - Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.
# MAGIC ---

# COMMAND ----------

import hashlib
import time

def measure_overlap(a, b):
  '''a and b are ndarrays of shape (len, 28, 28)'''
  a.flags.writeable = False; b.flags.writeable = False
  
  start = time.clock()
  
  a_hashed = [hashlib.sha1(i).hexdigest() for i in a]
  b_hashed = [hashlib.sha1(i).hexdigest() for i in b]
  a_set = set(a_hashed)
  b_set = set(b_hashed)
  
  overlap = set.intersection(a_set, b_set)
  overlap_aInb = filter(lambda x: x in b_set, a_hashed)
  overlap_bIna = filter(lambda x: x in a_set, b_hashed)
  
  return overlap, overlap_aInb, overlap_bIna, time.clock() - start

tv_dups, tInv, vInt, tv_time = measure_overlap(train_dataset, valid_dataset)
tt_dups, trInte, teIntr, tt_time = measure_overlap(train_dataset, test_dataset)

print("Between Train and Validation: %d overlaps, %d Train in Validation, %d Validation in Train, %.2f sec"%(len(tv_dups), len(tInv), len(vInt), tv_time))
print("Between Train and Test: %d overlaps, %d Train in Test, %d Test in Train, %.2f sec"%(len(tt_dups), len(trInte), len(teIntr), tt_time))

# COMMAND ----------

def sanitize(a, a_labels, b):
  '''sanitize a based on b'''
  start = time.clock()
  
  a_hashed = [hashlib.sha1(i).hexdigest() for i in a]
  b_hashed = [hashlib.sha1(i).hexdigest() for i in b]
  aNotInb = ~ np.in1d(a_hashed, b_hashed)
  
#   b_set = set([hashlib.sha1(i).hexdigest() for i in b])
#   a_dict = dict(zip(a_hashed, a))
#   aNotInb = filter(lambda x: x not in b_set, a_hashed)
#   a_sanitized = { k: a_dict[k] for k in aNotInb }

  return a[aNotInb], a_labels[aNotInb], time.clock() - start

valid_snt, valid_labels_snt, vt = sanitize(valid_dataset, valid_labels, train_dataset)
test_snt, test_labels_snt, tt = sanitize(test_dataset, test_labels, train_dataset)

print("Sanitized Validation dataset contains %d images, %.2f sec"%(len(valid_snt), vt))
print("Sanitized Test dataset contains %d images, %.2f sec"%(len(test_snt), tt))

# COMMAND ----------

# compress images from 28 by 28 to 14 by 14
train_small = np.array([np.round(ndimage.zoom(i, .5), 1) for i in train_dataset])
valid_small = np.array([np.round(ndimage.zoom(i, .5), 1) for i in valid_dataset])
test_small = np.array([np.round(ndimage.zoom(i, .5), 1) for i in test_dataset])

# COMMAND ----------

tv_dups, tInv, vInt, tv_time = measure_overlap(train_small, valid_small)
tt_dups, trInte, teIntr, tt_time = measure_overlap(train_small, test_small)

print("Between reduced Train and Validation: %d overlaps, %d Train in Validation, %d Validation in Train, %.2f sec"%(len(tv_dups), len(tInv), len(vInt), tv_time))
print("Between reduced Train and Test: %d overlaps, %d Train in Test, %d Test in Train, %.2f sec"%(len(tt_dups), len(trInte), len(teIntr), tt_time))

valid_small_snt, valid_small_labels_snt, vt = sanitize(valid_small, valid_labels, train_small)
test_small_snt, test_small_labels_snt, tt = sanitize(test_small, test_labels, train_small)

print("Sanitized Validation dataset contains %d images, %.2f sec"%(len(valid_small_snt), vt))
print("Sanitized Test dataset contains %d images, %.2f sec"%(len(test_small_snt), tt))

# COMMAND ----------

# MAGIC %md ---
# MAGIC Problem 6
# MAGIC ---------
# MAGIC 
# MAGIC Let's get an idea of what an off-the-shelf classifier can give you on this data. It's always good to check that there is something to learn, and that it's a problem that is not so trivial that a canned solution solves it.
# MAGIC 
# MAGIC Train a simple model on this data using 50, 100, 1000 and 5000 training samples. Hint: you can use the LogisticRegression model from sklearn.linear_model.
# MAGIC 
# MAGIC Optional question: train an off-the-shelf model on all the data!
# MAGIC 
# MAGIC ---

# COMMAND ----------

def simpleModel(data, labels, size):
  '''data is of shape (size, width, height)'''
  data = data.reshape(len(data), -1) # this reshapes the data into (size, width*height)
  r_list = np.random.choice(len(data), size, False)
  lr = LogisticRegression()
  lr.fit(data[r_list], labels[r_list])
  return lr, data[r_list], labels[r_list] # return the model, the data, and the predictions

# COMMAND ----------

s0, s1, s2, s3, s4 = [], [], [], [], []
myrange = [50, 100, 500, 1000, 5000]
for size in myrange:
  myfit, train_x, label_x = simpleModel(train_dataset, train_labels, size)
  s0.append(myfit.score(train_x.reshape(len(train_x), -1), label_x))
  s1.append(myfit.score(valid_dataset.reshape(len(valid_dataset), -1), valid_labels))
  s2.append(myfit.score(test_dataset.reshape(len(test_dataset), -1), test_labels))
  s3.append(myfit.score(valid_snt.reshape(len(valid_snt), -1), valid_labels_snt))
  s4.append(myfit.score(test_snt.reshape(len(test_snt), -1), test_labels_snt))

# COMMAND ----------

fig, ax = plt.subplots()
ax.plot(myrange, s0, myrange, s1, myrange, s2, myrange, s3, myrange, s4)
ax.legend(labels = ['Train', 'Validation', 'Test', 'Sanitized Validation', 'Sanitized Test'], loc = 4)
display(fig)

# COMMAND ----------

ss0, ss1, ss2, ss3, ss4 = [], [], [], [], []
for size in myrange:
  myfit, train_x, label_x = simpleModel(train_small, train_labels, size)
  ss0.append(myfit.score(train_x.reshape(len(train_x), -1), label_x))
  ss1.append(myfit.score(valid_small.reshape(len(valid_small), -1), valid_labels))
  ss2.append(myfit.score(test_small.reshape(len(test_small), -1), test_labels))
  ss3.append(myfit.score(valid_small_snt.reshape(len(valid_small_snt), -1), valid_small_labels_snt))
  ss4.append(myfit.score(test_small_snt.reshape(len(test_small_snt), -1), test_small_labels_snt))

# COMMAND ----------

fig, ax = plt.subplots()
ax.plot(myrange, ss0, myrange, ss1, myrange, ss2, myrange, ss3, myrange, ss4)
ax.legend(labels = ['Validation', 'Test', 'Sanitized Validation', 'Sanitized Test'], loc = 4)
display(fig)