#!/usr/bin/env python
# coding: utf-8

# In[1]:


from hangar import Repository

import numpy as np
import pickle
import gzip


# Download mnist from [here](https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz) and save it to a known path.
#
# ## Initialize the Repo

# In[2]:


repo = Repository('~/jjmachan/hangar_examples/mnist')
repo.init(user_name='jjmachan', user_email='jjmachan@g.com', remove_old=True)
repo


# In[3]:


repo


# In[4]:


co = repo.checkout(write=True)
co


# In[5]:


co


# ## Arraysets
# These are the structures that are used to store the data as numpy
# arrays. Hence only numeric data can be stored.

# In[6]:


co.arraysets


# In[7]:

# Load the dataset
with gzip.open('./mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='bytes')

# def rescale(array):
#     array = array * 256
#     rounded = np.round(array)
#     return rounded.astype(np.uint8())

# sample image and label for creating arrayset
sample_trimg = train_set[0][0]
sample_trlabel = np.array([train_set[1][0]])

# training images
trimgs = train_set[0]
trlabels = train_set[1]

data = [train_set, valid_set, test_set]


# In[12]:


sample_trimg


# In[13]:


# Train
co.arraysets.init_arrayset(name='mnist_training_images',
                           prototype=sample_trimg)
co.arraysets.init_arrayset(name='mnist_training_labels',
                           prototype=sample_trlabel)


# In[14]:


# Val
co.arraysets.init_arrayset(name='mnist_validation_images',
                           prototype=sample_trimg)
co.arraysets.init_arrayset(name='mnist_validation_labels',
                           prototype=sample_trlabel)

# Test
co.arraysets.init_arrayset(name='mnist_test_images',
                           prototype=sample_trimg)
co.arraysets.init_arrayset(name='mnist_test_labels',
                           prototype=sample_trlabel)


# In[15]:


arraysets_list = [('mnist_training_images', 'mnist_training_labels'),
       ('mnist_validation_images', 'mnist_validation_labels'),
        ('mnist_test_images', 'mnist_test_labels')]

for imgs, labels in arraysets_list:
    print(co.arraysets[imgs], co.arraysets[labels])


# In[17]:

for i, (imgs, labels) in enumerate(arraysets_list):
    print(i)
    img_aset, label_aset = co.arraysets[imgs], co.arraysets[labels]
    with img_aset, label_aset:
        for idx, image in enumerate(data[i][0]):
            img_aset.add(data=image, name=idx)
            label_aset.add(data=np.array([data[i][1][idx]]), name=idx)


# In[18]:


co.arraysets['mnist_training_images']


# In[19]:


co.commit('added all the mnist datasets')


# In[20]:


repo.log()


# In[21]:


co.close()
