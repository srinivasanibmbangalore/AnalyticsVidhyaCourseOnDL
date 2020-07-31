
# coding: utf-8

# In[44]:


get_ipython().system('pip install tqdm')


# In[80]:


get_ipython().system('pip install psutil')


# In[1]:


import argparse
import os
import shutil
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from psutil import virtual_memory


from keras import backend as K 
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation,InputLayer, Conv2D, MaxPool2D, Flatten
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers import GlobalMaxPool2D

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical


# In[2]:


train_csv_path='/storage/manvswoman/train.csv'
test_csv_path='/storage/manvswoman/test_fkwGUNG1.csv'
images_path='/storage/manvswoman/images/'
output_path='/storage/manvswoman/'


# In[3]:


num_files=0
clmn_names=['image_name','imgDirLocation']
imgDF = pd.DataFrame(columns=clmn_names)
imgDF.describe()


# In[4]:


get_ipython().system('ls -lrt /storage/manvswoman/images/train.csv')


# In[5]:


seed = 42
rng = np.random.RandomState(seed)
tr_data = pd.read_csv(train_csv_path)
tr_data.describe()


# In[27]:


tr_data.head()


# ## Prepare the Training and Test File Array from the given image data set

# In[34]:


name1=tr_data.iloc[1,0]
print(name1)


# In[7]:


path, dirs, files = next(os.walk(images_path))
file_count = len(files)
print(file_count)


# In[8]:


import fnmatch
a=len(fnmatch.filter(os.listdir(images_path),'*.jpg'))


# In[66]:


print(a)


# In[9]:


get_ipython().run_cell_magic('time', '', "X=[]\ni=0\nfor index,row in tr_data.iterrows():\n    file_path=images_path+'/'+row['image_names']\n    img = plt.imread(file_path)\n    img=cv2.imread(file_path)\n    #img_resized=cv2.resize(img,(112,112))\n    X.append(img)\n    i+=1\nX_a=np.array(X)\nX_a.shape\n    ")


# In[10]:


X_a.shape


# In[11]:


print(pd.__version__)


# In[12]:


Y_a=tr_data['class'].values
Y_a.shape

Y_a = to_categorical(Y_a)


# In[13]:


Y_a.shape


# In[14]:


used_ramgb=virtual_memory().available/1e9
print("Available Memory is " + str(used_ramgb) + " gb")


# ###Load the image and visualize it post converting it into RGB

# In[36]:


trialImgLoc=images_path+'/'+ tr_data.iloc[18,0]
print(trialImgLoc)
img=cv2.imread(trialImgLoc)
plt.imshow(img)
img.shape


# In[37]:


im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(im_rgb)


# In[38]:


used_ramgb=virtual_memory().available/1e9
print("Available Memory is " + str(used_ramgb) + " gb")


# ### Pre-Process the data for leveraging VGG16

# In[39]:


X_a.min(), X_a.max()


# In[40]:


#preprocess input images accordiing to requirements of VGG16 model since it takes -1 and +1 as min and max values
X_a = preprocess_input(X_a, mode='tf')   # Very very memory consuming function
used_ramgb=virtual_memory().available/1e9
print("Available Memory is " + str(used_ramgb) + " gb")


# In[41]:


X_a.min(), X_a.max()


# In[42]:


X_train, X_valid, y_train, y_valid=train_test_split(X_a,Y_a,test_size=0.3, random_state=seed)


# In[43]:


del X_a


# In[44]:


used_ramgb=virtual_memory().available/1e9
print("Available Memory is " + str(used_ramgb) + " gb")


# ### Load the Weights of the Pre-Trained VGG16 Model

# In[45]:


# creating model with pre trained imagenet weights
base_model = VGG16(weights='imagenet')
#shows model summary
base_model.summary()


# In[48]:



# creating a VGG16 model with imagenet pretrained weights , accepting input of shape (224,224,3)
# also remove the final layers from model(include_top= False)
base_model = VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
# show model summary
base_model.summary()


# ### Fine tune the model for the current problem of classifying between man and woman
# 1. Extract features
# 2. Flatten the data
# 3. Rescale features
# 4. Create a Neural Network Model
# 5. Compile the model
# 6. Train and Validate the model

# In[50]:


get_ipython().run_cell_magic('time', '', '# extract features using the pretrained VGG16 model\n# for training set\nbase_model_pred = base_model.predict(X_train)')


# In[51]:


get_ipython().run_cell_magic('time', '', '#for validation set\nbase_model_pred_valid = base_model.predict(X_valid)')


# In[52]:


#show shape of predictions
base_model_pred.shape


# In[53]:


#show shape of predictions
base_model_pred_valid.shape


# In[54]:


# flattening the model output to one dimension for every sample of training set
base_model_pred = base_model_pred.reshape(8537, 7*7*512)


# In[56]:


base_model_pred.shape


# In[55]:


# flattening the model output to one dimension for every sample of validation set
base_model_pred_valid = base_model_pred_valid.reshape(3659, 7*7*512)


# In[57]:


base_model_pred_valid.shape


# In[58]:


# checking the min and max of the extracted features
base_model_pred.min(), base_model_pred.max()


# In[59]:


#get maximum value from generated features
max_val = base_model_pred.max()


# In[60]:


#normalizing features generated from the VGG16 model to [0,1]
base_model_pred = base_model_pred / max_val
base_model_pred_valid = base_model_pred_valid / max_val
base_model_pred.min(), base_model_pred.max()


# In[61]:


#create a sequential model 
model = Sequential()
# add input layer to the model that accepts input of shape 7*7*512
model.add(InputLayer((7*7*512, )))
# add fully connected layer with 1024 neurons and relu activation
model.add(Dense(units=1024, activation='relu'))

# add fully connected layer with 2 neurons and relu activation
model.add(Dense(units=2, activation='softmax'))


# In[62]:


# compile the model
model.compile(optimizer='sgd', metrics=['accuracy'], loss='categorical_crossentropy')


# In[63]:


model.summary()


# In[64]:


model.fit(base_model_pred, y_train, epochs=100, validation_data=(base_model_pred_valid, y_valid))


# ### Get Predictions

# In[66]:


# get predictions
predictions = model.predict_classes(base_model_pred_valid)
#show predictions
print(predictions)

