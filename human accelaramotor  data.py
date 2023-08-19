#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization


# In[2]:


from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[4]:


file=open('E:\AI\human recognition accemolator\WISDM_ar_v1.1\WISDM_ar_v1.1_raw.txt')
lines=file.readlines()

processedList=[]

for i, line in enumerate(lines):
    try:
        line=line.split(',')
        last=line[5].split(';')[0]
        last=last.strip()
        if last =='':
            break;
        temp= [line[0],line[1],line[2],line[3],line[4],last]
        processedList.append(temp)
    except:
        print('Error at line number: ',i)


# In[5]:


processedList


# In[6]:


columns=['user','activity','time','x','y','z']


# In[7]:


data=pd.DataFrame(data=processedList,columns=columns)
data.head()


# In[8]:


data.shape


# In[9]:


data.info()


# In[10]:


data.isnull().sum()


# In[11]:


data['activity'].value_counts()


# In[12]:


data['x']=data['x'].astype('float')
data['y']=data['y'].astype('float')
data['z']=data['z'].astype('float')


# In[13]:


data.info()


# In[14]:


Fs=20


# In[15]:


activities=data['activity'].value_counts().index


# In[16]:


activities


# In[17]:


def plot_activity(activity, data):
    fig, (ax0, ax1, ax2)= plt.subplots(nrows=3, figsize=(15,7), sharex=True)
    plot_axis(ax0, data['time'], data['x'], 'X-Axis')
    plot_axis(ax1, data['time'], data['y'], 'y-Axis')
    plot_axis(ax2, data['time'], data['z'], 'z-Axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()

def plot_axis(ax,x,y,title):
    ax.plot(x,y,'g')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y)-np.std(y),max(y)+np.std(y)])
    ax.set_xlim(min(x),max(x))
    ax.grid(True)

for activity in activities: 
    data_for_plot= data[(data['activity']== activity)][:Fs*10]
    plot_activity(activity, data_for_plot)


# In[18]:


df=data.drop(['user','time'], axis=1).copy()
df.head()         


# In[19]:


df['activity'].value_counts()


# In[20]:


Walking=df[df['activity']=='Walking'].head(3555).copy()
Jogging=df[df['activity']=='Jogging'].head(3555).copy()
Upstairs=df[df['activity']=='Upstairs'].head(3555).copy()
Downstairs=df[df['activity']=='Downstairs'].head(3555).copy()
Sitting=df[df['activity']=='Sitting'].head(3555).copy()
Standing=df[df['activity']=='Standing'].copy()


# In[21]:


balanced_data=pd.DataFrame()
balanced_data=balanced_data.append([Walking, Jogging, Upstairs, Downstairs, Sitting, Standing])
balanced_data.shape


# In[22]:


balanced_data['activity'].value_counts()


# In[23]:


balanced_data.head()


# In[24]:


from sklearn.preprocessing import LabelEncoder


# In[25]:


label=LabelEncoder()


# In[26]:


balanced_data['label']=label.fit_transform(balanced_data['activity'])


# In[27]:


balanced_data.head()


# In[28]:


label.classes_


# In[29]:


x=balanced_data[['x','y','z']]
y=balanced_data['label']


# In[30]:


scaler=StandardScaler()
x=scaler.fit_transform(x)
scaled_x=pd.DataFrame(data=x,columns=['x','y','z'])
scaled_x['label']=y.values


# In[31]:


scaled_x


# In[32]:


import scipy.stats as stats


# In[33]:


Fs=20
frame_size=Fs*4
hop_size=Fs*2


# In[34]:


def get_frame(df,frame_size,hop_size):
    N_FEATURES=3
    
    frames=[]
    labels=[]
    for i in range(0, len(df) - frame_size, hop_size):
        x=df['x'].values[i:i+frame_size]
        y=df['y'].values[i:i+frame_size]
        z=df['z'].values[i:i+frame_size]
        label=stats.mode(df['label'][i:i+frame_size])[0][0]
        frames.append([x,y,z])
        labels.append(label)
    frames=np.asarray(frames).reshape(-1,frame_size,N_FEATURES)
    labels=np.asarray(labels)
    return frames, labels


# In[35]:


X, y= get_frame(scaled_x, frame_size, hop_size)


# In[36]:


X.shape, y.shape


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


# In[38]:


X_train.shape


# In[39]:


X_test.shape


# In[40]:


X_train[0].shape


# In[41]:


X_test[0].shape


# In[42]:


X_train=X_train.reshape(425,80,3,1)


# In[43]:


X_test=X_test.reshape(107,80,3,1)


# In[44]:


model=Sequential()


# In[45]:


model.add(Conv2D(16, (2,2), activation='relu', input_shape= X_train[0].shape))
model.add(Dropout(0.1))

model.add(Conv2D(32,(2,2),activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(6,activation='softmax'))


# In[46]:


model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[47]:


history=model.fit(X_train, y_train, epochs=10, validation_data=(X_test,y_test), verbose=1)


# In[59]:


model.summary()


# In[ ]:





# In[ ]:





# In[ ]:





# In[48]:


def plot_learningCurve(history, epochs):
    epoch_range=range(1, epochs+1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train','Val'], loc='upper left')
    plt.show()
    
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train','Val'], loc='upper left')
    plt.show()


# In[49]:


plot_learningCurve(history,10)


# In[50]:


from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


# In[51]:


y_predict = np.argmax(model.predict(X_test), axis=-1)


# In[52]:


mat= confusion_matrix(y_test, y_predict)
plot_confusion_matrix(conf_mat=mat, class_names=label.classes_, show_normed=False)


# In[53]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = [ 'Rollar skatting', 'Playing Volleyball','Skateboarding','Playing ice hockey','Playing Basketball']
students = [96.85,1.63,0.21,0.20,0.16]
ax.bar(langs,students)
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.title('Accuracy Of Algorithm')
plt.show()


# In[54]:


# importing library
import matplotlib.pyplot as plt
 
# function to add value labels
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')
 
if __name__ == '__main__':
   
    # creating data on which bar chart will be plot
    x = ['Rollar skatting', 'Playing Volleyball','Skateboarding','Playing ice hockey','Playing Basketball']
    y = [96.85,1.63,0.21,0.20,0.16]
     
    # setting figure size by using figure() function
    plt.figure(figsize = (10, 5))
     
    # making the bar chart on the data
    plt.bar(x, y)
     
    # calling the function to add value labels
    addlabels(x, y)
     
    # giving title to the plot
    plt.title("Activity Percentage of Output")
     
    # giving X and Y labels
    plt.xlabel("Activity")
    plt.ylabel("Percentage")
     
    # visualizing the plot
    plt.show()


# In[61]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print("Shape: ", X_train[0].shape)


# In[58]:





# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


# Split sizes
train_size = 0.7
val_size = 0.15
test_size = 0.15


# In[ ]:


print(X.shape, X_train.shape, X_test.shape)


# In[ ]:




