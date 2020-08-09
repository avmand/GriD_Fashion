# -*- coding: utf-8 -*-

import pandas as pd
import math
import numpy as np 
import itertools
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from keras.models import Sequential 
from keras import optimizers
#from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense  
from keras import applications  
from keras.utils.np_utils import to_categorical  
import matplotlib.pyplot as plt 
#import matplotlib.image as mpimg
import datetime
import time
import shutil
import os
from pathlib import Path
import glob



###############################GLOBAL VARIABLES START###############################
#CHANGE THESE BEFORE RUNNING


#For the csv files of all myntra images and their data
stylescsv = r'myntra_data_cnn\myntradataset\styles.csv'

#For the folder where the classified images should be moved to
filestartdest='/myntra_data_cnn/'

#For the folder where the images to be classified should be moved from
filestartsource='myntra_data_cnn\myntradataset\images\\'


#Would you like to add more images to the training set? yes or no
addmoreimages = 'yes'   
#For the folder where the images should be moved from
add_startoffile=['/myntra_data_cnn/myntra_additional_dataset_dresses',
                 '/myntra_data_cnn/myntra_additional_dataset_skirts']
#type of files .jpg, .png, .jfif, etc                 
add_typeoffile=['.jfif',
                 '.jfif']
#For the folder where the images should be moved to
add_startofdestfold=['/myntra_data_cnn/data/train/Dresses',
                 '/myntra_data_cnn/data/train/Skirts']
#categories to be moved to                
add_category=['Dresses',
              'Skirts']


#Where each set of classified images should be retrieved from
train_data_dir = '/myntra_data_cnn/data/train/'  
validation_data_dir = '/myntra_data_cnn/data/validate/'  
test_data_dir = '/myntra_data_cnn/data/test/'

#Location of test image
test_my_image_path = 'filename.jpg'



################################GLOBAL VARIABLES END################################




colnames=['id','gender', 'masterCategory','subCategory', 'articleType', 'baseColour', 'season', 'year', 'usage', 'productDisplayName']
dataset_csv = pd.read_csv(stylescsv,names=colnames, delimiter=',', error_bad_lines=False, header=None,usecols=['id','articleType'], na_values=" NaN")

dataset_csv = dataset_csv.dropna()
dataset_csv=dataset_csv[1:]

#print (dataset_csv[0:5])

#print(len(dataset_csv))
#print(0.60*len(dataset_csv))
#print((0.20*len(dataset_csv)+0.60*len(dataset_csv)))

nameslist={}
for i in range (1, len(dataset_csv)):
  if (dataset_csv['articleType'][i] not in nameslist.keys()):
    nameslist[dataset_csv['articleType'][i]]=1
  else:
    nameslist[dataset_csv['articleType'][i]]+=1

import operator
sorted_d = dict(sorted(nameslist.items(), key=operator.itemgetter(1),reverse=True))

#print(nameslist)
#print(sorted_d)

dataset_csv.sort_values(by=['articleType'], inplace=True)
final_df = dataset_csv.sort_values(by=['articleType'], ascending=True)

#print (final_df)
#print(nameslist['Skirts'])
#print(final_df[1:5])


final_df = final_df.reset_index(drop=True)
dataset_csv=final_df

#filestartdest='/content/myntrak2/myntra_data_cnn/'
#filestartsource='/content/myntrak2/myntra_data_cnn/myntradataset/images/'

for i in range(1, len(dataset_csv)):
    
  if (nameslist[dataset_csv['articleType'][i]]>100):
    if (i==1):
      name = dataset_csv['articleType'][i]
    if ((i==1) or (dataset_csv['articleType'][i-1] != dataset_csv['articleType'][i]) ):
      name = dataset_csv['articleType'][i]
      size = nameslist[name]
      train_sizemax = math.floor(0.60 * size)
      test_sizemax = math.floor(0.20 * size)+train_sizemax
      count=0
      fold = 'data/train'
    count+=1
    if (count>train_sizemax):
      fold = 'data/test'      
    if (count>test_sizemax):
      fold = 'data/validate'
      
    file0=filestartdest+fold+'/'+name
    my_file = Path(file0)
    if not (my_file.is_dir()):
      os.makedirs(file0)
    file2=filestartdest+fold+"/"+name
    file1=filestartsource+ dataset_csv['id'][i]+'.jpg'
    if dataset_csv['id'][i] not in ['39403','39410','39401','39425','12347']:
      if  not (Path(file2+"/"+dataset_csv['id'][i]+".jpg").is_file()):
        shutil.move(file1, file2)
##########################For adding new images################################
        
if (addmoreimages.lower() == 'yes'):
    for j in range(0,len(add_startoffile)):
        imagescloth=glob.glob( add_startoffile[j] + '/*' + add_typeoffile[j])
        for i in range (0, len(imagescloth)):
          shutil.move(imagescloth[i], add_startofdestfold[j])
          nameslist[add_category[j]]+=1
              
###############################################################################



#END OF PREPROCESSING/ADDING DATA


#Default dimensions we found online
img_width, img_height = 224, 224  
   
#Create a bottleneck file
top_model_weights_path = 'bottleneck_fc_model.h5' 



# loading up our datasets

#train_data_dir = '/content/myntrak2/myntra_data_cnn/data/train/'  
#validation_data_dir = '/content/myntrak2/myntra_data_cnn/data/validate/'  
#test_data_dir = '/content/myntrak2/myntra_data_cnn/data/test/'
   


# number of epochs to train top model  
epochs = 7 #this has been changed after multiple model run  
# batch size used by flow_from_directory and predict_generator  
batch_size = 50

#Loading vgc16 model
vgg16 = applications.VGG16(include_top=False, weights='imagenet')

datagen = ImageDataGenerator(rescale=1. / 255)  #needed to create the bottleneck .npy files

#__this can take an hour and half to run so only run it once. 
#once the npy files have been created, no need to run again. Convert this cell to a code cell to run.__

start = datetime.datetime.now()
   
generator = datagen.flow_from_directory(  
     train_data_dir,  
     target_size=(img_width, img_height),  
     batch_size=batch_size,  
     class_mode=None,  
     shuffle=False)  
   
nb_train_samples = len(generator.filenames)  
num_classes = len(generator.class_indices)  
   
predict_size_train = int(math.ceil(nb_train_samples / batch_size))  
   
bottleneck_features_train = vgg16.predict_generator(generator, predict_size_train)  
   
np.save('bottleneck_features_train.npy', bottleneck_features_train)
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)

print('*'*100)

#__this can take half an hour to run so only run it once. once the npy files have been created, no need to run again. Convert this cell to a code cell to run.__

start = datetime.datetime.now()
generator = datagen.flow_from_directory(  
     validation_data_dir,  
     target_size=(img_width, img_height),  
     batch_size=batch_size,  
     class_mode=None,  
     shuffle=False)  
   
nb_validation_samples = len(generator.filenames)  
   
predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))  
   
bottleneck_features_validation = vgg16.predict_generator(  
     generator, predict_size_validation)  
   
np.save('bottleneck_features_validation.npy', bottleneck_features_validation) 
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)

print('*'*100)

#__this can take half an hour to run so only run it once. once the npy files have been created, no need to run again. Convert this cell to a code cell to run.__

start = datetime.datetime.now()
generator = datagen.flow_from_directory(  
     test_data_dir,  
     target_size=(img_width, img_height),  
     batch_size=batch_size,  
     class_mode=None,  
     shuffle=False)  
   
nb_test_samples = len(generator.filenames)  
   
predict_size_test = int(math.ceil(nb_test_samples / batch_size))  
   
bottleneck_features_test = vgg16.predict_generator(  
     generator, predict_size_test)  
   
np.save('bottleneck_features_test.npy', bottleneck_features_test) 
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)

#training data
generator_top = datagen.flow_from_directory(  
         train_data_dir,  
         target_size=(img_width, img_height),  
         batch_size=batch_size,  
         class_mode='categorical',  
         shuffle=False)  
   
nb_train_samples = len(generator_top.filenames)  
num_classes = len(generator_top.class_indices)  
   
# load the bottleneck features saved earlier  
train_data = np.load('bottleneck_features_train.npy')  
   
# get the class lebels for the training data, in the original order  
train_labels = generator_top.classes  
   
# convert the training labels to categorical vectors  
train_labels = to_categorical(train_labels, num_classes=num_classes)

#validation data
generator_top = datagen.flow_from_directory(  
         validation_data_dir,  
         target_size=(img_width, img_height),  
         batch_size=batch_size,  
         class_mode=None,  
         shuffle=False)  
   
nb_validation_samples = len(generator_top.filenames)  
   
validation_data = np.load('bottleneck_features_validation.npy')  
   

validation_labels = generator_top.classes  
validation_labels = to_categorical(validation_labels, num_classes=num_classes)

#testing data
generator_top = datagen.flow_from_directory(  
         test_data_dir,  
         target_size=(img_width, img_height),  
         batch_size=batch_size,  
         class_mode=None,  
         shuffle=False)  
   
nb_test_samples = len(generator_top.filenames)  
   
test_data = np.load('bottleneck_features_test.npy')  
   

test_labels = generator_top.classes  
test_labels = to_categorical(test_labels, num_classes=num_classes)


start = datetime.datetime.now()
model = Sequential()  
model.add(Flatten(input_shape=train_data.shape[1:]))  
model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3)))  
model.add(Dropout(0.5))  
model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3)))  
model.add(Dropout(0.3)) 
model.add(Dense(num_classes, activation='softmax'))  

model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])  

history = model.fit(train_data, train_labels,  
      epochs=10,
      batch_size=batch_size,  
      validation_data=(validation_data, validation_labels))  

model.save_weights(top_model_weights_path)  

(eval_loss, eval_accuracy) = model.evaluate(  
 validation_data, validation_labels, batch_size=batch_size, verbose=1)

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))  
print("[INFO] Loss: {}".format(eval_loss))  
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)

#Model summary
model.summary()

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('accuracy')  
plt.xlabel('epoch')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss')  
plt.xlabel('epoch')
plt.legend()
plt.show()

model.evaluate(test_data, test_labels)

print('test data', test_data)
preds = np.round(model.predict(test_data),0) 
#to fit them into classification metrics and confusion metrics, some additional modificaitions are required
print('rounded test_labels', preds)

# clothes = ['Scarves']
names_tar=[]
for i in nameslist.keys():
  if (nameslist[i]>=100):
    names_tar.append(i)
classification_metrics = metrics.classification_report(test_labels, preds, target_names=names_tar )
print(classification_metrics)

#Since our data is in dummy format we put the numpy array into a dataframe and call idxmax axis=1 to return the column
# label of the maximum value thus creating a categorical variable
#Basically, flipping a dummy variable back to it's categorical variable
categorical_test_labels = pd.DataFrame(test_labels).idxmax(axis=1)
categorical_preds = pd.DataFrame(preds).idxmax(axis=1)

confusion_matrix= confusion_matrix(categorical_test_labels, categorical_preds)

def plot_confusion_matrix(cm, classes,
             normalize=False,
             title='Confusion matrix',
             cmap=plt.cm.Blues):
    #Add Normalization Option
    '''prints pretty confusion metric with normalization option '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
#     print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(confusion_matrix, ['Scarves','Ring'])

#Those numbers are all over the place. Now turning normalize= True
plot_confusion_matrix(confusion_matrix, 
                      ['Scarves','Ring'],
                     normalize=True)

"""############   Testing images on model   ###############"""

def read_image(file_path):
    print("[INFO] loading and preprocessing image...")  
    image = load_img(file_path, target_size=(224, 224))  
    image = img_to_array(image)  
    image = np.expand_dims(image, axis=0)
    image /= 255.  
    return image

def test_single_image(path):
#    clothes = ['Scarves','Ring']
    clothes = names_tar
    images = read_image(path)
    time.sleep(.5)
    bt_prediction = vgg16.predict(images)  
    preds = model.predict_proba(bt_prediction)
    for idx, animal, x in zip(range(0,6), clothes , preds[0]):
        print("ID: {}, Label: {} {}%".format(idx, animal, round(x*100,2) ))
    print('Final Decision:')
    time.sleep(.5)
    for x in range(3):
        print('.'*(x+1))
        time.sleep(.2)
    class_predicted = model.predict_classes(bt_prediction)
    class_dictionary = generator_top.class_indices  
    inv_map = {v: k for k, v in class_dictionary.items()}  
    print("ID: {}, Label: {}".format(class_predicted[0], inv_map[class_predicted[0]]))  
    return load_img(path)

#test_my_image_path = '/content/large_2019_04_03_Ella_Chynna_FeverFish15102.jpg'

test_single_image(test_my_image_path)
