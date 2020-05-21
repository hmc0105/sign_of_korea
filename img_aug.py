from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import os

cwd=os.getcwd()
train_datagen=ImageDataGenerator(horizontal_flip=True,shear_range=0.2,zoom_range=0.5,
width_shift_range=0.3,height_shift_range=0.3)

i=0

for batch in train_datagen.flow_from_directory(
    directory=cwd,subset='training',batch_size=300,seed=42,shuffle=True,class_mode='categorical',save_to_dir=cwd,save_format='jpeg') :
    i+=1
    if i>20:
        break

