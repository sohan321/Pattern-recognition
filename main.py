# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:58:04 2025

@author: sohan
"""
# read the MNIST dataset
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def findDist(image1, image2):
    image1=np.float32(image1)
    image2=np.float32(image2)
    difference=0
    total=0
    for i in range(0, 28):
        for j in range(0,28):
            val1=image1[i][j]
            val2=image2[i][j]
            difference= np.square(val1-val2)
            total =total + difference
    return np.sqrt(total)

def findDistFast(image1, image2):
    image1=np.float32(image1)
    image2=np.float32(image2)
    difference=image1-image2
    sq=difference * difference
    total=sq.sum()
    return np.sqrt(total)



def findClosest(image1, training_images):
    lowest=np.inf
    index=-1
    for i in range(0,1000):
        distance=findDistFast(image1, training_images[i])
        if lowest > distance:
            lowest=distance
            index=i
    return  index

def nearestNeighbor(image1, training_images, training_labels):
    index=findClosest(image1, training_images)
    label=training_labels[index]
    return label
    
def evaluate(test_images, test_labels):
    correct=0
    incorrect=0;
    for i in range(0,1000):
         ans=nearestNeighbor(test_images[i], training_images, training_labels)
         value=test_labels[i]
         if ans ==value:
             correct=correct+1
         else:
            incorrect=incorrect+1
    print(correct, "correct", incorrect, "incorrect")
         
        


dataset = tf.keras.datasets.mnist
(training_set, test_set) = dataset.load_data()
(training_images, training_labels) = training_set
(test_images, test_labels) = test_set

# display training image at position 17
training_index = 0
example_image = training_images[training_index]
label = training_labels[training_index]
correct =0
incorrect=0
count=0
for i in range(10):
    training_index=i
    example_image = training_images[training_index]
    label = training_labels[training_index]
    if label ==1 or label ==0: 
        count=count+1
        plt.imshow(example_image, cmap='gray')
        if training_images[i].mean() > 30:
            print("Guess: 0")
            if label==0:
                correct=correct+1
            else:
                incorrect=incorrect+1
        else:
            print("Guess: 1")
            if label==1:
                correct=correct+1
            else:
                incorrect=incorrect+1
            
       # print("dtype:", example_image.dtype)
       # print("shape:", example_image.shape)
        print("Class label:", label, "Training index:", training_index)

       # print(training_images[1].mean())
       # print(training_images[3].mean())
ratio=correct/count *100
#print(ratio,"%")

            
print(findDist(training_images[0], training_images[1]))
print("The closest image to 0 is", findClosest(test_images[0], training_images))
print(nearestNeighbor(test_images[0], training_images, training_labels))
plt.imshow(test_images[0], cmap='gray')
print(findClosest(test_images[5], training_images))
plt.imshow(training_images[773], cmap='gray')
evaluate(test_images, test_labels)
print(findDistFast(training_images[0], training_images[1]))

#%%
"""
model =tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(512,activation='tanh'),
    tf.keras.layers.Dense(10,activation="softmax")
    ])
model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print('\nTest accuracy: %.2f%%' %(test_acc * 100))
"""


"""
# Turning values into float32 for better accuracy

training_images=np.float32(training_images)
test_images=np.float32(test_images)
model =tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(512,activation='tanh'),
    tf.keras.layers.Dense(10,activation="softmax")
    ])
model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print('\nTest accuracy: %.2f%%' %(test_acc * 100))
"""
#Normalizing values for even better accuracy

training_images=training_images/255
test_images=test_images/255
model =tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(512,activation='tanh'),
    tf.keras.layers.Dense(10,activation="softmax")
    ])
model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print('\nTest accuracy: %.2f%%' %(test_acc * 100))

#%%
dataset = tf.keras.datasets.mnist
(training_set, test_set) = dataset.load_data()
(training_images, training_labels) = training_set
(test_images, test_labels) = test_set

training_images=training_images/255
test_images=test_images/255

training_images=np.float32(training_images)
test_images=np.float32(test_images)


training_images =np.expand_dims(training_images,-1)
test_images=np.expand_dims(test_images,-1)
model = tf.keras.Sequential([
tf.keras.Input(shape=training_images[0].shape),
tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
tf.keras.layers.Flatten(),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])
model.fit(training_images[0:100], training_labels[0:100], epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print('\nTest accuracy: %.2f%%' %(test_acc * 100))
    
    