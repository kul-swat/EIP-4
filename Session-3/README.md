### Final validation accuracy of base model is: 82.57
### Maximum validation accuracy of base model is: 83.06

### Defination of my model

model = Sequential()

model.add(SeparableConv2D(32, (3, 3), strides=(1, 1), border_mode='same', input_shape=(32, 32, 3))) #32x32x32 RF=3
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(SeparableConv2D(128, (3, 3), strides=(1, 1), border_mode='same'))#32x32x128 RF=5
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(SeparableConv2D(256, (3, 3), strides=(1, 1), border_mode='same')) #32x32x256 RF=7
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(MaxPooling2D(pool_size=(2, 2))) #16x16x256 RF=14
model.add(Convolution2D(32, 1, 1, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(SeparableConv2D(128, (3, 3), strides=(1, 1), border_mode='same')) #16x16x128 RF=16
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(SeparableConv2D(256, (3, 3), strides=(1, 1), border_mode='same')) #16x16x256 RF=18
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(MaxPooling2D(pool_size=(2, 2))) # 8x8x256, RF= 36
model.add(Convolution2D(64, 1, 1, activation='relu')) 
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 1, 1, activation='relu')) #8x8x10 RF=36
model.add(BatchNormalization())

model.add(GlobalAveragePooling2D())
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



### Training log of mu=y model

/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:12: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
  if sys.path[0] == '':
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:12: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., validation_data=(array([[[..., verbose=1, steps_per_epoch=390, epochs=50)`
  if sys.path[0] == '':
Epoch 1/50
390/390 [==============================] - 23s 59ms/step - loss: 1.6830 - acc: 0.4418 - val_loss: 1.5674 - val_acc: 0.4463
Epoch 2/50
390/390 [==============================] - 20s 52ms/step - loss: 1.3363 - acc: 0.5820 - val_loss: 1.3588 - val_acc: 0.5569
Epoch 3/50
390/390 [==============================] - 20s 52ms/step - loss: 1.1578 - acc: 0.6385 - val_loss: 1.3656 - val_acc: 0.5223
Epoch 4/50
390/390 [==============================] - 20s 51ms/step - loss: 1.0442 - acc: 0.6696 - val_loss: 1.3277 - val_acc: 0.5303
Epoch 5/50
390/390 [==============================] - 20s 52ms/step - loss: 0.9598 - acc: 0.6950 - val_loss: 1.0713 - val_acc: 0.6454
Epoch 6/50
390/390 [==============================] - 20s 52ms/step - loss: 0.8944 - acc: 0.7132 - val_loss: 1.0026 - val_acc: 0.6537
Epoch 7/50
390/390 [==============================] - 20s 51ms/step - loss: 0.8447 - acc: 0.7255 - val_loss: 1.1758 - val_acc: 0.5998
Epoch 8/50
390/390 [==============================] - 20s 51ms/step - loss: 0.7986 - acc: 0.7410 - val_loss: 1.3312 - val_acc: 0.5411
Epoch 9/50
390/390 [==============================] - 20s 51ms/step - loss: 0.7611 - acc: 0.7523 - val_loss: 1.1119 - val_acc: 0.6190
Epoch 10/50
390/390 [==============================] - 20s 52ms/step - loss: 0.7225 - acc: 0.7666 - val_loss: 0.9004 - val_acc: 0.6909
Epoch 11/50
390/390 [==============================] - 20s 51ms/step - loss: 0.6932 - acc: 0.7749 - val_loss: 0.7545 - val_acc: 0.7454
Epoch 12/50
390/390 [==============================] - 20s 51ms/step - loss: 0.6656 - acc: 0.7833 - val_loss: 0.7612 - val_acc: 0.7362
Epoch 13/50
390/390 [==============================] - 20s 51ms/step - loss: 0.6425 - acc: 0.7884 - val_loss: 0.8505 - val_acc: 0.7040
Epoch 14/50
390/390 [==============================] - 20s 51ms/step - loss: 0.6196 - acc: 0.7966 - val_loss: 0.9851 - val_acc: 0.6617
Epoch 15/50
390/390 [==============================] - 20s 51ms/step - loss: 0.6020 - acc: 0.8024 - val_loss: 0.9517 - val_acc: 0.6710
Epoch 16/50
390/390 [==============================] - 20s 51ms/step - loss: 0.5828 - acc: 0.8106 - val_loss: 0.7076 - val_acc: 0.7547
Epoch 17/50
390/390 [==============================] - 20s 51ms/step - loss: 0.5647 - acc: 0.8146 - val_loss: 0.8948 - val_acc: 0.6798
Epoch 18/50
390/390 [==============================] - 20s 51ms/step - loss: 0.5468 - acc: 0.8209 - val_loss: 0.8079 - val_acc: 0.7192
Epoch 19/50
390/390 [==============================] - 20s 51ms/step - loss: 0.5345 - acc: 0.8258 - val_loss: 0.6590 - val_acc: 0.7778
Epoch 20/50
390/390 [==============================] - 20s 51ms/step - loss: 0.5260 - acc: 0.8262 - val_loss: 0.8759 - val_acc: 0.6956
Epoch 21/50
390/390 [==============================] - 20s 52ms/step - loss: 0.5094 - acc: 0.8325 - val_loss: 0.6923 - val_acc: 0.7622
Epoch 22/50
390/390 [==============================] - 20s 51ms/step - loss: 0.4973 - acc: 0.8349 - val_loss: 0.7027 - val_acc: 0.7596
Epoch 23/50
390/390 [==============================] - 20s 51ms/step - loss: 0.4882 - acc: 0.8385 - val_loss: 0.7536 - val_acc: 0.7404
Epoch 24/50
390/390 [==============================] - 20s 51ms/step - loss: 0.4776 - acc: 0.8416 - val_loss: 0.7375 - val_acc: 0.7479
Epoch 25/50
390/390 [==============================] - 20s 51ms/step - loss: 0.4664 - acc: 0.8451 - val_loss: 0.7198 - val_acc: 0.7549
Epoch 26/50
390/390 [==============================] - 20s 51ms/step - loss: 0.4593 - acc: 0.8471 - val_loss: 0.6273 - val_acc: 0.7829
Epoch 27/50
390/390 [==============================] - 20s 51ms/step - loss: 0.4527 - acc: 0.8494 - val_loss: 0.7872 - val_acc: 0.7324
Epoch 28/50
390/390 [==============================] - 20s 51ms/step - loss: 0.4386 - acc: 0.8547 - val_loss: 0.5944 - val_acc: 0.7954
Epoch 29/50
390/390 [==============================] - 20s 51ms/step - loss: 0.4354 - acc: 0.8552 - val_loss: 0.5535 - val_acc: 0.8113
Epoch 30/50
390/390 [==============================] - 20s 51ms/step - loss: 0.4253 - acc: 0.8575 - val_loss: 0.8822 - val_acc: 0.6966
Epoch 31/50
390/390 [==============================] - 20s 51ms/step - loss: 0.4158 - acc: 0.8615 - val_loss: 0.6217 - val_acc: 0.7840
Epoch 32/50
390/390 [==============================] - 20s 51ms/step - loss: 0.4130 - acc: 0.8618 - val_loss: 0.6230 - val_acc: 0.7884
Epoch 33/50
390/390 [==============================] - 20s 51ms/step - loss: 0.4068 - acc: 0.8647 - val_loss: 0.6500 - val_acc: 0.7804
Epoch 34/50
390/390 [==============================] - 20s 51ms/step - loss: 0.4002 - acc: 0.8659 - val_loss: 0.6788 - val_acc: 0.7665
Epoch 35/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3886 - acc: 0.8699 - val_loss: 0.6862 - val_acc: 0.7660
Epoch 36/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3880 - acc: 0.8691 - val_loss: 0.5754 - val_acc: 0.8032
Epoch 37/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3778 - acc: 0.8732 - val_loss: 0.6130 - val_acc: 0.7865
Epoch 38/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3707 - acc: 0.8755 - val_loss: 0.6505 - val_acc: 0.7824
Epoch 39/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3705 - acc: 0.8757 - val_loss: 0.7389 - val_acc: 0.7492
Epoch 40/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3658 - acc: 0.8770 - val_loss: 0.6288 - val_acc: 0.7878
Epoch 41/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3639 - acc: 0.8776 - val_loss: 0.5990 - val_acc: 0.8045
Epoch 42/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3532 - acc: 0.8810 - val_loss: 0.7368 - val_acc: 0.7570
Epoch 43/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3526 - acc: 0.8810 - val_loss: 0.5494 - val_acc: 0.8114
Epoch 44/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3422 - acc: 0.8836 - val_loss: 0.6168 - val_acc: 0.7976
Epoch 45/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3442 - acc: 0.8828 - val_loss: 0.5695 - val_acc: 0.8125
Epoch 46/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3381 - acc: 0.8867 - val_loss: 0.5254 - val_acc: 0.8214
Epoch 47/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3315 - acc: 0.8883 - val_loss: 0.6058 - val_acc: 0.8013
Epoch 48/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3259 - acc: 0.8905 - val_loss: 0.5905 - val_acc: 0.8019
Epoch 49/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3225 - acc: 0.8915 - val_loss: 0.6025 - val_acc: 0.7970
Epoch 50/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3231 - acc: 0.8908 - val_loss: 0.5699 - val_acc: 0.8097
Model took 1002.75 seconds to train

Accuracy on test data is: 80.97

