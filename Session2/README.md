/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:1: UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.
  """Entry point for launching an IPython kernel.
Train on 60000 samples, validate on 10000 samples
Epoch 1/20
60000/60000 [==============================] - 46s 774us/step - loss: 0.3488 - acc: 0.9385 - val_loss: 0.0886 - val_acc: 0.9781
Epoch 2/20
60000/60000 [==============================] - 41s 678us/step - loss: 0.1097 - acc: 0.9741 - val_loss: 0.0578 - val_acc: 0.9837
Epoch 3/20
60000/60000 [==============================] - 40s 672us/step - loss: 0.0800 - acc: 0.9786 - val_loss: 0.0456 - val_acc: 0.9868
Epoch 4/20
60000/60000 [==============================] - 41s 683us/step - loss: 0.0649 - acc: 0.9822 - val_loss: 0.0466 - val_acc: 0.9852
Epoch 5/20
60000/60000 [==============================] - 40s 674us/step - loss: 0.0574 - acc: 0.9839 - val_loss: 0.0543 - val_acc: 0.9837
Epoch 6/20
60000/60000 [==============================] - 40s 674us/step - loss: 0.0528 - acc: 0.9853 - val_loss: 0.0319 - val_acc: 0.9899
Epoch 7/20
60000/60000 [==============================] - 40s 675us/step - loss: 0.0483 - acc: 0.9855 - val_loss: 0.0316 - val_acc: 0.9901
Epoch 8/20
60000/60000 [==============================] - 40s 674us/step - loss: 0.0448 - acc: 0.9869 - val_loss: 0.0347 - val_acc: 0.9888
Epoch 9/20
60000/60000 [==============================] - 41s 679us/step - loss: 0.0436 - acc: 0.9875 - val_loss: 0.0398 - val_acc: 0.9883
Epoch 10/20
60000/60000 [==============================] - 41s 675us/step - loss: 0.0400 - acc: 0.9883 - val_loss: 0.0260 - val_acc: 0.9921
Epoch 11/20
60000/60000 [==============================] - 41s 681us/step - loss: 0.0404 - acc: 0.9875 - val_loss: 0.0279 - val_acc: 0.9906
Epoch 12/20
60000/60000 [==============================] - 41s 684us/step - loss: 0.0373 - acc: 0.9889 - val_loss: 0.0270 - val_acc: 0.9913
Epoch 13/20
60000/60000 [==============================] - 41s 676us/step - loss: 0.0358 - acc: 0.9895 - val_loss: 0.0268 - val_acc: 0.9917
Epoch 14/20
60000/60000 [==============================] - 41s 676us/step - loss: 0.0356 - acc: 0.9897 - val_loss: 0.0282 - val_acc: 0.9910
Epoch 15/20
60000/60000 [==============================] - 40s 675us/step - loss: 0.0333 - acc: 0.9902 - val_loss: 0.0241 - val_acc: 0.9921
Epoch 16/20
60000/60000 [==============================] - 41s 679us/step - loss: 0.0328 - acc: 0.9899 - val_loss: 0.0308 - val_acc: 0.9898
Epoch 17/20
60000/60000 [==============================] - 41s 676us/step - loss: 0.0338 - acc: 0.9901 - val_loss: 0.0187 - val_acc: 0.9943
Epoch 18/20
60000/60000 [==============================] - 40s 675us/step - loss: 0.0311 - acc: 0.9903 - val_loss: 0.0250 - val_acc: 0.9926
Epoch 19/20
60000/60000 [==============================] - 41s 681us/step - loss: 0.0315 - acc: 0.9900 - val_loss: 0.0239 - val_acc: 0.9929
Epoch 20/20
60000/60000 [==============================] - 41s 683us/step - loss: 0.0296 - acc: 0.9909 - val_loss: 0.0191 - val_acc: 0.9925
<keras.callbacks.History at 0x7f53e7319588>

Logs of 20 Epochs are mentioned above.
Strategies to get 99.43% validation accuracy on 17th Epoch are: 
1.) Used kernel of size 14x14 and 30x30, applied max pooling after 3 layers of convolutions. 
2.) Batch Normalization is used in every layer. 
3.) Drop out is used when model was reacting over fitting. 
4.) Earlier 10x10 and 20x20 were used but model was unable to learn so used GAP and increased the size of kernel. 
5.) Used learning Rate and increased the value from 0.01 to 0.03 and got the accuracy. 
