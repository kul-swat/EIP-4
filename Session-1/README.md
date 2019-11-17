## Definitions-

**Convolution**- Combination of input data with Kernel to produce output of a single layer. It contributes on reading the input data set.

**Kernel/ Filter**- It is a random 3*3 matrix values, which convolutes with input data to produce output. The channel number of kernel depends on channel number of output of earlier layer.

**Epoch**- The number of times the layer is to be trained, to give higher perfomance but the data shouldn't be overfitting

**1x1 matrix**- It is the matrix which is to be taken, to extract the bold feature from input layer. Dimension of matrix depend on the output of max pooling.

**3x3 matrix**- It is also called filter matrix used to filter the input data to produce output data of 1 layer. Number of Kernel depends on number of channel of output of earlier layer

**Activation Function**- It is a function used to giv the output. Relu Activation function is used as it convert negative values to 1 so eliminates the negative values in predicting output.

**Feature Map**- It is also called Channel. All similar quantity is kept under Channel. Example as 'E-Extractor' would extract only 'E' from a sentence.

**Receptive Field**- It is set of pixels in input image which is responsible for creating one pixel in the output image.

