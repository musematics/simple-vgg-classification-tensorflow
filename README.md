# simple-vgg-classification-tensorflow
Simple VGG implementation for image classification using tensorflow and python. Easy to use for beginners like me. 

## environment
Python 2.7.12
Tensorflow 1.3.0

## data organization
Put your data according to the labels (1,2,3,...,n) into different folder.
Training:  ./example/train/1, ./example/train/2 ,... , ./example/train/n
Testing: ./example/test/1, ./example/test/2 ,... , ./example/test/n

Or one dataset using cross validation
./easy/1, ./easy/2,..., ./easy/n

To alter between one dataset cross validation and training & testing set, comment or uncomment the corresponding codes.

## note
The data in this repository is for demonstration, and they are insufficient for acquiring desirable performance.

## more information
For more information (in Chinese), please visit [my blog](https://blog.csdn.net/rocachilles/article/details/87894808)
