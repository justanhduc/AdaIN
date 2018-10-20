#Adaptive Instance Normalization for style transfer

This is a re-implementation of the paper "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization" by Huang et al.. 
The implementation follows most but not all details from the paper, including some hyperparameter settings, output activation, etc...

##Requirements

[Theano](http://deeplearning.net/software/theano/index.html)

[neuralnet](https://github.com/justanhduc/neuralnet)

[VGG pretrained weight file](https://github.com/ftokarev/tf-vgg-weights/raw/master/vgg19_weights_normalized.h5)

##Usages

Type help for more details. Basically there are two main functions as follows.

###Training

Download the MS COCO 2017 train and valid sets [here](http://cocodataset.org/#download). 
Next download the [Wikiart](www.cs-chan.com/source/ICIP2017/wikiart.zip) dataset and also the [metadata](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset)
To train a network using the default settings, use

```
python train_test.py path-to-MS-COCO-train path-to-Wikiart-top-folder --input_path_val path-to-MS-COCO-val --style_train_val_path path-to-Wikiart-metadata
``` 

###Testing
There are two testing modes: bulk testing or single sample pair testing.
To test one pair, simply use

```
python train_test.py path-to-input path-to-style --test_one --checkpoint_file checkpoint-file-name --checkpoint_folder folder containing the checkpoint file
```

For bulk testing, the code currently support Wikiart style validation only because of reusing the same script to read data. 
To test a folder of images, use 

```
python train_test.py path-to-input path-to-Wikiart-top-folder --test_bulk --style_train_val_path path-to-Wikiart-metadata --checkpoint_file checkpoint-file-name --checkpoint_folder folder containing the checkpoint file
```

## Examples
<p align='center'>
  <img src='examples/test input 0_6.jpg' width="140px">
  <img src='examples/test style 0_6.jpg' width="140px">
  <img src='examples/test output 0_6.jpg' width="140px">
  <img src='examples/test input 0_12.jpg' width="140px">
  <img src='examples/test style 0_12.jpg' width="140px">
  <img src='examples/test output 0_12.jpg' width="140px">
</p>

<p align='center'>
  <img src='examples/test input 4_8' width="140px">
  <img src='examples/test style 4_8.jpg' width="140px">
  <img src='examples/test output 4_8.jpg' width="140px">
  <img src='examples/test input 4_17.jpg' width="140px">
  <img src='examples/test style 4_17.png' width="140px">
  <img src='examples/test output 4_17.jpg' width="140px">
</p>

<p align='center'>
  <img src='examples/test input 8_3.jpg' width="140px">
  <img src='examples/test style 8_3.jpg' width="140px">
  <img src='examples/test output 8_3.jpg' width="140px">
  <img src='examples/test input_0.jpg' width="140px">
  <img src='examples/test style_0.jpg', width="140px">
  <img src='examples/test output_0.jpg' width="140px">
</p>

##References
"Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization" by Huang et al.

The pretrained VGG is taken from [this Tensorflow implementation]().

[ArtGAN](https://github.com/cs-chan/ArtGAN) (providing the Wikiart zip file).
