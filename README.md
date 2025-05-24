This is a clone of https://github.com/melihaltun/Eye_Tracking_with_Deep_CNN

Training DataSet : https://perceptualui.org/research/datasets/LPW/

Some adjustments have been made to make the training setup is easier.

Training and training setup
```commandline
$ selectTrainTestValData.py
$ getMeanIntensity.py
$ vid2Frames.py
$ trainModel.py

# to test on live images acquired from live camera
$ testVideoStream.py

# to fuse the split compressed model checkpoint for extraction
$ zip -s0 models_parts.zip --out models_fused.zip
# then extract models_fused into the same directory
```

There is an additional pretrained model for quick testing.

The aim was to make a POC and use the NNet to track eye movement and link it to a controller to enhance communication capabilities for people with severe motor handicaps (e.g. ALS)

Unless the model was overfitted during training, currently known issues are:
- either the image from the tested capturing device needs to be preprocesses in order to extract the eyes bounding boxes and then do the inference computation to get pupil's position
- or the image quality provided by the capturing device does not align with the image quality of the dataset used for training and thus generates bad results.

Tested on Ubuntu 20.04
