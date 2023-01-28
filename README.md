### General image captioning python application

-----

https://towardsdatascience.com/image-captioning-in-deep-learning-9cd23fb4d8d2

TF
https://www.tensorflow.org/tutorials/text/image_captioning

Keras
https://keras.io/examples/vision/image_captioning/

HuggingFace
https://huggingface.co/nlpconnect/vit-gpt2-image-captioning


### Method 1 - Image captioning based on COCO dataset

### Instructions

Install the `pycoctools` with pip. In case of errors consider installing the "dev" version of python.
`sudo apt install python3.X-dev`

`pip install pycocotools`

Alternatively clone this repo: https://github.com/cocodataset/cocoapi  
```
git clone https://github.com/cocodataset/cocoapi.git  
```

and setup the coco API (also described in the readme [here](https://github.com/cocodataset/cocoapi)) 
```
cd cocoapi/PythonAPI  
make  
cd ..
```

#### COCO data
Download data from: http://cocodataset.org/#download

* Under **Annotations**, download:
  * **2014 Train/Val annotations [241MB]** (extract captions_train2014.json and captions_val2014.json, and place at locations cocoapi/annotations/captions_train2014.json and cocoapi/annotations/captions_val2014.json, respectively)  
  * **2014 Testing Image info [1MB]** (extract image_info_test2014.json and place at location cocoapi/annotations/image_info_test2014.json)

* Under **Images**, download:
  * **2014 Train images [83K/13GB]** (extract the train2014 folder and place at location cocoapi/images/train2014/)
  * **2014 Val images [41K/6GB]** (extract the val2014 folder and place at location cocoapi/images/val2014/)
  * **2014 Test images [41K/6GB]** (extract the test2014 folder and place at location cocoapi/images/test2014/)

