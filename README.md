### General image captioning python application

----
#### Description

In this repo implementations of image captioning methods have been dockerized and prepared for deployment. 
The goal is to provide a simple and easy to use application for image captioning with a web interface.


-----

Overview of image captioning approaches 
https://towardsdatascience.com/image-captioning-in-deep-learning-9cd23fb4d8d2

TF
https://www.tensorflow.org/tutorials/text/image_captioning

Keras
https://keras.io/examples/vision/image_captioning/

HuggingFace
https://huggingface.co/nlpconnect/vit-gpt2-image-captioning

image understanding 
https://huggingface.co/docs/transformers/main/model_doc/bridgetower

---

### 1. Image captioning using the COCO dataset

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

#### Dataset for training - COCO (Common Objects in Context)
Download data from: http://cocodataset.org/#download

* Under **Annotations**, download:
  * **2017 Train/Val annotations [241MB]** (extract captions_train2017.json and captions_val2017.json, and place at locations cocoapi/annotations/captions_train2017.json and cocoapi/annotations/captions_val2017.json, respectively)  
  * **2017 Testing Image info [1MB]** (extract image_info_test2017.json and place at location cocoapi/annotations/image_info_test2017.json)

* Under **Images**, download:
  * **2017 Train images [83K/13GB]** (extract the train2017 folder and place at location cocoapi/images/train2017/)
  * **2017 Val images [41K/6GB]** (extract the val2017 folder and place at location cocoapi/images/val2017/)
  * **2017 Test images [41K/6GB]** (extract the test2017 folder and place at location cocoapi/images/test2017/)

#### Training

#### Deployment
The deployment of the application is on q cloud service providing a bare-bone instance (such as DigitalOcean Droplet or in German provider Strato)

- dockerize the application by running the build (see run_docker.sh)
- save the image as a tar file for transfer to the server
`sudo docker save -o PATH_TO_FILE.tar IMAGE_NAME`

- deliver the tar image to the server 
`rsync -avz PATH_TO_FILE.tar username@remote_server:directory`

- load the image on the server
`sudo docker load -i PATH_TO_FILE.tar`

- killing uvicorns (if running)
(`sudo kill $(pgrep -P PID)`)

#### Setting up a reverse proxy with nginx


### Web-server API interface using FastAPI

<img src="assets/api_Screenshot.png">
