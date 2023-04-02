# Project *Search for Mouse*
___
Project to experiment with: Object detection, Tensorflow. <br>
Using robot Vector (DDL/Anki's Robot: https://www.digitaldreamlabs.com/products/vector-robot)
<br>

## Description
After Vector's greeting and job assessment acknowledgement, he starts to turn in place and look
if he sees a mouse:
- If yes, he will move toward it, and try to *catch* it. 
  - Successful catch is determined from Vector's pitch sensor value.
- If not, after a complete 360 degrees rotation, he will stop searching


Here is a demonstration video: <br>
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/2PdNvz8b8aI/0.jpg)](https://www.youtube.com/watch?v=2PdNvz8b8aI)


## Some built details
The *object detection* part of this application has been made following :
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html


### Object Detection Model
Most of the work was to built the object detection model, which is not included in the repository (it is too large).<br>
In my case: 
- I prepared few hundreds images, taken with Vector's front camera. 
About 50%/50% with/without a mouse in the field of view.
- Annotate this dataset
- Use the pre-trained model *ssd_resnet50_v1* and this dataset to finalize training and model built. 
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html <br>
