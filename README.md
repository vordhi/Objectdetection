# Objectdetection
FireDetection With Object Detection The Goal of the project was to detect fire and smoke from a real-time 
feed from the cameras, such as CCTV cameras. The pre-trained model was used and retrained with 
fire and smoke images to achieve the desired result.

Use "requirements.txt" to recreate the environment

For testing Run  the "test.py" script. It will load the model and will open the webcam to detect the fire.
As a test you can show a fire image from youtube video from your mobile. You need to bring the mobile close to the webcam.
for the model to detect fire from a video in a mobile.
If you have a Labview 2014 installed. Uncomment the codeblocks marked in the "test.py" file.

### For installing the Tensorflow objectdetection package
firt clone the repository
> git clone https://github.com/tensorflow/models.git

Then Change directory as shown below:
> cd models/research
### Install TensorFlow Object Detection API.
> cp object_detection/packages/tf1/setup.py .

> python -m pip install .
