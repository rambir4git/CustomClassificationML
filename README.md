# CustomClassificationML
Android app to run custom tflite models

create a assets folder at CustomClassificationML/app/src/main/*
place crop_model.tflite and crop_labels.json in that assets folder.
NOTE:
1) if you change file names make sure to change them in code aswell
2) use the code from colab notebook to obtain .tflite and .json files for a trained model
      
Explicty grant all these permissions:
1) STORAGE
2) CAMERA
3) LOCATION
4) MICROPHONE

change following variables in MainActivity.java according to your model
1) inputImageSize - size of the input image for the model
2) classifications - number of different classication categories

Keep an eye on logcat for debugging, application TAG is CustomModel
