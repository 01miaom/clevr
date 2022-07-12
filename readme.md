![sample output](./image/output2.jpg)

## Describe

The captcha geometry recognition model based on yolo can currently only recognize geometry types such as cylinders, spheres, cones, cubes, etc. If you need to recognize more types of geometry such as numbers and letters, you can use this model as a pre-training model to retrain.

## Use this model

Download the ```.h5``` model and ```detection_config.json```, and place these two files in the same folder as the image and script, then run the following code in script:

```python
from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(".h5")
detector.setJsonPath("detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="test.jpg", output_image_path="output.jpg")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
```

## Train the model yourself

- How to Label Data

Data annotation can refer to the following open source projects. After the annotation, an xml file corresponding to the image will be generated.

https://github.com/tzutalin/labelImg

- How to train a model

https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/Custom/CUSTOMDETECTIONTRAINING.md

- How to use an already trained model

https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/Custom/CUSTOMDETECTION.md

## Credits

- open-sorce: https://github.com/tzutalin/labelImg
- open-sorce: https://github.com/OlafenwaMoses/ImageAI
