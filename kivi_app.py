'''
Camera Example
==============

This example demonstrates a simple use of the camera. It shows a window with
a buttoned labelled 'play' to turn the camera on and off. Note that
not finding a camera, perhaps because gstreamer is not installed, will
throw an exception during the kv language processing.

'''

# Uncomment these lines to see all the messages
# from kivy.logger import Logger
# import logging
# Logger.setLevel(logging.TRACE)

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
import time
from kivy.uix.camera import Camera
import os
from imageai.Classification import ImageClassification
import pyttsx3
engine = pyttsx3.init()


execution_path = os.getcwd()
prediction = ImageClassification()
prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath(os.path.join(execution_path, "mobilenet_v2.h5"))
prediction.loadModel()

camera = Camera(play=True, resolution=(224, 224), index=0)

def my_callback(dt):
    camera.export_to_png('env.png')
    predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "env.png"), result_count=10)
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(eachPrediction, " : ", eachProbability)
        if eachProbability > 45:
            engine.say(eachPrediction)
            engine.runAndWait()


Builder.load_string('''
<CameraClick>:
    orientation: 'vertical'
    Camera:
        id: camera
        resolution: (240, 240)
        play: True
    ToggleButton:
        text: 'Play'
        on_press: camera.play = not camera.play
        size_hint_y: None
        height: '48dp'
    Button:
        text: 'Capture'
        size_hint_y: None
        height: '48dp'
        on_press: root.capture()
''')




'''def capture_image(dt):
    camera = self.ids['camera']
    timestr = time.strftime("%Y%m%d_%H%M%S")
    camera.export_to_png('environment.png')'''

class CameraClick(BoxLayout):
    def capture(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        camera = self.ids['camera']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("IMG_{}.png".format(timestr))
        print("Captured")


class TestCamera(App):

    def build(self):
        Clock.schedule_interval(my_callback, 5) # Take images of environment ery 5 seconds



def run():
    time.sleep(5)
    TestCamera().run()


