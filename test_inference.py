from pycoral.utils import edgetpu
from pycoral.adapters import common
import tflite_runtime.interpreter as tflite
from PIL import Image
import time
import cv2 as cv
import numpy as np

model_file = 'models/vggface_quant.tflite'
img_file = 'images/reza.jpg'

# For pycoral
#interpreter = edgetpu.make_interpreter(model_file)

interpreter = tflite.Interpreter(model_file, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()
interpreter_output = interpreter.get_output_details()[0]
interpreter_input = interpreter.get_input_details()[0]


#size = common.input_size(interpreter)
#img = Image.open(img_file).convert('RGB').resize(size, 1)

img = cv.imread(img_file)[:,:,::-1]
img = np.expand_dims(img, axis=0)
# Predict vector
# vector = self.classifier.predict(face)[0]

for i in range(5):
    print (img.shape)
    t1 = time.time()
    # for pycoral
    # common.set_input(interpreter, img)
    interpreter.set_tensor(interpreter_input['index'], img)
    interpreter.invoke()

    t2 = time.time()
    print ((t2-t1))