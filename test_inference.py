from pycoral.utils import edgetpu
from pycoral.adapters import common
from PIL import Image
import time
import cv2 as cv

model_file = 'models/vggface_quant.tflite'
img_file = 'images/reza.jpg'

interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()


size = common.input_size(interpreter)
#img = Image.open(img_file).convert('RGB').resize(size, 1)
img = cv.imread(img_file)[:,:,::-1]

for i in range(5):
    print (size)
    t1 = time.time()
    common.set_input(interpreter, img)
    interpreter.invoke()

    t2 = time.time()
    print ((t2-t1))