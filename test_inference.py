from pycoral.utils import edgetpu
from pycoral.adapters import common
from PIL import Image

model_file = 'models/vggface_quant.tflite'
img_file = 'images/reza.jpg'

interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()

size = common.input_size(interpreter)
img = Image.open(img_file).convert('RGB').resize(size, 1)

