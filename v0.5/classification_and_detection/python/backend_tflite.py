"""
tflite backend (https://github.com/tensorflow/tensorflow/lite)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

from threading import Lock

import tensorflow as tf
from tensorflow.lite.python import interpreter as interpreter_wrapper
from tensorflow.lite.python.optimize import calibrator
from tensorflow.python.framework.dtypes import qint8 # quantized signed 8-bit integer

import backend


class BackendTflite(backend.Backend):
    def __init__(self):
        super(BackendTflite, self).__init__()
        self.sess = None
        self.lock = Lock()

    def version(self):
        return tf.__version__ + "/" + tf.__git_version__

    def name(self):
        return "tflite"

    def image_format(self):
        # tflite is always NHWC
        return "NHWC"

    def load(self, model_path, inputs=None, outputs=None):
        with open(model_path, 'rb') as f:
            self.quantizer = calibrator.Calibrator(f.read())
        self.sess = interpreter_wrapper.Interpreter(model_path=model_path)
        self.sess.allocate_tensors()
        # keep input/output name to index mapping
        self.input2index = {i["name"]: i["index"] for i in self.sess.get_input_details()}
        self.output2index = {i["name"]: i["index"] for i in self.sess.get_output_details()}
        # keep input/output names
        self.inputs = list(self.input2index.keys())
        self.outputs = list(self.output2index.keys())
        return self

    def predict(self, feed):
        def _input_gen():
            """Yield one batch of feed items.
                Returns:
                  [v] generator: A batch of N feed items.
            """
            for item in feed['input']:
                yield [[item]]

        self.lock.acquire()
        # generate quantized model
        self.quantized_model = self.quantizer.calibrate_and_quantize(
            _input_gen,
            allow_float=False,
            input_type=qint8,
            output_type=qint8,
        )
        # set inputs
        for k, v in self.input2index.items():
            self.sess.set_tensor(v, feed[k])
        self.sess.invoke()
        # get results
        res = [self.sess.get_tensor(v) for _, v in self.output2index.items()]
        self.lock.release()
        return res
