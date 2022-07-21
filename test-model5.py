import wavefile
import tensorflow as tf
import numpy as np

def callback(rec):
    window[:len(window)//2] = window[len(window)//2:]
    window[len(window)//2:] = rec
    data = tf.constant(window)
    spectrogram = tf.signal.stft(data, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    interpreter.set_tensor(input['index'], [spectrogram])
    interpreter.invoke()
    print(interpreter.get_tensor(output['index']))

samples = wavefile.load('/Users/tbacon/Desktop/tensor/data/mini_speech_commands/yes/0c2d2ffa_nohash_0.wav')[1][0]
window = np.zeros(16000, dtype=np.float32)

interpreter = tf.lite.Interpreter(model_path='./model.tflite')
interpreter.allocate_tensors()
output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.

callback(samples[:8000])
callback(samples[8000:16000])
callback(samples[:8000])
callback(samples[8000:16000])

