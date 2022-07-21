import wavefile
import tensorflow as tf

#16K Sample, Mono, 1 Second -> Float32 Values
samples = wavefile.load('/Users/tbacon/Desktop/tensor/data/mini_speech_commands/yes/0c2d2ffa_nohash_0.wav')[1][0]
data = tf.constant(samples)
spectrogram = tf.signal.stft(data, frame_length=255, frame_step=128)
spectrogram = tf.abs(spectrogram)
spectrogram = spectrogram[..., tf.newaxis]

interpreter = tf.lite.Interpreter(model_path='./model.tflite')
interpreter.allocate_tensors()
output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.
interpreter.set_tensor(input['index'], [spectrogram])
interpreter.invoke()
print(interpreter.get_tensor(output['index']))