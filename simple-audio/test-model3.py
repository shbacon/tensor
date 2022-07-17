import tensorflow as tf

audio_binary = tf.io.read_file('/Users/tbacon/Desktop/tensor/data/mini_speech_commands/yes/0c2d2ffa_nohash_0.wav')
audio, _ = tf.audio.decode_wav(contents=audio_binary)
waveform = tf.squeeze(audio, axis=-1)
input_len = 16000
waveform = waveform[:input_len]
zero_padding = tf.zeros([16000] - tf.shape(waveform),dtype=tf.float32)
waveform = tf.cast(waveform, dtype=tf.float32)
equal_length = tf.concat([waveform, zero_padding], 0)
spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
spectrogram = tf.abs(spectrogram)
spectrogram = spectrogram[..., tf.newaxis]

interpreter = tf.lite.Interpreter(model_path='./model.tflite')
interpreter.allocate_tensors()
output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.
interpreter.set_tensor(input['index'], spectrogram)
interpreter.invoke()
print(interpreter.get_tensor(output['index']))
