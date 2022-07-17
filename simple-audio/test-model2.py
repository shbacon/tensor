import tensorflow as tf
import os
import numpy as np
import pathlib

DATASET_PATH = 'data/mini_speech_commands'
data_dir = pathlib.Path(DATASET_PATH)
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
print('Commands:', commands)
AUTOTUNE = tf.data.AUTOTUNE
def get_spectrogram(waveform):
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros([16000] - tf.shape(waveform),dtype=tf.float32)
    waveform = tf.cast(waveform, dtype=tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def get_label(file_path):
    parts = tf.strings.split(input=file_path,sep=os.path.sep)
    return parts[-2]

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    label_id = tf.math.argmax(label == commands)
    return spectrogram, label_id

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(map_func=get_waveform_and_label,num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(map_func=get_spectrogram_and_label_id,num_parallel_calls=AUTOTUNE)
    return output_ds


interpreter = tf.lite.Interpreter(model_path='./model.tflite')
interpreter.allocate_tensors()
output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.

sample_ds = preprocess_dataset(['/Users/tbacon/Desktop/tensor/data/mini_speech_commands/yes/0c2d2ffa_nohash_0.wav','/Users/tbacon/Desktop/tensor/data/mini_speech_commands/test/7061-6-0-0.wav','/Users/tbacon/Desktop/tensor/data/mini_speech_commands/yes/0ff728b5_nohash_1.wav'])

for spectrogram, label in sample_ds.batch(1):
    #print(spectrogram)
    interpreter.set_tensor(input['index'], spectrogram)
    interpreter.invoke()
    #interpreter.get_tensor(output['index']).shape
    print(interpreter.get_tensor(output['index']))
