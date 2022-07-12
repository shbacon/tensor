from tflite_support.task import audio
from tflite_support.task import core
from tflite_support.task import processor

base_options = core.BaseOptions(file_name='/Users/sobacon/Desktop/tensor/birds_models/my_birds_model.tflite')
classification_options = processor.ClassificationOptions(max_results=1)
options = audio.AudioClassifierOptions(base_options=base_options, classification_options=classification_options)
classifier = audio.AudioClassifier.create_from_options(options)

audio_file = audio.TensorAudio.create_from_wav_file('/Users/sobacon/Desktop/tensor/dataset/small_birds_dataset/test/gun/gunshots0_00002_00.wav', classifier.required_input_buffer_size)
audio_result = classifier.classify(audio_file)
print(audio_result)
