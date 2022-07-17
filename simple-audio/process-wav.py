import librosa    
import os
import soundfile as sf

#Resample to 16K and Convert to Mono
sourceDir = './data/mini_speech_commands/fold1'
destDir = './test'
for filename in os.listdir(sourceDir):
    print(filename)
    f = os.path.join(sourceDir, filename)
    y, s = librosa.load(f, sr=16000) # Downsample to 16000
    sf.write(os.path.join(destDir, filename), y, 16000)


