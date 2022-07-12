import os

directory = './wavefiles'
f = open("wave.csv", "w")
for filename in os.listdir(directory):
    print(filename)
    f.write("gun,[],['song'],,,,,,,"+filename+",,5,,,,train\n")
    #f = os.path.join(directory, filename)
    #if os.path.isfile(f): print(f)
f.close()
