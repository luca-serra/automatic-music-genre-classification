
from __future__ import print_function
import librosa
import csv
import numpy as np
from sklearn import model_selection
import pandas as pd
import pywt
import pywt.data
from sklearn import preprocessing, svm, tree, ensemble, decomposition
from numpy.fft import fft, fftshift


# 1. Get the file path to the included audio example
filename = librosa.util.example_audio_file()

# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
y, sr = librosa.load("blues.00004.au")
# 3. Run the default beat tracker
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print(len(y),sr)
print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

# 4. Convert the frame indices of beat events into timestamps
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

print('Saving output to beat_times.csv')
librosa.output.times_csv('beat_times.csv', beat_times)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mfccs1 = librosa.feature.mfcc(S=librosa.power_to_db(S))
S=librosa.power_to_db(S)
mfccs = librosa.feature.mfcc(y, sr)
sc = librosa.feature.spectral_centroid(y=y, sr=sr)
az = librosa.feature.spectral_centroid(y=y, sr=sr, S=None, n_fft=2048, hop_length=512, freq=None)
print(az)


def mffcdata(xmfcc, a, b):
    avmfcc = 0
    varmfcc = 0
    for i in range(a):
        vari = 0
        avi = 0
        for j in range(b):
            avi += xmfcc[i][j]/b
        for j in range(b):
            vari += ((xmfcc[i][j] - avi)**2)/b
        avmfcc += avi/a
        varmfcc += vari/a
    return avmfcc, varmfcc


print(mffcdata(mfccs,1,20))
d = 15
n = 9000
data = [[0 for i in range(d)] for j in range(n)]
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz','metal','pop','reggae', 'rock']

for i in range(10):
    for j in range(100):
        print(j)
        if j < 10:
            az, sr = librosa.load(genres[i] + '.' + '0000' + str(j) + '.au')
        else:
            az, sr = librosa.load(genres[i] + '.' + '000' + str(j) + '.au')
        lensong = len(az)
        y1 = az[int(lensong * 0.1): int(lensong * 0.1) + int(3.5 * sr)]
        y2 = az[int(lensong * 0.2): int(lensong * 0.2) + int(3.5 * sr)]
        y3 = az[int(lensong * 0.3): int(lensong * 0.3) + int(3.5 * sr)]
        y4 = az[int(lensong * 0.4): int(lensong * 0.4) + int(3.5 * sr)]
        y5 = az[int(lensong * 0.5): int(lensong * 0.5) + int(3.5 * sr)]
        y6 = az[int(lensong * 0.6): int(lensong * 0.6) + int(3.5 * sr)]
        y7 = az[int(lensong * 0.7): int(lensong * 0.7) + int(3.5 * sr)]
        y8 = az[int(lensong * 0.8): int(lensong * 0.8) + int(3.5 * sr)]
        y = [y1, y2, y3, y4, y5, y6, y7, y8]
        winsize = int(0.02 * sr)

        for k in range(8):
            rank = (100 * i + j) * 8 + k
            tempo, beat_frames = librosa.beat.beat_track(y=y[k], sr=sr)
            data[rank][0] = tempo
            MFCCs = librosa.feature.mfcc(y=y[k], sr=sr, n_fft=winsize, n_mfcc=20)
            (a, b) = mffcdata(MFCCs, 20, len(MFCCs[0]))
            data[rank][1] = a
            data[rank][2] = b
            SC = librosa.feature.spectral_centroid(y=y[k], sr=sr, n_fft=winsize, freq=None)
            (a, b) = mffcdata(SC, 1, len(SC[0]))
            data[rank][3] = a
            data[rank][4] = b
            SB = librosa.feature.spectral_bandwidth(y=y[k], sr=sr, n_fft=winsize, freq=None, norm=True, p=2)
            (a, b) = mffcdata(SB, 1, len(SB[0]))
            data[rank][5] = a
            data[rank][6] = b
            SRO = librosa.feature.spectral_rolloff(y=y[k], sr=sr, n_fft=winsize, freq=None, roll_percent=0.85)
            (a, b) = mffcdata(SRO, 1, len(SRO[0]))
            data[rank][7] = a
            data[rank][8] = b
            SCO = librosa.feature.spectral_contrast(y=y[k], sr=sr, n_fft=winsize, freq=None, fmin=200.0,
                                                    n_bands=6, quantile=0.02, linear=False)
            (a, b) = mffcdata(SCO, 1, len(SCO[0]))
            data[rank][9] = a
            data[rank][10] = b
            SF = librosa.feature.spectral_flatness(y=y[k], n_fft=winsize, amin=1e-10, power=2.0)
            (a, b) = mffcdata(SF, 1, len(SF[0]))
            data[rank][11] = a
            data[rank][12] = b
            ZCR = librosa.feature.zero_crossing_rate(y=y[k], frame_length=winsize, center=True)
            (a, b) = mffcdata(ZCR, 1, len(ZCR[0]))
            data[rank][13] = a
            data[rank][14] = b

print('data',data[0])
df = pd.DataFrame(data, columns=['Tempo', 'MFCC average', 'MFCC variance',
                                 'Spectral centroid average', 'Spectral centroid variance',
                                 'Spectral bandwidth average', 'Spectral bandwidth variance',
                                 'Spectral roll-off average', 'Spectral roll-off variance', 'Spectral contrast average',
                                 'Spectral contrast variance', 'Spectral flatness average',
                                 'Spectral flatness variance', 'Zero crossing rate average',
                                 'Zero crossing rate variance'])
dfn = preprocessing.scale(df)
print(dfn)
with open('AFeaturesV1.csv', 'w') as myfile:
    myfile.write('Tempo, MFCC average, MFCC variance, Spectral centroid average, Spectral centroid variance, ' \
       'Spectral bandwidth average, Spectral bandwidth variance, Spectral roll-off average, Spectral roll-off variance, '
                 'Spectral contrast average, Spectral contrast variance, Spectral flatness average,'
                 'Spectral flatness variance, Zero crossing rate average, Zero crossing rate variance \n')
    for i in range(n):
        line = str()
        for j in range(d):
            line += str(dfn[i][j]) + ', '
        line += '\n'
        myfile.write(line)