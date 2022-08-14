import pyaudio
import numpy as np

#http://people.csail.mit.edu/hubert/pyaudio/
#special thanks to http://zulko.github.io/blog/2014/03/29/soundstretching-and-pitch-shifting-in-python/

def mimic(duration=300):

    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100

    #work with one huge chunk
    # CHUNK = 204800
    CHUNK = 65536
    RECORD_SECONDS = duration
    WAVE_OUTPUT_FILENAME = "file.wav"

    audio = pyaudio.PyAudio()
    
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,output=True,
                    frames_per_buffer=CHUNK)
    print ("* recording")

    def stretch(snd_array, factor, window_size, h):
        """ Stretches/shortens a sound, by some factor. """
        phase = np.zeros(window_size)
        hanning_window = np.hanning(window_size)
        result = np.zeros(int(len(snd_array) / factor + window_size))
        for i in np.arange(0, len(snd_array) - (window_size + h), h*factor):
            i = int(i)
            # Two potentially overlapping subarrays
            a1 = snd_array[i: i + window_size]
            a2 = snd_array[i + h: i + window_size + h]

            # The spectra of these arrays
            s1 = np.fft.fft(hanning_window * a1)
            s2 = np.fft.fft(hanning_window * a2)

            # Rephase all frequencies
            phase = (phase + np.angle(s2/s1)) % 2*np.pi

            a2_rephased = np.fft.ifft(np.abs(s2)*np.exp(1j*phase))
            i2 = int(i/factor)
            result[i2: i2 + window_size] += hanning_window*a2_rephased.real
        return result.astype('int16')

    def speedx(sound_array, factor):
        """ Multiplies the sound's speed by some `factor` """
        indices = np.round( np.arange(0, len(sound_array), factor) )
        indices = indices[indices < len(sound_array)].astype(int)
        return sound_array[ indices.astype(int) ]

    def pitchshift(snd_array, n, window_size=2**13, h=2**11):
        """ Changes the pitch of a sound by ``n`` semitones. """
        factor = 2**(1.0 * n / 12.0)
        stretched = stretch(snd_array, 1.0/factor, window_size, h)
        return speedx(stretched[window_size:], factor)

    def playAudio(audio, samplingRate, channels):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=samplingRate,
                        output=True)
        sound = (audio.astype(np.int16).tostring())
        stream.write(sound)

        stream.stop_stream()
        stream.close()
        p.terminate()
        return

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        # data = stream.read(CHUNK)
        # stream.write(data, CHUNK)
        data = stream.read(CHUNK)
        data = np.fromstring(data, dtype=np.int16)
        #make two times louder
        data *= 4
        pitched = pitchshift(data, 7)
        sound = (pitched.astype(np.int16).tostring())
        stream.write(sound)

    print ("* done recording")

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # #Tests
    # playAudio(data, RATE, CHANNELS)

    # pitched = pitchshift(data, -5)
    # playAudio(pitched, RATE, CHANNELS)

    # pitched = pitchshift(data, 5)
    # playAudio(pitched, RATE, CHANNELS)

# mimic(300)