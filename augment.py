import os
import librosa
import numpy as np
import soundfile as sf

def audio_to_spectrogram(audio_path, n_fft=2048, hop_length=512):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Convert audio to spectrogram with specified hop_length
    spectrogram = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    
    return spectrogram, sr, y

def spectrogram_to_audio(spectrogram, sr, y):
    # Convert spectrogram to waveform audio
    audio_reconstructed = librosa.istft(spectrogram, length=len(y), dtype=np.float32)
    
    return audio_reconstructed

import numpy as np

import numpy as np

def apply_pitch_shift(spectrogram, shift_range=5, num_shifts=1):
    # Clone the spectrogram
    shifted_spectrogram = np.copy(spectrogram)
    
    # Get the number of frequency bins in the spectrogram
    num_freq_bins = shifted_spectrogram.shape[0]
    
    # Apply pitch shifts
    for i in range(num_shifts):
        shift = np.random.randint(-shift_range, shift_range)
        shifted_spectrogram = np.roll(shifted_spectrogram, shift, axis=0)
        
        # Handle the part that wraps around by setting it to 0
        if shift > 0:
            shifted_spectrogram[:shift, :] = 0
        elif shift < 0:
            shifted_spectrogram[shift:, :] = 0
        
    return shifted_spectrogram


# Directory containing audio files
audio_dir = 'data/dcase2023t2/dev_data/raw/gearbox/train'



# Loop through each audio file in the input directory
for filename in os.listdir(audio_dir):
    if filename.endswith('.wav'):
        audio_path = os.path.join(audio_dir, filename)
        
        # Convert audio to spectrogram
        spectrogram, sr, y = audio_to_spectrogram(audio_path)
        
        # Apply time masking to the spectrogram
        masked_spectrogram = apply_pitch_shift(spectrogram)
        
        # Convert masked spectrogram back to audio
        reconstructed_audio = spectrogram_to_audio(masked_spectrogram, sr, y)
        
        # Save the reconstructed audio to a file
        output_audio_path = os.path.join(audio_dir, 'pitched_' + filename )
        sf.write(output_audio_path, reconstructed_audio, sr)

print("Time masking applied to all audio files in the directory.")