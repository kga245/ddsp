"""Utility functions for DDSP Core Functions tutorials.

Contains:
- Audio playback and saving
- Visualization helpers
- Common configurations
"""

import numpy as np
import matplotlib.pyplot as plt
import ddsp
import ddsp.core
import tensorflow as tf
import os

# Audio configuration
SAMPLE_RATE = 16000  # Can be modified based on needs

# Create outputs directory if it doesn't exist
os.makedirs('tutorials/01_core_functions/outputs', exist_ok=True)

def create_transfer_function(freqs, mags):
    """Helper to create transfer functions for filtering.
    
    Args:
        freqs: Frequency points
        mags: Magnitude response at each frequency
    
    Returns:
        Complex transfer function for FFT filtering
    """
    # Convert to tensors
    freqs = tf.convert_to_tensor(freqs, dtype=tf.float32)
    mags = tf.convert_to_tensor(mags, dtype=tf.float32)
    
    # Create complex transfer function
    return tf.complex(mags, tf.zeros_like(mags))

def play_audio(audio, sample_rate=SAMPLE_RATE, demo_name="output"):
    """Creates an audio file from the audio data.
    
    Args:
        audio: Audio data to save
        sample_rate: Sample rate of the audio
        demo_name: Name of the output file
    """
    print("Audio sample rate:", sample_rate)
    
    # Convert TensorFlow tensor to NumPy if needed
    if isinstance(audio, tf.Tensor):
        audio = audio.numpy()
    
    print("Audio shape:", audio.shape)
    print("Audio min/max:", audio.min(), audio.max())
    
    # Save audio to a file in the outputs directory
    audio_path = os.path.join('tutorials/01_core_functions/outputs', f"{demo_name}.wav")
    audio_array = np.array(audio)
    import soundfile as sf
    sf.write(audio_path, audio_array, sample_rate)
    print(f"Audio saved to {audio_path}")

def specplot(audio, sample_rate=SAMPLE_RATE, demo_name="spectrogram"):
    """Plot and save the spectrogram of the audio.
    
    Args:
        audio: Audio signal to plot
        sample_rate: Sample rate of the audio
        demo_name: Name for the saved spectrogram file (without extension)
    """
    if isinstance(audio, tf.Tensor):
        audio = audio.numpy()
    
    # Convert to mono if needed
    if len(audio.shape) > 1:
        if audio.shape[0] == 1:
            audio = audio[0]
        elif audio.shape[-1] == 1:
            audio = audio[..., 0]
    
    # Calculate spectrogram
    n_fft = 2048
    hop_length = 512
    
    # Compute spectrogram using scipy
    from scipy import signal
    frequencies, times, spectrogram = signal.spectrogram(
        audio,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        scaling='density'
    )
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(times, frequencies, 20 * np.log10(spectrogram + 1e-10))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Magnitude [dB]')
    plt.title(f'Spectrogram - {demo_name}')
    
    # Save plot with matching filename
    plot_path = os.path.join('tutorials/01_core_functions/outputs', f"{demo_name}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Spectrogram saved to {plot_path}")

def transfer_function(audio, sample_rate=SAMPLE_RATE):
    """Calculate the transfer function of the audio.
    
    Args:
        audio: Audio signal to analyze
        sample_rate: Sample rate of the audio
    
    Returns:
        frequencies: Array of frequency points
        magnitudes: Magnitude response at each frequency
    """
    if isinstance(audio, tf.Tensor):
        audio = audio.numpy()
    
    # Ensure audio is 3D [batch, time, channels]
    if len(audio.shape) == 1:
        audio = audio[np.newaxis, :, np.newaxis]
    elif len(audio.shape) == 2:
        audio = audio[np.newaxis, :, :]
    
    # Calculate FFT
    n_fft = 2048
    fft = np.fft.rfft(audio, n=n_fft, axis=1)
    magnitudes = np.abs(fft)
    
    # Calculate frequency points
    frequencies = np.linspace(0, sample_rate/2, n_fft//2 + 1)
    
    return frequencies, magnitudes

def plot_impulse_responses(impulse_response, desired_response=None, sample_rate=SAMPLE_RATE):
    """Plot impulse responses and their frequency responses.
    
    Args:
        impulse_response: The impulse response to plot
        desired_response: Optional target response to compare against
        sample_rate: Sample rate of the audio
    """
    if isinstance(impulse_response, tf.Tensor):
        impulse_response = impulse_response.numpy()
    
    # Calculate frequency response
    frequencies, magnitudes = transfer_function(impulse_response, sample_rate=sample_rate)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot time domain
    plt.subplot(121)
    plt.plot(impulse_response)
    plt.title('Impulse Response (Time Domain)')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    # Plot frequency domain
    plt.subplot(122)
    plt.plot(frequencies, magnitudes[0, :, 0], label='Actual')
    if desired_response is not None:
        plt.plot(frequencies, desired_response, '--', label='Desired')
    plt.title('Frequency Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def normalize_audio(audio, target_peak=0.8):
    """Normalize audio to a target peak amplitude.
    
    Args:
        audio: Audio signal to normalize
        target_peak: Target peak amplitude (default 0.8)
    
    Returns:
        Normalized audio signal
    """
    if isinstance(audio, tf.Tensor):
        peak = tf.reduce_max(tf.abs(audio))
        return (audio / peak) * target_peak
    else:
        peak = np.max(np.abs(audio))
        return (audio / peak) * target_peak

def smooth(signal, window_size=100):
    """Smooth a signal using a moving average.
    
    Args:
        signal: Signal to smooth
        window_size: Size of the smoothing window
    
    Returns:
        Smoothed signal
    """
    window = np.ones(window_size) / window_size
    if isinstance(signal, tf.Tensor):
        signal = signal.numpy()
    return np.convolve(signal, window, mode='same') 