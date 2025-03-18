"""DDSP Filter Examples

Demonstrates:
1. Basic filtering
2. Time-varying filters
3. Different filter types
4. White noise filtering
"""

import numpy as np
import tensorflow as tf
import ddsp
import ddsp.core
import ddsp.synths
import matplotlib.pyplot as plt
from utils import (
    play_audio, specplot, normalize_audio, SAMPLE_RATE,
    create_transfer_function
)
import os
from scipy import signal

# Create all output directories
os.makedirs('tutorials/01_core_functions/outputs/filters/noise', exist_ok=True)
os.makedirs('tutorials/01_core_functions/outputs/filters/musical', exist_ok=True)

def basic_filter_demo():
    """Demonstrate basic filter functionality."""
    print("\n=== Basic Filter Demo ===")
    
    duration = 4.0
    n_samples = int(duration * SAMPLE_RATE)
    
    # Create lowpass filter magnitudes
    freqs = tf.linspace(0.0, SAMPLE_RATE/2, 1025)
    cutoff = 1000.0
    magnitudes = tf.cast(freqs <= cutoff, tf.float32)[tf.newaxis, tf.newaxis, :]
    
    # Create filtered noise
    filtered = ddsp.synths.FilteredNoise(
        n_samples=n_samples,
        window_size=1024,
        scale_fn=None,
        initial_bias=-5.0,
    )
    audio = filtered(magnitudes)
    
    # Save outputs
    audio = normalize_audio(audio[0])
    play_audio(audio, demo_name="filters/noise/basic_filter")
    specplot(audio, demo_name="filters/noise/basic_filter")
    
    # Plot filter response
    plt.figure(figsize=(10, 4))
    plt.semilogx(freqs, 20 * tf.math.log(magnitudes[0, 0] + 1e-6) / tf.math.log(10.0))
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Basic Lowpass Filter Response')
    plt.savefig("tutorials/01_core_functions/outputs/filters/noise/basic_filter_response.png")
    plt.close()

def timevarying_filter_demo():
    """Demonstrate time-varying filter."""
    print("\n=== Time-Varying Filter Demo ===")
    
    duration = 4.0
    n_samples = int(duration * SAMPLE_RATE)
    
    # Create sweeping filter magnitudes
    freqs = tf.linspace(0.0, SAMPLE_RATE/2, 1025)
    cutoff = tf.linspace(100.0, 8000.0, n_samples)
    magnitudes = tf.cast(freqs[tf.newaxis, :] <= cutoff[:, tf.newaxis], tf.float32)
    
    # Create filtered noise
    filtered = ddsp.synths.FilteredNoise(
        n_samples=n_samples,
        window_size=1024,
        scale_fn=None,
        initial_bias=-5.0,
    )
    audio = filtered(magnitudes[tf.newaxis, :, :])
    
    # Save outputs
    audio = normalize_audio(audio[0])
    play_audio(audio, demo_name="filters/noise/timevarying_filter")
    specplot(audio, demo_name="filters/noise/timevarying_filter")

def filter_types_demo():
    """Demonstrate different filter types."""
    print("\n=== Filter Types Demo ===")
    
    duration = 4.0
    n_samples = int(duration * SAMPLE_RATE)
    
    # Create bandpass filter magnitudes
    freqs = tf.linspace(0.0, SAMPLE_RATE/2, 1025)
    center_freq = 1000.0
    bandwidth = 500.0
    magnitudes = tf.exp(-((freqs - center_freq) ** 2) / (2 * bandwidth ** 2))
    magnitudes = magnitudes[tf.newaxis, tf.newaxis, :]
    
    # Create filtered noise
    filtered = ddsp.synths.FilteredNoise(
        n_samples=n_samples,
        window_size=1024,
        scale_fn=None,
        initial_bias=-5.0,
    )
    audio = filtered(magnitudes)
    
    # Save outputs
    audio = normalize_audio(audio[0])
    play_audio(audio, demo_name="filters/noise/bandpass_filter")
    specplot(audio, demo_name="filters/noise/bandpass_filter")
    
    # Plot filter response
    plt.figure(figsize=(10, 4))
    plt.semilogx(freqs, 20 * tf.math.log(magnitudes[0, 0] + 1e-6) / tf.math.log(10.0))
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Bandpass Filter Response')
    plt.savefig("tutorials/01_core_functions/outputs/filters/noise/bandpass_filter_response.png")
    plt.close()

def noise_filter_demo():
    """Demonstrate noise filtering."""
    print("\n=== Noise Filter Demo ===")
    
    duration = 4.0
    n_samples = int(duration * SAMPLE_RATE)
    
    # Create bandpass filter around 440 Hz
    freqs = tf.linspace(0.0, SAMPLE_RATE/2, 1025)
    center_freq = 440.0
    bandwidth = 50.0
    magnitudes = tf.exp(-((freqs - center_freq) ** 2) / (2 * bandwidth ** 2))
    magnitudes = magnitudes[tf.newaxis, tf.newaxis, :]
    
    # Create filtered and unfiltered noise
    filtered = ddsp.synths.FilteredNoise(
        n_samples=n_samples,
        window_size=1024,
        scale_fn=None,
        initial_bias=-5.0,
    )
    
    # Generate filtered noise
    audio_filtered = filtered(magnitudes)
    audio_filtered = normalize_audio(audio_filtered[0])
    
    # Generate unfiltered noise
    audio_noisy = filtered(tf.ones_like(magnitudes))
    audio_noisy = normalize_audio(audio_noisy[0])
    
    # Save outputs
    play_audio(audio_noisy, demo_name="filters/noise/noisy_signal")
    play_audio(audio_filtered, demo_name="filters/noise/filtered_signal")
    specplot(audio_noisy, demo_name="filters/noise/noisy_signal")
    specplot(audio_filtered, demo_name="filters/noise/filtered_signal")
    
    # Plot filter response
    plt.figure(figsize=(10, 4))
    plt.semilogx(freqs, 20 * tf.math.log(magnitudes[0, 0] + 1e-6) / tf.math.log(10.0))
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Noise Filter Response (440 Hz Bandpass)')
    plt.savefig("tutorials/01_core_functions/outputs/filters/noise/noise_filter_response.png")
    plt.close()

def musical_filter_demo():
    """Demonstrate a lowpass filter sweep on a sustained chord."""
    print("\n=== Musical Filter Demo ===")
    
    duration = 4.0
    n_samples = int(duration * SAMPLE_RATE)
    n_frequencies = 1000  # Number of frequency bands for the filter
    n_frames = 250  # Number of time frames for the filter to change
    
    # Generate the chord first
    frequencies = np.array([261.63, 329.63, 392.00])  # C4, E4, G4
    frequency_envelopes = tf.broadcast_to(
        frequencies[np.newaxis, np.newaxis, :],
        [1, n_samples, 3])
    amplitude_envelopes = tf.ones([1, n_samples, 3], dtype=tf.float32) * 0.3
    
    audio = ddsp.core.oscillator_bank(
        frequency_envelopes=frequency_envelopes,
        amplitude_envelopes=amplitude_envelopes,
        sample_rate=SAMPLE_RATE
    )
    
    # Create time-varying filter magnitudes
    # This creates a sweeping bandpass filter that moves through the frequency spectrum
    magnitudes = []
    for w in np.linspace(4.0, 40.0, n_frames):
        response = tf.sin(tf.linspace(0.0, w, n_frequencies))**4.0
        magnitudes.append(response)
    magnitudes = tf.stack(magnitudes)[tf.newaxis, :, :]  # [batch, n_frames, n_frequencies]
    
    # Apply the filter using frequency_filter
    filtered = ddsp.core.frequency_filter(
        audio,
        magnitudes,
        window_size=256  # Smaller window size for better temporal resolution
    )
    
    # Save outputs
    play_audio(audio[0], demo_name="filters/musical/chord_original")
    play_audio(filtered[0], demo_name="filters/musical/chord_filtered")
    specplot(audio[0], demo_name="filters/musical/chord_original")
    specplot(filtered[0], demo_name="filters/musical/chord_filtered")
    
    # Plot filter response over time
    plt.figure(figsize=(10, 4))
    plt.matshow(np.rot90(magnitudes[0]), aspect='auto')
    plt.title('Filter Frequency Response Over Time')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.savefig("tutorials/01_core_functions/outputs/filters/musical/filter_response.png")
    plt.close()

def speech_filter_demo():
    """Demonstrate filter effects on speech-like frequencies."""
    print("\n=== Speech Filter Demo ===")
    
    duration = 4.0
    n_samples = int(duration * SAMPLE_RATE)
    
    # Create a vowel-like sound using harmonics
    f0 = 150  # fundamental frequency
    n_harmonics = 10
    harmonic_freqs = [f0 * (i + 1) for i in range(n_harmonics)]
    
    # Generate harmonics
    frequency_envelopes = tf.broadcast_to(
        tf.constant(harmonic_freqs)[tf.newaxis, tf.newaxis, :],
        [1, n_samples, n_harmonics])
    amplitude_envelopes = tf.ones([1, n_samples, n_harmonics], dtype=tf.float32) * 0.1
    
    audio = ddsp.core.oscillator_bank(
        frequency_envelopes=frequency_envelopes,
        amplitude_envelopes=amplitude_envelopes,
        sample_rate=SAMPLE_RATE
    )[0]
    
    # Process in chunks
    chunk_size = 2048
    hop_size = 1024
    n_chunks = (n_samples - chunk_size) // hop_size
    
    filtered = tf.zeros_like(audio)
    window = tf.signal.hann_window(chunk_size)
    
    for i in range(n_chunks):
        start = i * hop_size
        end = start + chunk_size
        chunk = audio[start:end] * window
        
        # Get current formant frequencies
        t = i / n_chunks
        formant1 = 730 * (1-t) + 270 * t    # First formant
        formant2 = 1090 * (1-t) + 2290 * t  # Second formant
        
        # FFT
        chunk_fft = tf.signal.rfft(chunk)
        freqs = tf.linspace(0.0, SAMPLE_RATE/2, chunk_size//2 + 1)
        
        # Create formant filter
        magnitudes = (tf.exp(-((freqs - formant1)**2)/(2*50**2)) + 
                     tf.exp(-((freqs - formant2)**2)/(2*50**2)))
        magnitudes = tf.cast(magnitudes, tf.complex64)
        
        chunk_filtered = tf.signal.irfft(chunk_fft * magnitudes) * window
        
        # Accumulate
        filtered = tf.tensor_scatter_nd_add(
            filtered,
            tf.reshape(tf.range(start, end), [-1, 1]),
            chunk_filtered
        )
    
    filtered = filtered / tf.reduce_max(tf.abs(filtered))
    
    # Save outputs
    play_audio(audio, demo_name="filters/musical/vowel_original")
    play_audio(filtered, demo_name="filters/musical/vowel_filtered")
    specplot(audio, demo_name="filters/musical/vowel_original")
    specplot(filtered, demo_name="filters/musical/vowel_filtered")

def noise_melody_filter_demo():
    """Demonstrate a resonant filter that creates a melody from noise."""
    print("\n=== Noise Melody Filter Demo ===")
    # [Previous noise filter code goes here, but saving to noise folder]
    # ... outputs go to 'filters/noise/noise_melody_*'

def melody_filter_demo():
    """Demonstrate different frequency band filtering on a melody."""
    print("\n=== Melody Filter Demo ===")
    
    duration = 5.0  # 5 bars of 1 second each
    n_samples = int(duration * SAMPLE_RATE)
    n_frequencies = 1000
    n_frames = 500  # 100 frames per second
    
    # Create a simple melody with harmonics
    melody_hz = 440 * np.array([1.0, 1.2, 1.5, 1.0, 1.2])  # Simple melody that repeats each bar
    frequencies = melody_hz[:, np.newaxis, np.newaxis]  # [notes, 1, 1]
    frequencies = frequencies[np.newaxis, :, :]  # [1, notes, 1]
    
    # Resample to smooth transitions
    frequency_envelope = ddsp.core.resample(frequencies, n_samples, method='cubic')
    
    # Generate rich harmonic content
    audio_signal = 0.0
    for harmonic in [1.0, 2.0, 3.0, 4.0, 5.0]:
        harmonic_freq = frequency_envelope * harmonic
        harmonic_amp = tf.ones_like(frequency_envelope) * (0.3 / harmonic)
        audio_signal += ddsp.core.oscillator_bank(
            frequency_envelopes=harmonic_freq,
            amplitude_envelopes=harmonic_amp,
            sample_rate=SAMPLE_RATE
        )
    
    # Reshape audio to [batch, samples]
    audio = tf.reshape(audio_signal, [1, -1])
    
    # Create 5 different filter shapes (one for each bar)
    magnitudes = []
    samples_per_bar = n_frames // 5
    
    for frame in range(n_frames):
        freqs = tf.linspace(0.0, SAMPLE_RATE/2, n_frequencies)
        bar_index = frame // samples_per_bar
        
        if bar_index == 0:
            # First bar: No filtering (flat response)
            response = tf.ones_like(freqs)
        elif bar_index == 1:
            # Second bar: Low-pass (cut high frequencies)
            cutoff = 1500
            response = tf.exp(-(freqs - 0)**2 / (2 * 500**2))
        elif bar_index == 2:
            # Third bar: High-pass (cut low frequencies)
            cutoff = 1000
            response = 1.0 - tf.exp(-(freqs - 0)**2 / (2 * 500**2))
        elif bar_index == 3:
            # Fourth bar: Band-pass (keep middle frequencies)
            center = 1500
            response = tf.exp(-(freqs - center)**2 / (2 * 300**2))
        else:
            # Fifth bar: Band-stop (cut middle frequencies)
            center = 1500
            response = 1.0 - tf.exp(-(freqs - center)**2 / (2 * 300**2))
            
        # Normalize response
        response = response / tf.reduce_max(response)
        magnitudes.append(response)
    
    magnitudes = tf.stack(magnitudes)[tf.newaxis, :, :]
    
    # Apply the filter
    filtered = ddsp.core.frequency_filter(
        audio,
        magnitudes,
        window_size=256
    )
    
    # Normalize outputs
    audio = audio[0] / tf.reduce_max(tf.abs(audio))
    filtered = filtered[0] / tf.reduce_max(tf.abs(filtered))
    
    # Save outputs
    play_audio(audio, demo_name="filters/musical/melody_original")
    play_audio(filtered, demo_name="filters/musical/melody_filtered")
    specplot(audio, demo_name="filters/musical/melody_original")
    specplot(filtered, demo_name="filters/musical/melody_filtered")
    
    # Plot filter response over time
    plt.figure(figsize=(10, 4))
    plt.matshow(np.rot90(magnitudes[0]), aspect='auto')
    plt.title('Filter Frequency Response Over Time')
    plt.xlabel('Time (5 bars)')
    plt.ylabel('Frequency')
    plt.savefig("tutorials/01_core_functions/outputs/filters/musical/melody_filter_response.png")
    plt.close()

if __name__ == "__main__":
    # Run original demos
    print("DDSP Filter Examples")
    print("===================")
    basic_filter_demo()
    timevarying_filter_demo()
    filter_types_demo()
    noise_filter_demo()
    
    # Run new musical demos
    print("\nDDSP Musical Filter Examples")
    print("===========================")
    musical_filter_demo()
    speech_filter_demo()
    melody_filter_demo() 