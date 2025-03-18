"""DDSP Core Functions Tutorial

This script introduces the core DDSP functions used to build synthesizers and processors.
We'll cover:
1. How to generate audio with oscillators
2. How to filter audio with frequency domain transfer functions
3. How to delay audio with variable length delays
4. How to use these elements to build more complex processors
"""

import numpy as np
import matplotlib.pyplot as plt
import ddsp
import ddsp.core
import tensorflow as tf
import os

# Create outputs directory if it doesn't exist
os.makedirs('tutorials/outputs', exist_ok=True)

# Helper function to play audio in the notebook
def play_audio(audio, sample_rate=16000, demo_name="output"):
    """Creates an audio player widget for the audio."""
    print("Audio sample rate:", sample_rate)
    
    # Convert TensorFlow tensor to NumPy if needed
    if isinstance(audio, tf.Tensor):
        audio = audio.numpy()
    
    print("Audio shape:", audio.shape)
    print("Audio min/max:", audio.min(), audio.max())
    
    # Save audio to a file in the outputs directory
    audio_path = os.path.join('tutorials/outputs', f"{demo_name}.wav")
    audio_array = np.array(audio)
    import soundfile as sf
    sf.write(audio_path, audio_array, sample_rate)
    print(f"Audio saved to {audio_path}")

# 1. Oscillator
def oscillator_demo():
    """Demonstrate the basic oscillator functionality."""
    print("\n=== Oscillator Demo ===")
    
    # Create a time-varying frequency signal (Hz)
    f0_hz = tf.linspace(0.0, 1000.0, 1000)  # Linear frequency sweep
    
    # Generate audio
    audio = ddsp.core.oscillator_bank(
        frequency_envelopes=f0_hz[tf.newaxis, :, tf.newaxis],
        amplitude_envelopes=tf.ones_like(f0_hz)[tf.newaxis, :, tf.newaxis],
        sample_rate=16000
    )
    audio = audio[0]
    
    play_audio(audio, demo_name="oscillator_output")
    print("Generated a linear frequency sweep from 0 to 1000 Hz")

# 2. Filter
def filter_demo():
    """Demonstrate audio filtering."""
    print("\n=== Filter Demo ===")
    
    # Generate a richer input signal with multiple harmonics
    duration = 4.0
    sample_rate = 16000
    n_samples = int(duration * sample_rate)
    time = np.linspace(0, duration, n_samples)
    
    # Create rich harmonic content
    f0_hz = 220.0  # A3 note
    audio = np.zeros_like(time)
    for harmonic in range(1, 16):  # More harmonics
        audio += np.sin(2 * np.pi * f0_hz * harmonic * time) / harmonic
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # Add batch dimension and convert to tensor
    audio = tf.convert_to_tensor(audio[np.newaxis, :], dtype=tf.float32)
    
    # Create a more dramatic time-varying lowpass filter
    n_frames = 100  # More frames for smoother movement
    n_freqs = 2048
    
    # Create a more dramatic filter sweep
    # Start with high frequencies, sweep down, then back up
    cutoff_trajectory = np.concatenate([
        np.linspace(1.0, 0.05, n_frames // 2),  # Sweep down
        np.linspace(0.05, 1.0, n_frames // 2),  # Sweep up
    ])
    
    # Create filter magnitudes with smooth rolloff
    magnitudes = np.zeros((1, n_frames, n_freqs))
    freqs = np.linspace(0, 1, n_freqs)
    
    for i in range(n_frames):
        cutoff = cutoff_trajectory[i]
        # Create smooth rolloff using sigmoid function
        magnitudes[0, i, :] = 1 / (1 + np.exp((freqs - cutoff) * 30))
    
    magnitudes = tf.convert_to_tensor(magnitudes, dtype=tf.float32)
    
    filtered_audio = ddsp.core.frequency_filter(
        audio,
        magnitudes=magnitudes,
        window_size=2048
    )
    
    filtered_audio = filtered_audio[0]
    
    # Ensure the output is normalized
    filtered_audio = filtered_audio / tf.reduce_max(tf.abs(filtered_audio)) * 0.8
    
    play_audio(filtered_audio, demo_name="filter_output")
    print("Applied a dramatic sweeping low-pass filter to a rich harmonic sound")

# 3. Variable Length Delay
def delay_demo():
    """Demonstrate variable length delay."""
    print("\n=== Delay Demo ===")
    
    # Create input pulses
    sample_rate = 16000
    duration = 4.0
    n_samples = int(duration * sample_rate)
    audio = np.zeros([1, n_samples])
    
    # Create 4 pulses of short sine bursts
    for i in range(4):
        start = i * 12000  # Space them out evenly
        t = np.linspace(0, 0.1, 1000)  # 100ms burst
        burst = np.sin(2 * np.pi * 440 * t) * 0.5
        burst *= np.exp(-t * 50)  # Exponential decay
        audio[0, start:start+1000] = burst
    
    # Create multiple delay layers with feedback
    delayed_audio = tf.convert_to_tensor(audio, dtype=tf.float32)
    output = delayed_audio
    
    # Parameters for delay
    n_delay = 8000  # 0.5 second delay
    feedback = 0.7  # Feedback amount
    
    # Apply multiple delay layers with feedback
    for i in range(5):  # 5 echo repetitions
        # Create modulated delay time
        phase = 0.5 + 0.3 * np.sin(2 * np.pi * 0.25 * np.linspace(0.0, 1.0, n_samples))
        phase = phase[np.newaxis, :]
        
        # Apply delay
        delayed = ddsp.core.variable_length_delay(
            audio=delayed_audio,
            phase=phase,
            max_length=n_delay
        )
        
        # Add to output with feedback
        output += delayed * (feedback ** (i + 1))
        delayed_audio = delayed
    
    # Normalize output
    output = output / tf.reduce_max(tf.abs(output)) * 0.8
    output = output[0]  # Remove batch dimension
    
    play_audio(output, demo_name="delay_output")
    print("Applied multiple delay layers with feedback to create echo effect")

def main():
    """Run all demos."""
    print("DDSP Core Functions Tutorial")
    print("===========================")
    
    oscillator_demo()
    filter_demo()
    delay_demo()

if __name__ == "__main__":
    main() 