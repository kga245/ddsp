"""DDSP Delay Examples

Demonstrates:
1. Basic delay
2. Flanger effect
3. Chorus effect
4. Vibrato effect
"""

import numpy as np
import tensorflow as tf
import ddsp
import ddsp.core
import matplotlib.pyplot as plt
from utils import play_audio, specplot, normalize_audio, SAMPLE_RATE
import os

# Create outputs directory for delays
os.makedirs('tutorials/01_core_functions/outputs/delays', exist_ok=True)

def basic_delay_demo():
    """Demonstrate basic delay functionality."""
    print("\n=== Basic Delay Demo ===")
    
    duration = 4.0
    n_samples = int(duration * SAMPLE_RATE)
    
    # Create input signal
    time = np.linspace(0, duration, n_samples)
    input_signal = np.sin(2 * np.pi * 440 * time)
    
    # Create delay (250ms with feedback)
    delay_samples = int(0.25 * SAMPLE_RATE)
    feedback = 0.4
    n_echoes = 5
    
    delay_buffer = np.zeros(n_samples + delay_samples)
    delay_buffer[:n_samples] = input_signal
    
    for i in range(n_echoes):
        delay_pos = i * delay_samples
        delay_buffer[delay_pos:delay_pos + n_samples] += \
            (feedback ** i) * input_signal
    
    audio = delay_buffer[:n_samples]
    
    # Save outputs
    audio = normalize_audio(audio)
    play_audio(audio, demo_name="delays/basic_delay")
    specplot(audio, demo_name="delays/basic_delay")

def flanger_demo():
    """Demonstrate flanger effect."""
    print("\n=== Flanger Demo ===")
    
    duration = 4.0
    n_samples = int(duration * SAMPLE_RATE)
    
    # Create input signal
    time = np.linspace(0, duration, n_samples)
    input_signal = np.sin(2 * np.pi * 440 * time)
    
    # Create modulated delay
    max_delay_samples = int(0.003 * SAMPLE_RATE)  # 3ms
    mod_freq = 0.5  # 0.5 Hz
    depth = 0.7
    
    delay_time = max_delay_samples * (0.5 + 0.5 * depth * 
                                    np.sin(2 * np.pi * mod_freq * time))
    
    # Apply delay
    audio = ddsp.core.variable_length_delay(
        input_signal[np.newaxis, :],
        delay_time[np.newaxis, :],
        max_length=max_delay_samples
    )[0]
    
    # Save outputs
    audio = normalize_audio(audio)
    play_audio(audio, demo_name="delays/flanger")
    specplot(audio, demo_name="delays/flanger")
    
    # Plot modulation
    plt.figure(figsize=(10, 4))
    plt.plot(time, delay_time / SAMPLE_RATE * 1000)
    plt.xlabel('Time (s)')
    plt.ylabel('Delay Time (ms)')
    plt.title('Flanger Modulation')
    plt.grid(True)
    plt.savefig("tutorials/01_core_functions/outputs/delays/flanger_modulation.png")
    plt.close()

def chorus_demo():
    """Demonstrate chorus effect."""
    print("\n=== Chorus Demo ===")
    
    duration = 4.0
    n_samples = int(duration * SAMPLE_RATE)
    
    # Create input signal
    time = np.linspace(0, duration, n_samples)
    input_signal = np.sin(2 * np.pi * 440 * time)
    
    # Create modulated delay
    max_delay_samples = int(0.025 * SAMPLE_RATE)  # 25ms
    mod_freq = 0.25  # 0.25 Hz
    depth = 0.3
    
    delay_time = max_delay_samples * (0.5 + 0.5 * depth * 
                                    np.sin(2 * np.pi * mod_freq * time))
    
    # Apply delay and mix with original
    delayed = ddsp.core.variable_length_delay(
        input_signal[np.newaxis, :],
        delay_time[np.newaxis, :],
        max_length=max_delay_samples
    )[0]
    
    audio = 0.7 * input_signal + 0.7 * delayed
    
    # Save outputs
    audio = normalize_audio(audio)
    play_audio(audio, demo_name="delays/chorus")
    specplot(audio, demo_name="delays/chorus")
    
    # Plot modulation
    plt.figure(figsize=(10, 4))
    plt.plot(time, delay_time / SAMPLE_RATE * 1000)
    plt.xlabel('Time (s)')
    plt.ylabel('Delay Time (ms)')
    plt.title('Chorus Modulation')
    plt.grid(True)
    plt.savefig("tutorials/01_core_functions/outputs/delays/chorus_modulation.png")
    plt.close()

def vibrato_demo():
    """Demonstrate vibrato effect."""
    print("\n=== Vibrato Demo ===")
    
    duration = 4.0
    n_samples = int(duration * SAMPLE_RATE)
    
    # Create input signal
    time = np.linspace(0, duration, n_samples)
    input_signal = np.sin(2 * np.pi * 440 * time)
    
    # Create modulated delay
    max_delay_samples = int(0.020 * SAMPLE_RATE)  # 20ms
    mod_freq = 5.0  # 5 Hz
    depth = 1.0
    
    delay_time = max_delay_samples * (0.5 + 0.5 * depth * 
                                    np.sin(2 * np.pi * mod_freq * time))
    
    # Apply delay
    audio = ddsp.core.variable_length_delay(
        input_signal[np.newaxis, :],
        delay_time[np.newaxis, :],
        max_length=max_delay_samples
    )[0]
    
    # Save outputs
    audio = normalize_audio(audio)
    play_audio(audio, demo_name="delays/vibrato")
    specplot(audio, demo_name="delays/vibrato")
    
    # Plot modulation
    plt.figure(figsize=(10, 4))
    plt.plot(time, delay_time / SAMPLE_RATE * 1000)
    plt.xlabel('Time (s)')
    plt.ylabel('Delay Time (ms)')
    plt.title('Vibrato Modulation')
    plt.grid(True)
    plt.savefig("tutorials/01_core_functions/outputs/delays/vibrato_modulation.png")
    plt.close()

if __name__ == "__main__":
    print("DDSP Delay Examples")
    print("==================")
    basic_delay_demo()
    flanger_demo()
    chorus_demo()
    vibrato_demo() 