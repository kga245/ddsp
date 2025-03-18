"""DDSP Oscillator Examples

Demonstrates:
1. Basic sine wave oscillator (frequency sweep)
2. Single modulated oscillator (clear frequency modulation)
3. Swarm of modulated oscillators
4. Wavetable synthesis
"""

import numpy as np
import tensorflow as tf
import ddsp
import ddsp.core
from utils import play_audio, specplot, normalize_audio, smooth, SAMPLE_RATE
import os

# Create outputs directory for oscillators
os.makedirs('tutorials/01_core_functions/outputs/oscillators', exist_ok=True)

def basic_oscillator_demo(sample_rate=16000, name_suffix=""):
    """Demonstrate basic oscillator functionality with a frequency sweep."""
    print(f"\n=== Basic Oscillator Demo ({name_suffix}) ===")
    
    duration = 4.0
    n_samples = int(duration * sample_rate)
    
    # Different sweep ranges and durations
    if name_suffix == "wide":
        f0_hz = tf.linspace(20.0, 8000.0, n_samples)  # Wider frequency range
    elif name_suffix == "log":
        f0_hz = tf.exp(tf.linspace(np.log(20.0), np.log(2000.0), n_samples))  # Logarithmic sweep
    elif name_suffix == "slow":
        f0_hz = tf.linspace(0.0, 500.0, n_samples)  # Slower/narrower sweep
    else:
        f0_hz = tf.linspace(0.0, 1000.0, n_samples)  # Original
    
    audio = ddsp.core.oscillator_bank(
        frequency_envelopes=f0_hz[tf.newaxis, :, tf.newaxis],
        amplitude_envelopes=tf.ones_like(f0_hz)[tf.newaxis, :, tf.newaxis],
        sample_rate=sample_rate
    )
    audio = audio[0]
    
    name = f"basic_sweep_{name_suffix}" if name_suffix else "basic_sweep"
    play_audio(audio, sample_rate=sample_rate, demo_name="oscillators/" + name)
    specplot(audio, sample_rate=sample_rate, demo_name="oscillators/" + name)
    print(f"Generated sweep with {name_suffix} parameters")

def single_modulated_oscillator_demo(sample_rate=16000, name_suffix=""):
    """Demonstrate a single frequency-modulated oscillator."""
    print(f"\n=== Single Modulated Oscillator Demo ({name_suffix}) ===")
    
    duration = 4.0
    n_samples = int(duration * sample_rate)
    time = np.linspace(0, duration, n_samples)
    
    # Different modulation parameters
    if name_suffix == "fast":
        center_freq = 500
        mod_depth = 200
        mod_speed = 5  # Faster modulation
    elif name_suffix == "deep":
        center_freq = 1000
        mod_depth = 800  # Deeper modulation
        mod_speed = 1
    elif name_suffix == "slow":
        center_freq = 500
        mod_depth = 200
        mod_speed = 0.25  # Slower modulation
    else:
        center_freq = 500
        mod_depth = 200
        mod_speed = 1
    
    # Create the modulated frequency
    frequency = center_freq + mod_depth * np.sin(2 * np.pi * mod_speed * time)
    frequencies = frequency[np.newaxis, :, np.newaxis]  # Add batch and channel dims
    
    # Generate audio
    audio = ddsp.core.oscillator_bank(
        frequency_envelopes=frequencies,
        amplitude_envelopes=tf.ones_like(frequencies) * 0.8,
        sample_rate=sample_rate
    )
    audio = audio[0]
    
    # Play and visualize
    name = f"single_modulated_{name_suffix}" if name_suffix else "single_modulated"
    play_audio(audio, sample_rate=sample_rate, demo_name="oscillators/" + name)
    specplot(audio, sample_rate=sample_rate, demo_name="oscillators/" + name)
    print(f"Generated single oscillator with {name_suffix} modulation")

def swarm_oscillator_demo(sample_rate=16000, name_suffix=""):
    """Demonstrate a swarm of sinusoids with random frequency modulation."""
    print(f"\n=== Swarm Oscillator Demo ({name_suffix}) ===")
    
    duration = 6.0
    n_samples = int(duration * sample_rate)
    
    # Different swarm configurations
    if name_suffix == "dense":
        n_oscillators = 200  # More oscillators
        freq_range = (100, 1000)  # Wider frequency range
        mod_depth = 100
        mod_speed = 0.5
    elif name_suffix == "sparse":
        n_oscillators = 50  # Fewer oscillators
        freq_range = (300, 600)  # Narrower frequency range
        mod_depth = 50
        mod_speed = 2
    elif name_suffix == "chaos":
        n_oscillators = 100
        freq_range = (50, 2000)  # Very wide range
        mod_depth = 400  # Deep modulation
        mod_speed = 3
    else:
        n_oscillators = 100
        freq_range = (200, 800)
        mod_depth = 200
        mod_speed = 1
    
    # Create frequency trajectories
    frequencies = np.random.uniform(freq_range[0], freq_range[1], 
                                  (1, n_samples, n_oscillators))
    
    # Add modulation
    time = np.linspace(0, duration, n_samples)
    for i in range(n_oscillators):
        mod = np.sin(2 * np.pi * mod_speed * time + np.random.uniform(0, 2*np.pi))
        mod = smooth(mod, 1000)
        frequencies[0, :, i] += mod * mod_depth
    
    # Create amplitude envelopes with fade in/out
    fade_samples = int(0.1 * sample_rate)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    base_amplitude = np.ones(n_samples) * (1.0 / n_oscillators)
    base_amplitude[:fade_samples] *= fade_in
    base_amplitude[-fade_samples:] *= fade_out
    amplitudes = np.tile(base_amplitude[:, np.newaxis], [1, n_oscillators])
    amplitudes = amplitudes[np.newaxis, :, :]
    
    audio = ddsp.core.oscillator_bank(
        frequency_envelopes=frequencies,
        amplitude_envelopes=amplitudes,
        sample_rate=sample_rate
    )
    audio = audio[0]
    
    name = f"swarm_{name_suffix}" if name_suffix else "swarm"
    play_audio(audio, sample_rate=sample_rate, demo_name="oscillators/" + name)
    specplot(audio, sample_rate=sample_rate, demo_name="oscillators/" + name)
    print(f"Generated swarm with {name_suffix} configuration")

def wavetable_demo(sample_rate=16000, name_suffix=""):
    """Demonstrate wavetable synthesis with different waveforms."""
    print(f"\n=== Wavetable Demo ({name_suffix}) ===")
    
    # Create wavetables
    n_samples = 2048
    t = tf.linspace(0.0, 2.0 * np.pi, n_samples)
    
    # Different wavetable configurations
    if name_suffix == "saw":
        # Sawtooth and square
        wave1 = tf.sin(t)  # Start with sine
        wave2 = 2 * (t/2/np.pi - tf.floor(t/2/np.pi)) - 1  # Sawtooth
        wave3 = tf.sign(tf.sin(t))  # Square
    elif name_suffix == "complex":
        # Complex harmonic content
        wave1 = tf.sin(t)  # Fundamental
        wave2 = tf.sin(t) + 0.5*tf.sin(2*t) + 0.33*tf.sin(3*t)  # Rich harmonics
        wave3 = tf.sign(tf.sin(t)) * tf.abs(tf.sin(t))  # Rectified
    else:
        # Original sine, triangle, square
        wave1 = tf.sin(t)
        wave2 = 2 * tf.abs(2 * (t/2/np.pi - tf.floor(t/2/np.pi + 0.5))) - 1
        wave3 = tf.sign(tf.sin(t))
    
    wavetable = tf.stack([wave1, wave2, wave3], axis=0)
    wavetable = wavetable[tf.newaxis, :, :]
    
    # Create a 6-second sound
    duration = 6.0
    n_samples = int(duration * sample_rate)
    
    # Different base frequencies
    if name_suffix == "high":
        f0 = 440  # A4
    elif name_suffix == "low":
        f0 = 110  # A2
    else:
        f0 = 220  # A3
    
    phase = tf.linspace(0.0, duration * f0, n_samples) % 1.0
    phase = phase[tf.newaxis, :, tf.newaxis]
    
    # Create transitions
    n_frames = 100
    morph = np.zeros((1, n_frames, 3))
    third = n_frames // 3
    morph[0, :third, 0] = 1.0
    morph[0, third:2*third, 1] = 1.0
    morph[0, 2*third:, 2] = 1.0
    
    # Add crossfades
    fade = 5
    morph[0, third-fade:third+fade, 0] = np.cos(np.linspace(0, np.pi/2, 2*fade))**2
    morph[0, third-fade:third+fade, 1] = np.sin(np.linspace(0, np.pi/2, 2*fade))**2
    morph[0, 2*third-fade:2*third+fade, 1] = np.cos(np.linspace(0, np.pi/2, 2*fade))**2
    morph[0, 2*third-fade:2*third+fade, 2] = np.sin(np.linspace(0, np.pi/2, 2*fade))**2
    
    wavetables = ddsp.core.resample(morph, n_samples) @ wavetable
    audio = ddsp.core.linear_lookup(phase, wavetables)
    audio = audio[0]
    
    name = f"wavetable_{name_suffix}" if name_suffix else "wavetable"
    play_audio(normalize_audio(audio), sample_rate=sample_rate, demo_name="oscillators/" + name)
    specplot(audio, sample_rate=sample_rate, demo_name="oscillators/" + name)
    print(f"Generated wavetable with {name_suffix} configuration")

if __name__ == "__main__":
    print("DDSP Oscillator Examples")
    print("=======================")
    
    # Basic sweep variations
    basic_oscillator_demo()  # Original
    basic_oscillator_demo(name_suffix="wide")  # Wider frequency range
    basic_oscillator_demo(name_suffix="log")   # Logarithmic sweep
    basic_oscillator_demo(name_suffix="slow")  # Slower sweep
    basic_oscillator_demo(sample_rate=48000, name_suffix="48k")  # High sample rate
    
    # Swarm variations
    swarm_oscillator_demo()  # Original
    swarm_oscillator_demo(name_suffix="dense")  # More oscillators
    swarm_oscillator_demo(name_suffix="sparse")  # Fewer oscillators
    swarm_oscillator_demo(name_suffix="chaos")  # Wide range, deep modulation
    swarm_oscillator_demo(sample_rate=48000, name_suffix="48k")  # High sample rate
    
    # Wavetable variations
    wavetable_demo()  # Original (sine, triangle, square)
    wavetable_demo(name_suffix="saw")  # Sine, saw, square
    wavetable_demo(name_suffix="complex")  # Complex harmonics
    wavetable_demo(name_suffix="high")  # Higher pitch
    wavetable_demo(name_suffix="low")   # Lower pitch
    wavetable_demo(sample_rate=48000, name_suffix="48k")  # High sample rate
    # Fast modulation version
    single_modulated_oscillator_demo(name_suffix="fast")
    # Deep modulation version
    single_modulated_oscillator_demo(name_suffix="deep")
    # Slow modulation version
    single_modulated_oscillator_demo(name_suffix="slow")
    # High sample rate version (48kHz)
    single_modulated_oscillator_demo(sample_rate=48000, name_suffix="48k") 