# DDSP Oscillator Demonstrations

This document presents various demonstrations of DDSP's oscillator capabilities, including basic frequency sweeps, modulated oscillators, oscillator swarms, and wavetable synthesis. Each demo includes spectrograms and audio playback.

## Basic Frequency Sweep Demonstrations

The basic sweep demonstrates a simple oscillator with varying frequency over time.

<details>
<summary>Basic Sweep Variations (click to expand)</summary>

### Original Sweep (0-1000 Hz)
![Basic Sweep Spectrogram](outputs/oscillators/basic_sweep.png)
<audio controls>
  <source src="outputs/oscillators/basic_sweep.wav" type="audio/wav">
</audio>

[Listen to Basic Sweep](outputs/oscillators/basic_sweep.wav)

### Wide Range Sweep (20-8000 Hz)
![Wide Sweep Spectrogram](outputs/oscillators/basic_sweep_wide.png)
<audio controls>
  <source src="outputs/oscillators/basic_sweep_wide.wav" type="audio/wav">
</audio>

### Logarithmic Sweep
![Logarithmic Sweep Spectrogram](outputs/oscillators/basic_sweep_log.png)
<audio controls>
  <source src="outputs/oscillators/basic_sweep_log.wav" type="audio/wav">
</audio>

### Slow Sweep (0-500 Hz)
![Slow Sweep Spectrogram](outputs/oscillators/basic_sweep_slow.png)
<audio controls>
  <source src="outputs/oscillators/basic_sweep_slow.wav" type="audio/wav">
</audio>

### High Sample Rate (48kHz)
![High Sample Rate Sweep Spectrogram](outputs/oscillators/basic_sweep_48k.png)
<audio controls>
  <source src="outputs/oscillators/basic_sweep_48k.wav" type="audio/wav">
</audio>

</details>

## Oscillator Swarm Demonstrations

The swarm demo creates a cloud of oscillators with random frequency modulation.

<details>
<summary>Swarm Variations (click to expand)</summary>

### Original Swarm (100 oscillators)
![Swarm Spectrogram](outputs/oscillators/swarm.png)
<audio controls>
  <source src="outputs/oscillators/swarm.wav" type="audio/wav">
</audio>

### Dense Swarm (200 oscillators)
![Dense Swarm Spectrogram](outputs/oscillators/swarm_dense.png)
<audio controls>
  <source src="outputs/oscillators/swarm_dense.wav" type="audio/wav">
</audio>

### Sparse Swarm (50 oscillators)
![Sparse Swarm Spectrogram](outputs/oscillators/swarm_sparse.png)
<audio controls>
  <source src="outputs/oscillators/swarm_sparse.wav" type="audio/wav">
</audio>

### Chaos Swarm (Wide range, deep modulation)
![Chaos Swarm Spectrogram](outputs/oscillators/swarm_chaos.png)
<audio controls>
  <source src="outputs/oscillators/swarm_chaos.wav" type="audio/wav">
</audio>

### High Sample Rate Swarm (48kHz)
![High Sample Rate Swarm Spectrogram](outputs/oscillators/swarm_48k.png)
<audio controls>
  <source src="outputs/oscillators/swarm_48k.wav" type="audio/wav">
</audio>

</details>

## Wavetable Synthesis Demonstrations

The wavetable demo shows smooth transitions between different waveforms.

<details>
<summary>Wavetable Variations (click to expand)</summary>

### Original (Sine → Triangle → Square)
![Wavetable Spectrogram](outputs/oscillators/wavetable.png)
<audio controls>
  <source src="outputs/oscillators/wavetable.wav" type="audio/wav">
</audio>

### Sawtooth Variation (Sine → Saw → Square)
![Sawtooth Wavetable Spectrogram](outputs/oscillators/wavetable_saw.png)
<audio controls>
  <source src="outputs/oscillators/wavetable_saw.wav" type="audio/wav">
</audio>

### Complex Harmonics
![Complex Wavetable Spectrogram](outputs/oscillators/wavetable_complex.png)
<audio controls>
  <source src="outputs/oscillators/wavetable_complex.wav" type="audio/wav">
</audio>

### High Pitch (440 Hz)
![High Pitch Wavetable Spectrogram](outputs/oscillators/wavetable_high.png)
<audio controls>
  <source src="outputs/oscillators/wavetable_high.wav" type="audio/wav">
</audio>

### Low Pitch (110 Hz)
![Low Pitch Wavetable Spectrogram](outputs/oscillators/wavetable_low.png)
<audio controls>
  <source src="outputs/oscillators/wavetable_low.wav" type="audio/wav">
</audio>

### High Sample Rate (48kHz)
![High Sample Rate Wavetable Spectrogram](outputs/oscillators/wavetable_48k.png)
<audio controls>
  <source src="outputs/oscillators/wavetable_48k.wav" type="audio/wav">
</audio>

</details>

## Reading the Spectrograms

- The vertical axis represents frequency (Hz)
- The horizontal axis represents time (seconds)
- Color intensity represents the amplitude of frequencies (brighter = stronger)
- The spectrograms use a logarithmic frequency scale to better show harmonic relationships

## Technical Details

- Default sample rate: 16000 Hz
- High sample rate examples: 48000 Hz
- All audio is normalized to prevent clipping
- Crossfades are applied between wavetable transitions to prevent clicks
