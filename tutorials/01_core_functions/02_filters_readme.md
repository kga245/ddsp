# DDSP Filter Demonstrations

## Noise-Based Filter Demonstrations

### Basic Filter Demo
Demonstrates a simple lowpass filter with 1000 Hz cutoff frequency applied to white noise.

![Basic Filter Spectrogram](outputs/filters/noise/basic_filter.png)
![Basic Filter Response](outputs/filters/noise/basic_filter_response.png)
<audio controls>
  <source src="outputs/filters/noise/basic_filter.wav" type="audio/wav">
</audio>

### Time-Varying Filter Demo
Shows a filter sweeping from 100 Hz to 8000 Hz over time, creating a dynamic filtering effect.

![Time-Varying Filter Spectrogram](outputs/filters/noise/timevarying_filter.png)
<audio controls>
  <source src="outputs/filters/noise/timevarying_filter.wav" type="audio/wav">
</audio>

### Filter Types Demo
Demonstrates a bandpass filter centered at 1000 Hz with 500 Hz bandwidth.

![Bandpass Filter Spectrogram](outputs/filters/noise/bandpass_filter.png)
![Bandpass Filter Response](outputs/filters/noise/bandpass_filter_response.png)
<audio controls>
  <source src="outputs/filters/noise/bandpass_filter.wav" type="audio/wav">
</audio>

### Noise Filter Demo
Shows the effect of a narrow bandpass filter (50 Hz bandwidth) centered at 440 Hz.

![Noise Filter Response](outputs/filters/noise/noise_filter_response.png)

#### Original Noise
![Noisy Signal Spectrogram](outputs/filters/noise/noisy_signal.png)
<audio controls>
  <source src="outputs/filters/noise/noisy_signal.wav" type="audio/wav">
</audio>

#### Filtered Noise
![Filtered Signal Spectrogram](outputs/filters/noise/filtered_signal.png)
<audio controls>
  <source src="outputs/filters/noise/filtered_signal.wav" type="audio/wav">
</audio>

## Musical Filter Demonstrations

### Musical Chord Filter
Demonstrates a sweeping filter effect on a C major chord (C4, E4, G4). The filter response creates a dynamic spectral movement through the harmonics.

![Original Chord Spectrogram](outputs/filters/musical/chord_original.png)
![Filtered Chord Spectrogram](outputs/filters/musical/chord_filtered.png)
<audio controls>
  <source src="outputs/filters/musical/chord_original.wav" type="audio/wav">
</audio>
<audio controls>
  <source src="outputs/filters/musical/chord_filtered.wav" type="audio/wav">
</audio>

### Speech Filter Demo
Creates vowel-like sounds by applying formant filtering to a harmonic tone. The filter morphs between two formant configurations over time.

![Original Vowel Spectrogram](outputs/filters/musical/vowel_original.png)
![Filtered Vowel Spectrogram](outputs/filters/musical/vowel_filtered.png)
<audio controls>
  <source src="outputs/filters/musical/vowel_original.wav" type="audio/wav">
</audio>
<audio controls>
  <source src="outputs/filters/musical/vowel_filtered.wav" type="audio/wav">
</audio>

### Melody Filter Demo
Demonstrates five different filter types on a simple melody:
1. No filtering (reference)
2. Lowpass filter (removes high frequencies)
3. Highpass filter (removes low frequencies)
4. Bandpass filter (keeps middle frequencies)
5. Bandstop filter (removes middle frequencies)

![Original Melody Spectrogram](outputs/filters/musical/melody_original.png)
![Filtered Melody Spectrogram](outputs/filters/musical/melody_filtered.png)
<audio controls>
  <source src="outputs/filters/musical/melody_original.wav" type="audio/wav">
</audio>
<audio controls>
  <source src="outputs/filters/musical/melody_filtered.wav" type="audio/wav">
</audio>

## Technical Details

### Filter Parameters
- Basic Filter: Lowpass with 1000 Hz cutoff
- Time-Varying Filter: Sweeping cutoff from 100 Hz to 8000 Hz
- Bandpass Filter: Center at 1000 Hz, 500 Hz bandwidth
- Noise Filter: Narrow bandpass around 440 Hz (50 Hz bandwidth)
- Speech Filter: Moving formants (F1: 730→270 Hz, F2: 1090→2290 Hz)
- Melody Filter: Various types with cutoffs at 1500 Hz

### Implementation Notes
- Uses FFT-based filtering via ddsp.core.frequency_filter
- All audio normalized to prevent clipping
- Sample rate: 16000 Hz
- Window sizes vary by demo (64-1024 samples)
- Gaussian-shaped filter responses for smooth transitions 