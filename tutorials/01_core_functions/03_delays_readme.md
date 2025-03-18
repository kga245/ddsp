# DDSP Delay-Based Effect Demonstrations

This document presents various demonstrations of DDSP's delay-based effects, including basic delay, flanger, chorus, and vibrato.

## Basic Delay Demo
Demonstrates a simple delay with feedback, creating multiple echoes.

![Basic Delay Spectrogram](outputs/delays/basic_delay.png)
<audio controls>
  <source src="outputs/delays/basic_delay.wav" type="audio/wav">
</audio>

[Listen to Basic Delay](outputs/delays/basic_delay.wav)

## Flanger Demo
Shows a short modulated delay that creates a sweeping comb filter effect.

![Flanger Spectrogram](outputs/delays/flanger.png)
![Flanger Modulation](outputs/delays/flanger_modulation.png)
<audio controls>
  <source src="outputs/delays/flanger.wav" type="audio/wav">
</audio>

[Listen to Flanger](outputs/delays/flanger.wav)

## Chorus Demo
Demonstrates a longer modulated delay mixed with the original signal.

![Chorus Spectrogram](outputs/delays/chorus.png)
![Chorus Modulation](outputs/delays/chorus_modulation.png)
<audio controls>
  <source src="outputs/delays/chorus.wav" type="audio/wav">
</audio>

[Listen to Chorus](outputs/delays/chorus.wav)

## Vibrato Demo
Shows pitch modulation using a modulated delay line.

![Vibrato Spectrogram](outputs/delays/vibrato.png)
![Vibrato Modulation](outputs/delays/vibrato_modulation.png)
<audio controls>
  <source src="outputs/delays/vibrato.wav" type="audio/wav">
</audio>

[Listen to Vibrato](outputs/delays/vibrato.wav)

## Technical Details

### Effect Parameters
- Basic Delay: 250ms delay time, 0.4 feedback ratio, 5 echoes
- Flanger: 3ms max delay, 0.5 Hz modulation rate
- Chorus: 25ms max delay, 0.25 Hz modulation rate
- Vibrato: 20ms max delay, 5 Hz modulation rate

### Implementation Notes
- Uses variable-length delay line for modulated effects
- Modulation visualized for time-varying effects
- All audio is normalized to prevent clipping
- 440 Hz sine wave used as input signal
- Sample rate: 16000 Hz

### Effect Characteristics
- **Flanger**: Creates a sweeping comb filter effect
- **Chorus**: Thickens the sound by mixing delayed and original signals
- **Vibrato**: Creates pitch modulation through delay modulation 