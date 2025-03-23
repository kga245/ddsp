# DDSP (Differentiable Digital Signal Processing)

DDSP is a powerful library that combines machine learning with traditional sound processing to create, transform, and analyze audio. Here's what you can do with it and how:

## 1. Timbre Transfer (Voice/Instrument Conversion)
Transform one sound into another using pre-trained models.

### How to Use
1. Open the [timbre_transfer.ipynb](ddsp/colab/demos/timbre_transfer.ipynb) notebook in Google Colab
2. Choose your input method:
   - Record directly through your microphone
   - Upload an audio file (.mp3 or .wav)
3. Select from available pre-trained models:
   - Violin
   - Flute
   - Flute2 (alternate flute model)
   - Trumpet
   - Tenor_Saxophone
   - Upload your own (custom trained model as .zip)

### Parameter Adjustments
1. Note Detection:
   - Threshold (0.0 to 2.0) - Controls detection sensitivity

2. Automatic Adjustments:
   - Quiet parts without notes (0-60 dB)
   - Force pitch to nearest note ("autotune" 0.0 to 1.0)

3. Manual Adjustments:
   - Pitch shift (-2 to +2 octaves)
   - Overall loudness (-20 to +20 dB)

### Best Practices
- Use clean, single-note recordings
- Avoid background noise
- Start with the default settings before tweaking parameters
- Be aware that results may sound unnatural if input audio differs significantly from training data
- For best results, match input audio characteristics to target instrument range

### Additional Resources
- For real-time applications: Check [Train_VST.ipynb](ddsp/colab/demos/Train_VST.ipynb)
- For custom models: See Section 2 (Train Your Own Sound Models)

## 2. Train Your Own Sound Models
Create custom sound transformations with your own audio samples.

### How to Use
1. Open the [train_autoencoder.ipynb](ddsp/colab/demos/train_autoencoder.ipynb) notebook
2. Prepare your training data (see Recording Guide)
3. Upload your data
4. Train the model
5. Export your trained model

### Recording Guide
#### Setup Requirements
- Decent quality microphone
- Quiet room with minimal echo
- Consistent microphone distance
- Consistent recording levels
- Minimum 16kHz sample rate (44.1kHz preferred)

#### What to Record
1. Basic Notes
   - Single notes held for 2-3 seconds each
   - Cover the full range of your instrument/voice
   - Include 3 dynamics for each note:
     * Soft (piano)
     * Medium (mezzo-forte)
     * Loud (forte)

2. Playing Techniques
   - Natural/normal playing style
   - Vibrato (if applicable)
   - Different articulations:
     * Staccato (short, detached)
     * Legato (smooth, connected)
     * Accent notes
   - Special techniques specific to your instrument:
     * Guitar: picking vs fingering
     * Violin: pizzicato, bowing positions
     * Voice: different vowel sounds
     * Wind instruments: different breath techniques

#### Recording Structure (10-12 minutes total)
1. Chromatic scale ascending (2 mins)
   - Play each note for 2 seconds
   - Leave 1 second gap between notes
   - Medium volume

2. Long notes at different dynamics (2 mins)
   - Choose 5-6 different notes across range
   - Play each note at 3 different volumes

3. Short notes/staccato (2 mins)
   - Quick notes across full range
   - Mix of dynamics

4. Musical phrases (2 mins)
   - Simple melodies
   - Natural playing style

5. Experimental techniques (2 mins)
   - Instrument-specific techniques
   - Unique timbres

6. Additional material (1-2 mins)
   - Extra takes of problematic sections
   - Alternative techniques

### Common Recording Pitfalls and Solutions

1. Environment Issues
   - ❌ Problem: Room echo
   - ✅ Solution: Record in a small room with soft furnishings
   
   - ❌ Problem: Background noise
   - ✅ Solution: Turn off AC/heating, close windows, put phones on silent

2. Technical Issues
   - ❌ Problem: Clipping/distortion
   - ✅ Solution: Keep input levels around -12dB peak
   
   - ❌ Problem: Too quiet
   - ✅ Solution: Aim for peaks around -6dB to -12dB

3. Performance Issues
   - ❌ Problem: Inconsistent timing
   - ✅ Solution: Use a metronome for pacing
   
   - ❌ Problem: Uneven volumes
   - ✅ Solution: Mark playing positions and mic distances

4. Content Issues
   - ❌ Problem: Missing parts of range
   - ✅ Solution: Use a written checklist of notes/techniques
   
   - ❌ Problem: Accidental polyphony
   - ✅ Solution: Wait for each note to fully stop before playing next

### File Preparation and Upload
1. Format Requirements
   - Save files as .wav or .mp3
   - Files can be separate or one long recording
   - Remove any lengthy silences
   - Name files with technique and take number

2. Upload Process
   - Create a folder in Google Drive
   - Upload all audio files to this folder
   - Connect the notebook to your Drive
   - Select your folder when prompted

### Training Process
1. Model Configuration
   - Choose model architecture
   - Set training parameters
   - Configure audio preprocessing

2. Training Monitoring
   - Watch for loss convergence
   - Monitor audio quality
   - Check for overfitting
   - Save checkpoints regularly

## 3. Pitch Detection
Analyze musical content using either DDSP or CREPE algorithms.

### How to Use
1. Open the [pitch_detection.ipynb](ddsp/colab/demos/pitch_detection.ipynb) notebook
2. Input your audio (record or upload)
3. Choose detection algorithm (DDSP or CREPE)
4. Adjust parameters
5. Analyze results

### Output Types
1. Visual Outputs
   - Pitch over time graph (MIDI notes)
   - Spectrogram visualization
   - Confidence levels plot
   - Comparison view of DDSP vs CREPE results

2. Audio Outputs
   - Original audio
   - Sinusoidal resynthesis (clean pitch representation)
   - Harmonic resynthesis (with overtones)
   - DDSP pitch audio
   - CREPE pitch audio

### Algorithm Comparison
1. DDSP (Differentiable Digital Signal Processing)
   - Self-supervised learning approach
   - Better at handling complex timbres
   - Provides additional harmonic information
   - More computationally efficient
   - Works well with singing and speech

2. CREPE (Convolutional REpresentation for Pitch Estimation)
   - Supervised learning approach
   - More accurate for simple monophonic sounds
   - Industry standard for pitch detection
   - Higher computational requirements
   - Better with noisy recordings

### Parameter Controls
1. Detection Settings
   - Confidence threshold (0.0-1.0)
   - Frame rate (speed of analysis)
   - Frequency range limits
   - Viterbi algorithm toggle (for smoother tracking)

2. Output Settings
   - Note quantization strength
   - Minimum note duration
   - Pitch rounding tolerance
   - Output format (Hz, MIDI, note names)

### Applications
1. Music Transcription
   ```
   Process:
   - Record or upload audio
   - Run pitch detection
   - Export MIDI or musical notation
   - Clean up results with confidence threshold
   - Quantize to musical grid if needed
   ```

2. Melody Extraction
   ```
   Process:
   - Process polyphonic audio
   - Filter for highest confidence pitches
   - Apply melodic smoothing
   - Extract main melodic line
   ```

3. Pitch Correction
   ```
   Process:
   - Analyze original pitch
   - Set target scale/notes
   - Adjust correction strength
   - Apply correction while maintaining timbre
   ```

### Best Practices
1. For Transcription:
   - Use high-quality recordings
   - Start with monophonic sources
   - Set appropriate confidence thresholds
   - Verify results aurally

2. For Melody Extraction:
   - Focus on clear lead elements
   - Use frequency filtering when needed
   - Consider musical context
   - Compare both DDSP and CREPE results

3. For Pitch Correction:
   - Work with clean source material
   - Use gentle correction settings first
   - Preserve natural pitch variations
   - Monitor for artifacts

### Limitations
- Less accurate with polyphonic audio
- May struggle with very noisy recordings
- Some instruments have complex pitch characteristics
- Real-time processing has higher latency
- Results may need manual cleanup

## 4. Sound Synthesis and Effects
Create and modify sounds using neural audio synthesis and processing chains.

### Core Components
1. Harmonic Synthesizer
   ```
   Purpose: Generate pitched sounds with controllable harmonics
   
   Controls:
   - Fundamental frequency (pitch)
   - Amplitude envelope
   - Harmonic distribution
   - Number of harmonics (20-100)
   
   Applications:
   - Tonal instrument synthesis
   - Voice synthesis
   - Harmonic sound design
   ```

2. Filtered Noise Generator
   ```
   Purpose: Create unpitched and textural sounds
   
   Controls:
   - Noise magnitude
   - Filter bank parameters
   - Spectral shape
   - Time-varying filtering
   
   Applications:
   - Breath sounds
   - Percussion synthesis
   - Texture generation
   - Environmental sounds
   ```

3. Effects Processing
   ```
   Purpose: Shape and transform audio signals
   
   Types:
   - Reverb (space simulation)
   - Add (signal mixing)
   - Multiply (amplitude modulation)
   - Filter (frequency shaping)
   ```

### Processing Architecture
1. Basic Chain Structure
   ```
   Input → Processor 1 → Processor 2 → ... → Output
   
   Example:
   Audio → Harmonic Synth → Noise Gen → Reverb → Final
   ```

2. Advanced Configurations
   ```
   Parallel Processing:
                → Process A →
   Input Signal → Process B → Combine → Output
                → Process C →
   
   Feedback Loops:
   Input → Process → Output
      ↑______________|
   ```

### Parameter Control
1. Direct Control
   - Manual parameter adjustment
   - MIDI controller mapping
   - Automation curves
   - Real-time modulation

2. ML-Based Control
   - Learned parameter mapping
   - Feature extraction
   - Gesture control
   - Automatic parameter optimization

3. Algorithmic Control
   - LFOs (Low Frequency Oscillators)
   - Envelope generators
   - Random/chaos generators
   - Mathematical functions

## General Guidelines

### System Requirements
- Google account (for Colab)
- GPU runtime enabled
- Stable internet connection
- Sufficient storage space
- Basic understanding of audio concepts

### Best Practices
1. Setup
   - Always use GPU runtime in Colab
   - Keep original audio clean and high-quality
   - Start with example audio before using your own
   - Save your work frequently

2. Performance
   - Optimize processing chains
   - Monitor CPU/GPU usage
   - Use appropriate buffer sizes
   - Consider latency requirements

3. Workflow
   - Start with simple configurations
   - Document successful techniques
   - Create backup copies
   - Test incrementally

### Common Issues and Solutions
1. Technical Issues
   - Colab disconnections: Save checkpoints frequently
   - Performance problems: Reduce model complexity
   - Audio glitches: Check buffer settings
   - Memory errors: Clear runtime and restart

2. Quality Issues
   - Poor results: Check input audio quality
   - Artifacts: Adjust processing parameters
   - Inconsistent output: Verify model settings
   - Latency: Optimize processing chain

### Additional Resources
1. Documentation
   - DDSP GitHub repository
   - Tutorial notebooks
   - API documentation
   - Example implementations

2. Community
   - GitHub issues
   - Discussion forums
   - User examples
   - Research papers

3. Tools
   - Audio editing software
   - MIDI controllers
   - Audio interfaces
   - Recording equipment

