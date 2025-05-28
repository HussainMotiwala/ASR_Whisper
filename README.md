# Speaker-Aware Audio Transcription System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Whisper](https://img.shields.io/badge/Whisper-OpenAI-green.svg)](https://github.com/openai/whisper)
[![PyAnnote](https://img.shields.io/badge/PyAnnote-Audio-orange.svg)](https://github.com/pyannote/pyannote-audio)

## Overview

A sophisticated speaker diarization and transcription system that combines OpenAI's Whisper for speech recognition with advanced speaker identification capabilities. The system provides automatic speaker detection, high-quality transcription with confidence scoring, and intelligent postprocessing for production-ready audio analysis.

## Key Features

### 🎯 **Advanced Speaker Diarization**
- **Primary**: PyAnnote.audio 3.1 for state-of-the-art speaker separation
- **Fallback**: SpeechBrain-based clustering for robust performance
- **Adaptive**: Simple VAD fallback ensures system reliability

### 🧠 **Intelligent Audio Processing**
- Multi-method audio loading (scipy, soundfile, librosa, pydub)
- Adaptive noise reduction with configurable strength
- Smart silence trimming and audio normalization
- Automatic sample rate conversion

### 📝 **Enhanced Transcription Pipeline**
- OpenAI Whisper integration with multiple model sizes
- Confidence-based segment filtering
- Context-aware transcription with bleeding prevention
- Multi-language support with intelligent language detection

### 🔧 **Production-Ready Features**
- Configurable postprocessing rules via JSON
- Comprehensive logging and error handling
- Memory-efficient chunk-based processing
- Detailed transcription metadata and statistics

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Audio Input   │───▶│  Speaker         │───▶│  Whisper        │
│                 │    │  Diarization     │    │  Transcription  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                         │
                              ▼                         ▼
                    ┌──────────────────┐    ┌─────────────────┐
                    │  PyAnnote.audio  │    │  Confidence     │
                    │  (Primary)       │    │  Filtering      │
                    └──────────────────┘    └─────────────────┘
                              │                         │
                              ▼                         ▼
                    ┌──────────────────┐    ┌─────────────────┐
                    │  SpeechBrain     │    │  Postprocessing │
                    │  (Fallback)      │    │  & Output       │
                    └──────────────────┘    └─────────────────┘
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)
- Hugging Face account with access to PyAnnote models

### Dependencies

```bash
# Core dependencies
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install openai-whisper
pip install pyannote.audio
pip install speechbrain
pip install scikit-learn

# Audio processing
pip install librosa
pip install noisereduce
pip install pydub
pip install soundfile
pip install scipy

# Utilities
pip install python-dotenv
pip install numpy
pip install pathlib
```

### Quick Install
```bash
git clone <repository-url>
cd speaker-aware-transcription
pip install -r requirements.txt
```

## Configuration

### Environment Variables
Create a `.env` file in the project root:

```env
HUGGINGFACE_TOKEN=your_hugging_face_token_here
```

### Hugging Face Setup
1. Create account at [https://huggingface.co](https://huggingface.co)
2. Accept conditions for `pyannote/speaker-diarization-3.1`
3. Generate access token at [https://hf.co/settings/tokens](https://hf.co/settings/tokens)
4. Add token to `.env` file

### Postprocessing Rules (Optional)
Create a JSON file with correction rules:

```json
{
  "correction_rules": [
    {
      "previous_word": "sign",
      "replace_word": "surrender",
      "confidence_threshold": 0.75
    },
    {
      "previous_word": "artificial intelligence",
      "replace_word": "AI",
      "confidence_threshold": 0.8
    }
  ]
}
```

## Usage

### Command Line Interface

#### Basic Usage
```bash
python transcribe_diarization2.py audio_file.wav
```

#### Advanced Configuration
```bash
python transcribe_diarization2.py audio_file.wav \
    --model large-v3 \
    --min-chunk-duration 0.5 \
    --min-segment-confidence 0.6 \
    --disable-context \
    --postprocessing-rules rules.json \
    --output custom_output.txt
```

#### Available Options
```bash
# Model Configuration
--model {tiny,base,small,medium,large,large-v2,large-v3}
--device {cpu,cuda}

# Diarization Settings
--min-chunk-duration FLOAT          # Minimum speaker segment duration (default: 0.5s)
--min-segment-confidence FLOAT      # Confidence threshold (default: 0.4)
--aggressive-segmentation           # Enhanced speaker separation

# Context Management
--context-words INT                 # Words for inter-speaker context (default: 10)
--disable-context                   # Prevent context bleeding between speakers

# Audio Processing
--no-noise-reduction               # Skip noise reduction
--noise-strength FLOAT             # Noise reduction intensity (0.0-1.0)

# Language Settings
--language {en,hi,auto}            # Force specific language
--restrict-languages BOOL          # Limit to English/Hindi (default: True)
--task {transcribe,translate}      # Transcribe or translate to English

# Output & Rules
--output PATH                      # Custom output file path
--postprocessing-rules PATH        # JSON rules file

# System Information
--info                             # Display model and system information
```

### Programmatic Usage

```python
from transcribe_diarization2 import SpeakerAwareTranscriber

# Initialize transcriber
transcriber = SpeakerAwareTranscriber(
    model_size="large-v3",
    device="cuda",
    restrict_languages=True,
    enable_context=False,
    min_segment_confidence=0.5,
    huggingface_token=None  # Loads from .env
)

# Process audio file
result = transcriber.process_audio_file(
    input_file="meeting_recording.wav",
    min_chunk_duration=0.5,
    noise_reduction=True,
    task="transcribe"
)

# Access detailed results
print(f"Diarization method: {result['diarization_method']}")
print(f"Speakers detected: {len(result['speaker_distribution'])}")
print(f"Overall confidence: {result['overall_confidence']:.3f}")
```

## Output Format

The system generates comprehensive transcription files with multiple sections:

### 1. Metadata Header
```
================================================================================
SPEAKER-AWARE TRANSCRIPTION WITH PYANNOTE PRIMARY + SPEECHBRAIN FALLBACK
================================================================================
Generated: 2024-01-15 14:30:22
Model: large-v3
Device: cuda
Diarization method: pyannote
Overall confidence: 0.847
Speaker time distribution:
  Speaker 1: 45.2s
  Speaker 2: 38.7s
```

### 2. Speaker-Grouped Transcription
```
[Speaker 1]:
Hello, welcome to today's meeting. I'd like to discuss our quarterly results...

[Speaker 2]:
Thank you for the introduction. I have some questions about the revenue figures...
```

### 3. Chronological Transcript with Timestamps
```
[  0.00s -   4.25s] [Speaker 1] [Conf: 0.892]: Hello, welcome to today's meeting.
[  4.50s -   8.75s] [Speaker 1] [Conf: 0.834]: I'd like to discuss our quarterly results.
[  9.00s -  12.30s] [Speaker 2] [Conf: 0.876]: Thank you for the introduction.
```

## Performance Optimization

### Hardware Recommendations
- **GPU**: NVIDIA RTX 3060 or higher for optimal performance
- **RAM**: 16GB+ recommended for large-v3 model
- **Storage**: SSD for faster model loading and audio processing

### Model Selection Guide
| Model Size | Parameters | VRAM Usage | Speed | Accuracy |
|------------|------------|------------|-------|----------|
| tiny       | 39M        | ~1GB      | 32x   | Good     |
| base       | 74M        | ~1GB      | 16x   | Better   |
| small      | 244M       | ~2GB      | 6x    | Good+    |
| medium     | 769M       | ~5GB      | 2x    | Better+  |
| large-v3   | 1550M      | ~10GB     | 1x    | Best     |

### Optimization Tips
- Use `--disable-context` for better speaker separation
- Increase `--min-segment-confidence` for cleaner output
- Reduce `--min-chunk-duration` for better speaker transitions
- Enable `--aggressive-segmentation` for overlapping speech

## Technical Specifications

### Supported Audio Formats
- WAV, MP3, FLAC, M4A, OGG
- Sample rates: 8kHz to 48kHz (auto-converted to 16kHz)
- Channels: Mono/Stereo (converted to mono)

### Language Support
- **Restricted Mode**: English (en) and Hindi (hi)
- **Full Mode**: 99+ languages supported by Whisper
- Automatic language detection with confidence scoring

### Diarization Methods
1. **PyAnnote.audio**: Neural speaker diarization with transformer architecture
2. **SpeechBrain**: ECAPA-TDNN embeddings with agglomerative clustering
3. **Simple VAD**: Librosa-based voice activity detection fallback

## Troubleshooting

### Common Issues

#### PyAnnote Authentication Error
```bash
# Error: Access denied for pyannote models
# Solution: Verify Hugging Face token and model access
```
1. Check `.env` file contains valid token
2. Verify model access at huggingface.co/pyannote/speaker-diarization-3.1
3. Ensure token has appropriate permissions

#### CUDA Out of Memory
```bash
# Error: CUDA out of memory
# Solutions:
--model medium          # Use smaller model
--device cpu           # Force CPU processing
```

#### Poor Speaker Separation
```bash
# Optimize diarization parameters:
--aggressive-segmentation
--min-chunk-duration 0.3
--disable-context
```

#### Low Transcription Quality
```bash
# Improve accuracy:
--min-segment-confidence 0.6
--model large-v3
--no-noise-reduction  # If audio is already clean
```

## API Reference

### SpeakerAwareTranscriber Class

#### Constructor Parameters
```python
SpeakerAwareTranscriber(
    model_size: str = "large-v3",
    device: str = None,
    restrict_languages: bool = True,
    postprocessing_rules: Union[str, dict] = None,
    context_words: int = 0,
    enable_context: bool = True,
    min_segment_confidence: float = 0.5,
    aggressive_segmentation: bool = False,
    huggingface_token: str = None
)
```

#### Key Methods
- `process_audio_file()`: Complete transcription pipeline
- `transcribe_with_speaker_diarization()`: Core transcription method
- `detect_speaker_segments()`: Speaker diarization only
- `get_model_info()`: System and model information

### Development Setup
```bash
git clone <repository-url>
cd speaker-aware-transcription
pip install -e .
pip install -r requirements-dev.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this system in academic research, please cite:

```bibtex
@software{speaker_aware_transcription,
  title={Speaker-Aware Audio Transcription System},
  author={Hussain Motiwala},
  year={2024},
  url={(https://github.com/HussainMotiwala/ASR_Whisper)}
}
```

## Acknowledgments

- OpenAI for the Whisper speech recognition model
- PyAnnote.audio team for speaker diarization capabilities
- SpeechBrain project for robust audio processing tools
- Hugging Face for model hosting and infrastructure


**Built with ❤️ for the audio processing and AI community**
