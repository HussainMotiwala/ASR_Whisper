import os
import librosa
import noisereduce as nr
import numpy as np
from pydub import AudioSegment
import whisper
import tempfile
from pathlib import Path
import argparse
from datetime import datetime
import torch
import warnings
from scipy.io import wavfile
from scipy import signal
import json
import re
from collections import Counter
warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables
load_dotenv()  # Add this line

# Set environment variable BEFORE importing to avoid Windows symlink issues
os.environ['HUGGINGFACE_HUB_CACHE_STRATEGY'] = 'copy'

# Primary diarization - pyannote
try:
    from pyannote.audio import Pipeline
    import torchaudio
    PYANNOTE_AVAILABLE = True
    print("✅ pyannote.audio available for primary speaker diarization")
except ImportError as e:
    PYANNOTE_AVAILABLE = False
    print(f"⚠️  pyannote.audio not available: {e}")
    print("   Install with: pip install pyannote.audio")

# Fallback diarization - SpeechBrain
try:
    from speechbrain.inference import EncoderClassifier, SpeakerRecognition
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    import scipy.spatial.distance as distance
    SPEECHBRAIN_AVAILABLE = True
    print("✅ SpeechBrain available for fallback speaker diarization")
except ImportError as e:
    SPEECHBRAIN_AVAILABLE = False
    print(f"⚠️  SpeechBrain not available: {e}")
    print("   Speaker diarization will use simple VAD fallback")
    print("   Install with: pip install speechbrain scikit-learn")

class PostProcessor:
    """Handles postprocessing of transcription text based on correction rules"""
    
    def __init__(self, rules_file=None, rules_dict=None):
        """
        Initialize postprocessor with correction rules
        
        Args:
            rules_file (str): Path to JSON file containing correction rules
            rules_dict (dict): Dictionary containing correction rules
        """
        self.correction_rules = []
        
        if rules_file and os.path.exists(rules_file):
            with open(rules_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.correction_rules = data.get('correction_rules', [])
        elif rules_dict:
            self.correction_rules = rules_dict.get('correction_rules', [])
        
        print(f"📝 Loaded {len(self.correction_rules)} postprocessing rules")
        
        # Sort rules by confidence threshold (descending) for better processing
        self.correction_rules.sort(key=lambda x: x.get('confidence_threshold', 0.5), reverse=True)
    
    def apply_corrections(self, text, confidence_score=1.0):
        """
        Apply correction rules to text based on confidence thresholds
        
        Args:
            text (str): Text to correct
            confidence_score (float): Confidence score of the text
            
        Returns:
            tuple: (corrected_text, corrections_applied)
        """
        if not text or not self.correction_rules:
            return text, []
        
        corrected_text = text
        corrections_applied = []
        
        for rule in self.correction_rules:
            previous_word = rule.get('previous_word', '')
            replace_word = rule.get('replace_word', '')
            threshold = rule.get('confidence_threshold', 0.5)
            
            # Only apply correction if confidence meets threshold
            if confidence_score <= threshold and previous_word and replace_word:
                # Case-insensitive matching but preserve case in replacement
                pattern = re.compile(re.escape(previous_word), re.IGNORECASE)
                matches = pattern.findall(corrected_text)
                
                if matches:
                    # Apply replacement
                    corrected_text = pattern.sub(replace_word, corrected_text)
                    corrections_applied.append({
                        'original': previous_word,
                        'corrected': replace_word,
                        'confidence_threshold': threshold,
                        'applied_at_confidence': confidence_score,
                        'occurrences': len(matches)
                    })
        
        return corrected_text, corrections_applied
    
    def get_context_text(self, text, max_words=0):
        """
        Get last few words from text to use as context for next chunk
        
        Args:
            text (str): Text to extract context from
            max_words (int): Maximum number of words to include
            
        Returns:
            str: Context text
        """
        if not text or max_words == 0:
            return ""
                
        words = text.strip().split()
        return " ".join(words[-max_words:])

class SpeakerAwareTranscriber:
    def __init__(self, model_size="large-v3", device=None, restrict_languages=True, 
                 postprocessing_rules=None, context_words=0, speechbrain_cache_dir="./models/speechbrain",
                 enable_context=True, min_segment_confidence=0.5, aggressive_segmentation=False,
                 huggingface_token=None):
        """
        Initialize the AudioTranscriber with pyannote primary and SpeechBrain fallback diarization
        
        Args:
            model_size (str): Whisper model size
            device (str): Device to run model on
            restrict_languages (bool): Restrict to English and Hindi only
            postprocessing_rules (dict or str): Postprocessing rules (dict or path to JSON file)
            context_words (int): Number of words from previous chunk to use as context
            speechbrain_cache_dir (str): Directory to cache SpeechBrain models locally
            enable_context (bool): Whether to use context from previous chunks (can cause bleeding)
            min_segment_confidence (float): Minimum confidence to include segments
            aggressive_segmentation (bool): Use more aggressive speaker segmentation
            huggingface_token (str): Hugging Face access token for pyannote models
        """
        print(f"Loading Whisper model: {model_size}")
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.restrict_languages = restrict_languages
        self.context_words = context_words
        self.speechbrain_cache_dir = speechbrain_cache_dir
        self.enable_context = enable_context
        self.min_segment_confidence = min_segment_confidence
        self.aggressive_segmentation = aggressive_segmentation
        self.huggingface_token = huggingface_token or os.getenv('HUGGINGFACE_TOKEN')
        
        # Create cache directory if it doesn't exist
        os.makedirs(speechbrain_cache_dir, exist_ok=True)
        
        # Initialize postprocessor (keeping existing system)
        if isinstance(postprocessing_rules, str):
            self.postprocessor = PostProcessor(rules_file=postprocessing_rules)
        elif isinstance(postprocessing_rules, dict):
            self.postprocessor = PostProcessor(rules_dict=postprocessing_rules)
        else:
            self.postprocessor = PostProcessor()  # No rules
        
        print(f"Using device: {device}")
        print(f"Context words for chunk processing: {context_words}")
        print(f"Context enabled: {enable_context}")
        print(f"Minimum segment confidence: {min_segment_confidence}")
        print(f"Aggressive segmentation: {aggressive_segmentation}")
        
        if restrict_languages:
            print("Language restriction: English and Hindi only")
            self.supported_languages = ['en', 'hi']
        else:
            print("Language restriction: All supported languages")
            self.supported_languages = None
        
        # Load the Whisper model
        try:
            if device == "cuda" and not torch.cuda.is_available():
                print("⚠️  CUDA requested but not available, falling back to CPU")
                device = "cpu"
                self.device = "cpu"
            
            self.model = whisper.load_model(model_size, device=device, download_root=None)
            print(f"✅ Model {model_size} loaded successfully on {device}")
        except Exception as e:
            print(f"❌ Failed to load model {model_size}: {e}")
            if "CUDA" in str(e) and device == "cuda":
                print("🔧 CUDA error detected, trying CPU...")
                device = "cpu"
                self.device = "cpu"
                try:
                    self.model = whisper.load_model(model_size, device="cpu")
                    print(f"✅ Model {model_size} loaded successfully on CPU")
                except Exception as e2:
                    print(f"❌ CPU fallback also failed: {e2}")
                    print("Falling back to base model on CPU...")
                    self.model = whisper.load_model("base", device="cpu")
            else:
                print("Falling back to base model...")
                self.model = whisper.load_model("base", device=device)
        
        self.model_size = model_size
        
        # Initialize speaker diarization (pyannote primary, SpeechBrain fallback)
        self.pyannote_pipeline = None
        self.speechbrain_speaker_model = None
        self.speechbrain_vad_model = None
        self.diarization_method = None
        
        self._init_speaker_diarization()
        
    def _init_speaker_diarization(self):
        """Initialize speaker diarization models - speechbrain first, then Pyannote fallback"""
        
        # Try speechbrain first (primary method)
        if SPEECHBRAIN_AVAILABLE:
            success = self._init_speechbrain_diarization()
            if success:
                self.diarization_method = "speechbrain"
                return
            else:
                print("⚠️  SpeechBrain initialization failed, falling back to Pyannote")
        
        # Fallback to Pyannote
        if PYANNOTE_AVAILABLE:
            success = self._init_pyannote_diarization()
            if success:
                self.diarization_method = "pyannote"
                return
        
        # Final fallback to simple VAD
        print("📢 Speaker diarization: Using simple VAD fallback")
        self.diarization_method = "simple_vad"
    
    def _init_pyannote_diarization(self):
        """Initialize pyannote speaker diarization pipeline"""
        try:
            print("🎯 Loading pyannote.audio speaker diarization pipeline...")
            
            if not self.huggingface_token:
                print("⚠️  No Hugging Face token provided for pyannote models")
                print("   You need to:")
                print("   1. Create account at https://huggingface.co")
                print("   2. Accept conditions for pyannote/speaker-diarization-3.1")
                print("   3. Create access token at https://hf.co/settings/tokens")
                print("   4. Pass token via --huggingface-token argument")
                return False
            
            # Load the pyannote pipeline
            self.pyannote_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.huggingface_token
            )
            
            # Enable TF32 for better performance (after loading pipeline)
            import torch
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("🚀 TF32 enabled for better pyannote performance")
            
            # Move to appropriate device
            if self.device == "cuda" and torch.cuda.is_available():
                self.pyannote_pipeline.to(torch.device("cuda"))
                print("🚀 Moved pyannote pipeline to CUDA")
            
            print("✅ pyannote.audio speaker diarization pipeline loaded successfully")
            return True
            
        except Exception as e:
            print(f"⚠️  Failed to load pyannote pipeline: {e}")
            if "access" in str(e).lower() or "token" in str(e).lower():
                print("   This appears to be an authentication issue.")
                print("   Please ensure you have:")
                print("   1. Accepted the model conditions on Hugging Face")
                print("   2. Provided a valid access token")
            return False
    
    def _init_speechbrain_diarization(self):
        """Initialize SpeechBrain speaker diarization models (fallback)"""
        try:
            print("🔊 Loading SpeechBrain models for speaker diarization (fallback)...")
            
            # Import LocalStrategy for Windows fix
            from speechbrain.utils.fetching import LocalStrategy
            
            # Load speaker embedding model
            print("  Loading speaker embedding model...")
            self.speechbrain_speaker_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join(self.speechbrain_cache_dir, "spkrec-ecapa-voxceleb"),
                run_opts={"device": self.device},
                local_strategy=LocalStrategy.COPY,  # Use COPY instead of SYMLINK
            )
            
            # Load VAD model for voice activity detection
            print("  Loading VAD model...")
            try:
                self.speechbrain_vad_model = EncoderClassifier.from_hparams(
                    source="speechbrain/vad-crdnn-libriparty",
                    savedir=os.path.join(self.speechbrain_cache_dir, "vad-crdnn-libriparty"),
                    run_opts={"device": self.device},
                    local_strategy=LocalStrategy.COPY,  # Use COPY instead of SYMLINK
                )
                print("✅ SpeechBrain speaker diarization models loaded successfully")
            except Exception as vad_error:
                print(f"⚠️  VAD model failed to load: {vad_error}")
                print("   Will use librosa-based VAD instead")
                self.speechbrain_vad_model = None
            
            return True
                
        except Exception as e:
            print(f"⚠️  Failed to load SpeechBrain models: {e}")
            return False
    
    def detect_speaker_segments(self, audio, sr=16000):
        """
        Detect speaker segments using the best available method
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            List of tuples: [(start_time, end_time, speaker_label), ...]
        """
        print(f"🎤 Running speaker diarization with method: {self.diarization_method}")
        
        if self.diarization_method == "pyannote":
            return self._diarize_with_pyannote(audio, sr)
        elif self.diarization_method == "speechbrain":
            return self._diarize_with_speechbrain(audio, sr)
        else:
            return self._simple_speaker_detection(audio, sr)
    
    def _diarize_with_pyannote(self, audio, sr=16000):
        """Use pyannote for speaker diarization"""
        print("🎯 Running speaker diarization with pyannote.audio...")
        
        try:
            # Convert audio to the format expected by pyannote
            if sr != 16000:
                # Resample to 16kHz if needed
                num_samples = int(len(audio) * 16000 / sr)
                audio = signal.resample(audio, num_samples)
                sr = 16000
            
            # Create a temporary file for pyannote (it expects file input)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                # Write audio to temporary file
                wavfile.write(temp_file.name, sr, (audio * 32767).astype(np.int16))
                temp_filepath = temp_file.name
            
            try:
                # Run diarization
                diarization = self.pyannote_pipeline(temp_filepath)
                
                # Convert pyannote output to our format and create speaker mapping
                raw_segments = []
                speaker_labels = set()
                
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    start_time = turn.start
                    end_time = turn.end
                    raw_segments.append((start_time, end_time, speaker))
                    speaker_labels.add(speaker)
                
                # Create consistent speaker mapping
                speaker_mapping = self._create_speaker_mapping(speaker_labels)
                print(f"🗣️  Speaker mapping: {speaker_mapping}")
                
                # Apply mapping to segments
                segments = []
                for start_time, end_time, original_speaker in raw_segments:
                    mapped_speaker = speaker_mapping[original_speaker]
                    segments.append((start_time, end_time, mapped_speaker))
                
                # Clean up temporary file
                os.unlink(temp_filepath)
                
                print(f"🎯 Generated {len(segments)} speaker segments with pyannote")
                
                # Show speaker distribution
                speakers = set([seg[2] for seg in segments])
                print(f"🗣️  Final speakers: {', '.join(sorted(speakers))}")
                
                return segments
                
            except Exception as e:
                # Clean up temporary file on error
                if os.path.exists(temp_filepath):
                    os.unlink(temp_filepath)
                raise e
                
        except Exception as e:
            print(f"❌ pyannote diarization failed: {e}")
            print("   Falling back to SpeechBrain diarization")
            
            # Fallback to SpeechBrain if available
            if self.speechbrain_speaker_model is not None:
                self.diarization_method = "speechbrain"
                return self._diarize_with_speechbrain(audio, sr)
            else:
                print("   Falling back to simple speaker detection")
                self.diarization_method = "simple_vad"
                return self._simple_speaker_detection(audio, sr)
    
    def _create_speaker_mapping(self, pyannote_speakers):
        """
        Create consistent mapping from pyannote speaker IDs to Speaker 1, Speaker 2, etc.
        
        Args:
            pyannote_speakers: Set of original pyannote speaker labels
            
        Returns:
            dict: mapping from original to standardized labels
        """
        sorted_speakers = sorted(list(pyannote_speakers))
        mapping = {}
        
        for i, original_speaker in enumerate(sorted_speakers):
            if i < 2:  # Only map first 2 speakers for now
                mapping[original_speaker] = f"Speaker {i + 1}"
            else:
                # If more than 2 speakers, assign to closest existing speaker
                mapping[original_speaker] = f"Speaker {(i % 2) + 1}"
        
        return mapping
    
    def _diarize_with_speechbrain(self, audio, sr=16000):
        """Use SpeechBrain for speaker diarization (fallback method)"""
        print("🔊 Running speaker diarization with SpeechBrain (fallback)...")
        
        try:
            # Step 1: Voice Activity Detection
            if self.speechbrain_vad_model is not None:
                speech_segments = self._speechbrain_vad(audio, sr)
            else:
                speech_segments = self._librosa_vad(audio, sr)
            
            if not speech_segments:
                print("⚠️  No speech segments detected")
                return []
            
            print(f"🗣️  Found {len(speech_segments)} speech segments")
            
            # Step 2: Extract speaker embeddings for each segment
            embeddings = []
            valid_segments = []
            
            for i, (start_time, end_time) in enumerate(speech_segments):
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                segment_audio = audio[start_sample:end_sample]
                
                # Minimum segment length
                min_length = sr * 0.5  # 0.5 seconds minimum
                if len(segment_audio) < min_length:
                    continue
                
                try:
                    # Convert to tensor and get embedding
                    segment_tensor = torch.tensor(segment_audio).unsqueeze(0).float()
                    if self.device == "cuda":
                        segment_tensor = segment_tensor.cuda()
                    
                    # Get speaker embedding
                    with torch.no_grad():
                        embedding = self.speechbrain_speaker_model.encode_batch(segment_tensor)
                        embeddings.append(embedding.squeeze().cpu().numpy())
                        valid_segments.append((start_time, end_time))
                        
                except Exception as e:
                    print(f"    ⚠️  Failed to get embedding for segment {i}: {e}")
                    continue
            
            if len(embeddings) < 2:
                print("⚠️  Not enough valid segments for clustering")
                # Return single speaker for all segments
                return [(start, end, "Speaker 1") for start, end in valid_segments]
            
            # Step 3: Cluster embeddings to identify speakers
            embeddings = np.array(embeddings)
            speaker_labels = self._cluster_speakers_speechbrain(embeddings)
            
            # Step 4: Create speaker segments with consistent labeling
            segments = []
            for (start_time, end_time), speaker_id in zip(valid_segments, speaker_labels):
                speaker_label = f"Speaker {speaker_id + 1}"
                segments.append((start_time, end_time, speaker_label))
            
            print(f"🎯 Generated {len(segments)} speaker segments with SpeechBrain")
            
            # Show speaker distribution
            speakers = set([seg[2] for seg in segments])
            print(f"🗣️  Detected speakers: {', '.join(sorted(speakers))}")
            
            return segments
            
        except Exception as e:
            print(f"❌ SpeechBrain diarization failed: {e}")
            print("   Falling back to simple speaker detection")
            self.diarization_method = "simple_vad"
            return self._simple_speaker_detection(audio, sr)
    
    def _speechbrain_vad(self, audio, sr=16000):
        """Use SpeechBrain VAD to detect speech segments"""
        try:
            # Convert audio to tensor
            audio_tensor = torch.tensor(audio).unsqueeze(0).float()
            if self.device == "cuda":
                audio_tensor = audio_tensor.cuda()
            
            # Get VAD predictions
            with torch.no_grad():
                speech_prob = self.speechbrain_vad_model.encode_batch(audio_tensor)
                speech_prob = torch.sigmoid(speech_prob).squeeze().cpu().numpy()
            
            # Parameters for VAD
            threshold = 0.6
            min_speech_duration = 0.5
            min_silence_duration = 0.5
            
            # Convert probabilities to segments
            speech_frames = speech_prob > threshold
            
            # Apply morphological operations to clean up detection
            import scipy.ndimage
            # Fill small gaps
            speech_frames = scipy.ndimage.binary_closing(speech_frames, structure=np.ones(int(sr * 0.2)))
            # Remove very short segments
            speech_frames = scipy.ndimage.binary_opening(speech_frames, structure=np.ones(int(sr * min_speech_duration)))
            
            # Find speech segments
            segments = []
            in_speech = False
            start_time = 0
            
            frame_duration = len(audio) / len(speech_frames) / sr
            
            for i, is_speech in enumerate(speech_frames):
                current_time = i * frame_duration
                
                if is_speech and not in_speech:
                    start_time = current_time
                    in_speech = True
                elif not is_speech and in_speech:
                    end_time = current_time
                    segment_duration = end_time - start_time
                    
                    if segment_duration >= min_speech_duration:
                        segments.append((start_time, end_time))
                    in_speech = False
            
            # Handle case where audio ends during speech
            if in_speech:
                end_time = len(audio) / sr
                segment_duration = end_time - start_time
                if segment_duration >= min_speech_duration:
                    segments.append((start_time, end_time))
            
            # Post-process: merge segments that are too close together
            merged_segments = []
            for start, end in segments:
                if merged_segments and start - merged_segments[-1][1] < min_silence_duration:
                    merged_segments[-1] = (merged_segments[-1][0], end)
                else:
                    merged_segments.append((start, end))
            
            print(f"    📊 VAD detection: {len(segments)} raw → {len(merged_segments)} merged segments")
            return merged_segments
            
        except Exception as e:
            print(f"⚠️  SpeechBrain VAD failed: {e}, using librosa VAD")
            return self._librosa_vad(audio, sr)
    
    def _librosa_vad(self, audio, sr=16000):
        """Librosa-based Voice Activity Detection"""
        top_db = 25
        frame_length = 4096
        hop_length = 1024
        min_duration = 0.5
        min_gap = 0.5
        
        intervals = librosa.effects.split(
            audio, 
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        segments = []
        
        for start_sample, end_sample in intervals:
            start_time = start_sample / sr
            end_time = end_sample / sr
            duration = end_time - start_time
            
            if duration >= min_duration:
                segments.append((start_time, end_time))
        
        # Merge segments that are too close
        merged_segments = []
        for start, end in segments:
            if merged_segments and start - merged_segments[-1][1] < min_gap:
                merged_segments[-1] = (merged_segments[-1][0], end)
            else:
                merged_segments.append((start, end))
        
        print(f"    📊 Librosa VAD: {len(segments)} raw → {len(merged_segments)} merged segments")
        return merged_segments
    
    def _cluster_speakers_speechbrain(self, embeddings, max_speakers=4):
        """Cluster speaker embeddings for SpeechBrain"""
        n_samples = len(embeddings)
        
        if n_samples <= 1:
            return [0] * n_samples
        
        max_clusters = min(max_speakers, n_samples, 4)
        
        best_score = -1
        best_labels = None
        best_n_clusters = 2
        
        for n_clusters in range(2, max_clusters + 1):
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='ward',
                    distance_threshold=None
                )
                labels = clustering.fit_predict(embeddings)
                
                # Calculate silhouette score
                if len(set(labels)) > 1:
                    score = silhouette_score(embeddings, labels)
                    # Prefer fewer clusters
                    penalty = (n_clusters - 2) * 0.1
                    adjusted_score = score - penalty
                    
                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_labels = labels
                        best_n_clusters = n_clusters
                        
            except Exception as e:
                print(f"    ⚠️  Clustering with {n_clusters} clusters failed: {e}")
                continue
        
        if best_labels is None:
            # Fallback: simple distance-based clustering
            print("    🔄 Using distance-based clustering fallback")
            best_labels = self._simple_distance_clustering(embeddings, threshold=0.3)
        
        print(f"    📊 Selected {best_n_clusters} speakers (adjusted score: {best_score:.3f})")
        return best_labels
    
    def _simple_distance_clustering(self, embeddings, threshold=0.3):
        """Simple distance-based clustering"""
        labels = [0]  # First embedding gets label 0
        centroids = [embeddings[0]]
        
        for embedding in embeddings[1:]:
            # Find closest centroid
            distances = [distance.cosine(embedding, centroid) for centroid in centroids]
            min_distance = min(distances)
            
            if min_distance < threshold:
                # Assign to existing cluster
                labels.append(distances.index(min_distance))
            else:
                # Create new cluster if under limit
                new_label = len(centroids)
                if new_label < 4:  # Max 4 speakers
                    labels.append(new_label)
                    centroids.append(embedding)
                else:
                    # Force assignment to closest existing cluster
                    labels.append(distances.index(min_distance))
        
        return labels
    
    def _simple_speaker_detection(self, audio, sr=16000):
        """Simple VAD-based speaker detection fallback"""
        print("🎤 Using simple VAD-based speaker detection...")
        
        # Use librosa for voice activity detection
        top_db = 25
        intervals = librosa.effects.split(audio, top_db=top_db, frame_length=2048, hop_length=512)
        
        segments = []
        current_speaker = "Speaker 1"  # Start with first speaker
        
        # Simple heuristic: alternate speakers on longer pauses
        for i, (start_sample, end_sample) in enumerate(intervals):
            start_time = start_sample / sr
            end_time = end_sample / sr
            
            # Simple heuristic: alternate speakers on longer pauses
            if i > 0:
                prev_end = intervals[i-1][1] / sr
                pause_duration = start_time - prev_end
                
                # If pause is longer than 1.5 seconds, assume speaker change
                if pause_duration > 1.5:
                    current_speaker = "Speaker 2" if current_speaker == "Speaker 1" else "Speaker 1"
            
            segments.append((start_time, end_time, current_speaker))
        
        print(f"🎯 Generated {len(segments)} speaker segments using VAD")
        return segments
    
    def create_speaker_chunks(self, speaker_segments, min_duration=0.5):
        """
        Create chunks from speaker segments with improved auto speaker detection
        
        Args:
            speaker_segments: List of (start, end, speaker) tuples from diarization
            min_duration: Minimum duration in seconds for valid chunks
            
        Returns:
            Tuple: (valid_chunks, all_speaker_changes)
        """
        print(f"📝 Processing {len(speaker_segments)} speaker segments...")
        
        # Step 1: Merge consecutive segments from same speaker with gap tolerance
        merged_segments = self._merge_consecutive_speaker_segments_improved(speaker_segments, max_gap=2.0)
        print(f"📊 Merged {len(speaker_segments)} raw segments → {len(merged_segments)} merged segments")
        
        # Step 2: Filter by minimum duration and create valid chunks
        valid_chunks = []
        all_speaker_changes = []
        
        for i, (start_time, end_time, speaker_label) in enumerate(merged_segments):
            duration = end_time - start_time
            
            # Record all speaker changes for reporting
            all_speaker_changes.append({
                'segment_number': i + 1,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'original_speaker': speaker_label,
                'user_label': speaker_label,  # Use detected speaker directly
                'valid': duration >= min_duration,
                'reason': 'Valid segment' if duration >= min_duration else f'Too short ({duration:.2f}s < {min_duration}s)'
            })
            
            # Only keep segments that meet minimum duration
            if duration >= min_duration:
                valid_chunks.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'speaker': speaker_label,  # Use detected speaker directly
                    'original_speaker': speaker_label,
                    'segment_number': i + 1
                })
                print(f"  ✅ Segment {i+1}: [{start_time:.2f}s - {end_time:.2f}s] = {speaker_label} ({duration:.2f}s)")
            else:
                print(f"  🗑️  Segment {i+1}: [{start_time:.2f}s - {end_time:.2f}s] = {speaker_label} - IGNORED ({duration:.2f}s < {min_duration}s)")
        
        print(f"📊 Speaker chunking summary:")
        print(f"   Total segments: {len(speaker_segments)}")
        print(f"   Merged segments: {len(merged_segments)}")
        print(f"   Valid chunks: {len(valid_chunks)}")
        print(f"   Ignored (too short): {len(merged_segments) - len(valid_chunks)}")
        
        return valid_chunks, all_speaker_changes
    
    def _merge_consecutive_speaker_segments_improved(self, segments, max_gap=2.0):
        """
        Merge consecutive segments from the same speaker with improved logic
        
        Args:
            segments: List of (start, end, speaker) tuples
            max_gap: Maximum gap in seconds to merge across
            
        Returns:
            List of merged segments
        """
        if not segments:
            return segments
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x[0])
        
        merged = []
        current_start, current_end, current_speaker = sorted_segments[0]
        
        for start, end, speaker in sorted_segments[1:]:
            gap = start - current_end
            
            # Merge if same speaker and gap is small enough
            if speaker == current_speaker and gap <= max_gap:
                # Extend current segment
                current_end = end
                print(f"    🔗 Merging {current_speaker}: gap of {gap:.2f}s")
            else:
                # Finish current segment and start new one
                merged.append((current_start, current_end, current_speaker))
                current_start, current_end, current_speaker = start, end, speaker
        
        # Add the last segment
        merged.append((current_start, current_end, current_speaker))
        
        return merged
    
    def transcribe_speaker_chunk(self, audio, chunk_info, previous_context="", task="transcribe"):
        """Transcribe a single speaker chunk with improved confidence filtering"""
        chunk_start = int(chunk_info['start_time'] * 16000)
        chunk_end = int(chunk_info['end_time'] * 16000)
        chunk_audio = audio[chunk_start:chunk_end]
        
        speaker = chunk_info['speaker']
        duration = chunk_info['duration']
        
        print(f"🎤 Transcribing {speaker}: [{chunk_info['start_time']:.2f}s - {chunk_info['end_time']:.2f}s] ({duration:.2f}s)")
        
        # Apply language restriction if enabled
        forced_language = None
        if self.restrict_languages:
            detected_lang = self._detect_language_with_restriction(chunk_audio)
            forced_language = detected_lang
            print(f"  🗣️  Detected language: {detected_lang}")
        else:
            detected_lang = None
        
        # Enhanced transcription options with better confidence settings
        options = {
            "language": detected_lang,
            "task": task,
            "verbose": False,
            "word_timestamps": True,
            "temperature": 0.0,
            "beam_size": 5,  # Increased beam size for better accuracy
            "patience": 2.0,  # Reduced for better consistency
            "fp16": torch.cuda.is_available(),
            "condition_on_previous_text": self.enable_context,
            "compression_ratio_threshold": 2.0,  # More aggressive compression ratio threshold
            "logprob_threshold": -1.0,  # Stricter threshold to filter out garbled text
            "no_speech_threshold": 0.8,  # Higher threshold to avoid false positives
        }
        
        # Add context as initial prompt if available and enabled
        if self.enable_context and previous_context:
            options["initial_prompt"] = previous_context
            context_preview = previous_context[:50] + "..." if len(previous_context) > 50 else previous_context
            print(f"  🔗 Using context: '{context_preview}'")
        elif not self.enable_context:
            print(f"  🚫 Context disabled to prevent bleeding")
        
        try:
            # Transcribe chunk
            result = self.model.transcribe(chunk_audio, **options)
            
            # Get final detected language and validate
            final_detected_lang = result.get("language", detected_lang or "unknown")
            
            # Strict validation and correction for language restriction
            if self.restrict_languages and final_detected_lang not in self.supported_languages:
                print(f"  🚫 Language '{final_detected_lang}' violates restriction, correcting...")
                if final_detected_lang in ['ur', 'pa', 'bn', 'ta', 'te', 'ml', 'kn', 'gu', 'or', 'as', 'ne', 'si', 'mr']:
                    final_detected_lang = 'hi'
                else:
                    final_detected_lang = 'en'
                result['language'] = final_detected_lang
                print(f"  🔧 Corrected to: {final_detected_lang}")
            
            # Calculate chunk confidence
            chunk_confidence = self._get_chunk_confidence(result)
            
            # Enhanced confidence filtering with content quality checks
            if 'segments' in result and result['segments']:
                filtered_segments = []
                for segment in result['segments']:
                    segment_confidence = self._get_segment_confidence(segment)
                    segment_text = segment.get('text', '').strip()
                    
                    # Check if segment meets confidence threshold
                    if segment_confidence >= self.min_segment_confidence:
                        # Additional quality checks for obvious errors
                        if self._is_valid_segment_content(segment_text):
                            filtered_segments.append(segment)
                        else:
                            print(f"    🗑️  Filtering invalid content: '{segment_text[:30]}...'")
                    else:
                        print(f"    🗑️  Filtering low confidence segment: {segment_confidence:.3f}")
                
                result['segments'] = filtered_segments
                if not filtered_segments:
                    print(f"    ⚠️  All segments filtered due to low confidence/quality")
            
            # Apply existing postprocessing to this chunk
            original_text = result.get('text', '')
            corrected_text, corrections = self.postprocessor.apply_corrections(
                original_text, 
                chunk_confidence
            )
            
            result['text'] = corrected_text
            result['original_text'] = original_text
            result['postprocessing_corrections'] = corrections
            
            if corrections:
                print(f"  📝 Applied {len(corrections)} corrections")
            
            # Adjust timestamps and add metadata to segments
            if 'segments' in result and result['segments']:
                for segment in result['segments']:
                    segment['start'] += chunk_info['start_time']
                    segment['end'] += chunk_info['start_time']
                    segment['speaker'] = speaker
                    segment['chunk_language'] = final_detected_lang
                    segment['chunk_confidence'] = chunk_confidence
                    segment['original_speaker'] = chunk_info['original_speaker']
                    
                    # Apply corrections to segment text too
                    if 'text' in segment:
                        segment_original = segment['text']
                        segment_corrected, _ = self.postprocessor.apply_corrections(
                            segment_original, 
                            chunk_confidence
                        )
                        segment['text'] = segment_corrected
                        segment['original_text'] = segment_original
            
            # Add chunk metadata
            result['speaker'] = speaker
            result['chunk_confidence'] = chunk_confidence
            result['chunk_start_time'] = chunk_info['start_time']
            result['chunk_end_time'] = chunk_info['end_time']
            result['chunk_duration'] = duration
            result['original_speaker'] = chunk_info['original_speaker']
            
            sample_text = corrected_text[:100] + "..." if len(corrected_text) > 100 else corrected_text
            print(f"  📄 {speaker}: {sample_text}")
            print(f"  📊 Confidence: {chunk_confidence:.3f}")
            
            return result
            
        except Exception as e:
            print(f"  ❌ Failed to transcribe {speaker} chunk: {e}")
            return None
    
    def _is_valid_segment_content(self, text):
        """
        Check if segment content is valid (not garbled/repetitive text)
        
        Args:
            text (str): Text to validate
            
        Returns:
            bool: True if content seems valid
        """
        if not text or len(text.strip()) < 3:
            return False
        
        text = text.strip().lower()
        
        # Filter out common garbled patterns
        invalid_patterns = [
            "we'll be right back",
            "thank you for watching",
            "highly exhibited", 
            "thank you very much",
            "mm-hmm",
            "uh-huh",
            "all right",
            "thank you thank you thank you",  # Repetitive
        ]
        
        # Check for exact matches or very similar patterns
        for pattern in invalid_patterns:
            if pattern in text and len(text) < len(pattern) + 10:  # Close match
                return False
        
        # Check for excessive repetition
        words = text.split()
        if len(words) > 3:
            word_counts = Counter(words)
            most_common_count = word_counts.most_common(1)[0][1]
            if most_common_count > len(words) * 0.5:  # More than 50% repetition
                return False
        
        return True
    
    def _get_segment_confidence(self, segment):
        """Calculate confidence for a single segment"""
        if 'words' in segment and segment['words']:
            word_confidences = [word.get('probability', 0.0) for word in segment['words'] 
                             if 'probability' in word]
            return np.mean(word_confidences) if word_confidences else 0.0
        return 0.0
    
    def transcribe_with_speaker_diarization(self, audio, task="transcribe", min_chunk_duration=0.5):
        """Main transcription method using speaker diarization with improved accuracy"""
        print("🎤 Starting speaker-aware transcription...")
        print(f"📊 Diarization method: {self.diarization_method}")
        print(f"📊 Settings: context={'enabled' if self.enable_context else 'disabled'}, "
              f"min_confidence={self.min_segment_confidence}, "
              f"min_chunk_duration={min_chunk_duration}s")
        
        sr = 16000
        
        # Step 1: Detect speaker segments
        speaker_segments = self.detect_speaker_segments(audio, sr)
        
        if not speaker_segments:
            print("❌ No speaker segments detected, falling back to single transcription")
            return self.transcribe_single_pass_with_postprocessing(audio, language=None, task=task)
        
        # Step 2: Create valid chunks using improved auto speaker detection
        valid_chunks, all_speaker_changes = self.create_speaker_chunks(
            speaker_segments, 
            min_duration=min_chunk_duration
        )
        
        if not valid_chunks:
            print("❌ No valid chunks after filtering, falling back to single transcription")
            return self.transcribe_single_pass_with_postprocessing(audio, language=None, task=task)
        
        # Step 3: Transcribe each valid chunk
        all_segments = []
        all_corrections = []
        chunk_results = []
        languages_detected = set()
        previous_context = ""
        
        for chunk_info in valid_chunks:
            result = self.transcribe_speaker_chunk(
                audio, 
                chunk_info, 
                previous_context=previous_context if self.enable_context else "",
                task=task
            )
            
            if result:
                chunk_results.append(result)
                
                # Collect segments
                if 'segments' in result and result['segments']:
                    all_segments.extend(result['segments'])
                
                # Collect corrections
                corrections = result.get('postprocessing_corrections', [])
                all_corrections.extend(corrections)
                
                # Track languages
                lang = result.get('language', 'unknown')
                if self.restrict_languages and lang not in self.supported_languages:
                    lang = 'hi' if lang in ['ur', 'pa', 'bn', 'ta', 'te', 'ml', 'kn', 'gu', 'or', 'as', 'ne', 'si', 'mr'] else 'en'
                languages_detected.add(lang)
                
                # Update context for next chunk only if context is enabled
                if self.enable_context:
                    previous_context = self.postprocessor.get_context_text(
                        result.get('text', ''), 
                        max_words=self.context_words
                    )
        
        # Step 4: Combine results
        combined_text = ' '.join([result.get('text', '').strip() for result in chunk_results if result.get('text', '').strip()])
        
        # Calculate overall confidence
        confidences = [result.get('chunk_confidence', 0.0) for result in chunk_results if result.get('chunk_confidence', 0.0) > 0]
        overall_confidence = np.mean(confidences) if confidences else 0.0
        
        # Determine primary language
        lang_counts = {}
        for segment in all_segments:
            lang = segment.get('chunk_language', 'unknown')
            if self.restrict_languages and lang not in self.supported_languages:
                lang = 'hi' if lang in ['ur', 'pa', 'bn', 'ta', 'te', 'ml', 'kn', 'gu', 'or', 'as', 'ne', 'si', 'mr'] else 'en'
            lang_counts[lang] = lang_counts.get(lang, 0) + len(segment.get('text', ''))
        
        primary_language = max(lang_counts.keys(), key=lambda k: lang_counts[k]) if lang_counts else 'unknown'
        
        # Final validation of primary language
        if self.restrict_languages and primary_language not in self.supported_languages:
            primary_language = 'en'
        
        # Enhanced summary
        print(f"\n📊 Speaker-aware transcription completed:")
        print(f"   Diarization method: {self.diarization_method}")
        print(f"   Total speaker segments: {len(speaker_segments)}")
        print(f"   Valid chunks transcribed: {len(valid_chunks)}")
        print(f"   Total segments after filtering: {len(all_segments)}")
        print(f"   Languages detected: {', '.join(sorted(languages_detected))}")
        print(f"   Overall confidence: {overall_confidence:.3f}")
        print(f"   Total corrections applied: {len(all_corrections)}")
        print(f"   Auto speaker detection: Enabled")
        
        # Show speaker distribution
        speaker_distribution = {}
        for chunk in valid_chunks:
            speaker = chunk['speaker']
            speaker_distribution[speaker] = speaker_distribution.get(speaker, 0) + chunk['duration']
        
        print(f"   Speaker time distribution:")
        for speaker, duration in sorted(speaker_distribution.items()):
            print(f"     {speaker}: {duration:.1f}s")
        
        return {
            'text': combined_text,
            'segments': all_segments,
            'language': 'mixed' if len(languages_detected) > 1 else primary_language,
            'languages_detected': sorted(list(languages_detected)),
            'language_distribution': lang_counts,
            'primary_language': primary_language,
            'overall_confidence': overall_confidence,
            'postprocessing_corrections': all_corrections,
            'speaker_diarization_enabled': True,
            'context_enabled': self.enable_context,
            'context_words': self.context_words,
            'total_speaker_segments': len(speaker_segments),
            'valid_chunks': len(valid_chunks),
            'ignored_chunks': len(speaker_segments) - len(valid_chunks),
            'all_speaker_changes': all_speaker_changes,
            'chunk_results': chunk_results,
            'diarization_method': self.diarization_method,
            'min_segment_confidence': self.min_segment_confidence,
            'aggressive_segmentation': self.aggressive_segmentation,
            'speaker_distribution': speaker_distribution,
            'auto_speaker_detection': True
        }
    
    def load_audio(self, file_path, target_sr=16000):
        """Load audio file and convert to the target sample rate"""
        print(f"Loading audio file: {file_path}")
        
        methods = [
            ("scipy", self._load_with_scipy),
            ("soundfile", self._load_with_soundfile),
            ("librosa", self._load_with_librosa),
            ("pydub", self._load_with_pydub),
        ]
        
        for method_name, method_func in methods:
            try:
                print(f"Trying {method_name}...")
                audio, sr = method_func(file_path, target_sr)
                print(f"✅ Successfully loaded with {method_name}: {len(audio)/sr:.2f} seconds at {sr}Hz")
                return audio, sr
            except Exception as e:
                print(f"❌ {method_name} failed: {e}")
                continue
        
        raise RuntimeError(f"Failed to load audio file {file_path} with all available methods")
    
    def _load_with_scipy(self, file_path, target_sr):
        """Load with scipy (WAV files only)"""
        if not file_path.lower().endswith('.wav'):
            raise ValueError("Scipy method only supports WAV files")
        
        orig_sr, audio = wavfile.read(file_path)
        
        # Convert to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float32) - 128) / 128.0
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample if needed
        if orig_sr != target_sr:
            num_samples = int(len(audio) * target_sr / orig_sr)
            audio = signal.resample(audio, num_samples)
        
        return audio, target_sr
    
    def _load_with_soundfile(self, file_path, target_sr):
        """Load with soundfile directly"""
        import soundfile as sf
        audio, orig_sr = sf.read(file_path, dtype='float32')
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Simple resampling if needed
        if orig_sr != target_sr:
            num_samples = int(len(audio) * target_sr / orig_sr)
            audio = signal.resample(audio, num_samples)
        
        return audio, target_sr
    
    def _load_with_librosa(self, file_path, target_sr):
        """Load with librosa"""
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio, sr
    
    def _load_with_pydub(self, file_path, target_sr):
        """Load with pydub"""
        audio_segment = AudioSegment.from_file(file_path)
        audio_segment = audio_segment.set_channels(1).set_frame_rate(target_sr)
        audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        audio = audio / (2**15)
        return audio, target_sr
    
    def reduce_noise(self, audio, sr, noise_reduction_strength=0.6):
        """Apply noise reduction to audio"""
        print("Applying noise reduction...")
        reduced_noise = nr.reduce_noise(
            y=audio, 
            sr=sr,
            stationary=False,
            prop_decrease=noise_reduction_strength
        )
        return reduced_noise
    
    def preprocess_audio(self, audio, sr):
        """Additional preprocessing: normalize volume, trim silence"""
        print("Preprocessing audio...")
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=25)
        
        return audio
    
    def _detect_language_with_restriction(self, audio_chunk):
        """Language detection with restriction to English and Hindi"""
        try:
            sample_length = min(int(16000 * 3), len(audio_chunk))
            sample_audio = audio_chunk[:sample_length]
            
            if len(sample_audio) < 8000:
                return 'en'
            
            result = self.model.transcribe(
                sample_audio,
                task="transcribe",
                verbose=False,
                temperature=0.0,
                language=None,
                condition_on_previous_text=False,
                no_speech_threshold=0.8,
                logprob_threshold=-1.0,
            )
            
            detected_lang = result.get('language', 'en')
            
            if self.restrict_languages:
                if detected_lang in self.supported_languages:
                    return detected_lang
                elif detected_lang in ['ur', 'pa', 'bn', 'ta', 'te', 'ml', 'kn', 'gu', 'or', 'as', 'ne', 'si', 'mr']:
                    print(f"  🔄 Mapping {detected_lang} to Hindi due to language restriction")
                    return 'hi'
                else:
                    print(f"  🔄 Mapping {detected_lang} to English due to language restriction")
                    return 'en'
            else:
                return detected_lang
                
        except Exception as e:
            print(f"Language detection failed: {e}, defaulting to English")
            return 'en'
    
    def _get_chunk_confidence(self, result):
        """Calculate overall confidence for a transcription result"""
        if not result or 'segments' not in result:
            return 0.0
        
        segments = result['segments']
        if not segments:
            return 0.0
        
        total_confidence = 0.0
        word_count = 0
        
        for segment in segments:
            if 'words' in segment and segment['words']:
                for word in segment['words']:
                    if 'probability' in word:
                        total_confidence += word['probability']
                        word_count += 1
        
        return total_confidence / word_count if word_count > 0 else 0.0
    
    def transcribe_single_pass_with_postprocessing(self, audio, language=None, task="transcribe"):
        """Single-pass transcription with postprocessing (fallback method)"""
        try:
            forced_language = None
            if self.restrict_languages and language is None:
                detected_lang = self._detect_language_with_restriction(audio)
                print(f"🗣️  Detected language (restricted): {detected_lang}")
                language = detected_lang
                forced_language = detected_lang
            
            options = {
                "language": language,
                "task": task,
                "verbose": False,
                "word_timestamps": True,
                "temperature": 0.0,
                "best_of": 5,
                "beam_size": 5,
                "patience": 2.0,
                "fp16": torch.cuda.is_available(),
                "condition_on_previous_text": False,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6
            }
            
            if language is None:
                print("🔍 Auto-detecting language...")
            
            result = self.model.transcribe(audio, **options)
            
            final_detected_lang = result.get("language", "unknown")
            print(f"🗣️  Final language: {final_detected_lang}")
            
            if self.restrict_languages and final_detected_lang not in self.supported_languages:
                if final_detected_lang in ['ur', 'pa', 'bn', 'ta', 'te', 'ml', 'kn', 'gu', 'or', 'as', 'ne', 'si', 'mr']:
                    result['language'] = 'hi'
                    final_detected_lang = 'hi'
                else:
                    result['language'] = 'en'
                    final_detected_lang = 'en'
                print(f"🔧 Corrected language to: {final_detected_lang}")
            
            # Apply postprocessing
            original_text = result.get('text', '')
            chunk_confidence = self._get_chunk_confidence(result)
            
            corrected_text, corrections = self.postprocessor.apply_corrections(
                original_text, 
                chunk_confidence
            )
            
            result['text'] = corrected_text
            result['original_text'] = original_text
            result['postprocessing_corrections'] = corrections
            
            if corrections:
                print(f"📝 Applied {len(corrections)} postprocessing corrections")
            
            # Add confidence information
            result['chunk_confidence'] = chunk_confidence
            result['overall_confidence'] = chunk_confidence
            result['speaker_diarization_enabled'] = False
            result['diarization_method'] = 'none'
            
            return result
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return None
    
    def save_speaker_aware_transcription(self, result, output_file, include_metadata=True, include_word_timestamps=True):
        """Save speaker-aware transcription to text file with comprehensive information"""
        with open(output_file, 'w', encoding='utf-8') as f:
            if include_metadata:
                f.write("=" * 80 + "\n")
                f.write("SPEAKER-AWARE TRANSCRIPTION WITH PYANNOTE PRIMARY + SPEECHBRAIN FALLBACK\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {self.model_size}\n")
                f.write(f"Device: {self.device}\n")
                f.write(f"Language restriction: {'English and Hindi only' if self.restrict_languages else 'All languages'}\n")
                f.write(f"Primary language: {result.get('language', 'unknown')}\n")
                f.write(f"Speaker diarization: {'Enabled' if result.get('speaker_diarization_enabled') else 'Disabled'}\n")
                f.write(f"Diarization method: {result.get('diarization_method', 'N/A')}\n")
                f.write(f"Context enabled: {result.get('context_enabled', False)}\n")
                f.write(f"Context words: {result.get('context_words', 0)}\n")
                f.write(f"Min segment confidence: {result.get('min_segment_confidence', 0.0)}\n")
                f.write(f"Aggressive segmentation: {result.get('aggressive_segmentation', False)}\n")
                f.write(f"Auto speaker detection: {result.get('auto_speaker_detection', False)}\n")
                
                # Speaker diarization summary
                if result.get('speaker_diarization_enabled'):
                    f.write(f"Total speaker segments detected: {result.get('total_speaker_segments', 0)}\n")
                    f.write(f"Valid chunks transcribed: {result.get('valid_chunks', 0)}\n")
                    f.write(f"Ignored chunks (too short): {result.get('ignored_chunks', 0)}\n")
                    
                    # Speaker time distribution
                    if 'speaker_distribution' in result:
                        f.write(f"Speaker time distribution:\n")
                        for speaker, duration in sorted(result['speaker_distribution'].items()):
                            f.write(f"  {speaker}: {duration:.1f}s\n")
                
                # Postprocessing information
                corrections = result.get('postprocessing_corrections', [])
                f.write(f"Postprocessing corrections applied: {len(corrections)}\n")
                
                # Overall confidence
                if 'overall_confidence' in result:
                    f.write(f"Overall confidence: {result['overall_confidence']:.3f}\n")
                
                # Language information
                if 'languages_detected' in result:
                    f.write(f"All languages detected: {', '.join(result['languages_detected'])}\n")
                    
                if 'language_distribution' in result:
                    f.write(f"Language distribution (by character count):\n")
                    total_chars = sum(result['language_distribution'].values())
                    for lang, count in result['language_distribution'].items():
                        percentage = (count / total_chars * 100) if total_chars > 0 else 0
                        f.write(f"  {lang}: {count} chars ({percentage:.1f}%)\n")
                
                f.write(f"Total segments: {len(result.get('segments', []))}\n")
                f.write("=" * 80 + "\n\n")
            
            f.write("FULL TRANSCRIPTION BY SPEAKER:\n")
            f.write("-" * 50 + "\n")
            
            current_speaker = None
            for segment in result.get("segments", []):
                speaker = segment.get('speaker', 'Unknown')
                text = segment.get('text', '').strip()
                
                if speaker != current_speaker:
                    f.write(f"\n[{speaker}]:\n")
                    current_speaker = speaker
                
                f.write(f"{text} ")
            
            f.write("\n\n")
            
            # Alternative format: chronological with timestamps
            f.write("CHRONOLOGICAL TRANSCRIPTION WITH TIMESTAMPS:\n")
            f.write("-" * 60 + "\n")
            
            for segment in result.get("segments", []):
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                speaker = segment.get('speaker', 'Unknown')
                text = segment.get('text', '').strip()
                confidence = segment.get('chunk_confidence', 0.0)
                
                f.write(f"[{start:6.2f}s - {end:6.2f}s] [{speaker}] [Conf: {confidence:.3f}]: {text}\n")
            
            f.write("\n")
        
        print(f"📄 Speaker-aware transcription saved to: {output_file}")
    
    def process_audio_file(self, input_file, output_file=None, noise_reduction=True, 
                          noise_strength=0.6, language=None, task="transcribe", 
                          min_chunk_duration=0.5):
        """
        Complete pipeline: load, process, transcribe with speaker diarization, and save
        """
        print(f"🎵 Processing: {input_file}")
        
        # Generate output filename if not provided
        if output_file is None:
            input_path = Path(input_file)
            suffix = f"transcription_{self.model_size.replace('-', '_')}_pyannote_primary"
            if self.restrict_languages:
                suffix += "_restricted_en_hi"
            if self.diarization_method:
                suffix += f"_{self.diarization_method}"
            if not self.enable_context:
                suffix += "_no_context"
            if self.aggressive_segmentation:
                suffix += "_aggressive"
            suffix += "_auto_speaker"
            output_file = input_path.parent / f"{input_path.stem}_{suffix}.txt"
        
        try:
            # Step 1: Load audio
            audio, sr = self.load_audio(input_file)
            
            # Step 2: Apply noise reduction if requested
            if noise_reduction:
                audio = self.reduce_noise(audio, sr, noise_strength)
            
            # Step 3: Preprocess audio (normalize, trim)
            audio = self.preprocess_audio(audio, sr)
            
            # Step 4: Transcribe with speaker diarization
            result = self.transcribe_with_speaker_diarization(
                audio, 
                task=task, 
                min_chunk_duration=min_chunk_duration
            )
            
            if result:
                # Step 5: Save transcription with speaker information
                self.save_speaker_aware_transcription(result, output_file)
                print("✅ Speaker-aware processing completed successfully!")
                
                # Enhanced completion summary
                print(f"🎤 Diarization method used: {result.get('diarization_method')}")
                if result.get('speaker_diarization_enabled'):
                    print(f"📊 Speaker segments: {result.get('total_speaker_segments')} total, {result.get('valid_chunks')} transcribed, {result.get('ignored_chunks')} ignored")
                
                if 'languages_detected' in result:
                    print(f"🌐 Languages found: {', '.join(result['languages_detected'])}")
                
                if 'overall_confidence' in result:
                    print(f"📊 Overall confidence: {result['overall_confidence']:.3f}")
                
                # Postprocessing summary
                corrections = result.get('postprocessing_corrections', [])
                if corrections:
                    print(f"📝 Postprocessing corrections applied: {len(corrections)}")
                
                # Context usage summary
                if result.get('context_enabled'):
                    print(f"🔗 Context enabled: {result.get('context_words', 0)} words between speakers")
                else:
                    print(f"🚫 Context disabled to prevent bleeding")
                
                # Auto speaker detection summary
                if result.get('auto_speaker_detection'):
                    print(f"🎯 Auto speaker detection: Enabled")
                    if 'speaker_distribution' in result:
                        print(f"📊 Speaker distribution:")
                        for speaker, duration in sorted(result['speaker_distribution'].items()):
                            print(f"     {speaker}: {duration:.1f}s")
                
                return output_file
            else:
                print("❌ Transcription failed")
                return None
                
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_model_info(self):
        """Get information about the loaded model and speaker diarization setup"""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "context_words": self.context_words,
            "context_enabled": self.enable_context,
            "min_segment_confidence": self.min_segment_confidence,
            "aggressive_segmentation": self.aggressive_segmentation,
            "parameters": f"~{self.get_parameter_count()}",
            "languages_supported": "English and Hindi only" if self.restrict_languages else "99+ languages including Hindi, English",
            "primary_diarization": "pyannote.audio" if PYANNOTE_AVAILABLE else "Not available",
            "fallback_diarization": "SpeechBrain" if SPEECHBRAIN_AVAILABLE else "Simple VAD only",
            "current_diarization_method": self.diarization_method,
            "confidence_scoring": "Enabled - chunk and segment level with filtering",
            "language_restriction": "Enabled" if self.restrict_languages else "Disabled",
            "postprocessing": f"Enabled - {len(self.postprocessor.correction_rules)} rules loaded",
            "context_bleeding_prevention": "Enabled" if not self.enable_context else "Disabled",
            "chunking_method": "Speaker-based (dynamic duration)",
            "huggingface_token_provided": "Yes" if self.huggingface_token else "No",
            "auto_speaker_detection": "Enabled - uses actual diarization results"
        }
    
    def get_parameter_count(self):
        """Estimate parameter count based on model size"""
        param_counts = {
            "tiny": "39M",
            "base": "74M", 
            "small": "244M",
            "medium": "769M",
            "large": "1550M",
            "large-v2": "1550M",
            "large-v3": "1550M"
        }
        return param_counts.get(self.model_size, "Unknown")

def main():
    parser = argparse.ArgumentParser(description='Speaker-aware transcriber with pyannote primary + SpeechBrain fallback')
    parser.add_argument('input_file', nargs='?', help='Input audio file path')
    parser.add_argument('--huggingface-token', required=False, default=None,
                       help='Hugging Face access token for pyannote models (required for pyannote diarization)')
    parser.add_argument('--postprocessing-rules', nargs='?', 
                       help='Path to JSON file containing postprocessing rules')
    parser.add_argument('--speechbrain-cache-dir', default='./models/speechbrain',
                       help='Directory to cache SpeechBrain models locally (default: ./models/speechbrain)')
    parser.add_argument('-o', '--output', help='Output text file path')
    parser.add_argument('-m', '--model', default='large-v3',
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                       help='Whisper model size (default: large-v3)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], 
                       help='Device to run model on (auto-detect if not specified)')
    parser.add_argument('--min-chunk-duration', type=float, default=0.5,
                       help='Minimum duration in seconds for valid speaker chunks (default: 0.5)')
    parser.add_argument('--context-words', type=int, default=10,
                       help='Number of words from previous speaker to use as context (default: 10)')
    parser.add_argument('--disable-context', action='store_true',
                       help='Disable context to prevent bleeding between speakers (recommended for better accuracy)')
    parser.add_argument('--min-segment-confidence', type=float, default=0.4,
                       help='Minimum confidence threshold for including segments (default: 0.4)')
    parser.add_argument('--aggressive-segmentation', action='store_true',
                       help='Use more aggressive speaker segmentation for better speaker separation')
    parser.add_argument('--no-noise-reduction', action='store_true', 
                       help='Skip noise reduction step')
    parser.add_argument('--noise-strength', type=float, default=0.6,
                       help='Noise reduction strength (0.0-1.0, default: 0.6)')
    parser.add_argument('--language', 
                       help='Language code (hi=Hindi, en=English, None=auto-detect for mixed languages)')
    parser.add_argument('--task', choices=['transcribe', 'translate'], default='transcribe',
                       help='Task: transcribe (keep original language) or translate (to English)')
    parser.add_argument('--restrict-languages', type=bool, default=True,
                       help='Restrict language detection to English and Hindi only')
    parser.add_argument('--info', action='store_true',
                       help='Show model information and exit')
    
    args = parser.parse_args()
    
    # Load postprocessing rules if provided
    postprocessing_rules = None
    if args.postprocessing_rules and os.path.exists(args.postprocessing_rules):
        postprocessing_rules = args.postprocessing_rules
        print(f"📝 Loading postprocessing rules from: {args.postprocessing_rules}")
    
    # Initialize speaker-aware transcriber with improved settings
    print("🚀 Initializing Speaker-Aware Transcriber with Auto Speaker Detection...")
    transcriber = SpeakerAwareTranscriber(
        model_size=args.model, 
        device=args.device,
        restrict_languages=args.restrict_languages,
        postprocessing_rules=postprocessing_rules,
        context_words=args.context_words,
        speechbrain_cache_dir=args.speechbrain_cache_dir,
        enable_context=not args.disable_context,  # Context is disabled if flag is set
        min_segment_confidence=args.min_segment_confidence,
        aggressive_segmentation=args.aggressive_segmentation,
        huggingface_token=args.huggingface_token
    )
    
    # Show model info if requested
    if args.info:
        info = transcriber.get_model_info()
        print("\n📋 Model Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        return
    
    # Check if input file is provided
    if not args.input_file:
        print("❌ Please provide an input audio file")
        print("Usage: python improved_transcriber.py your_audio.wav --huggingface-token YOUR_TOKEN")
        print("       python improved_transcriber.py your_audio.wav --huggingface-token YOUR_TOKEN --disable-context")
        print("       python improved_transcriber.py your_audio.wav --huggingface-token YOUR_TOKEN --min-chunk-duration 0.5")
        print()
        print("Key improvements:")
        print("- Auto speaker detection using actual diarization results")
        print("- Better segment merging with 2-second gap tolerance")
        print("- Enhanced confidence filtering (default 0.5)")
        print("- Content quality checks to filter garbled text")
        print("- Improved speaker consistency")
        print()
        print("To get pyannote working:")
        print("1. Create account at https://huggingface.co")
        print("2. Accept conditions for pyannote/speaker-diarization-3.1")
        print("3. Create access token at https://hf.co/settings/tokens")
        print("4. Pass token via --huggingface-token argument")
        return
    
    # Process the file with improved speaker diarization
    result = transcriber.process_audio_file(
        input_file=args.input_file,
        output_file=args.output,
        noise_reduction=not args.no_noise_reduction,
        noise_strength=args.noise_strength,
        language=args.language,
        task=args.task,
        min_chunk_duration=args.min_chunk_duration
    )
    
    if result:
        print(f"✅ Speaker-aware transcription completed!")
        print(f"📄 Saved to: {result}")
        
        # Show quick stats
        if os.path.exists(result):
            with open(result, 'r', encoding='utf-8') as f:
                content = f.read()
                word_count = len(content.split())
                print(f"📊 Output stats: {word_count} words, {len(content)} characters")
    else:
        print("❌ Processing failed")

# Example usage
if __name__ == "__main__":
    # For command line usage
    main()
    
    # For programmatic usage:hf_
    """
    # Example with improved speaker detection
    postprocessing_rules = {
        "correction_rules": [
            {
                "previous_word": "sign",
                "replace_word": "surrender", 
                "confidence_threshold": 0.75
            }
        ]
    }
    
    # Initialize transcriber with auto speaker detection
    transcriber = SpeakerAwareTranscriber(
        model_size="large-v3", 
        device="cuda", 
        restrict_languages=True,
        postprocessing_rules=postprocessing_rules,
        context_words=30,
        enable_context=False,  # Disable context to prevent bleeding
        min_segment_confidence=0.5,  # Higher confidence threshold
        aggressive_segmentation=False,  # Disabled for better merging
        huggingface_token=<your_huggingface_token>
    )
    
    # Process file with auto speaker detection and improved settings
    result = transcriber.process_audio_file(
        input_file="path/to/conversation.wav",
        min_chunk_duration=0.5,  # 0.5 second minimum
        noise_reduction=True,
        task="transcribe"
    )
    
    # Key improvements:
    # 1. Auto speaker detection using pyannote's actual speaker IDs
    # 2. Improved segment merging with gap tolerance (2 seconds)
    # 3. Enhanced confidence filtering to remove garbled text
    # 4. Content quality checks to filter obvious errors
    # 5. Better speaker consistency across segments
    # 6. Removed toggle logic in favor of actual diarization results
    # 7. Stricter logprob threshold (-0.8) to filter poor quality segments
    """