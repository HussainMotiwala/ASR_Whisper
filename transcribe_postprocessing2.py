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
warnings.filterwarnings("ignore", category=FutureWarning)

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
        if not text or max_words == 0:  # Add this line
            return "" # Return empty string if text is empty or max_words is 0
                
        words = text.strip().split()
        
        return " ".join(words[-max_words:])

class LocalAudioTranscriber:
    def __init__(self, model_size="large-v3", device=None, chunk_duration=15, restrict_languages=True, 
                 postprocessing_rules=None, context_words=0):
        """
        Initialize the AudioTranscriber with local Whisper model and postprocessing
        
        Args:
            model_size (str): Whisper model size
            device (str): Device to run model on
            chunk_duration (int): Duration in seconds for processing chunks
            restrict_languages (bool): Restrict to English and Hindi only
            postprocessing_rules (dict or str): Postprocessing rules (dict or path to JSON file)
            context_words (int): Number of words from previous chunk to use as context
        """
        print(f"Loading Whisper model: {model_size}")
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.chunk_duration = chunk_duration
        self.restrict_languages = restrict_languages
        self.context_words = context_words
        
        # Initialize postprocessor
        if isinstance(postprocessing_rules, str):
            self.postprocessor = PostProcessor(rules_file=postprocessing_rules)
        elif isinstance(postprocessing_rules, dict):
            self.postprocessor = PostProcessor(rules_dict=postprocessing_rules)
        else:
            self.postprocessor = PostProcessor()  # No rules
        
        print(f"Using device: {device}")
        print(f"Chunk duration for mixed-language processing: {chunk_duration}s")
        print(f"Context words for chunk processing: {context_words}")
        
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
        
    def load_audio(self, file_path, target_sr=16000):
        """
        Load audio file and convert to the target sample rate with multiple fallback methods
        """
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
    
    def _filter_low_confidence_segments(self, segments, min_confidence=0.01):
        """Filter out segments with very low confidence"""
        filtered_segments = []
        
        for i, segment in enumerate(segments):  # ✅ Fixed: Added enumerate to get index
            if 'words' in segment and segment['words']:
                word_confidences = [w.get('probability', 0.0) for w in segment['words'] if 'probability' in w]
                if word_confidences:
                    segment_confidence = np.mean(word_confidences)
                    if segment_confidence >= min_confidence:
                        filtered_segments.append(segment)
                    else:
                        print(f"    🗑️  Filtered segment {i}: '{segment.get('text', '')[:50]}...' (conf: {segment_confidence:.3f})")
                else:
                    filtered_segments.append(segment)
            else:
                filtered_segments.append(segment)
        
        return filtered_segments
    
    def _is_valid_transcription(self, result):
        """Filter out repetitive or low-quality transcriptions"""
        text = result.get('text', '').strip()
        
        if not text:
            return False
        
        # Check for excessive repetition
        words = text.split()
        if len(words) > 5:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            max_repeat = max(word_counts.values())
            if max_repeat / len(words) > 0.60:  # More than 60% repetition
                return False
        
        # Check compression ratio
        compression_ratio = result.get('compression_ratio', 0)
        if compression_ratio > 2.4:
            return False
            
        return True
    
    def _get_segment_confidence(self, segment):
        """Calculate average confidence for a segment"""
        if 'words' not in segment:
            return 0.5
        
        words = segment['words']
        if not words:
            return 0.5
        
        total_confidence = sum(word.get('probability', 0.5) for word in words)
        return total_confidence / len(words)
    

    def _remove_overlapping_segments(self, segments, overlap_threshold=0.7, confidence_threshold=0.05, 
                                    similarity_threshold=0.8, min_overlap_duration=0.5):
        """
        Remove duplicate segments from overlapping chunks with improved logic and logging
        
        Args:
            segments: List of segments to process
            overlap_threshold: Percentage overlap to consider "significant" (default: 0.7)
            confidence_threshold: Minimum confidence difference to prefer one segment (default: 0.05)
            similarity_threshold: Text similarity threshold for near-duplicates (default: 0.8)
            min_overlap_duration: Minimum overlap duration in seconds to consider (default: 0.5)
        """
        if not segments:
            return segments
        
        print(f"🔍 Processing {len(segments)} segments for overlap removal...")
        
        # Sort segments by start time
        segments.sort(key=lambda x: x['start'])
        
        filtered_segments = []
        removed_segments = []
        
        for i, segment in enumerate(segments):
            segment_start = segment['start']
            segment_end = segment['end']
            segment_duration = segment_end - segment_start
            segment_text = segment.get('text', '').strip()
            
            if not filtered_segments:
                # First segment, always add
                filtered_segments.append(segment)
                continue
            
            # Check for overlap with the last added segment
            last_segment = filtered_segments[-1]
            last_start = last_segment['start']
            last_end = last_segment['end']
            last_duration = last_end - last_start
            last_text = last_segment.get('text', '').strip()
            
            # Calculate overlap
            overlap_start = max(segment_start, last_start)
            overlap_end = min(segment_end, last_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            # Calculate overlap percentages
            overlap_pct_current = (overlap_duration / segment_duration) if segment_duration > 0 else 0
            overlap_pct_last = (overlap_duration / last_duration) if last_duration > 0 else 0
            
            print(f"  Segment {i}: [{segment_start:.2f}-{segment_end:.2f}] vs Last: [{last_start:.2f}-{last_end:.2f}]")
            print(f"    Overlap: {overlap_duration:.2f}s ({overlap_pct_current:.1%} of current, {overlap_pct_last:.1%} of last)")
            print(f"    Current: '{segment_text[:50]}...'")
            print(f"    Last: '{last_text[:50]}...'")
            
            # Determine if this is a significant overlap that needs handling
            if overlap_duration > min_overlap_duration and (overlap_pct_current > overlap_threshold or overlap_pct_last > overlap_threshold):
                # Significant overlap detected - need to decide which to keep
                
                # Get confidence scores
                current_confidence = segment.get('chunk_confidence', self._get_segment_confidence(segment))
                last_confidence = last_segment.get('chunk_confidence', self._get_segment_confidence(last_segment))
                
                # Multiple criteria for deciding which segment to keep:
                
                # 1. Check for exact text duplicates first
                if segment_text.lower() == last_text.lower():
                    print(f"    🗑️  EXACT DUPLICATE: Keeping higher confidence segment")
                    if current_confidence > last_confidence:
                        removed_segments.append(filtered_segments.pop())  # Remove last
                        filtered_segments.append(segment)  # Add current
                        print(f"    ✅ Replaced with current (conf: {current_confidence:.3f} > {last_confidence:.3f})")
                    else:
                        removed_segments.append(segment)  # Skip current
                        print(f"    ✅ Kept last (conf: {last_confidence:.3f} >= {current_confidence:.3f})")
                    continue
                
                # 2. Check text similarity (rough measure)
                text_similarity = self._calculate_text_similarity(segment_text, last_text)
                
                if text_similarity > similarity_threshold:  # Very similar text
                    print(f"    🔄 SIMILAR TEXT (similarity: {text_similarity:.2f}): Using confidence to decide")
                    confidence_diff = abs(current_confidence - last_confidence)
                    
                    if confidence_diff > confidence_threshold:  # Significant confidence difference
                        if current_confidence > last_confidence:
                            removed_segments.append(filtered_segments.pop())
                            filtered_segments.append(segment)
                            print(f"    ✅ Replaced with current (conf: {current_confidence:.3f} >> {last_confidence:.3f})")
                        else:
                            removed_segments.append(segment)
                            print(f"    ✅ Kept last (conf: {last_confidence:.3f} >> {current_confidence:.3f})")
                    else:
                        # Similar confidence - keep the longer segment
                        if segment_duration > last_duration:
                            removed_segments.append(filtered_segments.pop())
                            filtered_segments.append(segment)
                            print(f"    ✅ Replaced with current (longer: {segment_duration:.2f}s > {last_duration:.2f}s)")
                        else:
                            removed_segments.append(segment)
                            print(f"    ✅ Kept last (longer: {last_duration:.2f}s >= {segment_duration:.2f}s)")
                    continue
                
                # 3. Partial overlap with different content - try to merge or keep both
                elif overlap_pct_current < 0.9 and overlap_pct_last < 0.9:
                    # Not complete overlap - might be legitimate partial overlap
                    print(f"    🔗 PARTIAL OVERLAP with different content - attempting smart handling")
                    
                    # If one segment is much shorter and mostly overlapped, remove it
                    if overlap_pct_current > 0.8 and segment_duration < 2.0:
                        removed_segments.append(segment)
                        print(f"    🗑️  Removed short overlapping current segment")
                        continue
                    elif overlap_pct_last > 0.8 and last_duration < 2.0:
                        removed_segments.append(filtered_segments.pop())
                        filtered_segments.append(segment)
                        print(f"    🗑️  Removed short overlapping last segment, added current")
                        continue
                    
                    # Try to merge if the segments seem complementary
                    merged_segment = self._try_merge_segments(last_segment, segment)
                    if merged_segment:
                        removed_segments.append(filtered_segments.pop())  # Remove last
                        filtered_segments.append(merged_segment)  # Add merged
                        print(f"    🔗 MERGED segments into one")
                        continue
                    
                    # If can't merge, keep both but warn
                    filtered_segments.append(segment)
                    print(f"    ⚠️  KEEPING BOTH segments (partial overlap with different content)")
                    continue
                
                # 4. High overlap - use confidence as tiebreaker
                else:
                    print(f"    🎯 HIGH OVERLAP: Using confidence as tiebreaker")
                    confidence_diff = abs(current_confidence - last_confidence)
                    
                    # Only replace if current is significantly better
                    if current_confidence > last_confidence + confidence_threshold:  # Configurable confidence threshold
                        removed_segments.append(filtered_segments.pop())
                        filtered_segments.append(segment)
                        print(f"    ✅ Replaced with current (conf: {current_confidence:.3f} > {last_confidence:.3f})")
                    else:
                        removed_segments.append(segment)
                        print(f"    ✅ Kept last (conf: {last_confidence:.3f} >= {current_confidence:.3f})")
                    continue
            
            else:
                # No significant overlap - keep both segments
                filtered_segments.append(segment)
                if overlap_duration > 0:
                    print(f"    ✅ MINOR OVERLAP: Keeping both segments")
                else:
                    print(f"    ✅ NO OVERLAP: Adding segment")
        
        # Summary
        removed_count = len(removed_segments)
        if removed_count > 0:
            print(f"📊 Overlap removal summary:")
            print(f"   Original segments: {len(segments)}")
            print(f"   Removed segments: {removed_count}")
            print(f"   Final segments: {len(filtered_segments)}")
            
            # Show what was removed
            if removed_count <= 5:  # Don't spam if too many
                print(f"   Removed content:")
                for i, removed in enumerate(removed_segments[:5]):
                    removed_text = removed.get('text', '').strip()
                    print(f"     {i+1}. [{removed['start']:.2f}-{removed['end']:.2f}s]: '{removed_text[:60]}...'")
        else:
            print(f"✅ No overlapping segments removed")
        
        return filtered_segments

    def _calculate_text_similarity(self, text1, text2):
        """
        Calculate rough text similarity between two strings
        """
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

    def _try_merge_segments(self, segment1, segment2):
        """
        Try to intelligently merge two overlapping segments
        """
        # Only merge if they're from the same chunk and language
        if (segment1.get('chunk_number') != segment2.get('chunk_number') or 
            segment1.get('chunk_language') != segment2.get('chunk_language')):
            return None
        
        text1 = segment1.get('text', '').strip()
        text2 = segment2.get('text', '').strip()
        
        # Simple merge strategy: if one text contains the other, use the longer one
        if text1 in text2:
            # text2 contains text1, use text2
            merged = segment2.copy()
            merged['start'] = min(segment1['start'], segment2['start'])
            merged['end'] = max(segment1['end'], segment2['end'])
            merged['text'] = text2
            merged['merged_from'] = [segment1.get('start'), segment2.get('start')]
            return merged
        elif text2 in text1:
            # text1 contains text2, use text1
            merged = segment1.copy()
            merged['start'] = min(segment1['start'], segment2['start'])
            merged['end'] = max(segment1['end'], segment2['end'])
            merged['text'] = text1
            merged['merged_from'] = [segment1.get('start'), segment2.get('start')]
            return merged
        
        # Could add more sophisticated merging logic here
        return None

    def _get_segment_confidence(self, segment):
        """Calculate average confidence for a segment with better handling"""
        if 'words' not in segment or not segment['words']:
            return segment.get('chunk_confidence', 0.5)  # Use chunk confidence as fallback
        
        words = segment['words']
        confidences = [word.get('probability', 0.5) for word in words if 'probability' in word]
        
        if not confidences:
            return segment.get('chunk_confidence', 0.5)
        
        return sum(confidences) / len(confidences)
    
    def transcribe_with_local_whisper(self, audio, language=None, task="transcribe", force_chunking=False):
        """Transcribe audio using local Whisper model with postprocessing and context"""
        print("Transcribing with local Whisper...")
        print(f"Audio length: {len(audio)/16000:.2f} seconds")
        
        if self.restrict_languages:
            print("🔒 Language restriction: English and Hindi only")
        
        # For mixed language or longer audio, use chunk processing with context
        if (len(audio) > 30 * 16000 or force_chunking) and language is None:
            print("🔀 Using chunk-based processing with postprocessing and context")
            return self.transcribe_mixed_language_chunks_with_context(audio, task)
        else:
            return self.transcribe_single_pass_with_postprocessing(audio, language, task)
    
    def transcribe_single_pass_with_postprocessing(self, audio, language=None, task="transcribe"):
        """Single-pass transcription with postprocessing"""
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
                "beam_size": 5,
                "patience": 2.0,
                "fp16": torch.cuda.is_available(),
                "condition_on_previous_text": False,
                "compression_ratio_threshold": 2.0,
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
            
            # Filter low confidence segments
            if 'segments' in result:
                result['segments'] = self._filter_low_confidence_segments(result['segments'])
            
            # Add confidence information
            result['chunk_confidence'] = chunk_confidence
            result['chunk_number'] = 1
            result['total_chunks'] = 1
            result['overall_confidence'] = chunk_confidence
            result['chunk_confidences'] = [{
                'chunk_number': 1,
                'start_time': 0.0,
                'end_time': len(audio) / 16000,
                'confidence': chunk_confidence,
                'language': final_detected_lang
            }]
            
            return result
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return None
    
    def transcribe_mixed_language_chunks_with_context(self, audio, task="transcribe"):
        """Process long mixed-language audio in chunks with postprocessing and FIXED adaptive context"""
        print(f"🔀 Processing mixed-language audio in {self.chunk_duration}s chunks with FIXED ADAPTIVE overlap...")
        
        sr = 16000
        chunk_samples = self.chunk_duration * sr
        initial_overlap_samples = 0 * sr  # Starting overlap
        
        all_segments = []
        languages_detected = set()
        chunk_count = 0
        chunk_confidences = []
        all_corrections = []
        
        # Context from previous chunk
        previous_context = ""
        
        # Track overlap changes for debugging
        overlap_history = []
        
        # 🔧 FIXED: Dynamic chunking with adaptive overlap calculated AFTER each chunk
        start = 0
        current_overlap = initial_overlap_samples  # Initialize with default overlap
        
        while start < len(audio):
            chunk_count += 1
            
            # Calculate chunk boundaries with CURRENT overlap
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            chunk_start_time = start / sr
            
            # Check if chunk is too short
            if len(chunk) < sr * 0.5:
                print(f"  ⚠️  Chunk {chunk_count} too short ({len(chunk)/sr:.1f}s), stopping")
                break
            
            print(f"📝 Processing chunk {chunk_count}: [{chunk_start_time:.1f}s - {end/sr:.1f}s] (overlap: {current_overlap/sr:.1f}s)")
            
            # Show context if available
            if previous_context:
                context_preview = previous_context[:50] + "..." if len(previous_context) > 50 else previous_context
                print(f"🔗 Using context ({len(previous_context.split())} words): '{context_preview}'")
            
            # Apply language restriction if enabled
            forced_language = None
            if self.restrict_languages:
                detected_lang = self._detect_language_with_restriction(chunk)
                forced_language = detected_lang
            else:
                detected_lang = None
            
            languages_detected.add(detected_lang or 'auto')
            
            # Enhanced options with context
            options = {
                "language": detected_lang,
                "task": task,
                "verbose": False, 
                "word_timestamps": True,
                "temperature": 0.0,
                "beam_size": 7,
                "patience": 2.4,
                "fp16": torch.cuda.is_available(),
                "condition_on_previous_text": True,
                "compression_ratio_threshold": 2.0,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
            }
            
            # Add context as initial prompt if available
            if previous_context:
                options["initial_prompt"] = previous_context
            
            try:
                # Transcribe current chunk
                result = self.model.transcribe(chunk, **options)
                
                # Get final detected language and validate
                final_detected_lang = result.get("language", detected_lang or "unknown")
                
                # Strict validation and correction
                if self.restrict_languages and final_detected_lang not in self.supported_languages:
                    print(f"  🚫 Chunk language '{final_detected_lang}' violates restriction, correcting...")
                    if final_detected_lang in ['ur', 'pa', 'bn', 'ta', 'te', 'ml', 'kn', 'gu', 'or', 'as', 'ne', 'si', 'mr']:
                        final_detected_lang = 'hi'
                    else:
                        final_detected_lang = 'en'
                    result['language'] = final_detected_lang
                    print(f"  🔧 Corrected to: {final_detected_lang}")
                
                languages_detected.add(final_detected_lang)
                
                # Calculate chunk confidence
                chunk_confidence = self._get_chunk_confidence(result)
                
                # Apply postprocessing to this chunk
                original_text = result.get('text', '')
                corrected_text, corrections = self.postprocessor.apply_corrections(
                    original_text, 
                    chunk_confidence
                )
                
                result['text'] = corrected_text
                result['original_text'] = original_text
                result['postprocessing_corrections'] = corrections
                
                # Apply corrections to individual segments too
                if 'segments' in result and result['segments']:
                    for segment in result['segments']:
                        if 'text' in segment:
                            segment_original = segment['text']
                            segment_corrected, _ = self.postprocessor.apply_corrections(
                                segment_original, 
                                chunk_confidence
                            )
                            segment['text'] = segment_corrected
                            segment['original_text'] = segment_original
                
                if corrections:
                    print(f"  📝 Applied {len(corrections)} corrections to chunk {chunk_count}")
                    all_corrections.extend(corrections)
                
                # 🔧 FIXED: Calculate NEXT overlap based on CURRENT chunk's confidence
                next_overlap = current_overlap  # Default: keep same overlap
                overlap_change_reason = "maintaining"
                
                if chunk_confidence < 0.25:
                    next_overlap = int(3.0 * sr)  # High overlap for low confidence
                    overlap_change_reason = "LOW confidence - increasing overlap"
                else:
                    next_overlap = int(2.0 * sr)  # Maximum overlap for very low confidence
                    overlap_change_reason = "Medium confidence - standard overlap"
                
                # Log overlap determination
                if next_overlap != current_overlap:
                    print(f"  🔄 Next overlap will be: {current_overlap/sr:.1f}s → {next_overlap/sr:.1f}s ({overlap_change_reason})")
                    print(f"  📈 Current chunk confidence: {chunk_confidence:.3f}")
                else:
                    print(f"  📊 Next overlap unchanged: {next_overlap/sr:.1f}s ({overlap_change_reason})")
                
                # Validate transcription quality
                if self._is_valid_transcription(result):
                    print(f"  🗣️  Chunk language: {final_detected_lang}")
                    print(f"  📊 Chunk confidence: {chunk_confidence:.3f}")
                    
                    if corrected_text.strip():
                        sample_preview = corrected_text[:100] + "..." if len(corrected_text) > 100 else corrected_text
                        print(f"  📄 Sample (corrected): {sample_preview}")
                    
                    # Store comprehensive chunk information with NEXT overlap info
                    chunk_info = {
                        'chunk_number': chunk_count,
                        'start_time': chunk_start_time,
                        'end_time': end/sr,
                        'confidence': chunk_confidence,
                        'language': final_detected_lang,
                        'corrections_applied': len(corrections),
                        'overlap_used': current_overlap / sr,
                        'next_overlap_determined': next_overlap / sr,
                        'overlap_change_reason': overlap_change_reason
                    }
                    chunk_confidences.append(chunk_info)
                    
                    # Filter low confidence segments
                    segments_before = len(result.get('segments', []))
                    if 'segments' in result:
                        result['segments'] = self._filter_low_confidence_segments(result['segments'])
                    segments_after = len(result.get('segments', []))
                    
                    if segments_before != segments_after:
                        print(f"  🗑️  Filtered {segments_before - segments_after} low-confidence segments")
                    
                    # Adjust timestamps and add metadata
                    if 'segments' in result and result['segments']:
                        for segment in result['segments']:
                            segment['start'] += chunk_start_time
                            segment['end'] += chunk_start_time
                            segment['chunk_language'] = final_detected_lang
                            segment['chunk_number'] = chunk_count
                            segment['chunk_confidence'] = chunk_confidence
                            segment['chunk_overlap_used'] = current_overlap / sr
                        
                        all_segments.extend(result['segments'])
                        print(f"  ✅ Added {len(result['segments'])} segments to final transcript")
                    else:
                        print(f"  ⚠️  No segments remaining after filtering!")
                    
                    # Update context for next chunk
                    previous_context = self.postprocessor.get_context_text(
                        corrected_text, 
                        max_words=self.context_words
                    )
                    
                else:
                    print(f"  ⚠️  Chunk {chunk_count} filtered out (low quality)")
                    next_overlap = int(3.0 * sr)  # Use maximum overlap after failed chunk
                    overlap_change_reason = "FAILED chunk - maximum overlap for recovery"
                    
                    chunk_confidences.append({
                        'chunk_number': chunk_count,
                        'start_time': chunk_start_time,
                        'end_time': end/sr,
                        'confidence': 0.0,
                        'language': 'filtered',
                        'status': 'filtered_low_quality',
                        'overlap_used': current_overlap / sr,
                        'next_overlap_determined': next_overlap / sr,
                        'overlap_change_reason': overlap_change_reason
                    })
                    
            except Exception as e:
                print(f"  ❌ Chunk {chunk_count} failed: {e}")
                next_overlap = int(6.0 * sr)  # Use maximum overlap after error
                overlap_change_reason = "ERROR recovery - maximum overlap"
                
                chunk_confidences.append({
                    'chunk_number': chunk_count,
                    'start_time': chunk_start_time,
                    'end_time': end/sr,
                    'confidence': 0.0,
                    'language': 'error',
                    'error': str(e),
                    'overlap_used': current_overlap / sr,
                    'next_overlap_determined': next_overlap / sr,
                    'overlap_change_reason': overlap_change_reason
                })
            
            # Track overlap usage for summary
            overlap_history.append({
                'chunk': chunk_count,
                'overlap_used': current_overlap / sr,
                'next_overlap': next_overlap / sr,
                'confidence': chunk_confidences[-1]['confidence'] if chunk_confidences else 0.0,
                'reason': overlap_change_reason
            })
            
            # 🔧 CRITICAL FIX: Calculate next start position using CURRENT overlap (before updating it)
            step_size = chunk_samples - next_overlap
            next_start = start + step_size
            
            print(f"  📍 Next chunk: start={next_start/sr:.1f}s (step={step_size/sr:.1f}s, current_overlap={current_overlap/sr:.1f}s)")
            
            # 🔧 FIXED: Update overlap for NEXT iteration AFTER calculating next start
            current_overlap = next_overlap
            
            # Update start for next iteration
            start = next_start
        
        # 🎯 FIXED SUMMARY: Report corrected adaptive overlap usage
        print(f"\n🔗 FIXED ADAPTIVE OVERLAP SUMMARY:")
        print(f"   Total chunks processed: {chunk_count}")
        if overlap_history:
            overlaps_used = [h['overlap_used'] for h in overlap_history]
            overlaps_next = [h['next_overlap'] for h in overlap_history]
            print(f"   Overlap range used: {min(overlaps_used):.1f}s - {max(overlaps_used):.1f}s")
            print(f"   Average overlap used: {sum(overlaps_used)/len(overlaps_used):.1f}s")
            
            # Show confidence-based adaptations
            adaptations = [h for h in overlap_history if "confidence" in h['reason'].lower()]
            if adaptations:
                print(f"   Confidence-based adaptations: {len(adaptations)}")
                for i, adapt in enumerate(adaptations[:5]):  # Show first 5 adaptations
                    print(f"     Chunk {adapt['chunk']}: conf={adapt['confidence']:.3f} → next_overlap={adapt['next_overlap']:.1f}s ({adapt['reason']})")
            
            # Show error/failure recoveries
            recoveries = [h for h in overlap_history if "error" in h['reason'].lower() or "failed" in h['reason'].lower()]
            if recoveries:
                print(f"   Error/failure recoveries: {len(recoveries)}")
        
        # Process overlapping segments
        print(f"\n📊 Total segments before overlap removal: {len(all_segments)}")
        segments_before_overlap = len(all_segments)
        
        all_segments = self._remove_overlapping_segments(
            all_segments,
            overlap_threshold=0.8,
            confidence_threshold=0.03,
            similarity_threshold=0.9,
            min_overlap_duration=0.3
        )
        
        segments_after_overlap = len(all_segments)
        if segments_before_overlap != segments_after_overlap:
            removed_count = segments_before_overlap - segments_after_overlap
            print(f"🗑️  Removed {removed_count} overlapping segments")
        
        # Combine all segments
        combined_text = ' '.join([seg.get('text', '').strip() for seg in all_segments if seg.get('text', '').strip()])
        
        # Calculate overall confidence
        valid_confidences = [c['confidence'] for c in chunk_confidences if c['confidence'] > 0]
        overall_confidence = np.mean(valid_confidences) if valid_confidences else 0.0
        
        # Final language validation
        if self.restrict_languages:
            languages_detected = {lang for lang in languages_detected if lang in self.supported_languages or lang == 'auto'}
        
        # Enhanced reporting
        print(f"\n🌐 Languages detected: {', '.join(sorted(languages_detected - {'auto'}))}")
        print(f"📊 Overall confidence: {overall_confidence:.3f}")
        print(f"📝 Total postprocessing corrections: {len(all_corrections)}")
        print(f"📄 Final transcript length: {len(combined_text)} characters")
        
        # Determine primary language
        lang_counts = {}
        for seg in all_segments:
            lang = seg.get('chunk_language', 'unknown')
            if self.restrict_languages and lang not in self.supported_languages:
                lang = 'hi' if lang in ['ur', 'pa', 'bn', 'ta', 'te', 'ml', 'kn', 'gu', 'or', 'as', 'ne', 'si', 'mr'] else 'en'
            lang_counts[lang] = lang_counts.get(lang, 0) + len(seg.get('text', ''))
        
        primary_language = max(lang_counts.keys(), key=lambda k: lang_counts[k]) if lang_counts else 'unknown'
        
        # Final validation of primary language
        if self.restrict_languages and primary_language not in self.supported_languages:
            primary_language = 'en'
        
        return {
            'text': combined_text,
            'segments': all_segments,
            'language': 'mixed' if len(languages_detected - {'auto'}) > 1 else primary_language,
            'languages_detected': sorted(list(languages_detected - {'auto'})),
            'language_distribution': lang_counts,
            'chunks_processed': chunk_count,
            'primary_language': primary_language,
            'chunk_confidences': chunk_confidences,
            'overall_confidence': overall_confidence,
            'total_chunks': chunk_count,
            'postprocessing_corrections': all_corrections,
            'context_enabled': True,
            'context_words': self.context_words,
            'adaptive_overlap_enabled': True,
            'overlap_history': overlap_history,  # NEW: Include overlap decision history
        }
    
    def save_transcription_with_confidence(self, result, output_file, include_metadata=True, include_word_timestamps=True):
        """Save transcription to text file with confidence scores and postprocessing information"""
        with open(output_file, 'w', encoding='utf-8') as f:
            if include_metadata:
                f.write("=" * 80 + "\n")
                f.write("ENHANCED TRANSCRIPTION REPORT WITH POSTPROCESSING\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {self.model_size}\n")
                f.write(f"Device: {self.device}\n")
                f.write(f"Language restriction: {'English and Hindi only' if self.restrict_languages else 'All languages'}\n")
                f.write(f"Primary language: {result.get('language', 'unknown')}\n")
                f.write(f"Context enabled: {result.get('context_enabled', False)}\n")
                f.write(f"Context words: {result.get('context_words', 0)}\n")
                
                # Postprocessing information
                corrections = result.get('postprocessing_corrections', [])
                f.write(f"Postprocessing corrections applied: {len(corrections)}\n")
                
                # Overall confidence
                if 'overall_confidence' in result:
                    f.write(f"Overall confidence: {result['overall_confidence']:.3f}\n")
                
                # Enhanced language information for mixed content
                if 'languages_detected' in result:
                    f.write(f"All languages detected: {', '.join(result['languages_detected'])}\n")
                    
                if 'language_distribution' in result:
                    f.write(f"Language distribution (by character count):\n")
                    total_chars = sum(result['language_distribution'].values())
                    for lang, count in result['language_distribution'].items():
                        percentage = (count / total_chars * 100) if total_chars > 0 else 0
                        f.write(f"  {lang}: {count} chars ({percentage:.1f}%)\n")
                
                if 'chunks_processed' in result:
                    f.write(f"Chunks processed: {result['chunks_processed']}\n")
                
                f.write(f"Total segments: {len(result.get('segments', []))}\n")
                f.write("=" * 80 + "\n\n")
            
            # Postprocessing corrections summary
            corrections = result.get('postprocessing_corrections', [])
            if corrections:
                f.write("POSTPROCESSING CORRECTIONS APPLIED:\n")
                f.write("-" * 60 + "\n")
                f.write("Original → Corrected (Confidence Threshold)\n")
                f.write("-" * 60 + "\n")
                
                correction_summary = {}
                for correction in corrections:
                    key = f"{correction['original']} → {correction['corrected']}"
                    if key not in correction_summary:
                        correction_summary[key] = {
                            'count': 0,
                            'threshold': correction['confidence_threshold'],
                            'applied_confidence': correction.get('applied_at_confidence', 0)
                        }
                    correction_summary[key]['count'] += correction.get('occurrences', 1)
                
                for correction_text, info in correction_summary.items():
                    f.write(f"{correction_text} ")
                    f.write(f"(threshold: {info['threshold']:.2f}, ")
                    f.write(f"applied at: {info['applied_confidence']:.2f}, ")
                    f.write(f"occurrences: {info['count']})\n")
                f.write("\n")
            
            # Chunk confidence summary with postprocessing info
            if 'chunk_confidences' in result:
                f.write("CHUNK ANALYSIS WITH POSTPROCESSING:\n")
                f.write("-" * 80 + "\n")
                f.write("Chunk  [Time Range     ] Language Confidence Corrections Status\n")
                f.write("-" * 80 + "\n")
                
                for chunk_info in result['chunk_confidences']:
                    f.write(f"{chunk_info['chunk_number']:3d}    ")
                    f.write(f"[{chunk_info['start_time']:6.2f}s - {chunk_info['end_time']:6.2f}s] ")
                    f.write(f"{chunk_info.get('language', 'unknown'):8s} ")
                    f.write(f"{chunk_info['confidence']:8.3f}   ")
                    f.write(f"{chunk_info.get('corrections_applied', 0):11d}   ")
                    
                    # Status
                    if 'error' in chunk_info:
                        f.write(f"ERROR: {chunk_info['error'][:30]}...")
                    elif chunk_info.get('status') == 'filtered_low_quality':
                        f.write(f"FILTERED (low quality)")
                    else:
                        f.write(f"OK")
                    
                    f.write("\n")
                f.write("\n")
            
            # Write clean transcription
            f.write("FULL TRANSCRIPTION (POSTPROCESSED):\n")
            f.write("-" * 50 + "\n")
            f.write(result["text"].strip())
            f.write("\n\n")
            
            # Show original vs corrected if single chunk
            if result.get('original_text') and result.get('text') != result.get('original_text'):
                f.write("ORIGINAL TRANSCRIPTION (BEFORE POSTPROCESSING):\n")
                f.write("-" * 50 + "\n")
                f.write(result["original_text"].strip())
                f.write("\n\n")
            
            # Write segments with timestamps, language information, and confidence
            if result.get("segments"):
                f.write("DETAILED TRANSCRIPTION WITH CONFIDENCE SCORES:\n")
                f.write("-" * 75 + "\n")
                
                current_language = None
                for i, segment in enumerate(result["segments"]):
                    start = segment["start"]
                    end = segment["end"]
                    text = segment["text"].strip()
                    
                    # Language and confidence info
                    chunk_lang = segment.get('chunk_language', '')
                    chunk_num = segment.get('chunk_number', '')
                    chunk_conf = segment.get('chunk_confidence', 0.0)
                    segment_conf = self._get_segment_confidence(segment)
                    
                    # Show language changes
                    if chunk_lang and chunk_lang != current_language:
                        f.write(f"\n--- LANGUAGE: {chunk_lang.upper()} ---\n")
                        current_language = chunk_lang
                    
                    # Format with all information
                    f.write(f"[{start:6.2f}s - {end:6.2f}s] ")
                    f.write(f"[{chunk_lang}] ")
                    f.write(f"[Chunk {chunk_num}] ")
                    f.write(f"[ChunkConf: {chunk_conf:.3f}] ")
                    f.write(f"[SegConf: {segment_conf:.3f}]")
                    f.write(f": {text}\n")
                
                f.write("\n")
            
            # Write word-level timestamps if available and requested
            if include_word_timestamps and result.get("segments"):
                f.write("WORD-LEVEL TIMESTAMPS WITH CONFIDENCE:\n")
                f.write("-" * 70 + "\n")
                
                for segment in result["segments"]:
                    if "words" in segment:
                        chunk_lang = segment.get('chunk_language', 'unknown')
                        f.write(f"\n--- Segment ({chunk_lang}) ---\n")
                        
                        for word_info in segment["words"]:
                            word = word_info.get("word", "").strip()
                            start = word_info.get("start", 0)
                            end = word_info.get("end", 0)
                            confidence = word_info.get("probability", 0)
                            f.write(f"{start:6.2f}-{end:6.2f}s: {word:<15} (conf: {confidence:.3f}) [{chunk_lang}]\n")
                
                f.write("\n")
            
            # Summary statistics for mixed language content
            if 'languages_detected' in result and len(result['languages_detected']) > 1:
                f.write("MIXED LANGUAGE ANALYSIS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"This audio contains {len(result['languages_detected'])} languages: {', '.join(result['languages_detected'])}\n")
                f.write(f"Primary language: {result.get('primary_language', 'unknown')}\n")
                if 'chunks_processed' in result:
                    f.write(f"Processed in {result['chunks_processed']} chunks for better language detection\n")
                if result.get('context_enabled'):
                    f.write(f"Context from previous chunks enabled ({result.get('context_words', 0)} words)\n")
                f.write("\n")
            
            # Quality metrics
            f.write("QUALITY METRICS:\n")
            f.write("-" * 30 + "\n")
            if 'chunk_confidences' in result:
                error_chunks = len([c for c in result['chunk_confidences'] if 'error' in c])
                filtered_chunks = len([c for c in result['chunk_confidences'] if c.get('status') == 'filtered_low_quality'])
                successful_chunks = result.get('chunks_processed', 0) - error_chunks - filtered_chunks
                
                f.write(f"Chunks with errors: {error_chunks}\n")
                f.write(f"Chunks filtered (low quality): {filtered_chunks}\n")
                f.write(f"Successful chunks: {successful_chunks}\n")
                
                if 'overall_confidence' in result:
                    f.write(f"Overall confidence: {result['overall_confidence']:.3f}\n")
                    
                    # Confidence categories
                    high_conf = len([c for c in result['chunk_confidences'] if c['confidence'] >= 0.8])
                    med_conf = len([c for c in result['chunk_confidences'] if 0.5 <= c['confidence'] < 0.8])
                    low_conf = len([c for c in result['chunk_confidences'] if 0 < c['confidence'] < 0.5])
                    
                    f.write(f"High confidence chunks (≥0.8): {high_conf}\n")
                    f.write(f"Medium confidence chunks (0.5-0.8): {med_conf}\n")
                    f.write(f"Low confidence chunks (<0.5): {low_conf}\n")
                
                # Postprocessing statistics
                total_corrections = sum([c.get('corrections_applied', 0) for c in result['chunk_confidences']])
                f.write(f"Total postprocessing corrections: {total_corrections}\n")
                
                chunks_with_corrections = len([c for c in result['chunk_confidences'] if c.get('corrections_applied', 0) > 0])
                f.write(f"Chunks with corrections: {chunks_with_corrections}\n")
            
            f.write("\n")
        
        print(f"📄 Enhanced transcription with postprocessing saved to: {output_file}")
    
    def process_audio_file(self, input_file, output_file=None, noise_reduction=True, 
                          noise_strength=0.6, language=None, task="transcribe", force_chunking=False):
        """
        Complete pipeline: load, process, transcribe, and save with postprocessing and context
        """
        print(f"🎵 Processing: {input_file}")
        
        # Generate output filename if not provided
        if output_file is None:
            input_path = Path(input_file)
            suffix = "enhanced_postprocessed"
            if self.restrict_languages:
                suffix += "_restricted_en_hi"
            if language is None:
                suffix += "_mixed_lang"
            else:
                suffix += f"_lang_{language}"
            output_file = input_path.parent / f"{input_path.stem}_transcription_{self.model_size.replace('-', '_')}_{suffix}.txt"
        
        try:
            # Step 1: Load audio
            audio, sr = self.load_audio(input_file)
            
            # Step 2: Apply noise reduction if requested
            if noise_reduction:
                audio = self.reduce_noise(audio, sr, noise_strength)
            
            # Step 3: Preprocess audio (normalize, trim)
            audio = self.preprocess_audio(audio, sr)
            
            # Step 4: Transcribe with local Whisper (with postprocessing and context)
            result = self.transcribe_with_local_whisper(
                audio, 
                language=language, 
                task=task, 
                force_chunking=force_chunking
            )
            
            if result:
                # Step 5: Save transcription with enhanced language information, confidence scores, and postprocessing info
                self.save_transcription_with_confidence(result, output_file)
                print("✅ Processing completed successfully!")
                
                # Enhanced completion summary
                if 'languages_detected' in result:
                    print(f"🌐 Languages found: {', '.join(result['languages_detected'])}")
                if 'language_distribution' in result:
                    total_chars = sum(result['language_distribution'].values())
                    for lang, count in result['language_distribution'].items():
                        percentage = (count / total_chars * 100) if total_chars > 0 else 0
                        print(f"📊 {lang}: {percentage:.1f}% of content")
                
                if 'overall_confidence' in result:
                    print(f"📊 Overall confidence: {result['overall_confidence']:.3f}")
                
                # Postprocessing summary
                corrections = result.get('postprocessing_corrections', [])
                if corrections:
                    print(f"📝 Postprocessing corrections applied: {len(corrections)}")
                    
                    # Show most common corrections
                    correction_counts = {}
                    for correction in corrections:
                        key = f"{correction['original']} → {correction['corrected']}"
                        correction_counts[key] = correction_counts.get(key, 0) + correction.get('occurrences', 1)
                    
                    if correction_counts:
                        most_common = sorted(correction_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                        print("📝 Most common corrections:")
                        for correction, count in most_common:
                            print(f"   {correction} ({count} times)")
                
                # Context usage summary
                if result.get('context_enabled'):
                    print(f"🔗 Context enabled: {result.get('context_words', 0)} words from previous chunks")
                
                # Quality summary
                if 'chunk_confidences' in result:
                    error_chunks = len([c for c in result['chunk_confidences'] if 'error' in c])
                    filtered_chunks = len([c for c in result['chunk_confidences'] if c.get('status') == 'filtered_low_quality'])
                    if error_chunks > 0:
                        print(f"⚠️  Chunks with errors: {error_chunks}")
                    if filtered_chunks > 0:
                        print(f"🗑️  Chunks filtered: {filtered_chunks}")
                
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
        """Get information about the loaded model"""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "chunk_duration": f"{self.chunk_duration}s",
            "context_words": self.context_words,
            "parameters": f"~{self.get_parameter_count()}",
            "languages_supported": "English and Hindi only" if self.restrict_languages else "99+ languages including Hindi, English",
            "mixed_language_support": "Optimized for Hindi-English code-switching",
            "confidence_scoring": "Enabled - chunk and segment level",
            "language_restriction": "Enabled" if self.restrict_languages else "Disabled",
            "postprocessing": f"Enabled - {len(self.postprocessor.correction_rules)} rules loaded",
            "context_enabled": "Previous chunk context for better accuracy"
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
    parser = argparse.ArgumentParser(description='Enhanced transcriber with postprocessing, context, and confidence scoring')
    parser.add_argument('input_file', nargs='?', help='Input audio file path')
    parser.add_argument('--postprocessing-rules', nargs='?', 
                       help='Path to JSON file containing postprocessing rules')
    parser.add_argument('-o', '--output', help='Output text file path')
    parser.add_argument('-m', '--model', default='large-v3',
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                       help='Whisper model size (default: large-v3)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], 
                       help='Device to run model on (auto-detect if not specified)')
    parser.add_argument('--chunk-duration', type=int, default=10,
                       help='Duration in seconds for processing chunks (default: 15)')
    parser.add_argument('--context-words', type=int, default=0,
                       help='Number of words from previous chunk to use as context (default: 2)')
    parser.add_argument('--no-noise-reduction', action='store_true', 
                       help='Skip noise reduction step')
    parser.add_argument('--noise-strength', type=float, default=0.6,
                       help='Noise reduction strength (0.0-1.0, default: 0.6)')
    parser.add_argument('--language', 
                       help='Language code (hi=Hindi, en=English, None=auto-detect for mixed languages)')
    parser.add_argument('--task', choices=['transcribe', 'translate'], default='transcribe',
                       help='Task: transcribe (keep original language) or translate (to English)')
    parser.add_argument('--force-chunking', action='store_true',
                       help='Force chunk-based processing even for short audio (better for mixed languages)')
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
    
    # Initialize transcriber with enhanced features
    print("🚀 Initializing Enhanced Whisper Transcriber with Postprocessing...")
    transcriber = LocalAudioTranscriber(
        model_size=args.model, 
        device=args.device,
        chunk_duration=args.chunk_duration,
        restrict_languages=args.restrict_languages,
        postprocessing_rules=postprocessing_rules,
        context_words=args.context_words
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
        print("Usage: python enhanced_transcriber.py your_audio.wav")
        print("       python enhanced_transcriber.py your_audio.wav --postprocessing-rules rules.json")
        return
    
    # Process the file with enhanced features
    result = transcriber.process_audio_file(
        input_file=args.input_file,
        output_file=args.output,
        noise_reduction=not args.no_noise_reduction,
        noise_strength=args.noise_strength,
        language=args.language,
        task=args.task,
        force_chunking=args.force_chunking
    )
    
    if result:
        print(f"✅ Enhanced transcription with postprocessing completed!")
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
    
    # For programmatic usage:
    """
    # Example postprocessing rules (can also be loaded from JSON file)
    postprocessing_rules = {
        "correction_rules": [
            {
                "previous_word": "day",
                "replace_word": "digit", 
                "confidence_threshold": 0.5
            },
            {
                "previous_word": "सब्सक्राइब",
                "replace_word": "surrender",
                "confidence_threshold": 0.95
            }
        ]
    }
    
    # Initialize enhanced transcriber with postprocessing and context
    transcriber = LocalAudioTranscriber(
        model_size="large-v3", 
        device="cuda", 
        chunk_duration=15,
        restrict_languages=True,
        postprocessing_rules=postprocessing_rules,  # NEW: Postprocessing rules
        context_words=30  # NEW: Context from previous chunks
    )
    
    # Process mixed language file with postprocessing and context
    result = transcriber.process_audio_file(
        input_file="path/to/mixed_hindi_english.wav",
        output_file="enhanced_transcript_with_postprocessing.txt",
        noise_reduction=True,
        noise_strength=0.6,
        language=None,  # Auto-detect
        task="transcribe",
        force_chunking=True  # Force chunking for better accuracy
    )
    
    # Key new features added:
    # 1. PostProcessor class for applying correction rules based on confidence
    # 2. Context from previous postprocessed chunks improves accuracy
    # 3. Real-time postprocessing during transcription (not post-processing)
    # 4. Detailed reporting of corrections applied
    # 5. Context-aware transcription with previous chunk information
    # 6. Confidence-based correction application
    # 7. Enhanced output with before/after comparison
    
    # The transcriber now:
    # - Applies postprocessing to each chunk as it's transcribed
    # - Uses the postprocessed text from previous chunk as context
    # - Only applies corrections when confidence meets threshold
    # - Provides detailed reporting of all corrections made
    # - Maintains chunk-by-chunk processing for better language detection
    """