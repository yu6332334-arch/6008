# ============================================================================
# audio_engine_v2.py
# ============================================================================
"""
Enhanced audio processing engine with:
- Proper thread synchronization and lifecycle management
- Ordered component shutdown to prevent resource leaks
- Memory-efficient audio processing
- Enhanced error recovery
- Clean resource cleanup
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import threading
import time
import numpy as np
import pyaudio
from typing import Optional, List, Dict, Any
import traceback
from pathlib import Path

# Import optimized modules (will use v2 versions when available)
from command_manager import CommandManager

# Lightweight debug logger for packaged builds
_STT_DEBUG_ENABLED = bool(getattr(sys, 'frozen', False) or os.environ.get('DEBUG_STT') == '1')
def _stt_log(message: str):
    if not _STT_DEBUG_ENABLED:
        return
    try:
        # Log next to the executable when frozen; otherwise current dir
        base_dir = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path.cwd()
        log_path = base_dir / 'stt_debug.log'
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    except Exception:
        pass

# Try to import enhanced TTS engine first
try:
    from tts_engine_v2 import TTSEngine
    print("[AudioEngine] Using enhanced TTS engine v2")
except ImportError:
    from tts_engine import TTSEngine
    print("[AudioEngine] Using standard TTS engine")

from model_manager import ModelManager

# Detect Whisper backend
BACKEND = None
try:
    from faster_whisper import WhisperModel
    BACKEND = "faster-whisper"
    print("[AudioEngine] Using faster-whisper backend")
except ImportError:
    try:
        import whisper
        BACKEND = "openai-whisper" 
        print("[AudioEngine] Using openai-whisper backend")
    except ImportError:
        BACKEND = "none"
        print("[AudioEngine] WARNING: No Whisper backend available")

class AudioEngine:
    """
    Enhanced high-performance audio processing engine.
    
    Key Improvements:
    - Proper component initialization order
    - Thread-safe state management with RLock
    - Graceful shutdown with dependency ordering
    - Memory leak prevention
    - Enhanced error handling and recovery
    - Async initialization to prevent UI blocking
    """
    
    # State constants
    WAKE_STATE_INACTIVE = 0
    WAKE_STATE_ACTIVE = 1
    
    def __init__(self, model_size="base", language=None, device="cpu"):
        print("[AudioEngine] Initializing...")
        
        # Configuration
        self.model_size = model_size
        self.language = language
        self.last_detected_language = None
        self.last_language_probability = None
        self.device = device
        
        # State management (thread-safe with RLock for reentrant locking)
        self._state_lock = threading.RLock()
        self.wake_state = self.WAKE_STATE_INACTIVE
        self._processing = False
        self._shutting_down = False
        
        # Performance optimizations
        self._last_command_time = 0
        self._command_cooldown = 2.0  # Prevent duplicate processing
        
        # Audio configuration
        self.chunk = 1024
        self.sample_rate = 16000
        self.channels = 1
        self.format = pyaudio.paInt16
        
        # Component initialization flags
        self._components_initialized = {
            "command_manager": False,
            "tts_engine": False,
            "model_manager": False,
            "model_loaded": False
        }
        
        # Initialize components with proper error handling
        print("[AudioEngine] Initializing components...")
        
        # 1. Command Manager (lightweight, no dependencies)
        try:
            self.cmd_hotword_mgr = CommandManager()
            self._components_initialized["command_manager"] = True
            print("[AudioEngine] ✓ Command manager initialized")
        except Exception as e:
            print(f"[AudioEngine] ✗ Command manager init error: {e}")
            self.cmd_hotword_mgr = None
        
        # 2. TTS Engine (independent, can fail gracefully)
        try:
            self.tts_mgr = TTSEngine()
            self._components_initialized["tts_engine"] = True
            print("[AudioEngine] ✓ TTS engine initialized")
            
            # CRITICAL FIX: Start TTS worker thread
            self.tts_mgr.start()
            print("[AudioEngine] ✓ TTS worker started")
        except Exception as e:
            print(f"[AudioEngine] ✗ TTS engine init error: {e}")
            traceback.print_exc()
            self.tts_mgr = None
        
        # 3. Model Manager (required for speech recognition)
        try:
            self.model_mgr = ModelManager()
            self._components_initialized["model_manager"] = True
            print("[AudioEngine] ✓ Model manager initialized")
        except Exception as e:
            print(f"[AudioEngine] ✗ Model manager init error: {e}")
            traceback.print_exc()
            self.model_mgr = None
        
        # Model loading state
        self.model = None
        self._model_ready = False
        self._model_init_thread = None
        
        # Error tracking
        self._last_error = None
        self._last_success_time = None
        self._error_count = 0
        
        # Resource tracking for cleanup
        self._active_threads = []
        self._audio_resources = []
        
        # Initialize model asynchronously
        self._init_model_async()
        
        _stt_log("AudioEngine initialized")
        print("[AudioEngine] Initialization complete")
    
    def _init_model_async(self):
        """Initialize model asynchronously to prevent UI blocking"""
        def _init():
            try:
                print(f"[AudioEngine] Loading model: {self.model_size}")
                
                if not self.model_mgr:
                    print("[AudioEngine] Model manager not available")
                    return
                
                self.model = self.model_mgr.load_model(self.model_size)
                
                with self._state_lock:
                    self._model_ready = self.model is not None
                    if self._model_ready:
                        self._components_initialized["model_loaded"] = True
                
                if self._model_ready:
                    print(f"[AudioEngine] ✓ Model {self.model_size} loaded successfully")
                    # Removed duplicate "System Ready" announcement - will be called once from main_voice_app.py
                else:
                    print(f"[AudioEngine] ✗ Failed to load model {self.model_size}")
                    self._last_error = f"Model {self.model_size} load failed"
                    
            except Exception as e:
                print(f"[AudioEngine] Model initialization error: {e}")
                traceback.print_exc()
                self._last_error = str(e)
                self._error_count += 1
        
        # Start in background thread
        self._model_init_thread = threading.Thread(
            target=_init,
            daemon=True,
            name="ModelInit"
        )
        self._model_init_thread.start()
        self._active_threads.append(self._model_init_thread)
    
    def is_model_ready(self) -> bool:
        """Check if model is ready for use"""
        with self._state_lock:
            return self._model_ready and not self._shutting_down
    
    def wait_for_model(self, timeout: float = 30.0) -> bool:
        """Wait for model to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_model_ready():
                return True
            if self._shutting_down:
                return False
            time.sleep(0.1)
        return False
    
    def set_wake_state(self, state: int):
        """Set wake state (thread-safe)"""
        if state not in [self.WAKE_STATE_INACTIVE, self.WAKE_STATE_ACTIVE]:
            return False
        
        with self._state_lock:
            if self._shutting_down:
                return False
            self.wake_state = state
        
        state_name = "ACTIVE" if state == 1 else "INACTIVE"
        print(f"[AudioEngine] Wake state: {state_name}")
        return True
    
    def get_wake_state(self) -> int:
        """Get current wake state"""
        with self._state_lock:
            return self.wake_state
    
    def is_active(self) -> bool:
        """Check if in active state"""
        return self.get_wake_state() == self.WAKE_STATE_ACTIVE
    
    def is_processing(self) -> bool:
        """Check if currently processing"""
        with self._state_lock:
            return self._processing
    
    def record_audio(self, duration: float = 3.0) -> Optional[np.ndarray]:
        """
        Record audio with optimized performance and proper resource cleanup.
        Enhanced error handling to prevent resource leaks.
        """
        if duration <= 0 or self._shutting_down:
            return None
        
        frames = []
        audio_interface = None
        stream = None
        
        try:
            # Initialize audio interface
            audio_interface = pyaudio.PyAudio()
            
            # Get default input device with fallback
            try:
                device_info = audio_interface.get_default_input_device_info()
                device_index = device_info['index']
            except Exception as e:
                print(f"[AudioEngine] Using default input device (error: {e})")
                device_index = None
            
            # Open audio stream
            stream = audio_interface.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk,
                input_device_index=device_index
            )
            
            # Track resource for cleanup
            self._audio_resources.append((stream, audio_interface))
            
            # Calculate frames to read
            frames_to_read = int(self.sample_rate / self.chunk * duration)
            
            # Record audio
            for i in range(frames_to_read):
                if self._shutting_down:
                    break
                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    frames.append(data)
                except Exception as e:
                    # Continue recording even if some frames fail
                    continue
            
            if not frames:
                _stt_log("No audio frames captured")
                return None
            
            # Convert to numpy array with proper dtype
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Basic quality check
            rms = np.sqrt(np.mean(audio_data**2))
            print(f"[AudioEngine] Recorded RMS={rms:.6f}")
            _stt_log(f"Recorded {len(frames)} frames; RMS={rms:.6f}")
            if rms < 0.001:
                print("[AudioEngine] Audio too quiet")
                _stt_log("Audio too quiet below threshold")
                return None
            
            return audio_data
        
        except Exception as e:
            print(f"[AudioEngine] Audio recording error: {e}")
            self._last_error = str(e)
            self._error_count += 1
            return None
        
        finally:
            # CRITICAL: Clean up resources immediately
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception as e:
                    print(f"[AudioEngine] Stream cleanup error: {e}")
            
            if audio_interface:
                try:
                    audio_interface.terminate()
                except Exception as e:
                    print(f"[AudioEngine] Audio interface cleanup error: {e}")
            
            # Remove from tracking
            try:
                self._audio_resources.remove((stream, audio_interface))
            except:
                pass
    
    def detect_wake_word(self, audio_data: np.ndarray, wake_word: str = "susie") -> bool:
        """
        Optimized wake word detection with proper concurrency control.
        """
        if audio_data is None or not self._model_ready or self._shutting_down:
            return False
        
        # Prevent concurrent processing
        with self._state_lock:
            if self._processing:
                return False
            self._processing = True
        
        try:
            # Get hotword boost
            boost_phrases = [wake_word]
            if self.cmd_hotword_mgr:
                boost_phrases.extend(self.cmd_hotword_mgr.get_boost_phrases(3))
            
            # Transcribe with optimized settings
            text = self._transcribe_optimized(audio_data, boost_phrases, fast_mode=True)
            
            if text:
                text = text.strip().lower()
                detected = wake_word.lower() in text
                
                if detected:
                    print(f"[AudioEngine] Wake word '{wake_word}' detected")
                    _stt_log(f"Wake word detected in text='{text}'")
                    # Quick TTS feedback
                    if self.tts_mgr:
                        self.tts_mgr.speak_status("please speak")
                    return True
            
            _stt_log("Wake word not detected")
            return False
        
        except Exception as e:
            print(f"[AudioEngine] Wake word detection error: {e}")
            self._last_error = str(e)
            return False
        
        finally:
            with self._state_lock:
                self._processing = False
    
    def transcribe(self, audio_data: np.ndarray) -> Optional[str]:
        """
        Optimized transcription with enhanced error handling.
        """
        if not self.is_active() or audio_data is None or not self._model_ready or self._shutting_down:
            return None
        
        # Prevent concurrent processing
        with self._state_lock:
            if self._processing:
                return None
            self._processing = True
        
        try:
            # Get hotword boost phrases
            boost_phrases = []
            if self.cmd_hotword_mgr:
                boost_phrases = self.cmd_hotword_mgr.get_boost_phrases(10)
            
            # Transcribe with optimization
            text = self._transcribe_optimized(audio_data, boost_phrases, fast_mode=False)
            
            if text:
                text = text.strip()
                # Apply deduplication to prevent repetitive text
                text = self._deduplicate_text(text)
                
                # FIX #5: Filter phrases longer than 4 words (allows "open robot cell", "open camera 1")
                text = self._filter_by_word_count(text, max_words=4)
                if not text:
                    print(f"[AudioEngine] Transcription rejected (>4 words)")
                    _stt_log(f"Transcription rejected: exceeds word limit")
                    return None
                
                print(f"[AudioEngine] Transcription: '{text}'")
                _stt_log(f"Transcription result: '{text}'")
                
                # Track success
                self._last_success_time = time.time()
                self._error_count = 0  # Reset error count on success
                return text
            
            return None
        
        except Exception as e:
            print(f"[AudioEngine] Transcription error: {e}")
            traceback.print_exc()
            self._last_error = str(e)
            self._error_count += 1
            _stt_log(f"Transcription exception: {e}")
            return None
        
        finally:
            with self._state_lock:
                self._processing = False
    
    def _transcribe_optimized(self, audio_data: np.ndarray, boost_phrases: List[str], fast_mode: bool = False) -> Optional[str]:
        """
        Optimized transcription with backend-specific optimizations.
        """
        if not self.model or self._shutting_down:
            return None
        
        try:
            initial_prompt = ", ".join(boost_phrases) if boost_phrases else None
            
            if BACKEND == "faster-whisper":
                # Optimized parameters for performance
                beam_size = 1 if fast_mode else 2  # Reduced for speed
                
                # FIX #6: Enable VAD filter with proper error handling
                # Try to use VAD; fallback gracefully if dependencies missing
                use_vad = True
                vad_error_msg = None
                try:
                    # Test if VAD model file exists in frozen builds
                    if getattr(sys, 'frozen', False):
                        import os
                        vad_model_path = os.path.join(sys._MEIPASS, 'faster_whisper', 'assets', 'silero_vad_v6.onnx')
                        if not os.path.exists(vad_model_path):
                            use_vad = False
                            vad_error_msg = f"VAD model not found at {vad_model_path}"
                            _stt_log(vad_error_msg)
                        else:
                            _stt_log(f"VAD model found at {vad_model_path}")
                except Exception as e:
                    use_vad = False
                    vad_error_msg = f"VAD check failed: {e}"
                    _stt_log(vad_error_msg)

                try:
                    segments, info = self.model.transcribe(
                        audio_data,
                        language=self.language,
                        beam_size=beam_size,
                        vad_filter=use_vad,
                        vad_parameters={
                            "threshold": 0.5,
                            "min_speech_duration_ms": 250,
                            "min_silence_duration_ms": 100,
                            "speech_pad_ms": 30
                        } if use_vad else None,
                        initial_prompt=initial_prompt,
                        temperature=0.0,  # Deterministic
                        compression_ratio_threshold=2.4,
                        log_prob_threshold=-1.0,
                        no_speech_threshold=0.6,
                        condition_on_previous_text=False,  # CRITICAL: Prevent repetition
                        word_timestamps=False  # Faster processing
                    )
                except Exception as transcribe_error:
                    # If VAD causes error, retry without VAD
                    if use_vad and "silero" in str(transcribe_error).lower():
                        _stt_log(f"VAD error detected, retrying without VAD: {transcribe_error}")
                        use_vad = False
                        segments, info = self.model.transcribe(
                            audio_data,
                            language=self.language,
                            beam_size=beam_size,
                            vad_filter=False,
                            initial_prompt=initial_prompt,
                            temperature=0.0,
                            compression_ratio_threshold=2.4,
                            log_prob_threshold=-1.0,
                            no_speech_threshold=0.6,
                            condition_on_previous_text=False,
                            word_timestamps=False
                        )
                    else:
                        raise
                
                if use_vad:
                    _stt_log("VAD filter enabled successfully")
                else:
                    _stt_log("VAD filter disabled (dependencies unavailable)")
                
                self.last_detected_language = getattr(info, "language", None)
                self.last_language_probability = getattr(info, "language_probability", None)
                _stt_log(f"Language detected: {self.last_detected_language}, probability={self.last_language_probability}")
                # CRITICAL FIX: Convert generator to list ONCE
                # Aggressively limit segments and detect repetition
                segments_list = []
                seen_texts = set()  # Track unique segments
                word_count = 0
                
                for i, seg in enumerate(segments):
                    if i >= 3 or self._shutting_down or word_count > 20:  # Limit by segment count AND word count
                        break
                    
                    seg_text = seg.text.strip()
                    if not seg_text:
                        continue
                    
                    # Skip if we've seen this exact text
                    seg_lower = seg_text.lower()
                    if seg_lower in seen_texts:
                        print(f"[AudioEngine] Skipping duplicate segment: '{seg_text}'")
                        break  # Stop entirely if we see duplication
                    
                    segments_list.append(seg_text)
                    seen_texts.add(seg_lower)
                    word_count += len(seg_text.split())
                
                text = " ".join(segments_list)
                _stt_log(f"Segments collected: {len(segments_list)}; words={word_count}; text='{text}'")
                
            elif BACKEND == "openai-whisper":
                # OpenAI Whisper parameters
                result = self.model.transcribe(
                    audio_data,
                    language=self.language,
                    fp16=False,
                    initial_prompt=initial_prompt
                )
                text = result.get("text", "")
                
            else:
                return None
            
            result_text = text.strip() if text else None
            if not result_text:
                _stt_log("Transcribe returned empty text")
            return result_text
        
        except Exception as e:
            print(f"[AudioEngine] Transcription backend error: {e}")
            _stt_log(f"Transcription backend exception: {e}")
            return None
    
    def match_command(self, text: str, commands: List[str]) -> Optional[str]:
        """
        Match command with cooldown, deduplication, and aggregation.
        FIX #7: Uses command aggregation when multiple instances detected.
        """
        if not text or not commands or self._shutting_down:
            return None
        
        # Remove repetitive patterns
        text = self._deduplicate_text(text)
        
        # Check cooldown
        current_time = time.time()
        if current_time - self._last_command_time < self._command_cooldown:
            print(f"[AudioEngine] Command cooldown active ({self._command_cooldown}s)")
            return None
        
        # FIX #7: Try command aggregation first (for multiple command instances)
        aggregated = self._aggregate_commands(text, commands)
        if aggregated:
            # Verify aggregated command is valid
            if self.cmd_hotword_mgr:
                self.cmd_hotword_mgr.record_usage(aggregated, success=True)
            self._last_command_time = current_time
            print(f"[AudioEngine] Command aggregated: '{text}' -> '{aggregated}'")
            return aggregated
        
        # Fallback: Find match using standard fuzzy matching
        matched_command = None
        if self.cmd_hotword_mgr:
            matched_command = self.cmd_hotword_mgr.find_best_match(text)
        
        if matched_command:
            # Record usage and update cooldown
            if self.cmd_hotword_mgr:
                self.cmd_hotword_mgr.record_usage(matched_command, success=True)
            self._last_command_time = current_time
            print(f"[AudioEngine] Command matched: '{text}' -> '{matched_command}'")
        else:
            print(f"[AudioEngine] No command match for: '{text}'")
        
        return matched_command
    
    def _deduplicate_text(self, text: str) -> str:
        """
        Aggressively remove repetitive patterns from transcribed text.
        e.g., "open camera open camera open camera" -> "open camera"
        IMPROVED: Detect and reject long hallucinations early
        """
        if not text:
            return text
        
        # Split into words
        words = text.split()
        if not words:
            return text
        
        # CRITICAL FIX: Reject extremely long transcriptions (likely hallucination)
        # Valid commands are 1-4 words, so >15 words is definitely wrong
        if len(words) > 15:
            print(f"[AudioEngine] Hallucination detected ({len(words)} words), taking first phrase only")
            _stt_log(f"Hallucination rejected: {len(words)} words")
            # Return only first 2-3 words (likely the actual command)
            return ' '.join(words[:3])
        
        # Detect repetition patterns
        # Check for sequences like "word1 word2 word1 word2 word1 word2"
        for pattern_len in range(1, min(6, len(words) // 2 + 1)):  # Check patterns up to 5 words
            pattern = words[:pattern_len]
            pattern_str = ' '.join(pattern).lower()
            
            # Count how many times this pattern repeats at the start
            repeat_count = 0
            idx = 0
            while idx + pattern_len <= len(words):
                current = ' '.join(words[idx:idx + pattern_len]).lower()
                if current == pattern_str:
                    repeat_count += 1
                    idx += pattern_len
                else:
                    break
            
            # If pattern repeats 3+ times, it's a hallucination - return just the pattern
            if repeat_count >= 3:
                print(f"[AudioEngine] Detected repetition pattern (x{repeat_count}): '{pattern_str}'")
                _stt_log(f"Repetition detected: '{pattern_str}' x{repeat_count}")
                return ' '.join(pattern)
        
        # Remove consecutive duplicate words
        unique_words = [words[0]]
        for word in words[1:]:
            if word.lower() != unique_words[-1].lower():
                unique_words.append(word)
        
        # If we have less than 40% unique content, it's likely hallucination
        if len(unique_words) < len(words) * 0.4:
            print(f"[AudioEngine] Low unique word ratio ({len(unique_words)}/{len(words)}), truncating")
            _stt_log(f"Low uniqueness: {len(unique_words)}/{len(words)}")
            # Take only first phrase (max 10 words)
            return ' '.join(unique_words[:10])
        
        return ' '.join(unique_words)
    
    def _filter_by_word_count(self, text: str, max_words: int = 3) -> Optional[str]:
        """
        FIX #5: Filter transcription to max_words limit.
        IMPROVED: Clean punctuation first, allow up to 4 words for commands like "open camera 1"
        """
        if not text:
            return text
        
        # Clean punctuation and extra whitespace before counting
        import string
        cleaned = text.translate(str.maketrans('', '', string.punctuation))
        words = cleaned.split()
        word_count = len(words)
        
        # Be more lenient: allow up to 4 words (covers "open robot cell", "open camera 1", etc.)
        # Only reject if significantly longer (5+ words likely not a valid command)
        if word_count > 4:
            print(f"[AudioEngine] Text exceeds word limit ({word_count} words): '{text}'")
            _stt_log(f"Rejected: {word_count} words > 4 limit")
            return None
        
        return text
    
    def _aggregate_commands(self, text: str, available_commands: List[str]) -> Optional[str]:
        """
        FIX #7: Command aggregation and disambiguation logic.
        When transcription contains multiple command instances, select by:
        1. Prefer longer/more specific commands (e.g., "open camera 1" over "open camera")
        2. Highest frequency (mode)
        3. Tie-breaker: first occurrence
        
        Args:
            text: Raw transcription that may contain multiple commands
            available_commands: List of valid command phrases
        
        Returns:
            Single disambiguated command, or None if no commands found
        """
        if not text or not available_commands:
            return None
        
        text_lower = text.lower()
        found_commands = []
        
        # PRIORITY FIX: Sort commands by length (longest first) to match specific commands first
        # This prevents "open camera" from matching when "open camera 1" is present
        sorted_commands = sorted(available_commands, key=lambda x: len(x), reverse=True)
        
        # Find all command occurrences in text
        matched_positions = set()  # Track positions already matched by longer commands
        
        for cmd in sorted_commands:
            cmd_lower = cmd.lower()
            # Count occurrences, but skip if position overlaps with a longer match
            start_pos = 0
            while True:
                pos = text_lower.find(cmd_lower, start_pos)
                if pos == -1:
                    break
                
                # Check if this position overlaps with an already-matched longer command
                cmd_end = pos + len(cmd_lower)
                overlaps = False
                for matched_start, matched_end in matched_positions:
                    if not (cmd_end <= matched_start or pos >= matched_end):
                        overlaps = True
                        break
                
                if not overlaps:
                    found_commands.append((pos, cmd))
                    matched_positions.add((pos, cmd_end))
                
                start_pos = pos + 1
        
        if not found_commands:
            return None
        
        # Count frequency of each command
        from collections import Counter
        command_list = [cmd for _, cmd in found_commands]
        command_freq = Counter(command_list)
        
        # Find command(s) with highest frequency
        max_freq = max(command_freq.values())
        top_commands = [cmd for cmd, freq in command_freq.items() if freq == max_freq]
        
        # If tie in frequency, prefer longer command, then first occurrence
        if len(top_commands) > 1:
            # First, prefer longer commands (more specific)
            max_length = max(len(cmd) for cmd in top_commands)
            longest_commands = [cmd for cmd in top_commands if len(cmd) == max_length]
            
            if len(longest_commands) == 1:
                selected = longest_commands[0]
                print(f"[AudioEngine] Command selected (longest): '{selected}' (freq={max_freq})")
                _stt_log(f"Aggregation: '{selected}' selected as longest from {len(top_commands)} tied commands")
                return selected
            
            # If still tied, select by first occurrence
            first_pos = float('inf')
            selected = longest_commands[0]
            for pos, cmd in found_commands:
                if cmd in longest_commands and pos < first_pos:
                    first_pos = pos
                    selected = cmd
            print(f"[AudioEngine] Tie broken by first occurrence: '{selected}' (freq={max_freq})")
            _stt_log(f"Aggregation: '{selected}' selected from {len(top_commands)} tied commands")
            return selected
        else:
            selected = top_commands[0]
            print(f"[AudioEngine] Command aggregated: '{selected}' (freq={max_freq})")
            _stt_log(f"Aggregation: '{selected}' selected with freq={max_freq}")
            return selected
    
    def reset_wake_state(self):
        """Reset to inactive state"""
        with self._state_lock:
            self.wake_state = self.WAKE_STATE_INACTIVE
            self._processing = False
        
        print("[AudioEngine] Wake state reset to INACTIVE")
        
        # TTS feedback
        if self.tts_mgr and not self._shutting_down:
            self.tts_mgr.speak_status("stand by")
    
    # Command management interface
    def add_command(self, text: str) -> bool:
        """Add new command"""
        if self.cmd_hotword_mgr:
            return self.cmd_hotword_mgr.add_command(text)
        return False
    
    def remove_command(self, text: str) -> bool:
        """Remove command"""
        if self.cmd_hotword_mgr:
            return self.cmd_hotword_mgr.remove_command(text)
        return False
    
    def get_all_commands(self) -> List[str]:
        """Get all commands"""
        if self.cmd_hotword_mgr:
            return self.cmd_hotword_mgr.get_all_commands()
        return []
    
    def train_command(self, text: str) -> float:
        """Train command"""
        if self.cmd_hotword_mgr:
            return self.cmd_hotword_mgr.train_command(text)
        return 0.0
    
    def get_training_count(self, text: str) -> int:
        """Get training count"""
        if self.cmd_hotword_mgr:
            return self.cmd_hotword_mgr.get_training_count(text)
        return 0
    
    # Model management interface
    def switch_model(self, model_name: str) -> bool:
        """Switch model asynchronously"""
        if not model_name or model_name == self.model_size or self._shutting_down:
            return True
        
        print(f"[AudioEngine] Switching model: {self.model_size} -> {model_name}")
        
        # Switch asynchronously to prevent blocking
        def _switch():
            try:
                with self._state_lock:
                    self._model_ready = False
                    self._processing = True
                
                if not self.model_mgr:
                    print("[AudioEngine] Model manager not available")
                    return False
                
                new_model = self.model_mgr.load_model(model_name)
                
                with self._state_lock:
                    if new_model:
                        self.model = new_model
                        self.model_size = model_name
                        self._model_ready = True
                        print(f"[AudioEngine] ✓ Model switched to {model_name}")
                        return True
                    else:
                        print(f"[AudioEngine] ✗ Failed to switch to {model_name}")
                        return False
            
            except Exception as e:
                print(f"[AudioEngine] Model switch error: {e}")
                traceback.print_exc()
                return False
            
            finally:
                with self._state_lock:
                    self._processing = False
        
        # Run in background
        thread = threading.Thread(
            target=_switch,
            daemon=True,
            name=f"ModelSwitch-{model_name}"
        )
        thread.start()
        self._active_threads.append(thread)
        return True
    
    def get_current_model(self) -> str:
        """Get current model name"""
        return self.model_size
    
    def get_available_models(self) -> List[str]:
        """Get available model names"""
        if self.model_mgr:
            return self.model_mgr.get_available_models()
        return []
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        with self._state_lock:
            status = {
                "wake_state": "ACTIVE" if self.wake_state == self.WAKE_STATE_ACTIVE else "INACTIVE",
                "wake_state_raw": self.wake_state,
                "processing": self._processing,
                "shutting_down": self._shutting_down,
                "model_size": self.model_size,
                "backend": BACKEND,
                "model_loaded": self._model_ready,
                "model_path": "Local cache" if self._model_ready else "Not loaded",
                "model_error": self._last_error or "None",
                "error_count": self._error_count,
                "components": self._components_initialized.copy(),
                "features": {
                    "tts_enabled": self._components_initialized.get("tts_engine", False),
                    "hotwords_enabled": self._components_initialized.get("command_manager", False),
                    "offline_mode": True
                }
            }
        
        # Add command statistics
        try:
            if self.cmd_hotword_mgr:
                cmd_stats = self.cmd_hotword_mgr.get_statistics()
                status["commands"] = cmd_stats
        except:
            status["commands"] = {"total": 0, "most_used": [], "highest_weight": []}
        
        # Add TTS info
        try:
            if self.tts_mgr:
                tts_info = self.tts_mgr.get_engine_info()
                status["tts"] = tts_info
        except:
            status["tts"] = {"engine_type": "unknown", "enabled": False}
        
        # Add model info
        try:
            if self.model_mgr:
                model_info = self.model_mgr.get_system_info()
                status["local_models"] = model_info
        except:
            status["local_models"] = {"available": [], "error": "Unknown"}
        
        # Add timestamps
        status["last_success"] = (
            time.ctime(self._last_success_time)
            if self._last_success_time
            else "None"
        )
        status["last_error"] = self._last_error or "None"
        
        return status
    
    def shutdown(self):
        """
        Enhanced clean shutdown with proper component ordering.
        CRITICAL for preventing resource leaks and application hangs.
        """
        print("[AudioEngine] Shutting down...")
        
        # Set shutdown flag
        with self._state_lock:
            if self._shutting_down:
                print("[AudioEngine] Already shutting down")
                return
            self._shutting_down = True
            self.wake_state = self.WAKE_STATE_INACTIVE
            self._processing = False
            self._model_ready = False
        
        # Shutdown order is CRITICAL - reverse of initialization
        # 1. Stop TTS first (has worker thread)
        if hasattr(self, 'tts_mgr') and self.tts_mgr:
            try:
                print("[AudioEngine] Shutting down TTS engine...")
                self.tts_mgr.shutdown()
                print("[AudioEngine] ✓ TTS engine shut down")
            except Exception as e:
                print(f"[AudioEngine] TTS shutdown error: {e}")
        
        # 2. Clean up audio resources
        print("[AudioEngine] Cleaning up audio resources...")
        for stream, interface in list(self._audio_resources):
            try:
                if stream:
                    stream.stop_stream()
                    stream.close()
                if interface:
                    interface.terminate()
            except Exception as e:
                print(f"[AudioEngine] Audio resource cleanup error: {e}")
        self._audio_resources.clear()
        
        # 3. Clear model references (helps with memory cleanup)
        with self._state_lock:
            self.model = None
        
        # 4. Shutdown command manager (has file I/O)
        if hasattr(self, 'cmd_hotword_mgr') and self.cmd_hotword_mgr:
            try:
                print("[AudioEngine] Shutting down command manager...")
                if hasattr(self.cmd_hotword_mgr, 'shutdown'):
                    self.cmd_hotword_mgr.shutdown()
                print("[AudioEngine] ✓ Command manager shut down")
            except Exception as e:
                print(f"[AudioEngine] Command manager shutdown error: {e}")
        
        # 5. Shutdown model manager last
        if hasattr(self, 'model_mgr') and self.model_mgr:
            try:
                print("[AudioEngine] Shutting down model manager...")
                if hasattr(self.model_mgr, 'shutdown'):
                    self.model_mgr.shutdown()
                print("[AudioEngine] ✓ Model manager shut down")
            except Exception as e:
                print(f"[AudioEngine] Model manager shutdown error: {e}")
        
        print("[AudioEngine] Shutdown complete")


# Test the enhanced audio engine
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Enhanced AudioEngine v2")
    print("=" * 60)
    
    engine = AudioEngine()
    
    # Wait for model to load
    print("\nWaiting for model to load...")
    if engine.wait_for_model(15):
        print("✓ Model loaded successfully")
        
        # Test commands
        commands = engine.get_all_commands()
        print(f"\nCommands available: {len(commands)}")
        
        # Test system status
        status = engine.get_system_status()
        print(f"\nSystem status:")
        print(f"  Wake state: {status['wake_state']}")
        print(f"  Model: {status['model_size']}")
        print(f"  Backend: {status['backend']}")
        print(f"  Components: {status['components']}")
        
    else:
        print("✗ Model loading timed out")
    
    # Shutdown
    print("\nShutting down...")
    engine.shutdown()
    print("\n✓ Test complete!")
