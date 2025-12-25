---
sidebar_position: 1
---

# Whisper Integration

## Audio Processing with OpenAI Whisper

This chapter covers integrating OpenAI Whisper for audio processing in humanoid robots. OpenAI Whisper is a state-of-the-art speech recognition model that can transcribe, translate, and understand audio in multiple languages. For humanoid robots, Whisper enables natural language interaction, command recognition, and contextual understanding of spoken instructions.

Whisper's architecture is based on a Transformer sequence-to-sequence model that maps audio spectrograms to text tokens. The model has been trained on 680,000 hours of multilingual and multitask supervised data, making it highly robust to accents, background noise, and technical language.

## Introduction to Whisper

Whisper models come in different sizes, each optimized for different performance requirements:

- **tiny**: 39M parameters, suitable for edge devices
- **base**: 74M parameters, good balance of speed and accuracy
- **small**: 244M parameters, higher accuracy
- **medium**: 769M parameters, high accuracy
- **large**: 1550M parameters, highest accuracy

For humanoid robots, the choice depends on computational resources and real-time requirements. The tiny and base models are often sufficient for command recognition, while larger models provide better accuracy for complex conversations.

```python
import whisper
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import librosa
import io
from dataclasses import dataclass
import threading
import queue
import time
from scipy import signal
import webrtcvad

@dataclass
class AudioConfig:
    """Configuration for audio processing"""
    sample_rate: int = 16000
    chunk_duration: float = 1.0  # seconds
    vad_mode: int = 1  # WebRTC VAD mode (0-3)
    silence_threshold: float = 0.3  # seconds of silence before stopping
    min_speech_duration: float = 0.5  # minimum speech duration
    max_buffer_duration: float = 10.0  # maximum buffer size in seconds

class AudioPreprocessor:
    """Preprocess audio for Whisper with noise reduction and VAD"""

    def __init__(self, config: AudioConfig):
        self.config = config
        self.vad = webrtcvad.Vad(config.vad_mode)

        # Noise reduction parameters
        self.noise_floor = 0.01
        self.snr_threshold = 10.0  # Signal-to-noise ratio threshold

        # Audio buffer for continuous processing
        self.audio_buffer = np.array([])
        self.max_buffer_size = int(config.max_buffer_duration * config.sample_rate)

        # Threading for real-time processing
        self.processing_lock = threading.Lock()

    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Preprocess audio data for Whisper"""
        # Normalize audio
        audio_data = self._normalize_audio(audio_data)

        # Apply noise reduction
        audio_data = self._reduce_noise(audio_data)

        # Apply high-pass filter to remove low-frequency noise
        audio_data = self._high_pass_filter(audio_data)

        return audio_data

    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio to optimal range for Whisper"""
        # Normalize to [-1, 1]
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        return audio_data

    def _reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply basic noise reduction"""
        # Simple spectral subtraction for noise reduction
        # Calculate power spectrum
        fft = np.fft.fft(audio_data)
        power_spectrum = np.abs(fft) ** 2

        # Estimate noise floor (assuming first 10% is noise)
        noise_floor_idx = int(len(power_spectrum) * 0.1)
        if noise_floor_idx > 0:
            noise_floor = np.mean(power_spectrum[:noise_floor_idx])
            noise_floor = max(self.noise_floor, noise_floor)

            # Subtract noise from spectrum
            enhanced_spectrum = np.maximum(power_spectrum - noise_floor, 0)

            # Apply Wiener filtering
            wiener_gain = enhanced_spectrum / (enhanced_spectrum + noise_floor)

            # Apply gain to original spectrum
            enhanced_fft = fft * np.sqrt(wiener_gain)

            # Inverse FFT
            audio_data = np.real(np.fft.ifft(enhanced_fft))

        return audio_data

    def _high_pass_filter(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply high-pass filter to remove low-frequency noise"""
        # Design high-pass filter
        nyquist = self.config.sample_rate / 2
        cutoff = 100.0  # Hz
        normalized_cutoff = cutoff / nyquist

        # Create Butterworth filter
        b, a = signal.butter(4, normalized_cutoff, btype='high', analog=False)

        # Apply filter
        filtered_audio = signal.filtfilt(b, a, audio_data)

        return filtered_audio

    def detect_voice_activity(self, audio_chunk: np.ndarray) -> bool:
        """Detect voice activity in audio chunk using WebRTC VAD"""
        # Convert to 16-bit PCM
        audio_int16 = (audio_chunk * 32767).astype(np.int16)

        # Ensure proper length (10, 20, or 30 ms frames)
        frame_duration = 20  # ms
        frame_size = int(self.config.sample_rate * frame_duration / 1000)

        if len(audio_int16) < frame_size:
            # Pad with zeros if too short
            padding = frame_size - len(audio_int16)
            audio_int16 = np.pad(audio_int16, (0, padding), mode='constant')

        # Process in frames
        frames = []
        for i in range(0, len(audio_int16), frame_size):
            frame = audio_int16[i:i+frame_size]
            if len(frame) == frame_size:
                frames.append(self.vad.is_speech(frame.tobytes(), self.config.sample_rate))

        # Voice activity if majority of frames have speech
        if frames:
            vad_ratio = sum(frames) / len(frames)
            return vad_ratio > 0.3  # At least 30% of frames should have speech
        return False

    def add_audio_chunk(self, audio_chunk: np.ndarray):
        """Add audio chunk to internal buffer"""
        with self.processing_lock:
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])

            # Limit buffer size
            max_samples = self.max_buffer_size
            if len(self.audio_buffer) > max_samples:
                self.audio_buffer = self.audio_buffer[-max_samples:]

    def get_audio_segment(self, duration: float) -> Optional[np.ndarray]:
        """Get audio segment of specified duration from buffer"""
        with self.processing_lock:
            required_samples = int(duration * self.config.sample_rate)
            if len(self.audio_buffer) >= required_samples:
                segment = self.audio_buffer[-required_samples:]
                return segment
            return None

class WhisperProcessor:
    """Process audio using Whisper model with real-time capabilities"""

    def __init__(self, model_size: str = "base", device: str = "cuda"):
        self.model_size = model_size
        self.device = device

        # Load Whisper model
        self.model = whisper.load_model(model_size, device=device)

        # Audio configuration
        self.config = AudioConfig()

        # Preprocessor
        self.preprocessor = AudioPreprocessor(self.config)

        # Threading for continuous processing
        self.processing_thread = None
        self.processing_queue = queue.Queue()
        self.running = False

        # Results queue for processed transcriptions
        self.results_queue = queue.Queue()

    def transcribe_audio(self, audio_data: np.ndarray, language: str = "en") -> Dict:
        """Transcribe audio using Whisper"""
        # Preprocess audio
        processed_audio = self.preprocessor.preprocess_audio(audio_data)

        # Convert to float32 if needed
        if processed_audio.dtype != np.float32:
            processed_audio = processed_audio.astype(np.float32)

        # Transcribe using Whisper
        result = self.model.transcribe(
            processed_audio,
            language=language,
            fp16=(self.device == "cuda")
        )

        return result

    def transcribe_with_options(self, audio_data: np.ndarray,
                              language: str = "en",
                              task: str = "transcribe",
                              temperature: float = 0.0,
                              best_of: int = 5,
                              beam_size: int = 5) -> Dict:
        """Transcribe audio with advanced options"""
        # Preprocess audio
        processed_audio = self.preprocessor.preprocess_audio(audio_data)

        # Convert to float32 if needed
        if processed_audio.dtype != np.float32:
            processed_audio = processed_audio.astype(np.float32)

        # Create options
        options = whisper.DecodingOptions(
            language=language,
            task=task,
            temperature=temperature,
            best_of=best_of,
            beam_size=beam_size,
            fp16=(self.device == "cuda")
        )

        # Decode
        result = self.model.decode(processed_audio, options)

        return {
            "text": result.text,
            "segments": result.tokens,
            "language": result.language
        }

    def start_continuous_processing(self):
        """Start continuous audio processing in background thread"""
        self.running = True
        self.processing_thread = threading.Thread(target=self._continuous_processing_loop)
        self.processing_thread.start()

    def stop_continuous_processing(self):
        """Stop continuous audio processing"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()

    def _continuous_processing_loop(self):
        """Continuous processing loop for real-time transcription"""
        while self.running:
            try:
                # Get audio from queue
                audio_chunk = self.processing_queue.get(timeout=1.0)

                # Add to buffer
                self.preprocessor.add_audio_chunk(audio_chunk)

                # Check for voice activity
                vad_active = self.preprocessor.detect_voice_activity(audio_chunk)

                if vad_active:
                    # Get audio segment for processing
                    segment = self.preprocessor.get_audio_segment(duration=5.0)  # 5 seconds
                    if segment is not None:
                        # Transcribe the segment
                        result = self.transcribe_audio(segment)

                        # Add result to results queue
                        self.results_queue.put(result)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in continuous processing: {e}")
                continue

class WhisperROSInterface:
    """ROS 2 interface for Whisper integration"""

    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
        from sensor_msgs.msg import AudioData
        from audio_common_msgs.msg import AudioData as AudioDataMsg

        # Initialize ROS node
        self.node = Node('whisper_audio_processor')

        # Parameters
        self.node.declare_parameter('model_size', 'base')
        self.node.declare_parameter('device', 'cuda')
        self.node.declare_parameter('sample_rate', 16000)

        self.model_size = self.node.get_parameter('model_size').value
        self.device = self.node.get_parameter('device').value
        self.sample_rate = self.node.get_parameter('sample_rate').value

        # Initialize Whisper processor
        self.whisper_processor = WhisperProcessor(self.model_size, self.device)

        # Publishers and subscribers
        self.audio_sub = self.node.create_subscription(
            AudioDataMsg, 'audio_input', self.audio_callback, 10)
        self.transcription_pub = self.node.create_publisher(
            String, 'transcription', 10)

        # Start continuous processing
        self.whisper_processor.start_continuous_processing()

        self.node.get_logger().info('Whisper ROS interface initialized')

    def audio_callback(self, msg: AudioDataMsg):
        """Handle incoming audio data"""
        # Convert audio data to numpy array
        audio_data = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0

        # Add to processing queue
        self.whisper_processor.processing_queue.put(audio_data)

        # Check for results
        try:
            while True:
                result = self.whisper_processor.results_queue.get_nowait()

                # Publish transcription
                transcription_msg = String()
                transcription_msg.data = result['text']
                self.transcription_pub.publish(transcription_msg)

        except queue.Empty:
            pass  # No results available yet

    def shutdown(self):
        """Shutdown the ROS interface"""
        self.whisper_processor.stop_continuous_processing()

def main():
    """Main function for standalone operation"""
    import rclpy

    rclpy.init()

    # Create Whisper ROS interface
    whisper_interface = WhisperROSInterface()

    try:
        rclpy.spin(whisper_interface.node)
    except KeyboardInterrupt:
        pass
    finally:
        whisper_interface.shutdown()
        whisper_interface.node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Audio Preprocessing

Effective audio preprocessing is crucial for optimal Whisper performance, especially in robotic environments where background noise, reverberation, and varying acoustic conditions can significantly impact recognition accuracy.

```python
import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
import librosa
from typing import Tuple, Optional
import webrtcvad
import pyaudio
import threading

class AdvancedAudioPreprocessor:
    """Advanced audio preprocessing for Whisper with multiple noise reduction techniques"""

    def __init__(self, sample_rate: int = 16000, frame_duration: float = 0.03):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration)

        # WebRTC VAD for voice activity detection
        self.vad = webrtcvad.Vad(2)  # Aggressive mode

        # Spectral subtraction parameters
        self.noise_floor = 0.01
        self.alpha = 0.9  # Noise estimation smoothing factor
        self.beta = 0.05  # Over-subtraction factor
        self.smoothing_factor = 0.8

        # Noise estimate (will be updated during processing)
        self.noise_estimate = None
        self.initialized = False

        # Audio buffer for continuous processing
        self.audio_buffer = np.array([])

    def preprocess_audio_batch(self, audio_data: np.ndarray) -> np.ndarray:
        """Preprocess audio in batch mode"""
        # Resample if needed
        if self.sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=self.sample_rate, target_sr=16000)
            self.sample_rate = 16000

        # Normalize audio
        audio_data = self._normalize_audio(audio_data)

        # Apply noise reduction
        audio_data = self._spectral_subtraction(audio_data)

        # Apply Wiener filtering
        audio_data = self._wiener_filter(audio_data)

        # Apply high-pass filter
        audio_data = self._high_pass_filter(audio_data)

        return audio_data

    def preprocess_audio_stream(self, audio_chunk: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Preprocess audio chunk in streaming mode with VAD"""
        # Normalize
        audio_chunk = self._normalize_audio(audio_chunk)

        # Apply noise reduction
        audio_chunk = self._spectral_subtraction_stream(audio_chunk)

        # Apply high-pass filter
        audio_chunk = self._high_pass_filter(audio_chunk)

        # Detect voice activity
        vad_active = self._detect_voice_activity(audio_chunk)

        return audio_chunk, vad_active

    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio to optimal range"""
        # Peak normalization
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
            # Scale to optimal range for Whisper (-0.8 to 0.8)
            audio_data = audio_data * 0.8

        return audio_data

    def _spectral_subtraction(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply spectral subtraction noise reduction"""
        # Convert to frequency domain
        fft_data = fft(audio_data)
        power_spectrum = np.abs(fft_data) ** 2

        # Estimate noise floor using minimum statistics
        if self.noise_estimate is None:
            self.noise_estimate = power_spectrum.copy()

        # Update noise estimate
        self.noise_estimate = self.alpha * self.noise_estimate + (1 - self.alpha) * power_spectrum

        # Apply spectral subtraction
        enhanced_spectrum = np.maximum(power_spectrum - self.beta * self.noise_estimate, 0)

        # Reconstruct signal
        magnitude = np.sqrt(enhanced_spectrum)
        enhanced_fft = magnitude * np.exp(1j * np.angle(fft_data))

        # Convert back to time domain
        enhanced_audio = np.real(ifft(enhanced_fft))

        return enhanced_audio.astype(np.float32)

    def _spectral_subtraction_stream(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Apply spectral subtraction for streaming audio"""
        # Convert to frequency domain
        fft_chunk = fft(audio_chunk)
        power_spectrum = np.abs(fft_chunk) ** 2

        if self.noise_estimate is None:
            # Initialize noise estimate with first chunk
            self.noise_estimate = power_spectrum.copy()
            self.initialized = True

        # Update noise estimate (only during silence periods for streaming)
        if not self.initialized:
            self.noise_estimate = self.alpha * self.noise_estimate + (1 - self.alpha) * power_spectrum
        else:
            # Use more conservative update during speech
            self.noise_estimate = 0.99 * self.noise_estimate + 0.01 * power_spectrum

        # Apply spectral subtraction
        enhanced_spectrum = np.maximum(power_spectrum - self.beta * self.noise_estimate, 0)

        # Reconstruct signal
        magnitude = np.sqrt(enhanced_spectrum)
        enhanced_fft = magnitude * np.exp(1j * np.angle(fft_chunk))

        # Convert back to time domain
        enhanced_audio = np.real(ifft(enhanced_fft))

        return enhanced_audio.astype(np.float32)

    def _wiener_filter(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply Wiener filtering for noise reduction"""
        # Compute STFT
        f, t, Zxx = signal.stft(audio_data, fs=self.sample_rate, nperseg=512)

        # Estimate noise PSD (assuming first portion is noise)
        noise_frames = min(10, Zxx.shape[1])
        noise_psd = np.mean(np.abs(Zxx[:, :noise_frames]) ** 2, axis=1, keepdims=True)

        # Compute Wiener gain
        signal_psd = np.maximum(np.abs(Zxx) ** 2 - noise_psd, 0)
        wiener_gain = signal_psd / (signal_psd + noise_psd)

        # Apply gain
        enhanced_Zxx = Zxx * wiener_gain

        # Inverse STFT
        _, enhanced_audio = signal.istft(enhanced_Zxx, fs=self.sample_rate, nperseg=512)

        return enhanced_audio.astype(np.float32)

    def _high_pass_filter(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply high-pass filter to remove low-frequency noise"""
        # Design high-pass filter
        nyquist = self.sample_rate / 2
        cutoff = 80.0  # Hz (remove very low frequencies that don't contain speech)
        normalized_cutoff = cutoff / nyquist

        # Create Butterworth filter
        b, a = signal.butter(4, normalized_cutoff, btype='high', analog=False)

        # Apply filter
        filtered_audio = signal.filtfilt(b, a, audio_data)

        return filtered_audio

    def _detect_voice_activity(self, audio_chunk: np.ndarray) -> bool:
        """Detect voice activity using WebRTC VAD"""
        # Convert to 16-bit PCM
        audio_int16 = (audio_chunk * 32767).astype(np.int16)

        # Ensure proper frame size (10, 20, or 30 ms)
        frame_size = int(self.sample_rate * self.frame_duration)
        if len(audio_int16) < frame_size:
            # Pad with zeros
            padding = frame_size - len(audio_int16)
            audio_int16 = np.pad(audio_int16, (0, padding), mode='constant')

        # Process in frames
        frames = []
        for i in range(0, len(audio_int16), frame_size):
            frame = audio_int16[i:i+frame_size]
            if len(frame) == frame_size:
                try:
                    is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
                    frames.append(is_speech)
                except:
                    # Skip problematic frames
                    continue

        # Voice activity if majority of frames have speech
        if frames:
            vad_ratio = sum(frames) / len(frames)
            return vad_ratio > 0.2  # At least 20% of frames should have speech
        return False

    def adaptive_noise_estimation(self, audio_data: np.ndarray, vad_active: bool):
        """Adaptively estimate noise based on VAD results"""
        if not vad_active and len(audio_data) > 0:
            # This is likely a noise segment, update noise estimate
            fft_data = fft(audio_data)
            power_spectrum = np.abs(fft_data) ** 2

            if self.noise_estimate is None:
                self.noise_estimate = power_spectrum.copy()
            else:
                # Update noise estimate with slower adaptation
                self.noise_estimate = 0.95 * self.noise_estimate + 0.05 * power_spectrum

class RealTimeAudioProcessor:
    """Real-time audio processor for continuous Whisper integration"""

    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 0.5):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)

        # Preprocessor
        self.preprocessor = AdvancedAudioPreprocessor(sample_rate)

        # Audio buffer
        self.audio_buffer = np.array([])
        self.max_buffer_duration = 10.0  # 10 seconds max
        self.max_buffer_size = int(sample_rate * self.max_buffer_duration)

        # Voice activity detection
        self.vad_active = False
        self.speech_start_time = None
        self.silence_start_time = None

        # Threading
        self.processing_lock = threading.Lock()
        self.audio_queue = queue.Queue()

    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """Process incoming audio chunk and return speech segments for transcription"""
        with self.processing_lock:
            # Preprocess chunk
            processed_chunk, vad_active = self.preprocessor.preprocess_audio_stream(audio_chunk)

            # Add to buffer
            self.audio_buffer = np.concatenate([self.audio_buffer, processed_chunk])

            # Limit buffer size
            if len(self.audio_buffer) > self.max_buffer_size:
                self.audio_buffer = self.audio_buffer[-self.max_buffer_size:]

            # Update VAD state
            current_time = time.time()
            if vad_active:
                # Speech detected
                if not self.vad_active:
                    # Just started speaking
                    self.speech_start_time = current_time
                    self.silence_start_time = None

                self.vad_active = True
            else:
                # Silence detected
                if self.vad_active:
                    # Just stopped speaking
                    self.silence_start_time = current_time

                self.vad_active = False

            # Update noise estimate
            self.preprocessor.adaptive_noise_estimation(processed_chunk, vad_active)

            # Check if we have a complete speech segment to return
            if (not self.vad_active and
                self.silence_start_time and
                (current_time - self.silence_start_time) > 0.5 and  # 0.5s silence threshold
                self.speech_start_time):

                # Extract speech segment
                speech_duration = self.silence_start_time - self.speech_start_time
                required_samples = int(speech_duration * self.sample_rate)

                if len(self.audio_buffer) >= required_samples:
                    speech_segment = self.audio_buffer[-required_samples:]

                    # Clear buffer after extracting speech
                    self.audio_buffer = np.array([])
                    self.speech_start_time = None
                    self.silence_start_time = None

                    return speech_segment

            return None

# Example usage of audio preprocessing
def example_audio_preprocessing():
    """Example of using the audio preprocessing pipeline"""
    # Create preprocessor
    preprocessor = AdvancedAudioPreprocessor()

    # Load example audio file
    # audio, sr = librosa.load("example_audio.wav", sr=16000)

    # For this example, create synthetic audio with noise
    duration = 5.0  # seconds
    t = np.linspace(0, duration, int(16000 * duration))
    # Create a simple sine wave with noise
    clean_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    noise = np.random.normal(0, 0.1, len(clean_audio))
    noisy_audio = clean_audio + noise

    # Preprocess the audio
    processed_audio = preprocessor.preprocess_audio_batch(noisy_audio)

    print(f"Original audio shape: {noisy_audio.shape}")
    print(f"Processed audio shape: {processed_audio.shape}")

    return processed_audio

def example_real_time_processing():
    """Example of real-time audio processing"""
    processor = RealTimeAudioProcessor()

    # Simulate processing of audio chunks
    chunk_duration = 0.1  # 100ms chunks
    chunk_size = int(16000 * chunk_duration)

    for i in range(100):  # Simulate 10 seconds of audio
        # Generate synthetic audio chunk
        t = np.linspace(i * chunk_duration, (i + 1) * chunk_duration, chunk_size)
        chunk = 0.3 * np.sin(2 * np.pi * 880 * t)  # 880 Hz tone
        chunk += np.random.normal(0, 0.05, chunk_size)  # Add noise

        # Process chunk
        speech_segment = processor.process_audio_chunk(chunk)

        if speech_segment is not None:
            print(f"Detected speech segment of {len(speech_segment) / 16000:.2f} seconds")
            # Here you would send the segment to Whisper for transcription
```

## Speech Recognition

Speech recognition with Whisper involves not just transcription but also language identification, translation, and understanding of context. For humanoid robots, this extends to command interpretation and intent recognition.

```python
import whisper
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import json
import asyncio
from dataclasses import dataclass
import re
from transformers import pipeline
import spacy

@dataclass
class RecognitionResult:
    """Result from speech recognition"""
    text: str
    language: str
    confidence: float
    segments: List[Dict]
    timestamp: float
    command_intent: Optional[str] = None
    entities: Optional[List[Dict]] = None

class CommandInterpreter:
    """Interpret recognized speech as commands for humanoid robots"""

    def __init__(self):
        # Load spaCy model for NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Define command patterns
        self.command_patterns = {
            'move': [
                r'go to (.+)',
                r'move to (.+)',
                r'walk to (.+)',
                r'go (.+)',
                r'move (.+)',
                r'walk (.+)'
            ],
            'grasp': [
                r'pick up (.+)',
                r'grab (.+)',
                r'get (.+)',
                r'pick (.+)',
                r'take (.+)'
            ],
            'navigation': [
                r'find (.+)',
                r'locate (.+)',
                r'look for (.+)',
                r'search for (.+)'
            ],
            'interaction': [
                r'talk to (.+)',
                r'speak to (.+)',
                r'hello (.+)',
                r'hi (.+)'
            ],
            'action': [
                r'wave',
                r'nod',
                r'shake',
                r'dance',
                r'sit',
                r'stand',
                r'jump'
            ]
        }

        # Location keywords
        self.location_keywords = {
            'kitchen', 'living room', 'bedroom', 'bathroom', 'office',
            'dining room', 'hallway', 'garage', 'garden', 'entrance'
        }

        # Object keywords
        self.object_keywords = {
            'cup', 'bottle', 'book', 'phone', 'keys', 'wallet',
            'apple', 'banana', 'water', 'milk', 'bread', 'chair'
        }

    def extract_command_intent(self, text: str) -> Tuple[Optional[str], List[Dict]]:
        """Extract command intent and entities from text"""
        if self.nlp:
            doc = self.nlp(text.lower())
        else:
            doc = None

        # Find command type
        command_type = None
        entities = []

        for cmd_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text.lower())
                if match:
                    command_type = cmd_type
                    # Extract the argument
                    arg = match.group(1).strip()
                    entities.append({
                        'type': 'argument',
                        'value': arg,
                        'start': match.start(1),
                        'end': match.end(1)
                    })
                    break
            if command_type:
                break

        # If no pattern matched, try NLP approach
        if not command_type and doc:
            # Look for verbs that indicate actions
            for token in doc:
                if token.pos_ == 'VERB':
                    # Check if this verb is in our command patterns
                    for cmd_type, patterns in self.command_patterns.items():
                        for pattern in patterns:
                            if token.lemma_ in pattern:
                                command_type = cmd_type
                                break
                        if command_type:
                            break
                    if command_type:
                        break

        # Extract additional entities using NLP
        if doc:
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'GPE', 'ORG', 'MONEY', 'TIME', 'DATE']:
                    entities.append({
                        'type': ent.label_,
                        'value': ent.text,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })

            # Extract location and object entities
            for token in doc:
                if token.text in self.location_keywords:
                    entities.append({
                        'type': 'location',
                        'value': token.text,
                        'start': token.idx,
                        'end': token.idx + len(token.text)
                    })
                elif token.text in self.object_keywords:
                    entities.append({
                        'type': 'object',
                        'value': token.text,
                        'start': token.idx,
                        'end': token.idx + len(token.text)
                    })

        return command_type, entities

class WhisperSpeechRecognizer:
    """Advanced speech recognition using Whisper with command interpretation"""

    def __init__(self, model_size: str = "base", device: str = "cuda"):
        self.model_size = model_size
        self.device = device

        # Load Whisper model
        self.model = whisper.load_model(model_size, device=device)

        # Command interpreter
        self.command_interpreter = CommandInterpreter()

        # Audio preprocessing
        self.preprocessor = AdvancedAudioPreprocessor()

        # Language identification (if needed)
        self.language_identification = pipeline(
            "language-detection",
            model="papluca/xlm-roberta-base-language-detection"
        ) if model_size != "large" else None

    def recognize_speech(self, audio_data: np.ndarray,
                        language: Optional[str] = None,
                        temperature: float = 0.0,
                        best_of: int = 5,
                        beam_size: int = 5) -> RecognitionResult:
        """Recognize speech and interpret as command"""
        # Preprocess audio
        processed_audio = self.preprocessor.preprocess_audio_batch(audio_data)

        # Convert to float32
        if processed_audio.dtype != np.float32:
            processed_audio = processed_audio.astype(np.float32)

        # Transcribe with Whisper
        if language:
            result = self.model.transcribe(
                processed_audio,
                language=language,
                temperature=temperature,
                best_of=best_of,
                beam_size=beam_size,
                fp16=(self.device == "cuda")
            )
        else:
            # Let Whisper detect language automatically
            result = self.model.transcribe(
                processed_audio,
                temperature=temperature,
                best_of=best_of,
                beam_size=beam_size,
                fp16=(self.device == "cuda")
            )

        # Extract text and language
        text = result.get('text', '').strip()
        detected_language = result.get('language', 'unknown')

        # Calculate confidence (simplified approach)
        confidence = self._estimate_confidence(result, text)

        # Extract segments
        segments = result.get('segments', [])

        # Interpret command
        command_intent, entities = self.command_interpreter.extract_command_intent(text)

        # Create result
        recognition_result = RecognitionResult(
            text=text,
            language=detected_language,
            confidence=confidence,
            segments=segments,
            timestamp=time.time(),
            command_intent=command_intent,
            entities=entities
        )

        return recognition_result

    def recognize_batch(self, audio_segments: List[np.ndarray]) -> List[RecognitionResult]:
        """Recognize multiple audio segments in batch"""
        results = []

        for audio_segment in audio_segments:
            result = self.recognize_speech(audio_segment)
            results.append(result)

        return results

    def _estimate_confidence(self, whisper_result: Dict, text: str) -> float:
        """Estimate confidence of transcription"""
        if not text.strip():
            return 0.0

        # Simple confidence estimation based on various factors
        confidence = 1.0

        # Length-based confidence (very short transcriptions might be unreliable)
        if len(text.strip()) < 3:
            confidence *= 0.5
        elif len(text.strip()) > 100:
            confidence *= 0.9  # Longer texts might have more errors

        # Check for common error patterns
        if any(word in text.lower() for word in ['you know', 'um', 'uh', 'like']):
            confidence *= 0.8  # Filler words might indicate uncertainty

        # Check for repeated words (might indicate decoding errors)
        words = text.lower().split()
        if len(words) > 1 and len(set(words)) / len(words) < 0.7:  # Too many repetitions
            confidence *= 0.6

        return max(0.0, min(1.0, confidence))

    def translate_speech(self, audio_data: np.ndarray, target_language: str = "en") -> RecognitionResult:
        """Translate speech to target language"""
        # Preprocess audio
        processed_audio = self.preprocessor.preprocess_audio_batch(audio_data)

        # Convert to float32
        if processed_audio.dtype != np.float32:
            processed_audio = processed_audio.astype(np.float32)

        # Translate with Whisper
        result = self.model.transcribe(
            processed_audio,
            task="translate",
            language="auto",  # Detect source language automatically
            temperature=0.0,
            best_of=5,
            beam_size=5,
            fp16=(self.device == "cuda")
        )

        # Interpret the translated result
        text = result.get('text', '').strip()
        source_language = result.get('language', 'unknown')

        # Calculate confidence
        confidence = self._estimate_confidence(result, text)

        # Extract segments
        segments = result.get('segments', [])

        # Interpret command
        command_intent, entities = self.command_interpreter.extract_command_intent(text)

        # Create result
        recognition_result = RecognitionResult(
            text=text,
            language=target_language,  # Target language
            confidence=confidence,
            segments=segments,
            timestamp=time.time(),
            command_intent=command_intent,
            entities=entities
        )

        return recognition_result

    def detect_language(self, audio_data: np.ndarray) -> str:
        """Detect language of audio (alternative method)"""
        if self.language_identification:
            # Preprocess and convert to text first
            processed_audio = self.preprocessor.preprocess_audio_batch(audio_data)
            # This is a simplified approach - in practice, you'd use audio-based language detection
            pass

        # For now, return result from Whisper
        result = self.model.transcribe(processed_audio, fp16=(self.device == "cuda"))
        return result.get('language', 'unknown')

class ContextAwareRecognizer:
    """Context-aware speech recognition that considers conversation history"""

    def __init__(self, whisper_recognizer: WhisperSpeechRecognizer):
        self.whisper_recognizer = whisper_recognizer
        self.conversation_history = []
        self.context_window = 5  # Keep last 5 interactions

    def recognize_with_context(self, audio_data: np.ndarray,
                             current_context: Optional[Dict] = None) -> RecognitionResult:
        """Recognize speech with context awareness"""
        # Perform initial recognition
        result = self.whisper_recognizer.recognize_speech(audio_data)

        # Apply context-aware corrections if needed
        if current_context:
            result = self._apply_context_corrections(result, current_context)

        # Update conversation history
        self.conversation_history.append({
            'timestamp': result.timestamp,
            'text': result.text,
            'intent': result.command_intent,
            'entities': result.entities
        })

        # Limit history size
        if len(self.conversation_history) > self.context_window:
            self.conversation_history = self.conversation_history[-self.context_window:]

        return result

    def _apply_context_corrections(self, result: RecognitionResult, context: Dict) -> RecognitionResult:
        """Apply context-aware corrections to recognition result"""
        # This is a simplified implementation
        # In a real system, you would use more sophisticated context modeling

        text = result.text

        # Example: If context indicates we're in a kitchen, correct common misrecognitions
        if context.get('location') == 'kitchen':
            # Correct common kitchen-related misrecognitions
            corrections = {
                'water': ['water', 'what are', 'wonder'],
                'milk': ['milk', 'mill', 'make'],
                'bread': ['bread', 'bred', 'read']
            }

            for correct_word, possible_mistakes in corrections.items():
                for mistake in possible_mistakes:
                    if mistake in text.lower() and correct_word in context.get('available_objects', []):
                        text = re.sub(r'\b' + mistake + r'\b', correct_word, text, flags=re.IGNORECASE)

        # Update result with corrected text
        result.text = text

        # Re-interpret command with corrected text
        command_intent, entities = self.whisper_recognizer.command_interpreter.extract_command_intent(text)
        result.command_intent = command_intent
        result.entities = entities

        return result

    def get_conversation_context(self) -> Dict:
        """Get current conversation context"""
        return {
            'history': self.conversation_history[-3:],  # Last 3 interactions
            'common_entities': self._extract_common_entities(),
            'topic': self._infer_current_topic()
        }

    def _extract_common_entities(self) -> List[str]:
        """Extract commonly mentioned entities from conversation history"""
        entities = []
        for interaction in self.conversation_history:
            if interaction.get('entities'):
                for entity in interaction['entities']:
                    entities.append(entity['value'])

        # Return most common entities
        from collections import Counter
        entity_counts = Counter(entities)
        return [entity for entity, count in entity_counts.most_common(5)]

    def _infer_current_topic(self) -> str:
        """Infer current conversation topic"""
        if not self.conversation_history:
            return "general"

        # Simple topic inference based on recent entities and intents
        recent_intents = [h.get('intent', 'unknown') for h in self.conversation_history[-3:]]
        intent_counts = Counter(recent_intents)

        # Return most common intent as topic
        most_common_intent = intent_counts.most_common(1)
        if most_common_intent:
            return most_common_intent[0][0]

        return "general"

# Example usage of speech recognition
def example_speech_recognition():
    """Example of using the speech recognition system"""
    # Initialize recognizer
    recognizer = WhisperSpeechRecognizer(model_size="base")

    # For this example, we'll create synthetic audio
    # In practice, you would load actual audio data
    duration = 3.0  # seconds
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Create synthetic "speech" (in practice, this would be actual recorded audio)
    synthetic_audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # Base tone
    synthetic_audio += 0.2 * np.sin(2 * np.pi * 660 * t)  # Harmonic
    synthetic_audio += np.random.normal(0, 0.05, len(synthetic_audio))  # Noise

    # Recognize speech
    result = recognizer.recognize_speech(synthetic_audio)

    print(f"Recognized text: {result.text}")
    print(f"Language: {result.language}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Command intent: {result.command_intent}")
    print(f"Entities: {result.entities}")

    return result

def example_context_aware_recognition():
    """Example of context-aware speech recognition"""
    # Initialize recognizer
    whisper_rec = WhisperSpeechRecognizer(model_size="base")
    context_rec = ContextAwareRecognizer(whisper_rec)

    # Simulate a conversation in a kitchen context
    context = {
        'location': 'kitchen',
        'available_objects': ['water', 'milk', 'bread', 'apple', 'banana']
    }

    # For this example, create synthetic audio
    duration = 2.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    synthetic_audio = 0.3 * np.sin(2 * np.pi * 523 * t)  # C note
    synthetic_audio += np.random.normal(0, 0.05, len(synthetic_audio))

    # Recognize with context
    result = context_rec.recognize_with_context(synthetic_audio, context)

    print(f"Context-aware recognition result:")
    print(f"Text: {result.text}")
    print(f"Intent: {result.command_intent}")
    print(f"Context: {context_rec.get_conversation_context()}")

    return result
```

## Integration with ROS 2

Integrating Whisper with ROS 2 enables humanoid robots to process audio in real-time and respond to voice commands. This integration involves creating ROS 2 nodes for audio capture, preprocessing, transcription, and command execution.

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import AudioData
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus
from audio_common_msgs.msg import AudioData as AudioDataMsg
import whisper
import numpy as np
from typing import Dict, List, Optional
import threading
import queue
import time
from dataclasses import dataclass

@dataclass
class RobotCommand:
    """Command for humanoid robot"""
    command_type: str
    arguments: Dict
    confidence: float
    timestamp: float

class AudioInputNode(Node):
    """ROS 2 node for audio input capture"""

    def __init__(self):
        super().__init__('audio_input_node')

        # Parameters
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('chunk_duration', 0.5)
        self.declare_parameter('device_index', -1)  # Use default device

        self.sample_rate = self.get_parameter('sample_rate').value
        self.chunk_duration = self.get_parameter('chunk_duration').value
        self.device_index = self.get_parameter('device_index').value

        # Publisher for audio data
        qos_profile = QoSProfile(depth=10)
        qos_profile.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.audio_pub = self.create_publisher(AudioDataMsg, 'audio_input', qos_profile)

        # Initialize audio input
        try:
            import pyaudio
            self.pyaudio = pyaudio
            self.audio = pyaudio.PyAudio()

            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=int(self.sample_rate * self.chunk_duration),
                input_device_index=self.device_index if self.device_index >= 0 else None
            )

            # Start audio capture timer
            self.timer = self.create_timer(self.chunk_duration, self.capture_audio)

            self.get_logger().info('Audio input node initialized')

        except ImportError:
            self.get_logger().error('PyAudio not available. Install with: pip install pyaudio')
            self.stream = None
            self.audio = None

    def capture_audio(self):
        """Capture audio chunk and publish to ROS"""
        if self.stream is None:
            return

        try:
            # Read audio data
            data = self.stream.read(int(self.sample_rate * self.chunk_duration), exception_on_overflow=False)

            # Create and publish audio message
            audio_msg = AudioDataMsg()
            audio_msg.data = data
            audio_msg.info.channels = 1
            audio_msg.info.sample_rate = self.sample_rate
            audio_msg.info.encoding = 'PCM_16'
            audio_msg.info.step = 2  # 16-bit = 2 bytes

            self.audio_pub.publish(audio_msg)

        except Exception as e:
            self.get_logger().error(f'Error capturing audio: {e}')

class WhisperTranscriptionNode(Node):
    """ROS 2 node for Whisper-based speech transcription"""

    def __init__(self):
        super().__init__('whisper_transcription_node')

        # Parameters
        self.declare_parameter('model_size', 'base')
        self.declare_parameter('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.declare_parameter('language', 'en')
        self.declare_parameter('enable_translation', False)

        self.model_size = self.get_parameter('model_size').value
        self.device = self.get_parameter('device').value
        self.language = self.get_parameter('language').value
        self.enable_translation = self.get_parameter('enable_translation').value

        # Initialize Whisper
        try:
            self.whisper_model = whisper.load_model(self.model_size, device=self.device)
            self.get_logger().info(f'Whisper model {self.model_size} loaded on {self.device}')
        except Exception as e:
            self.get_logger().error(f'Failed to load Whisper model: {e}')
            self.whisper_model = None
            return

        # Initialize audio preprocessor
        self.preprocessor = AdvancedAudioPreprocessor(sample_rate=16000)

        # Subscribers
        self.audio_sub = self.create_subscription(
            AudioDataMsg, 'audio_input', self.audio_callback, 10)

        # Publishers
        self.transcription_pub = self.create_publisher(String, 'transcription', 10)
        self.command_pub = self.create_publisher(String, 'robot_command', 10)
        self.confidence_pub = self.create_publisher(Float32, 'transcription_confidence', 10)

        # Internal state
        self.audio_buffer = np.array([])
        self.speech_segmenter = RealTimeAudioProcessor()

        self.get_logger().info('Whisper transcription node initialized')

    def audio_callback(self, msg: AudioDataMsg):
        """Handle incoming audio data"""
        if self.whisper_model is None:
            return

        try:
            # Convert audio data to numpy array
            audio_data = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0

            # Process with speech segmenter to detect complete utterances
            speech_segment = self.speech_segmenter.process_audio_chunk(audio_data)

            if speech_segment is not None:
                # Transcribe the speech segment
                self.transcribe_audio(speech_segment)

        except Exception as e:
            self.get_logger().error(f'Error processing audio: {e}')

    def transcribe_audio(self, audio_segment: np.ndarray):
        """Transcribe audio segment using Whisper"""
        try:
            # Preprocess audio
            processed_audio = self.preprocessor.preprocess_audio_batch(audio_segment)

            # Ensure correct format
            if processed_audio.dtype != np.float32:
                processed_audio = processed_audio.astype(np.float32)

            # Transcribe with Whisper
            if self.enable_translation:
                result = self.whisper_model.transcribe(
                    processed_audio,
                    task="translate",
                    language=self.language,
                    fp16=(self.device == "cuda")
                )
            else:
                result = self.whisper_model.transcribe(
                    processed_audio,
                    language=self.language,
                    fp16=(self.device == "cuda")
                )

            # Extract text
            text = result.get('text', '').strip()

            if text:
                # Publish transcription
                transcription_msg = String()
                transcription_msg.data = text
                self.transcription_pub.publish(transcription_msg)

                # Estimate and publish confidence
                confidence = self._estimate_transcription_confidence(result, text)
                confidence_msg = Float32()
                confidence_msg.data = confidence
                self.confidence_pub.publish(confidence_msg)

                # Interpret as command and publish if high confidence
                if confidence > 0.7:  # Only publish commands with high confidence
                    command_intent, entities = self.interpret_command(text)
                    if command_intent:
                        command_msg = String()
                        command_msg.data = f"{command_intent}:{text}"
                        self.command_pub.publish(command_msg)

                self.get_logger().info(f'Transcribed: "{text}" (confidence: {confidence:.2f})')

        except Exception as e:
            self.get_logger().error(f'Error in transcription: {e}')

    def _estimate_transcription_confidence(self, result: Dict, text: str) -> float:
        """Estimate confidence of transcription"""
        if not text.strip():
            return 0.0

        # Simple confidence estimation
        confidence = 1.0

        # Length-based adjustment
        if len(text.strip()) < 3:
            confidence *= 0.3
        elif len(text.strip()) > 100:
            confidence *= 0.9

        # Check for common filler words
        if any(word in text.lower() for word in ['you know', 'um', 'uh', 'like', 'so']):
            confidence *= 0.7

        return max(0.0, min(1.0, confidence))

    def interpret_command(self, text: str) -> Tuple[Optional[str], List[Dict]]:
        """Interpret transcribed text as robot command"""
        # This would use the CommandInterpreter from the previous section
        interpreter = CommandInterpreter()
        return interpreter.extract_command_intent(text)

class CommandExecutionNode(Node):
    """ROS 2 node for executing robot commands from speech"""

    def __init__(self):
        super().__init__('command_execution_node')

        # Subscribers
        self.command_sub = self.create_subscription(
            String, 'robot_command', self.command_callback, 10)

        # Publishers for robot control
        self.navigation_pub = self.create_publisher(PoseStamped, 'navigation_goal', 10)
        self.action_pub = self.create_publisher(String, 'robot_action', 10)

        # Internal state
        self.command_queue = queue.Queue()
        self.command_thread = threading.Thread(target=self.command_worker)
        self.command_thread.daemon = True
        self.command_thread.start()

        self.get_logger().info('Command execution node initialized')

    def command_callback(self, msg: String):
        """Handle incoming robot command"""
        try:
            # Parse command
            command_parts = msg.data.split(':', 1)
            if len(command_parts) >= 2:
                command_type = command_parts[0]
                command_text = command_parts[1]

                # Create robot command
                robot_cmd = RobotCommand(
                    command_type=command_type,
                    arguments={'text': command_text},
                    confidence=1.0,  # This would come from transcription confidence
                    timestamp=time.time()
                )

                # Add to queue for processing
                self.command_queue.put(robot_cmd)

                self.get_logger().info(f'Command received: {command_type} - {command_text}')

        except Exception as e:
            self.get_logger().error(f'Error parsing command: {e}')

    def command_worker(self):
        """Worker thread for processing commands"""
        while rclpy.ok():
            try:
                # Get command from queue (with timeout to allow graceful shutdown)
                command = self.command_queue.get(timeout=1.0)

                # Execute command
                self.execute_command(command)

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Error in command worker: {e}')

    def execute_command(self, command: RobotCommand):
        """Execute robot command based on type"""
        if command.command_type == 'move':
            self.execute_navigation_command(command)
        elif command.command_type == 'grasp':
            self.execute_manipulation_command(command)
        elif command.command_type == 'action':
            self.execute_action_command(command)
        else:
            self.get_logger().info(f'Unknown command type: {command.command_type}')

    def execute_navigation_command(self, command: RobotCommand):
        """Execute navigation command"""
        target_location = command.arguments.get('text', '').lower()

        # Simple location mapping (in a real system, this would use a map)
        location_map = {
            'kitchen': (2.0, 1.0, 0.0),
            'living room': (0.0, 0.0, 0.0),
            'bedroom': (-2.0, 1.0, 0.0),
            'bathroom': (-1.0, -1.0, 0.0)
        }

        if target_location in location_map:
            x, y, theta = location_map[target_location]

            # Create navigation goal
            goal = PoseStamped()
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.header.frame_id = 'map'
            goal.pose.position.x = x
            goal.pose.position.y = y
            goal.pose.position.z = 0.0

            # Convert theta to quaternion
            import math
            goal.pose.orientation.z = math.sin(theta / 2.0)
            goal.pose.orientation.w = math.cos(theta / 2.0)

            # Publish navigation goal
            self.navigation_pub.publish(goal)
            self.get_logger().info(f'Navigating to {target_location} at ({x}, {y})')
        else:
            self.get_logger().info(f'Unknown location: {target_location}')

    def execute_manipulation_command(self, command: RobotCommand):
        """Execute manipulation command"""
        object_name = command.arguments.get('text', '').lower()

        # In a real system, this would trigger arm/leg movements
        action_msg = String()
        action_msg.data = f'grasping:{object_name}'
        self.action_pub.publish(action_msg)

        self.get_logger().info(f'Attempting to grasp {object_name}')

    def execute_action_command(self, command: RobotCommand):
        """Execute action command"""
        action_name = command.arguments.get('text', '').lower()

        action_msg = String()
        action_msg.data = f'action:{action_name}'
        self.action_pub.publish(action_msg)

        self.get_logger().info(f'Performing action: {action_name}')

def main(args=None):
    """Main function to run the Whisper ROS integration"""
    rclpy.init(args=args)

    # Create nodes
    audio_node = AudioInputNode()
    whisper_node = WhisperTranscriptionNode()
    command_node = CommandExecutionNode()

    # Create executor and add nodes
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(audio_node)
    executor.add_node(whisper_node)
    executor.add_node(command_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        if hasattr(audio_node, 'stream') and audio_node.stream:
            audio_node.stream.stop_stream()
            audio_node.stream.close()
        if hasattr(audio_node, 'audio'):
            audio_node.audio.terminate()

        audio_node.destroy_node()
        whisper_node.destroy_node()
        command_node.destroy_node()

        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Real-time Processing

Real-time processing of audio for Whisper integration requires careful consideration of latency, computational efficiency, and accuracy. The following implementation demonstrates how to achieve real-time performance while maintaining transcription quality.

```python
import asyncio
import threading
import time
from collections import deque
import numpy as np
from typing import Deque, Optional, Callable
import queue

class RealTimeWhisperProcessor:
    """Real-time Whisper processor with streaming audio support"""

    def __init__(self,
                 model_size: str = "base",
                 device: str = "cuda",
                 buffer_duration: float = 5.0,  # seconds
                 hop_length: float = 0.5):      # seconds

        self.model_size = model_size
        self.device = device
        self.buffer_duration = buffer_duration
        self.hop_length = hop_length

        # Load model
        self.model = whisper.load_model(model_size, device=device)

        # Audio processing
        self.sample_rate = 16000
        self.buffer_size = int(buffer_duration * self.sample_rate)
        self.hop_size = int(hop_length * self.sample_rate)

        # Audio buffer for streaming
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_ptr = 0
        self.audio_available = threading.Event()

        # Processing queue
        self.process_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # Threading
        self.processing_thread = None
        self.running = False

        # Callbacks
        self.transcription_callback: Optional[Callable] = None
        self.partial_result_callback: Optional[Callable] = None

        # Preprocessing
        self.preprocessor = AdvancedAudioPreprocessor(sample_rate=self.sample_rate)

    def start_processing(self):
        """Start real-time processing"""
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def stop_processing(self):
        """Stop real-time processing"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()

    def add_audio_chunk(self, audio_chunk: np.ndarray):
        """Add audio chunk to processing buffer"""
        # Convert to float32 if needed
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Preprocess chunk
        processed_chunk = self.preprocessor.preprocess_audio_batch(audio_chunk)

        # Add to circular buffer
        chunk_len = len(processed_chunk)

        if self.buffer_ptr + chunk_len <= self.buffer_size:
            self.audio_buffer[self.buffer_ptr:self.buffer_ptr + chunk_len] = processed_chunk
            self.buffer_ptr += chunk_len
        else:
            # Wrap around
            first_part = self.buffer_size - self.buffer_ptr
            self.audio_buffer[self.buffer_ptr:] = processed_chunk[:first_part]
            self.audio_buffer[:chunk_len - first_part] = processed_chunk[first_part:]
            self.buffer_ptr = (self.buffer_ptr + chunk_len) % self.buffer_size

    def _processing_loop(self):
        """Main processing loop for real-time transcription"""
        last_hop = 0

        while self.running:
            try:
                current_pos = self.buffer_ptr

                # Check if we have enough audio to process
                if current_pos >= self.hop_size:
                    # Extract audio segment for processing
                    start_idx = (current_pos - self.hop_size) % self.buffer_size
                    end_idx = current_pos

                    if start_idx < end_idx:
                        audio_segment = self.audio_buffer[start_idx:end_idx]
                    else:
                        # Handle wrap-around
                        audio_segment = np.concatenate([
                            self.audio_buffer[start_idx:],
                            self.audio_buffer[:end_idx]
                        ])

                    # Process the segment
                    self._process_audio_segment(audio_segment)

                    # Move hop forward
                    last_hop = current_pos

                    # Small delay to prevent busy waiting
                    time.sleep(0.01)
                else:
                    # Not enough audio, wait a bit
                    time.sleep(0.1)

            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(0.1)

    def _process_audio_segment(self, audio_segment: np.ndarray):
        """Process a single audio segment with Whisper"""
        try:
            # Ensure proper format
            if len(audio_segment) < 100:  # Too short to process
                return

            # Transcribe the segment
            result = self.model.transcribe(
                audio_segment,
                fp16=(self.device == "cuda")
            )

            text = result.get('text', '').strip()

            if text and len(text) > 2:  # Only process non-empty, meaningful text
                # Calculate confidence
                confidence = self._estimate_confidence(result, text)

                # Create result
                transcription_result = {
                    'text': text,
                    'language': result.get('language', 'unknown'),
                    'confidence': confidence,
                    'timestamp': time.time(),
                    'segments': result.get('segments', [])
                }

                # Call callback if available
                if self.transcription_callback:
                    self.transcription_callback(transcription_result)

                # Add to result queue
                self.result_queue.put(transcription_result)

        except Exception as e:
            print(f"Error processing audio segment: {e}")

    def _estimate_confidence(self, result: Dict, text: str) -> float:
        """Estimate confidence of transcription"""
        if not text.strip():
            return 0.0

        confidence = 1.0

        # Length-based confidence
        if len(text.strip()) < 3:
            confidence *= 0.3
        elif len(text.strip()) > 100:
            confidence *= 0.9

        # Check for common filler words
        if any(word in text.lower() for word in ['you know', 'um', 'uh', 'like', 'so']):
            confidence *= 0.7

        return max(0.0, min(1.0, confidence))

    def set_transcription_callback(self, callback: Callable):
        """Set callback for completed transcriptions"""
        self.transcription_callback = callback

    def set_partial_result_callback(self, callback: Callable):
        """Set callback for partial results (not implemented in this basic version)"""
        self.partial_result_callback = callback

    def get_results(self) -> List[Dict]:
        """Get available transcription results"""
        results = []
        try:
            while True:
                result = self.result_queue.get_nowait()
                results.append(result)
        except queue.Empty:
            pass
        return results

class StreamingAudioProcessor:
    """Advanced streaming audio processor with VAD and real-time Whisper"""

    def __init__(self, model_size: str = "base", device: str = "cuda"):
        self.model_size = model_size
        self.device = device

        # Real-time processor
        self.rt_processor = RealTimeWhisperProcessor(model_size, device)

        # Voice activity detection
        self.vad_processor = RealTimeAudioProcessor()

        # Conversation context
        self.context_aware = ContextAwareRecognizer(
            WhisperSpeechRecognizer(model_size, device)
        )

        # Results buffer
        self.results_buffer = deque(maxlen=10)  # Keep last 10 results

    def process_streaming_audio(self, audio_chunk: np.ndarray) -> Optional[Dict]:
        """Process streaming audio chunk with VAD and transcription"""
        # First, use VAD to detect speech segments
        speech_segment = self.vad_processor.process_audio_chunk(audio_chunk)

        if speech_segment is not None:
            # We have a complete speech segment, process with Whisper
            result = self.context_aware.recognize_with_context(
                speech_segment,
                current_context=self.context_aware.get_conversation_context()
            )

            # Add to results buffer
            result_dict = {
                'text': result.text,
                'language': result.language,
                'confidence': result.confidence,
                'command_intent': result.command_intent,
                'entities': result.entities,
                'timestamp': result.timestamp
            }

            self.results_buffer.append(result_dict)

            return result_dict

        return None

    def start_streaming(self):
        """Start streaming processing"""
        self.rt_processor.start_processing()

    def stop_streaming(self):
        """Stop streaming processing"""
        self.rt_processor.stop_processing()

    def get_recent_results(self, n: int = 5) -> List[Dict]:
        """Get recent transcription results"""
        return list(self.results_buffer)[-n:]

# Example of real-time processing
def example_real_time_whisper():
    """Example of real-time Whisper processing"""
    import pyaudio
    import wave

    # Initialize real-time processor
    processor = StreamingAudioProcessor(model_size="base")
    processor.start_streaming()

    # Setup audio input (simplified example)
    chunk_duration = 0.5  # 500ms chunks
    sample_rate = 16000
    chunk_size = int(sample_rate * chunk_duration)

    # Simulate audio streaming
    print("Starting real-time processing... (Press Ctrl+C to stop)")

    try:
        for i in range(100):  # Simulate 50 seconds of audio
            # In a real application, this would come from microphone input
            # For this example, we'll generate synthetic audio
            t = np.linspace(i * chunk_duration, (i + 1) * chunk_duration, chunk_size)
            # Create audio with some speech-like characteristics
            audio_chunk = 0.2 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
            audio_chunk += 0.1 * np.sin(2 * np.pi * 660 * t)  # Harmonic
            audio_chunk += np.random.normal(0, 0.05, chunk_size)  # Noise

            # Process the chunk
            result = processor.process_streaming_audio(audio_chunk)

            if result and result['confidence'] > 0.5:
                print(f"Transcription: '{result['text']}' (Conf: {result['confidence']:.2f})")

            time.sleep(chunk_duration)  # Simulate real-time pacing

    except KeyboardInterrupt:
        print("\nStopping real-time processing...")

    finally:
        processor.stop_streaming()
        recent_results = processor.get_recent_results()
        print(f"\nRecent results: {len(recent_results)} transcriptions processed")

def example_ros_integration():
    """Example of integrating real-time processing with ROS 2"""
    # This would be part of a ROS 2 node
    processor = RealTimeWhisperProcessor(model_size="base")

    def transcription_handler(result):
        """Handle completed transcriptions"""
        print(f"Real-time transcription: {result['text']}")
        # In ROS context, this would publish to a topic
        # e.g., self.transcription_publisher.publish(String(data=result['text']))

    # Set up callback
    processor.set_transcription_callback(transcription_handler)

    # Start processing
    processor.start_processing()

    # Simulate adding audio chunks (in ROS, this would come from audio subscriber)
    for i in range(20):
        # Simulate audio chunk
        chunk_size = int(16000 * 0.5)  # 500ms chunk
        t = np.linspace(i * 0.5, (i + 1) * 0.5, chunk_size)
        audio_chunk = 0.3 * np.sin(2 * np.pi * 523 * t)  # Musical note
        audio_chunk += np.random.normal(0, 0.05, chunk_size)

        processor.add_audio_chunk(audio_chunk)
        time.sleep(0.1)  # Small delay

    # Get results
    results = processor.get_results()
    print(f"Processed {len(results)} segments")

    processor.stop_processing()

# Run examples if this file is executed directly
if __name__ == "__main__":
    print("Running real-time Whisper examples...")
    example_real_time_whisper()
```

## Summary

Whisper integration enables humanoid robots to understand and respond to spoken commands through sophisticated audio processing, speech recognition, and natural language understanding. The implementation includes:

1. **Audio Preprocessing**: Advanced noise reduction, voice activity detection, and signal enhancement techniques to improve transcription accuracy in robotic environments.

2. **Speech Recognition**: Real-time transcription using OpenAI Whisper models with confidence estimation and command interpretation capabilities.

3. **ROS 2 Integration**: Complete ROS 2 nodes for audio capture, transcription, and command execution that can be integrated into humanoid robot systems.

4. **Real-time Processing**: Efficient streaming audio processing with minimal latency while maintaining high transcription accuracy.

The system is designed to be robust in real-world environments with background noise, reverberation, and varying acoustic conditions. The modular architecture allows for customization based on specific robot platforms and application requirements, making it suitable for a wide range of humanoid robot applications requiring natural language interaction.