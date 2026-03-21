"""
╔══════════════════════════════════════════════════════════════════════╗
║         InterviewLens — Multimodal Confidence & Nervousness          ║
║         Analyzer for Virtual Interview Systems                       ║
║                                                                      ║
║  Features:                                                           ║
║  • 3D head pose estimation (yaw, pitch, roll) via MediaPipe          ║
║  • ML-based emotion recognition (DeepFace / FER)                     ║
║  • Face-size normalized landmark features                            ║
║  • Temporal averaging over sliding windows (noise reduction)         ║
║  • Optional: Audio analysis (speaking rate, pitch, pauses)           ║
║  • Final confidence + nervousness scores with report export          ║
╚══════════════════════════════════════════════════════════════════════╝

Install dependencies:
    pip install opencv-python mediapipe deepface fer numpy scipy \
                pyaudio librosa soundfile pandas matplotlib tqdm
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import json
import warnings
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple
from pathlib import Path

warnings.filterwarnings("ignore")

# MediaPipe compatibility:
# Newer builds may expose only Tasks API and not legacy `solutions`.
MP_VERSION = getattr(mp, "__version__", "unknown")
try:
    from mediapipe import solutions as mp_solutions
except Exception:
    mp_solutions = None

# ── Optional imports (graceful degradation) ──────────────────────────
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Video
    camera_index: int = 0
    frame_width:  int = 1280
    frame_height: int = 720
    fps_target:   int = 30

    # Temporal window
    temporal_window: int = 15       # frames to average over
    history_maxlen:  int = 300      # ~10s at 30fps

    # Head pose thresholds (degrees)
    yaw_threshold:   float = 20.0
    pitch_threshold: float = 15.0
    roll_threshold:  float = 12.0

    # Blink detection
    ear_threshold:   float = 0.20   # eye aspect ratio below = blink
    blink_cooldown:  float = 0.20   # seconds between blinks

    # Nervousness weights
    w_pose:     float = 0.30
    w_blink:    float = 0.20
    w_gaze:     float = 0.20
    w_sway:     float = 0.15
    w_emotion:  float = 0.15

    # Audio (optional)
    audio_sample_rate:  int = 16000
    audio_chunk_size:   int = 1024
    audio_window_secs:  float = 3.0

    # Output
    output_dir:  str = "interview_results"
    show_mesh:   bool = True
    show_plots:  bool = True


# ══════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class HeadPose:
    yaw:   float = 0.0   # left/right rotation
    pitch: float = 0.0   # up/down tilt
    roll:  float = 0.0   # side tilt

@dataclass
class EmotionScores:
    angry:    float = 0.0
    disgust:  float = 0.0
    fear:     float = 0.0
    happy:    float = 0.0
    sad:      float = 0.0
    surprise: float = 0.0
    neutral:  float = 0.0

@dataclass
class FrameFeatures:
    timestamp:    float = 0.0
    face_found:   bool  = False
    face_scale:   float = 1.0        # inter-ocular distance

    # 3D head pose
    pose:         HeadPose = field(default_factory=HeadPose)

    # Eye features (normalized)
    ear_left:     float = 0.0
    ear_right:    float = 0.0
    ear_avg:      float = 0.0
    blink:        bool  = False

    # Gaze shift (normalized iris offset)
    gaze_x:       float = 0.0
    gaze_y:       float = 0.0
    gaze_magnitude: float = 0.0

    # Head motion (frame-to-frame)
    head_sway:    float = 0.0

    # Facial geometry (all normalized by face_scale)
    mouth_aperture:    float = 0.0
    lip_corner_pull:   float = 0.0   # smile proxy
    brow_furrow:       float = 0.0
    brow_raise:        float = 0.0
    face_symmetry:     float = 1.0

    # ML emotions
    emotions:     EmotionScores = field(default_factory=EmotionScores)
    dominant_emotion: str = "neutral"

    # Derived scores
    confidence_score:   float = 0.0
    nervousness_score:  float = 0.0

@dataclass
class AudioFeatures:
    timestamp:      float = 0.0
    rms_energy:     float = 0.0
    pitch_hz:       float = 0.0
    pitch_std:      float = 0.0
    speaking_rate:  float = 0.0     # syllables/sec estimate
    pause_ratio:    float = 0.0     # fraction of silence
    voice_tremor:   float = 0.0     # pitch instability


# ══════════════════════════════════════════════════════════════════════
#  LANDMARK INDEX CONSTANTS (MediaPipe FaceMesh)
# ══════════════════════════════════════════════════════════════════════

class LM:
    # Eyes
    L_EYE_OUTER = 33;  L_EYE_INNER = 133
    L_EYE_TOP   = 159; L_EYE_BOT   = 145
    R_EYE_OUTER = 362; R_EYE_INNER = 263
    R_EYE_TOP   = 386; R_EYE_BOT   = 374

    # Irises (only with refine_landmarks=True)
    L_IRIS      = [468, 469, 470, 471, 472]
    R_IRIS      = [473, 474, 475, 476, 477]

    # Mouth
    MOUTH_LEFT  = 61;  MOUTH_RIGHT = 291
    MOUTH_TOP   = 13;  MOUTH_BOT   = 14

    # Nose
    NOSE_TIP    = 1;   NOSE_BASE   = 5

    # Brows
    L_BROW_INNER = 55; R_BROW_INNER = 285
    L_BROW_MID   = 52; R_BROW_MID   = 282
    L_BROW_OUTER = 46; R_BROW_OUTER = 276

    # Face outline
    CHIN        = 152; FOREHEAD    = 10
    L_CHEEK     = 234; R_CHEEK     = 454


# ══════════════════════════════════════════════════════════════════════
#  GEOMETRY UTILITIES
# ══════════════════════════════════════════════════════════════════════

def lm_xyz(landmarks, idx, w=1.0, h=1.0) -> np.ndarray:
    """Extract landmark as numpy array [x, y, z]."""
    p = landmarks[idx]
    return np.array([p.x * w, p.y * h, p.z])

def dist3d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def face_scale(landmarks) -> float:
    """Inter-ocular distance — stable face size normalizer."""
    lo = lm_xyz(landmarks, LM.L_EYE_OUTER)
    ro = lm_xyz(landmarks, LM.R_EYE_OUTER)
    return dist3d(lo, ro) + 1e-6

def eye_aspect_ratio(landmarks, left: bool) -> float:
    """
    EAR = vertical_distance / horizontal_distance
    Values < 0.20 typically indicate a blink.
    """
    if left:
        top = lm_xyz(landmarks, LM.L_EYE_TOP)
        bot = lm_xyz(landmarks, LM.L_EYE_BOT)
        out = lm_xyz(landmarks, LM.L_EYE_OUTER)
        inn = lm_xyz(landmarks, LM.L_EYE_INNER)
    else:
        top = lm_xyz(landmarks, LM.R_EYE_TOP)
        bot = lm_xyz(landmarks, LM.R_EYE_BOT)
        out = lm_xyz(landmarks, LM.R_EYE_OUTER)
        inn = lm_xyz(landmarks, LM.R_EYE_INNER)
    return dist3d(top, bot) / (dist3d(out, inn) + 1e-6)


# ══════════════════════════════════════════════════════════════════════
#  3D HEAD POSE ESTIMATOR
# ══════════════════════════════════════════════════════════════════════

class HeadPoseEstimator:
    """
    Estimates yaw/pitch/roll using solvePnP with a canonical 3D face model.
    This is the correct approach — not 2D ratio heuristics.
    """

    # 6 canonical 3D face model points (mm, approximate)
    MODEL_POINTS = np.array([
        [  0.0,    0.0,    0.0  ],   # Nose tip
        [  0.0,  -63.6,  -12.5 ],   # Chin
        [-43.3,   32.7,  -26.0 ],   # Left eye left corner
        [ 43.3,   32.7,  -26.0 ],   # Right eye right corner
        [-28.9,  -28.9,  -24.1 ],   # Left mouth corner
        [ 28.9,  -28.9,  -24.1 ],   # Right mouth corner
    ], dtype=np.float64)

    def __init__(self, frame_w: int, frame_h: int):
        self.w = frame_w
        self.h = frame_h
        focal = frame_w
        cx, cy = frame_w / 2, frame_h / 2
        self.camera_matrix = np.array([
            [focal, 0,     cx],
            [0,     focal, cy],
            [0,     0,     1 ],
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1))

    def estimate(self, landmarks) -> HeadPose:
        # 2D image points corresponding to MODEL_POINTS
        img_pts = np.array([
            [landmarks[LM.NOSE_TIP].x    * self.w, landmarks[LM.NOSE_TIP].y    * self.h],
            [landmarks[LM.CHIN].x        * self.w, landmarks[LM.CHIN].y        * self.h],
            [landmarks[LM.L_EYE_OUTER].x * self.w, landmarks[LM.L_EYE_OUTER].y * self.h],
            [landmarks[LM.R_EYE_OUTER].x * self.w, landmarks[LM.R_EYE_OUTER].y * self.h],
            [landmarks[LM.MOUTH_LEFT].x  * self.w, landmarks[LM.MOUTH_LEFT].y  * self.h],
            [landmarks[LM.MOUTH_RIGHT].x * self.w, landmarks[LM.MOUTH_RIGHT].y * self.h],
        ], dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(
            self.MODEL_POINTS, img_pts,
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return HeadPose()

        rot_mat, _ = cv2.Rodrigues(rot_vec)
        # Decompose rotation matrix → Euler angles
        sy = np.sqrt(rot_mat[0,0]**2 + rot_mat[1,0]**2)
        singular = sy < 1e-6
        if not singular:
            pitch = np.degrees(np.arctan2( rot_mat[2,1], rot_mat[2,2]))
            yaw   = np.degrees(np.arctan2(-rot_mat[2,0], sy))
            roll  = np.degrees(np.arctan2( rot_mat[1,0], rot_mat[0,0]))
        else:
            pitch = np.degrees(np.arctan2(-rot_mat[1,2], rot_mat[1,1]))
            yaw   = np.degrees(np.arctan2(-rot_mat[2,0], sy))
            roll  = 0.0

        return HeadPose(yaw=float(yaw), pitch=float(pitch), roll=float(roll))


# ══════════════════════════════════════════════════════════════════════
#  ML EMOTION RECOGNIZER
# ══════════════════════════════════════════════════════════════════════

class EmotionRecognizer:
    """
    Uses DeepFace (primary) or FER (fallback) for CNN-based emotion
    recognition.  Falls back to geometry-based AU approximation if
    neither is available.
    """

    def __init__(self):
        self.backend = None
        self.fer_detector = None
        self._init_backend()

    def _init_backend(self):
        if DEEPFACE_AVAILABLE:
            self.backend = "deepface"
            print("[EmotionRecognizer] Using DeepFace (CNN-based)")
        elif FER_AVAILABLE:
            self.fer_detector = FER(mtcnn=False)
            self.backend = "fer"
            print("[EmotionRecognizer] Using FER (CNN-based)")
        else:
            self.backend = "geometry"
            print("[EmotionRecognizer] Using geometry-based AU fallback")

    def predict(self, frame_bgr: np.ndarray,
                landmarks=None, scale: float = 1.0) -> EmotionScores:
        """Returns normalized emotion probabilities summing to ~1."""

        if self.backend == "deepface":
            return self._from_deepface(frame_bgr)
        elif self.backend == "fer":
            return self._from_fer(frame_bgr)
        else:
            return self._from_geometry(landmarks, scale)

    def _from_deepface(self, frame_bgr) -> EmotionScores:
        try:
            result = DeepFace.analyze(
                frame_bgr,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )
            e = result[0]["emotion"] if isinstance(result, list) else result["emotion"]
            total = sum(e.values()) + 1e-9
            return EmotionScores(
                angry   = e.get("angry",   0) / total,
                disgust = e.get("disgust",  0) / total,
                fear    = e.get("fear",     0) / total,
                happy   = e.get("happy",    0) / total,
                sad     = e.get("sad",      0) / total,
                surprise= e.get("surprise", 0) / total,
                neutral = e.get("neutral",  0) / total,
            )
        except Exception:
            return EmotionScores(neutral=1.0)

    def _from_fer(self, frame_bgr) -> EmotionScores:
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self.fer_detector.detect_emotions(rgb)
            if not result:
                return EmotionScores(neutral=1.0)
            e = result[0]["emotions"]
            total = sum(e.values()) + 1e-9
            return EmotionScores(
                angry   = e.get("angry",   0) / total,
                disgust = e.get("disgust",  0) / total,
                fear    = e.get("fear",     0) / total,
                happy   = e.get("happy",    0) / total,
                sad     = e.get("sad",      0) / total,
                surprise= e.get("surprise", 0) / total,
                neutral = e.get("neutral",  0) / total,
            )
        except Exception:
            return EmotionScores(neutral=1.0)

    def _from_geometry(self, landmarks, scale: float) -> EmotionScores:
        """AU-proxy based fallback when no ML model is available."""
        if landmarks is None:
            return EmotionScores(neutral=1.0)

        mw = dist3d(lm_xyz(landmarks, LM.MOUTH_LEFT),
                    lm_xyz(landmarks, LM.MOUTH_RIGHT)) / scale
        ma = dist3d(lm_xyz(landmarks, LM.MOUTH_TOP),
                    lm_xyz(landmarks, LM.MOUTH_BOT)) / scale
        lb = dist3d(lm_xyz(landmarks, LM.L_BROW_MID),
                    lm_xyz(landmarks, LM.L_EYE_TOP)) / scale
        rb = dist3d(lm_xyz(landmarks, LM.R_BROW_MID),
                    lm_xyz(landmarks, LM.R_EYE_TOP)) / scale
        brow_raise = (lb + rb) / 2
        brow_furrow_dist = dist3d(lm_xyz(landmarks, LM.L_BROW_INNER),
                                   lm_xyz(landmarks, LM.R_BROW_INNER)) / scale

        # Heuristic mapping (not as accurate as CNN but far better than
        # single lip-distance ratios)
        happy    = min(1.0, max(0, (mw - 0.25) * 4))
        surprise = min(1.0, max(0, (ma - 0.04) * 15 + (brow_raise - 0.08) * 10))
        angry    = min(1.0, max(0, (0.18 - brow_furrow_dist) * 8))
        fear     = min(1.0, max(0, (brow_raise - 0.10) * 8))
        sad      = min(1.0, max(0, (0.27 - mw) * 4))
        neutral  = max(0, 1.0 - happy - surprise - angry - fear - sad)

        scores = [happy, surprise, angry, fear, sad, 0.0, neutral]
        total  = sum(scores) + 1e-9
        return EmotionScores(
            happy=happy/total, surprise=surprise/total,
            angry=angry/total, fear=fear/total,
            sad=sad/total, disgust=0.0, neutral=neutral/total
        )


# ══════════════════════════════════════════════════════════════════════
#  AUDIO ANALYZER (optional)
# ══════════════════════════════════════════════════════════════════════

class AudioAnalyzer:
    """
    Real-time audio feature extraction:
    - RMS energy (loudness)
    - Fundamental frequency (pitch) via PYIN
    - Pitch variability (tremor / confidence)
    - Pause ratio (hesitation)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.buffer = deque(maxlen=int(cfg.audio_sample_rate * cfg.audio_window_secs))
        self.pa = None
        self.stream = None
        self.running = False

    def start(self):
        if not PYAUDIO_AVAILABLE or not AUDIO_AVAILABLE:
            print("[AudioAnalyzer] PyAudio/librosa not available — skipping audio.")
            return False
        try:
            self.pa = pyaudio.PyAudio()
            self.stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.cfg.audio_sample_rate,
                input=True,
                frames_per_buffer=self.cfg.audio_chunk_size,
                stream_callback=self._callback
            )
            self.running = True
            print("[AudioAnalyzer] Started audio capture")
            return True
        except Exception as e:
            print(f"[AudioAnalyzer] Failed to start: {e}")
            return False

    def _callback(self, in_data, frame_count, time_info, status):
        samples = np.frombuffer(in_data, dtype=np.float32)
        self.buffer.extend(samples)
        return (None, pyaudio.paContinue)

    def extract(self) -> Optional[AudioFeatures]:
        if not self.running or len(self.buffer) < self.cfg.audio_sample_rate:
            return None
        audio = np.array(self.buffer, dtype=np.float32)
        sr    = self.cfg.audio_sample_rate

        rms = float(np.sqrt(np.mean(audio**2)))
        silence_thresh = 0.01
        pause_ratio = float(np.mean(np.abs(audio) < silence_thresh))

        try:
            f0, voiced, _ = librosa.pyin(
                audio, fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'), sr=sr
            )
            f0_valid = f0[voiced] if voiced is not None else np.array([])
            pitch_hz  = float(np.nanmean(f0_valid)) if len(f0_valid) else 0.0
            pitch_std = float(np.nanstd(f0_valid))  if len(f0_valid) else 0.0
            # Voice tremor = coefficient of variation of pitch
            voice_tremor = (pitch_std / (pitch_hz + 1e-6)) if pitch_hz > 0 else 0.0
        except Exception:
            pitch_hz, pitch_std, voice_tremor = 0.0, 0.0, 0.0

        # Rough speaking rate: zero-crossing density in voiced segments
        zc = librosa.zero_crossings(audio, pad=False)
        speaking_rate = float(np.mean(zc)) * sr / 10.0  # rough proxy

        return AudioFeatures(
            timestamp     = time.time(),
            rms_energy    = rms,
            pitch_hz      = pitch_hz,
            pitch_std     = pitch_std,
            speaking_rate = speaking_rate,
            pause_ratio   = pause_ratio,
            voice_tremor  = voice_tremor,
        )

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.pa:
            self.pa.terminate()


# ══════════════════════════════════════════════════════════════════════
#  TEMPORAL SMOOTHER
# ══════════════════════════════════════════════════════════════════════

class TemporalSmoother:
    """
    Maintains a sliding window of FrameFeatures and returns
    per-field averages to reduce frame-to-frame noise.
    """

    def __init__(self, window: int):
        self.window = window
        self.buffer: deque = deque(maxlen=window)

    def push(self, f: FrameFeatures):
        self.buffer.append(f)

    def smooth(self) -> Optional[FrameFeatures]:
        if not self.buffer:
            return None
        n = len(self.buffer)

        avg = FrameFeatures()
        avg.face_found = any(f.face_found for f in self.buffer)
        if not avg.face_found:
            return avg

        valid = [f for f in self.buffer if f.face_found]
        if not valid:
            return avg
        nv = len(valid)

        # Scalar averages
        for attr in ('ear_avg', 'gaze_magnitude', 'head_sway',
                     'mouth_aperture', 'lip_corner_pull', 'brow_furrow',
                     'brow_raise', 'face_symmetry', 'face_scale'):
            setattr(avg, attr, np.mean([getattr(f, attr) for f in valid]))

        # Head pose
        avg.pose = HeadPose(
            yaw   = np.mean([f.pose.yaw   for f in valid]),
            pitch = np.mean([f.pose.pitch for f in valid]),
            roll  = np.mean([f.pose.roll  for f in valid]),
        )

        # Emotions
        for em in ('angry','disgust','fear','happy','sad','surprise','neutral'):
            val = np.mean([getattr(f.emotions, em) for f in valid])
            setattr(avg.emotions, em, val)

        avg.dominant_emotion = max(
            asdict(avg.emotions), key=lambda k: getattr(avg.emotions, k)
        )
        avg.timestamp = valid[-1].timestamp

        return avg


# ══════════════════════════════════════════════════════════════════════
#  SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════

class ScoringEngine:
    """
    Fuses all features into final confidence / nervousness scores.
    Optionally incorporates audio features for true multimodal output.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def compute(self, avg: FrameFeatures,
                audio: Optional[AudioFeatures] = None,
                blink_rate: float = 0.0) -> Tuple[float, float]:
        """
        Returns (confidence_score, nervousness_score) in [0, 100].
        """
        if not avg.face_found:
            return 0.0, 0.0

        cfg = self.cfg

        # ── 1. Pose deviation penalty ──────────────────────────────
        yaw_pen   = min(1.0, abs(avg.pose.yaw)   / cfg.yaw_threshold)
        pitch_pen = min(1.0, abs(avg.pose.pitch)  / cfg.pitch_threshold)
        roll_pen  = min(1.0, abs(avg.pose.roll)   / cfg.roll_threshold)
        pose_score = (yaw_pen * 0.5 + pitch_pen * 0.3 + roll_pen * 0.2)

        # ── 2. Blink rate score ────────────────────────────────────
        # Normal: 15–20 blinks/min. High > 22 = anxiety signal.
        normal_rate = 17.0
        blink_penalty = min(1.0, max(0, (blink_rate - normal_rate) / 20.0))

        # ── 3. Gaze stability ─────────────────────────────────────
        gaze_penalty = min(1.0, avg.gaze_magnitude * 40.0)

        # ── 4. Head sway ──────────────────────────────────────────
        sway_penalty = min(1.0, avg.head_sway * 150.0)

        # ── 5. Emotion-based nervousness ──────────────────────────
        em = avg.emotions
        emo_nervousness = min(1.0, em.fear * 0.4 + em.angry * 0.2 +
                              em.sad * 0.2 + em.disgust * 0.1 + em.surprise * 0.1)
        emo_confidence = min(0.3, em.happy * 0.3 + em.neutral * 0.1)

        # ── 6. Audio modality (optional) ──────────────────────────
        audio_penalty = 0.0
        audio_bonus   = 0.0
        if audio is not None:
            audio_penalty  = min(1.0, audio.pause_ratio * 1.5)
            audio_penalty += min(0.3, audio.voice_tremor * 2.0)
            # Consistent energy = confidence bonus
            audio_bonus    = min(0.2, audio.rms_energy * 5.0)

        # ── 7. Weighted nervousness ───────────────────────────────
        w = cfg
        nervousness = (
            w.w_pose    * pose_score     +
            w.w_blink   * blink_penalty  +
            w.w_gaze    * gaze_penalty   +
            w.w_sway    * sway_penalty   +
            w.w_emotion * emo_nervousness
        )
        if audio is not None:
            nervousness = nervousness * 0.8 + audio_penalty * 0.2

        nervousness_score = min(100.0, nervousness * 100.0)

        # ── 8. Confidence (inverse + bonuses) ─────────────────────
        symmetry_bonus = (avg.face_symmetry - 0.5) * 0.15
        smile_bonus    = avg.lip_corner_pull * 0.2
        confidence_score = max(0.0, min(100.0,
            100.0 - nervousness_score * 0.75
            + emo_confidence * 20.0
            + symmetry_bonus * 10.0
            + smile_bonus    * 10.0
            + audio_bonus    * 15.0
        ))

        return round(confidence_score, 1), round(nervousness_score, 1)

    def label(self, confidence: float, nervousness: float) -> str:
        if confidence > 70 and nervousness < 30:
            return "HIGHLY CONFIDENT"
        elif confidence > 55 and nervousness < 45:
            return "CONFIDENT"
        elif confidence > 40 and nervousness < 55:
            return "MODERATELY CONFIDENT"
        elif nervousness > 65:
            return "HIGHLY NERVOUS"
        elif nervousness > 45:
            return "NERVOUS"
        else:
            return "UNCERTAIN"


# ══════════════════════════════════════════════════════════════════════
#  VISUALIZER
# ══════════════════════════════════════════════════════════════════════

class Visualizer:
    """Draws real-time overlays on the OpenCV frame."""

    mp_drawing = mp_solutions.drawing_utils if mp_solutions else None
    mp_drawing_styles = mp_solutions.drawing_styles if mp_solutions else None
    mp_face_mesh = mp_solutions.face_mesh if mp_solutions else None

    # Colors
    GREEN  = (0, 229, 160)
    ORANGE = (0, 107, 255)
    RED    = (71, 71, 255)
    CYAN   = (229, 178, 0)
    WHITE  = (230, 234, 240)
    GRAY   = (100, 112, 130)

    def draw(self, frame: np.ndarray, features: FrameFeatures,
             avg: FrameFeatures, blink_rate: float,
             conf: float, nerv: float) -> np.ndarray:

        h, w = frame.shape[:2]
        overlay = frame.copy()

        if not features.face_found:
            cv2.putText(overlay, "No face detected", (w//2 - 120, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.RED, 2)
            return overlay

        # ── Confidence / Nervousness bars ─────────────────────────
        self._draw_score_bar(overlay, "CONFIDENCE",  conf, (20, 40),  w=180, color=self.GREEN)
        self._draw_score_bar(overlay, "NERVOUSNESS", nerv, (20, 100), w=180,
                             color=self.RED if nerv > 60 else (0, 190, 255))

        # ── Head pose display ─────────────────────────────────────
        pose = avg.pose
        cv2.putText(overlay, f"YAW:{pose.yaw:+.1f} PITCH:{pose.pitch:+.1f} ROLL:{pose.roll:+.1f}",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.CYAN, 1)

        # ── Blink rate ────────────────────────────────────────────
        blink_color = self.GREEN if blink_rate < 20 else (self.ORANGE if blink_rate < 28 else self.RED)
        cv2.putText(overlay, f"BLINK: {blink_rate:.0f}/min",
                    (w - 180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blink_color, 1)

        # ── Dominant emotion ──────────────────────────────────────
        em_color = self.GREEN if avg.dominant_emotion in ("happy", "neutral") else self.ORANGE
        cv2.putText(overlay, f"EMOTION: {avg.dominant_emotion.upper()}",
                    (w - 220, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, em_color, 1)

        # ── Status label ──────────────────────────────────────────
        engine = ScoringEngine(Config())
        label = engine.label(conf, nerv)
        label_color = self.GREEN if "CONFIDENT" in label else self.RED
        cv2.putText(overlay, label, (20, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, label_color, 2)

        # Corner brackets
        self._draw_brackets(overlay, w, h)

        return overlay

    def _draw_score_bar(self, frame, label, value, pos,
                        w=180, color=(0,229,160)):
        x, y = pos
        # Background
        cv2.rectangle(frame, (x, y - 14), (x + w, y + 6),
                      (30, 35, 45), -1)
        # Fill
        fill_w = int(w * value / 100)
        cv2.rectangle(frame, (x, y - 14), (x + fill_w, y + 6),
                      color, -1)
        # Text
        cv2.putText(frame, f"{label}: {value:.0f}%", (x + 4, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 234, 240), 1)

    def _draw_brackets(self, frame, w, h):
        size, thick = 25, 2
        c = self.GREEN
        for (x, y, dx, dy) in [(0,0,1,1),(w-1,0,-1,1),(0,h-1,1,-1),(w-1,h-1,-1,-1)]:
            cv2.line(frame, (x, y), (x + dx*size, y), c, thick)
            cv2.line(frame, (x, y), (x, y + dy*size), c, thick)

    @staticmethod
    def draw_mesh(frame, face_landmarks):
        if (Visualizer.mp_drawing is None or
            Visualizer.mp_drawing_styles is None or
            Visualizer.mp_face_mesh is None):
            return

        Visualizer.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=Visualizer.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=Visualizer.mp_drawing_styles
                .get_default_face_mesh_tesselation_style()
        )


# ══════════════════════════════════════════════════════════════════════
#  REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════

class ReportGenerator:
    """Saves JSON + optional matplotlib charts of session data."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(exist_ok=True)

    def save(self, history: List[Dict], session_secs: float):
        if not history:
            print("[Report] No data to save.")
            return

        ts = time.strftime("%Y%m%d_%H%M%S")

        # ── JSON ──────────────────────────────────────────────────
        report = {
            "session_duration_sec": round(session_secs, 1),
            "total_frames":         len(history),
            "summary": {
                "avg_confidence":   round(np.mean([h["confidence"] for h in history]), 1),
                "avg_nervousness":  round(np.mean([h["nervousness"] for h in history]), 1),
                "max_nervousness":  round(max(h["nervousness"] for h in history), 1),
                "min_confidence":   round(min(h["confidence"] for h in history), 1),
                "avg_blink_rate":   round(np.mean([h.get("blink_rate",0) for h in history]), 1),
                "dominant_emotion": self._mode([h.get("dominant_emotion","neutral") for h in history]),
            },
            "frames": history
        }

        json_path = self.out_dir / f"session_{ts}.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[Report] Saved JSON → {json_path}")

        # ── Charts ────────────────────────────────────────────────
        if MATPLOTLIB_AVAILABLE and self.cfg.show_plots:
            self._plot(report, ts)

    def _mode(self, lst):
        from collections import Counter
        return Counter(lst).most_common(1)[0][0] if lst else "neutral"

    def _plot(self, report, ts):
        frames   = report["frames"]
        times    = [f["timestamp"] - frames[0]["timestamp"] for f in frames]
        conf     = [f["confidence"]  for f in frames]
        nerv     = [f["nervousness"] for f in frames]
        blink    = [f.get("blink_rate", 0) for f in frames]
        yaw      = [f.get("yaw", 0) for f in frames]

        fig = plt.figure(figsize=(14, 8), facecolor="#07080d")
        fig.suptitle("InterviewLens — Session Report", color="#e8eaf0",
                     fontsize=14, fontweight="bold")

        gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.3)
        axes_cfg = dict(facecolor="#0d1018")

        # Confidence vs Nervousness
        ax1 = fig.add_subplot(gs[0, :], **axes_cfg)
        ax1.plot(times, conf, color="#00e5a0", linewidth=1.5, label="Confidence")
        ax1.plot(times, nerv, color="#ff4757", linewidth=1.5, label="Nervousness")
        ax1.fill_between(times, conf, alpha=0.12, color="#00e5a0")
        ax1.fill_between(times, nerv, alpha=0.12, color="#ff4757")
        ax1.set_ylim(0, 100); ax1.set_xlabel("Time (s)", color="#6b7280")
        ax1.set_ylabel("Score", color="#6b7280")
        ax1.set_title("Confidence & Nervousness Over Time", color="#e8eaf0")
        ax1.legend(facecolor="#131722", labelcolor="#e8eaf0", edgecolor="#2a3045")
        self._style_ax(ax1)

        # Blink rate
        ax2 = fig.add_subplot(gs[1, 0], **axes_cfg)
        ax2.plot(times, blink, color="#7b6fff", linewidth=1.2)
        ax2.axhline(20, color="#ffbb00", linewidth=0.8, linestyle="--", alpha=0.6)
        ax2.set_title("Blink Rate (blinks/min)", color="#e8eaf0")
        ax2.set_xlabel("Time (s)", color="#6b7280")
        self._style_ax(ax2)

        # Head yaw
        ax3 = fig.add_subplot(gs[1, 1], **axes_cfg)
        ax3.plot(times, yaw, color="#ff6b35", linewidth=1.2)
        ax3.axhline( 20, color="#ffbb00", linewidth=0.8, linestyle="--", alpha=0.6)
        ax3.axhline(-20, color="#ffbb00", linewidth=0.8, linestyle="--", alpha=0.6)
        ax3.set_title("Head Yaw (°)", color="#e8eaf0")
        ax3.set_xlabel("Time (s)", color="#6b7280")
        self._style_ax(ax3)

        plt.tight_layout()
        chart_path = Path(self.cfg.output_dir) / f"charts_{ts}.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="#07080d")
        plt.show()
        print(f"[Report] Saved chart → {chart_path}")

    def _style_ax(self, ax):
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a3045")
        ax.tick_params(colors="#6b7280")
        ax.yaxis.label.set_color("#6b7280")


# ══════════════════════════════════════════════════════════════════════
#  MAIN ANALYZER
# ══════════════════════════════════════════════════════════════════════

class InterviewAnalyzer:
    """
    Orchestrates the full pipeline:
      Camera → FaceMesh → HeadPose + EmotionML + AudioAnalysis
      → TemporalSmoothing → ScoringEngine → Visualization + Report
    """

    def __init__(self, cfg: Config = None):
        self.cfg      = cfg or Config()
        if not mp_solutions or not hasattr(mp_solutions, "face_mesh"):
            raise RuntimeError(
                f"Installed mediapipe=={MP_VERSION} does not expose legacy "
                "`mediapipe.solutions.face_mesh` required by this script.\n"
                "Install a solutions-enabled build, for example:\n"
                "  pip uninstall -y mediapipe\n"
                "  pip install mediapipe==0.10.9"
            )
        self.mp_fm    = mp_solutions.face_mesh
        self.pose_est = None
        self.emotion  = EmotionRecognizer()
        self.smoother = TemporalSmoother(self.cfg.temporal_window)
        self.scoring  = ScoringEngine(self.cfg)
        self.viz      = Visualizer()
        self.audio    = AudioAnalyzer(self.cfg)
        self.report   = ReportGenerator(self.cfg)

        # Session state
        self.history:     List[Dict] = []
        self.conf_history = deque(maxlen=self.cfg.history_maxlen)
        self.nerv_history = deque(maxlen=self.cfg.history_maxlen)
        self.blink_count  = 0
        self.last_blink   = 0.0
        self.session_start = 0.0
        self.prev_nose_pos: Optional[np.ndarray] = None

    # ── Public API ───────────────────────────────────────────────────

    def run_live(self):
        """Start real-time webcam analysis. Press Q to quit."""
        cap = cv2.VideoCapture(self.cfg.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cfg.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)

        if not cap.isOpened():
            print("[Error] Cannot open camera. Check camera_index in Config.")
            return

        self.audio.start()
        self.session_start = time.time()

        print("\n" + "═"*60)
        print("  InterviewLens — Press Q to quit, R to reset stats")
        print("═"*60 + "\n")

        with self.mp_fm.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,        # enables iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.pose_est = HeadPoseEstimator(w, h)

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)  # mirror

                # ── Process frame ─────────────────────────────────
                features = self._process_frame(frame, face_mesh)
                self.smoother.push(features)
                avg = self.smoother.smooth() or features

                # ── Scores ────────────────────────────────────────
                session_secs = time.time() - self.session_start
                blink_rate   = (self.blink_count / max(1, session_secs)) * 60
                audio_feat   = self.audio.extract()
                conf, nerv   = self.scoring.compute(avg, audio_feat, blink_rate)
                avg.confidence_score  = conf
                avg.nervousness_score = nerv

                self.conf_history.append(conf)
                self.nerv_history.append(nerv)

                # Record every 15 frames
                if len(self.history) == 0 or time.time() - self.history[-1]["timestamp"] > 0.5:
                    self.history.append({
                        "timestamp":    time.time(),
                        "confidence":   conf,
                        "nervousness":  nerv,
                        "blink_rate":   round(blink_rate, 1),
                        "yaw":          round(avg.pose.yaw, 1),
                        "pitch":        round(avg.pose.pitch, 1),
                        "roll":         round(avg.pose.roll, 1),
                        "dominant_emotion": avg.dominant_emotion,
                        "gaze_magnitude":   round(avg.gaze_magnitude, 4),
                    })

                # ── Draw mesh ─────────────────────────────────────
                if self.cfg.show_mesh and features.face_found:
                    pass  # MediaPipe mesh drawing via process result

                # ── Overlay ───────────────────────────────────────
                display = self.viz.draw(frame, features, avg, blink_rate, conf, nerv)
                cv2.imshow("InterviewLens", display)

                # ── Console update ────────────────────────────────
                if len(self.conf_history) % 15 == 0:
                    label = self.scoring.label(conf, nerv)
                    print(f"  [{session_secs:6.1f}s]  "
                          f"Confidence: {conf:5.1f}%  "
                          f"Nervousness: {nerv:5.1f}%  "
                          f"| {label:<20}  "
                          f"| YAW {avg.pose.yaw:+.0f}°  "
                          f"| BLINK {blink_rate:.0f}/min  "
                          f"| {avg.dominant_emotion.upper()}")

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self._reset_stats()
                    print("  [Reset] Session statistics cleared.")

        cap.release()
        cv2.destroyAllWindows()
        self.audio.stop()

        # ── Final report ──────────────────────────────────────────
        session_secs = time.time() - self.session_start
        self._print_summary(session_secs)
        self.report.save(self.history, session_secs)

    def run_on_video(self, video_path: str):
        """Analyze a pre-recorded video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Error] Cannot open video: {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.session_start = time.time()

        print(f"\n[VideoAnalysis] Processing {video_path} ({total_frames} frames @ {fps:.1f}fps)")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.pose_est = HeadPoseEstimator(w, h)

        with self.mp_fm.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as face_mesh:

            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                features = self._process_frame(frame, face_mesh)
                self.smoother.push(features)
                avg = self.smoother.smooth() or features

                video_time   = frame_idx / fps
                blink_rate   = (self.blink_count / max(1, video_time)) * 60 if video_time > 0 else 0
                conf, nerv   = self.scoring.compute(avg, None, blink_rate)

                self.history.append({
                    "timestamp":    video_time,
                    "confidence":   conf,
                    "nervousness":  nerv,
                    "blink_rate":   round(blink_rate, 1),
                    "yaw":          round(avg.pose.yaw, 1),
                    "pitch":        round(avg.pose.pitch, 1),
                    "roll":         round(avg.pose.roll, 1),
                    "dominant_emotion": avg.dominant_emotion,
                })

                frame_idx += 1
                if frame_idx % 30 == 0:
                    print(f"  Frame {frame_idx}/{total_frames} "
                          f"| Conf {conf:.0f}% | Nerv {nerv:.0f}%")

        cap.release()
        session_secs = frame_idx / fps
        self._print_summary(session_secs)
        self.report.save(self.history, session_secs)

    # ── Internal ─────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray, face_mesh) -> FrameFeatures:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = face_mesh.process(rgb)
        rgb.flags.writeable = True

        ff = FrameFeatures(timestamp=time.time(), face_found=False)

        if not results.multi_face_landmarks:
            return ff

        ff.face_found = True
        lms = results.multi_face_landmarks[0].landmark

        # ── Scale normalization ───────────────────────────────────
        ff.face_scale = face_scale(lms)

        # ── 3D Head Pose (solvePnP) ───────────────────────────────
        ff.pose = self.pose_est.estimate(lms)

        # ── Eye Aspect Ratio + blink ──────────────────────────────
        ff.ear_left  = eye_aspect_ratio(lms, left=True)
        ff.ear_right = eye_aspect_ratio(lms, left=False)
        ff.ear_avg   = (ff.ear_left + ff.ear_right) / 2.0

        now = time.time()
        if ff.ear_avg < self.cfg.ear_threshold and (now - self.last_blink) > self.cfg.blink_cooldown:
            ff.blink = True
            self.blink_count += 1
            self.last_blink  = now

        # ── Gaze (iris landmark centroid offset) ──────────────────
        if len(lms) > 477:
            lc = np.mean([lm_xyz(lms, i) for i in LM.L_IRIS], axis=0)
            rc = np.mean([lm_xyz(lms, i) for i in LM.R_IRIS], axis=0)
            le_center = (lm_xyz(lms, LM.L_EYE_OUTER) + lm_xyz(lms, LM.L_EYE_INNER)) / 2
            re_center = (lm_xyz(lms, LM.R_EYE_OUTER) + lm_xyz(lms, LM.R_EYE_INNER)) / 2
            gaze = ((lc - le_center) + (rc - re_center)) / 2
            ff.gaze_x, ff.gaze_y = gaze[0], gaze[1]
            ff.gaze_magnitude = float(np.linalg.norm(gaze[:2]))

        # ── Head sway (frame-to-frame motion) ─────────────────────
        nose = lm_xyz(lms, LM.NOSE_TIP)
        if self.prev_nose_pos is not None:
            ff.head_sway = float(np.linalg.norm(nose - self.prev_nose_pos))
        self.prev_nose_pos = nose

        # ── Normalized facial geometry ────────────────────────────
        sc = ff.face_scale
        ff.mouth_aperture = dist3d(lm_xyz(lms, LM.MOUTH_TOP),
                                   lm_xyz(lms, LM.MOUTH_BOT)) / sc
        ff.lip_corner_pull = (dist3d(lm_xyz(lms, LM.MOUTH_LEFT),
                                     lm_xyz(lms, LM.MOUTH_RIGHT)) /
                              dist3d(lm_xyz(lms, LM.L_CHEEK),
                                     lm_xyz(lms, LM.R_CHEEK)))

        li_brow = dist3d(lm_xyz(lms, LM.L_BROW_INNER), lm_xyz(lms, LM.R_BROW_INNER)) / sc
        ff.brow_furrow = max(0.0, 0.20 - li_brow) * 5.0

        lb_raise = dist3d(lm_xyz(lms, LM.L_BROW_MID), lm_xyz(lms, LM.L_EYE_TOP)) / sc
        rb_raise = dist3d(lm_xyz(lms, LM.R_BROW_MID), lm_xyz(lms, LM.R_EYE_TOP)) / sc
        ff.brow_raise = (lb_raise + rb_raise) / 2.0

        ff.face_symmetry = 1.0 - min(1.0, abs(ff.ear_left - ff.ear_right) * 10.0)

        # ── ML Emotion (every 5th frame for performance) ──────────
        if int(ff.timestamp * 10) % 5 == 0:
            ff.emotions = self.emotion.predict(frame if hasattr(self, '_frame') else
                                               cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                               lms, sc)
        ff.dominant_emotion = max(asdict(ff.emotions),
                                   key=lambda k: getattr(ff.emotions, k))

        return ff

    def _reset_stats(self):
        self.blink_count   = 0
        self.session_start = time.time()
        self.prev_nose_pos = None
        self.history.clear()
        self.conf_history.clear()
        self.nerv_history.clear()
        self.smoother.buffer.clear()

    def _print_summary(self, session_secs: float):
        if not self.history:
            return
        avg_conf = np.mean([h["confidence"]  for h in self.history])
        avg_nerv = np.mean([h["nervousness"] for h in self.history])
        label    = self.scoring.label(avg_conf, avg_nerv)
        print("\n" + "═"*60)
        print("  SESSION SUMMARY")
        print("═"*60)
        print(f"  Duration       : {session_secs:.1f}s")
        print(f"  Avg Confidence : {avg_conf:.1f}%")
        print(f"  Avg Nervousness: {avg_nerv:.1f}%")
        print(f"  Assessment     : {label}")
        print(f"  Total frames   : {len(self.history)}")
        print("═"*60 + "\n")


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="InterviewLens — Multimodal Confidence Analyzer")
    parser.add_argument("--video",   type=str, default=None,  help="Path to video file (omit for webcam)")
    parser.add_argument("--camera",  type=int, default=0,     help="Camera index (default: 0)")
    parser.add_argument("--no-mesh", action="store_true",     help="Disable face mesh overlay")
    parser.add_argument("--no-plot", action="store_true",     help="Disable matplotlib charts")
    parser.add_argument("--window",  type=int, default=15,    help="Temporal smoothing window (frames)")
    parser.add_argument("--output",  type=str, default="interview_results", help="Output directory")
    args = parser.parse_args()

    cfg = Config(
        camera_index     = args.camera,
        show_mesh        = not args.no_mesh,
        show_plots       = not args.no_plot,
        temporal_window  = args.window,
        output_dir       = args.output,
    )

    analyzer = InterviewAnalyzer(cfg)

    if args.video:
        analyzer.run_on_video(args.video)
    else:
        analyzer.run_live()
