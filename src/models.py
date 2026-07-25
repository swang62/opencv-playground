"""YOLOE-26 model loading, warming, and routing."""

from __future__ import annotations

import logging
import threading

import torch
from ultralytics import YOLO
from ultralytics.nn.text_model import MobileCLIPTS

from src import config
from src.body import BodyEngine
from src.face import FaceEngine

logger = logging.getLogger(__name__)

# Text encoder shared across all ModelBundle instances (one is enough).
text_encoder = None


def get_device() -> str:
    """Select ``"mps"`` when available, otherwise ``"cpu"``."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def warmup_model(model, device: str):
    """Warm a single model by running one dummy prediction."""
    dummy = torch.zeros(
        (1, 3, config.INFERENCE_SIZE, config.INFERENCE_SIZE), device=device
    )
    model.predict(dummy, verbose=False)


def get_text_encoder():
    """Return the shared MobileCLIPTS text encoder, built once on CPU."""
    global text_encoder
    if text_encoder is None:
        text_encoder = MobileCLIPTS(
            torch.device("cpu"), weight=f"{config.MODELS_DIR}/mobileclip2_b.ts"
        )
    return text_encoder


def encode_text(texts: list[str]) -> torch.Tensor:
    """Encode text on CPU, return (1, N, 512) tensor on MPS.

    Avoids MPS shader compilation overhead on the text encoder.
    """
    enc = get_text_encoder()
    tokens = enc.tokenize(texts)
    txt_feats = enc.encode_text(tokens).detach().cpu()
    # Shape: (N, 512) -> (1, N, 512)
    return txt_feats.unsqueeze(0)


class ModelBundle:
    """Routes frames to the prompted or prompt-free model.

    All models are loaded and warmed up at startup.
    """

    def __init__(self, prompted, promptfree_path: str, device: str):
        self.prompted = prompted
        self._promptfree = None
        self._promptfree_path = promptfree_path
        self._promptfree_lock = threading.Lock()
        self.device = device
        self._last_target: str = ""
        self._text_prompt_embedding_cache: dict[str, torch.Tensor] = {}
        self._face_engine: FaceEngine | None = None
        self._body_engine: BodyEngine | None = None

    _prompt_thread: threading.Thread | None = None
    _prompt_pending: str = ""

    @property
    def promptfree(self):
        """Lazy-load and cache the prompt-free model on first access."""
        if self._promptfree is None:
            with self._promptfree_lock:
                if self._promptfree is not None:
                    return self._promptfree
                m = YOLO(self._promptfree_path)
                m.to(self.device)
                warmup_model(m, self.device)
                self._promptfree = m
                logger.info("Prompt-free model ready")
        return self._promptfree

    @property
    def face_engine(self):
        if self._face_engine is None:
            self._face_engine = FaceEngine()
        return self._face_engine

    @property
    def body_engine(self):
        if self._body_engine is None:
            self._body_engine = BodyEngine()
        return self._body_engine

    def set_prompt(self, target: str):
        """Set a new text prompt, using cached embeddings if available."""
        if not target or target == self._last_target:
            return
        cached = self._text_prompt_embedding_cache.get(target)
        if cached is not None:
            self.prompted.set_classes([target], embeddings=cached)
            self._last_target = target
            return
        self._prompt_pending = target
        if self._prompt_thread is None or not self._prompt_thread.is_alive():
            self._prompt_thread = threading.Thread(
                target=self.do_set_prompt, daemon=True
            )
            self._prompt_thread.start()

    def do_set_prompt(self):
        target = self._prompt_pending
        # Encode text to raw embeddings on CPU.
        raw = encode_text([target]).to(self.device)
        # Run through the YOLOE head to get final tpe.
        head = self.prompted.model.model[-1]
        tpe = head.get_tpe(raw)
        self._text_prompt_embedding_cache[target] = tpe
        self.prompted.set_classes([target], embeddings=tpe)
        self._last_target = target

    def predict(self, frame, mode: str, **kwargs):
        """Run inference through the model matching *mode*."""
        if mode == "face":
            return self.face_engine.process(
                frame,
                show_labels=kwargs.get("show_labels", True),
            )
        if mode == "find":
            return self.prompted.predict(frame, **kwargs)
        return self.promptfree.predict(frame, **kwargs)

    def warmup(self):
        """Warm all primary models at startup."""
        logger.info("Warming up prompted model...")
        warmup_model(self.prompted, self.device)
        logger.info("Pre-loading text encoder (mobileclip2) on CPU...")
        try:
            raw = encode_text(["dummy"]).to(self.device)
            head = self.prompted.model.model[-1]
            tpe = head.get_tpe(raw)
            self._text_prompt_embedding_cache["dummy"] = tpe
            self.prompted.set_classes(["dummy"], embeddings=tpe)
            self._last_target = "dummy"
            logger.info("Text encoder ready")
        except Exception as exc:
            logger.warning("Text encoder pre-load failed: %s", exc)
            logger.warning("First text prompt may trigger a download")

        logger.info("Pre-loading prompt-free model...")
        try:
            _ = self.promptfree
        except Exception as exc:
            logger.warning("Prompt-free model load failed: %s", exc)

        logger.info("Pre-loading face engine (MediaPipe + UniFace)...")
        try:
            self.face_engine.warmup()
        except Exception as exc:
            logger.warning("Face engine load failed: %s", exc)

        logger.info("Pre-loading body engine (Pose + Hand)...")
        try:
            self.body_engine.warmup()
        except Exception as exc:
            logger.warning("Body engine load failed: %s", exc)

        logger.info("All models ready")


def load_model_bundle(prompted_path: str, promptfree_path: str) -> ModelBundle:
    """Load the prompted checkpoint, move to device, warm up, and return a bundle.

    Requires network access on first call (downloads checkpoint files).
    Subsequent calls use locally cached ``.pt`` files.
    """
    device = get_device()
    prompted = YOLO(prompted_path)
    prompted.to(device)

    bundle = ModelBundle(prompted, promptfree_path, device)
    bundle.warmup()
    return bundle
