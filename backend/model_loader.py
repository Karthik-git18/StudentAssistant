"""
model_loader.py
===============
Loads Qwen2.5-0.5B-Instruct and SentenceTransformer ONCE into global singletons.
"""

import logging
import traceback
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR        = Path(__file__).parent.parent / "models" / "Qwen2.5-0.5B-Instruct"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOKENIZER_MAX_LEN = 512


class LocalLLM:
    """Wrapper around a local Causal-LM with a single .generate() entry-point."""

    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = str(model_dir)
        self.device    = self._detect_device()
        self._load()

    def _detect_device(self):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        return "cpu"

    def _load(self):
        try:
            logger.info("[LLM] Loading from %s on device=%s", self.model_dir, self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                use_fast=False,
                model_max_length=TOKENIZER_MAX_LEN,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            logger.info("[LLM] Tokenizer loaded")

            dtype = torch.float16 if self.device in ("mps", "cuda") else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                dtype=dtype,
            )
            self.model.to(self.device)
            self.model.eval()

            logger.info("[LLM] Model loaded successfully on %s", self.device)

        except Exception as exc:
            logger.error("[LLM] Load failed: %s\n%s", exc, traceback.format_exc())
            raise RuntimeError(f"Failed to load model from {self.model_dir}: {exc}") from exc

    def generate(
        self,
        prompt: str,
        max_new_tokens: int  = 180,
        temperature: float   = 0.2,
        do_sample: bool      = False,
        top_p: float         = 0.9,
        repetition_penalty: float = 1.15,
        max_tokens: int      = 180,
    ) -> str:
        """
        Generate text from `prompt` using eval mode and torch.inference_mode().
        """
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=TOKENIZER_MAX_LEN,
                padding=False,
            ).to(self.device)

            prompt_len = inputs.input_ids.shape[1]

            gen_kwargs = {
                "max_new_tokens"    : max_new_tokens,
                "do_sample"         : do_sample,
                "repetition_penalty": repetition_penalty,
                "eos_token_id"      : self.tokenizer.eos_token_id,
                "pad_token_id"      : self.tokenizer.pad_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"]       = top_p

            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            return self.tokenizer.decode(
                outputs[0][prompt_len:], skip_special_tokens=True
            )

        except Exception as exc:
            logger.error("[LLM] generate() failed: %s\n%s", exc, traceback.format_exc())
            raise RuntimeError(f"LLM generation failed: {exc}") from exc


class TransformersEmbedder:
    """Mean-pool CLS embedder via vanilla Transformers."""

    def __init__(self, model_name=EMBED_MODEL_NAME):
        self.model_name = model_name
        self.device = "cpu"
        self._load()

    def _load(self):
        logger.info("[EMBED] Loading TransformersEmbedder from %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model     = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info("[EMBED] Fallback embedder ready")

    def encode(self, texts, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=256, return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = self.model(**enc, return_dict=True)
        hidden = out.last_hidden_state
        mask   = enc["attention_mask"].unsqueeze(-1).float()
        vecs   = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return vecs.cpu().numpy()


_llm: LocalLLM | None = None
_llm_error: str | None = None
_embedder = None
_faiss_cache: dict = {}


def get_llm() -> LocalLLM | None:
    global _llm, _llm_error
    if _llm is None and _llm_error is None:
        try:
            _llm = LocalLLM()
        except Exception as exc:
            _llm_error = str(exc)
            logger.error("[LLM] Could not load: %s", exc)
    return _llm


def get_llm_error() -> str | None:
    return _llm_error


def get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("[EMBED] Loading SentenceTransformer…")
            _embedder = SentenceTransformer(EMBED_MODEL_NAME)
            logger.info("[EMBED] SentenceTransformer ready")
        except Exception as exc:
            logger.warning("[EMBED] SentenceTransformer failed (%s). Falling back.", exc)
            try:
                _embedder = TransformersEmbedder()
            except Exception as fb_exc:
                logger.error("[EMBED] Fallback also failed: %s", fb_exc)
                _embedder = None
    return _embedder


def cache_faiss_index(user_id, index, chunks):
    _faiss_cache[user_id] = {"index": index, "chunks": chunks}


def get_cached_faiss(user_id):
    entry = _faiss_cache.get(user_id, {})
    return entry.get("index"), entry.get("chunks")


def clear_faiss_cache(user_id=None):
    if user_id:
        _faiss_cache.pop(user_id, None)
    else:
        _faiss_cache.clear()
