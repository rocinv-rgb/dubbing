"""
pipeline/config.py - Konfigurácia a konštanty
"""

import os
import torch
from pathlib import Path

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/workspace/models"))
CACHE_DIR  = Path(os.environ.get("CACHE_DIR",  "/workspace/cache"))
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# --- Konfigurácia zlučovania segmentov ---
# Prepisatelne z job inputu (pozri run_dubbing_pipeline parameter pause_marker)
PAUSE_MARKER: str | None = None   # None = ignoruj pauzy, "..." = vloz marker medzi vety
MAX_MERGE_PAUSE_S: float = 1.0    # max pauza medzi segmentmi na zlucenie (s)
MAX_MERGE_BLOCK_S: float = 7.0    # max dlzka zluceneho bloku (s)
SEGMENT_GAP_MS: float = 75.0      # buffer (ms) — TTS skonci aspon X ms pred zaciatkom dalsej vety
