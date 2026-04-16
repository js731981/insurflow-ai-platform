from __future__ import annotations

# Public demo thumbnails (no bundled binaries in the Space).
DEMO_IMAGES = {
    "no_damage": "https://images.unsplash.com/photo-1510557880182-3d4d3cba35a5",
    "minor_crack": "https://images.unsplash.com/photo-1580910051074-3eb694886505",
    "major_crack": "https://images.unsplash.com/photo-1598327105666-5b89351aff97",
}

# Hugging Face Spaces typically do not allow running local Ollama.
USE_LLM = False

# If `models/model.pth` exists, the demo will try to load it.
# Otherwise it falls back to a lightweight heuristic image analysis.
USE_CNN = True

TARGET_CLASSES = ["no_damage", "minor_crack", "major_crack"]

# Simple decision thresholds (demo-only).
LOW_AMOUNT_USD = 500.0
HIGH_AMOUNT_USD = 5000.0

# Fraud score bands.
APPROVE_MAX_SCORE = 0.35
INVESTIGATE_MAX_SCORE = 0.70

# Image-to-risk mapping (used when an image is provided).
SEVERITY_RISK = {"low": 0.1, "medium": 0.35, "high": 0.65}

