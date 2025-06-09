"""Tokenizer loading utilities to avoid repeated loading."""

import logging

logger = logging.getLogger(__name__)


def load_tokenizer_for_model(model_name: str, trust_remote_code: bool = True):
    """Load the appropriate tokenizer for the given model.

    Returns None if tokenizer loading fails or is not needed.
    """
    if not model_name:
        return None

    try:
        if model_name.startswith("voyage-"):
            # Use AutoTokenizer from transformers for Voyage models
            from transformers import AutoTokenizer

            tokenizer_name = f"voyageai/{model_name}"
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info(f"Loaded Voyage tokenizer: {tokenizer_name}")
            return tokenizer
        elif not model_name.startswith("models/"):  # Local models (not Gemini)
            # Use transformers for local models
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )
            logger.info(f"Loaded tokenizer for local model: {model_name}")
            return tokenizer
    except Exception as e:
        logger.warning(f"Failed to load tokenizer for {model_name}: {e}")
        logger.warning("Token counting may be less accurate")

    return None
