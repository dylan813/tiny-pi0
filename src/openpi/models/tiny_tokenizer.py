import logging
import numpy as np
import sentencepiece
from transformers import AutoProcessor

import openpi.shared.download as download

class TinyFASTTokenizer:
    """A simplified tokenizer for the Tiny Pi0 FAST model."""
    
    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = "physical-intelligence/fast"):
        self._max_len = max_len

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # Instantiate FAST tokenizer
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            # For tiny model, we'll use a simpler approach for action tokens
            # Just encode the raw action values as strings
            action_str = " ".join([f"{a:.4f}" for a in actions.flatten()])
            action_tokens = self._paligemma_tokenizer.encode(f"Action: {action_str}")
            postfix_tokens = action_tokens + self._paligemma_tokenizer.encode("|")
        else:
            postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [0] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + [False] * (self._max_len - tokens_len)
            ar_mask = ar_mask + [0] * (self._max_len - tokens_len)
            loss_mask = loss_mask + [False] * (self._max_len - tokens_len)
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        """Extract actions from predicted output tokens."""
        try:
            # Decode predicted output tokens
            decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())
            
            # Extract actions from decoded tokens
            if "Action: " not in decoded_tokens:
                logging.warning("No 'Action:' found in decoded tokens, returning zeros")
                return np.zeros((action_horizon, action_dim), dtype=np.float32)
            
            # Extract the action string
            action_str = decoded_tokens.split("Action: ")[1].split("|")[0].strip()
            
            # Parse the action values
            try:
                action_values = [float(val) for val in action_str.split() if val.strip()]
                
                # Always ensure we return the correct shape
                actions = np.zeros((action_horizon, action_dim), dtype=np.float32)
                
                # Fill in as many values as we have
                if action_values:
                    flat_actions = actions.flatten()
                    for i, val in enumerate(action_values):
                        if i < len(flat_actions):
                            flat_actions[i] = val
                    
                    # Reshape back to the expected dimensions
                    actions = flat_actions.reshape(action_horizon, action_dim)
                
                return actions
            except Exception as e:
                logging.warning(f"Error parsing action values: {e}")
                logging.warning(f"Action string: {action_str}")
                return np.zeros((action_horizon, action_dim), dtype=np.float32)
                
        except Exception as e:
            logging.warning(f"Error extracting actions: {e}")
            return np.zeros((action_horizon, action_dim), dtype=np.float32) 