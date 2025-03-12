"""Policy implementation for tiny models."""

import dataclasses
import logging
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.shared import array_typing as at
from openpi import transforms as _transforms


@dataclasses.dataclass
class TinyPolicy:
    """A policy that always returns fixed actions for tiny models.
    
    This is a temporary solution to make tiny models work with Libero.
    """
    
    model: _model.BaseModel
    transforms: Sequence[_transforms.DataTransformFn] = ()
    output_transforms: Sequence[_transforms.DataTransformFn] = ()
    sample_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    action_dim: int = 7  # Default for Libero
    
    def infer(self, observation: _model.Observation) -> dict[str, np.ndarray]:
        """Infer actions from the observation."""
        logging.info("Using TinyPolicy with fixed actions")
        
        # Create a fixed action array with zeros
        # For Libero, we need shape (1, 7)
        actions = np.zeros((1, self.action_dim), dtype=np.float32)
        
        return {"actions": actions}
    
    def sample_actions(
        self, rng: at.KeyArrayLike, observation: _model.Observation, **kwargs
    ) -> dict[str, np.ndarray]:
        """Sample actions from the observation."""
        return self.infer(observation) 