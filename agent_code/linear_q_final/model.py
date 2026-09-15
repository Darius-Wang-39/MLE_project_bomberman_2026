"""Linear model with migration from the original 17-feature checkpoint."""

from pathlib import Path
import numpy as np

from agent_code.linear_q_agent.model import LinearQModel as BaseLinearQModel


class LinearQModel(BaseLinearQModel):
    @classmethod
    def load(cls, path, feature_count):
        path = Path(path)
        if not path.exists():
            return cls(feature_count)
        with np.load(path) as data:
            old = np.asarray(data["weights"], dtype=np.float64)
            updates = int(np.asarray(data["updates"]).reshape(-1)[0])
        if old.size > feature_count:
            raise ValueError("Checkpoint has more features than this model")
        weights = np.zeros(feature_count, dtype=np.float64)
        weights[:old.size] = old
        return cls(feature_count, weights=weights, updates=updates)
