import numpy as np

class JaxPolicy:
  policy: dict[str, list[float]]
  
  def __init__(self, policy: dict[str, list[float]] = None) -> None:
    self.policy = {}
    # Using default {} broke things
    if policy is not None:
      self.policy = policy
  
  def __getitem__(self, key: str) -> list[float]:
    return self.policy[key]
  
  def __setitem__(self, key: str, value: list[float]) -> None:
    self.policy[key] = value

def stringify(a: list[float]) -> str:
  return ",".join(str(i) for i in a)

def destringify(a: str) -> list[float]:
  return np.array([float(i) for i in a.split(",")])