from enum import Enum
from typing import Sequence
import chex
import numpy as np 
import jax.numpy as jnp
import jax.lax as lax
import jax

import pickle
  
class SimilarityMetric(str, Enum):
  POLICY = "policy"
  VALUE = "value"
  POLICY_VALUE = "policy_value"
  LEGAL_ACTIONS = "legal_actions"
  LEGAL_POLICY = "legal_policy"
  LEGAL_POLICY_VALUE = "legal_policy_value"
  ACTION_HISTORY = "action_history"
  ACTION_HISTORY_POLICY = "action_history_policy"
  ACTION_HISTORY_LEGAL = "action_history_legal"
  ACTION_HISTORY_LEGAL_POLICY = "action_history_legal_policy"
  ISET_VECTOR = "iset_vector"
  ISET_POLICY = "iset_policy"
 
 
class DynamicsType(str, Enum):
  ISET = "iset"
  PUBLIC_STATE = "public_state"
  
class CategoricalSample(str, Enum):
  PURE_SOFTMAX = "pure_softmax"
  SINGLE_CATEGORY = "single_category"
  STRAIGHT_THROUGH = "straight_through"
  GUMBEL_SOFTMAX = "gumbel_softmax"
  GUMBEL_THROUGH = "gumbel_through"
  
  
chex.dataclass(frozen=True)
class TrainConfig:
  
  batch_size: int = 32
  
  trajectory_max: int = 6
  sampling_epsilon: float = 0.0
  
  train_rnad: bool = True
  train_transformations: bool = True
  train_mvs: bool = True
  train_abstraction: bool = True
  train_dynamics: bool = True
  train_legal_actions: bool = True
  
  
  use_abstraction: bool = False
  abstraction_amount: int = 10
  abstraction_size: int = 32
  similarity_metric: SimilarityMetric = SimilarityMetric.POLICY_VALUE
  similarity_noise: float = 0.02
  
  abstraction_soft_k_means_temperature: float = 1.0
  abstraction_soft_k_means_closeness_assignment: float = 0.5
  abstraction_soft_k_means_repulsive_force: float = 3.0
  abstraction_hard_k_means_closeness: float = 0.2
  transformation_soft_k_means_temperature: float = 1.0
  transformation_soft_k_means_closeness_assignment: float = 0.5
  transformation_soft_k_means_repulsive_force: float = 3.0
  
  dynamics_type: DynamicsType = DynamicsType.PUBLIC_STATE
  
  ps_encoder_hidden_size: int = 128
  ps_decoder_hidden_size: int = 64
  iset_hidden_size: int = 64
  dynamics_hidden_size: int = 64
  similarity_hidden_size: int = 64
  mvs_hidden_size: int = 64
  legal_actions_hidden_size: int = 64
  transformation_hidden_size: int = 128
  rnad_hidden_size: int = 256
  
  transformations: int = 10
  matrix_valued_states: bool = True
  
  c_iset_vtrace: float = 1.0
  rho_iset_vtrace: float = np.inf
  c_state_vtrace: float = 1.0
  rho_state_vtrace: float = np.inf
  
  eta_regularization: float = 0.2
  entropy_schedule_repeats: Sequence[int] = (1,)
  entropy_schedule_size: Sequence[int] = (2000,)
  
  learning_rate: float = 3e-4
  target_network_update: float = 1e-3
  seed: int = 42

 
 
@chex.dataclass(frozen=True)
class TimeStep():
  
  valid: chex.Array = () # [..., 1]
  player: chex.Array = () # [..., 1]
  public_state: chex.Array = () # [..., PS]
  
  # We store observation for both players, to have some information even when the opponent is playing.
  obs: chex.Array = () # [..., Player, O]
  legal: chex.Array = () # [..., A]
  action: chex.Array = () # [..., A]
  policy: chex.Array = () # [..., A]
  
  reward: chex.Array = () # [..., 1] Reward after playing an action
  

def normalize_loss(x: chex.Array, valid: chex.Array):
  # TODO: Verify this works
  normalization_term = jnp.sum(valid)
  beyond_valid_size = jnp.prod(jnp.array(x.shape[len(valid.shape):]))
  normalized_loss = x / (beyond_valid_size * normalization_term + (normalization_term == 0))
  return jnp.sum(normalized_loss)
  
  
def compute_abstraction_soft_kmeans(real: chex.Array, pred: chex.Array, valid: chex.Array, temperature: float, closeness_assignment: float, repulsive_force: float, hard_k_means_closeness: float):
  chex.assert_shape((real,), (*pred.shape[:-2], pred.shape[-1]))
  cluster_difference = lax.stop_gradient(jnp.expand_dims(real, -2)) - pred
  
  cluster_difference = cluster_difference * valid[..., None, None]
  
  cluster_distance = jnp.sum(cluster_difference ** 2, axis=-1)
  cluster_distance = cluster_distance + (cluster_distance < 1e-15)
  cluster_distance = cluster_distance ** 0.5
  
  closest = jnp.min(cluster_distance, -1, keepdims=True)
  
  soft_assignments = jax.nn.softmax(-cluster_distance * temperature, axis=-1)
  
  soft_assignments = jnp.where(jnp.logical_and(cluster_distance < closeness_assignment, cluster_distance > (closest + 1e-10)), -soft_assignments * repulsive_force, soft_assignments)
  
  hard_assignments = (cluster_distance <= (closest + 1e-10)).astype(jnp.float32)
  
  cluster_soft_assignement = jnp.where(closest < hard_k_means_closeness, hard_assignments, soft_assignments)
  
  cluster_each_other_distance = pred[..., :, None, :] - pred[..., None, :, :]
  cluster_each_other_distance = jnp.sum(cluster_each_other_distance ** 2, axis=-1)
  
  separation_loss = jnp.maximum(0, closeness_assignment - cluster_each_other_distance) * 0.2
  
  cluster_pullback_loss = jnp.sum(pred ** 2, axis=-1) * 1e-3
  
  separation_loss = normalize_loss(separation_loss, valid)
  cluster_pullback_loss = normalize_loss(cluster_pullback_loss, valid)
  
  cluster_loss = jnp.mean(cluster_difference ** 2, axis=-1)
  cluster_loss = jnp.sum(cluster_loss * cluster_soft_assignement, axis=-1) * valid 
  cluster_loss = cluster_loss * jnp.sqrt(pred.shape[-2])
  
  return jnp.mean(cluster_loss) + separation_loss + cluster_pullback_loss, cluster_soft_assignement
  
  

def sample_categorical(logits: chex.Array, rng_key, sample_type: CategoricalSample = CategoricalSample.GUMBEL_THROUGH, gumbel_temperature: float = 0.5):
  '''Produces a weigths from logits. Behaves differently depending on sample_type. gumbel_temperature is only used for Gumbel related methods.'''
  if sample_type == CategoricalSample.PURE_SOFTMAX:
    # Does propagate gradients, but the chosen weight may be too different then the real samples.
    return jax.nn.softmax(logits, axis=-1)
  if sample_type == CategoricalSample.SINGLE_CATEGORY:
    # Does not propagate gradient through the logits
    # Should be equivalent to jax.random.choice(rng_key, jax.nn.softmax(logits))
    category = jax.random.categorical(rng_key, logits, axis=-1)
    return jax.nn.one_hot(category, logits.shape[-1])
  elif sample_type == CategoricalSample.STRAIGHT_THROUGH:
    # The trick which propagates the gradients through the softmax which is immedietly subtracted so it does not affect the computation
    probability = jax.nn.softmax(logits, axis=-1)
    sample = jax.random.categorical(rng_key, logits, axis=-1)
    sample_one_hot = jax.nn.one_hot(sample, logits.shape[-1])
    return sample_one_hot + probability - lax.stop_gradient(probability)
  elif sample_type == CategoricalSample.GUMBEL_SOFTMAX:
    # The gumbel noise to produce soft samples, but more spikey than the actual samples
    gumbel_noise = jax.random.gumbel(rng_key, logits.shape)
    logits_with_noise = logits + gumbel_noise
    logits_with_noise = logits_with_noise / gumbel_temperature
    return jax.nn.softmax(logits_with_noise, axis=-1)
  elif sample_type == CategoricalSample.GUMBEL_THROUGH:
    # Combines Gumbel and Straight Through, should be the best of both worlds, but I am not sure.
    gumbel_noise = jax.random.gumbel(rng_key, logits.shape)
    logits_with_noise = logits + gumbel_noise
    
    sample = jnp.argmax(logits_with_noise, axis=-1) # You can take argmax later, but this is more clean.
    sample_one_hot = jax.nn.one_hot(sample, logits.shape[-1])
    
    logits_with_noise = logits_with_noise / gumbel_temperature
    probability = jax.nn.softmax(logits_with_noise, axis=-1)
    return sample_one_hot + probability - lax.stop_gradient(probability)
  else:
    raise ValueError(f"Invalid sample type: {sample_type}")

def save_model(model, path): 
  with open(path, "wb") as f:
    pickle.dump(model, f)
    
def load_model(path):
  with open(path, "rb") as f:
    return pickle.load(f)
  