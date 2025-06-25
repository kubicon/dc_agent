# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A version of JaxDarkChessCFR,
which aims to speed up the tree construction process
"""

# pylint: disable=g-importing-member
import functools

import chex
import jax
import jax.numpy as jnp
import numpy as np

from cfr.utils import JaxPolicy, stringify, num_distinct_actions
from cfr.jax_chess_cfr import regret_matching, update_regrets, update_regrets_plus, JAX_CFR_SIMULTANEOUS_UPDATE, TERMINAL_PLAYER_ID
from jax_chess import DarkChessGame, DarkChessGameState


@chex.dataclass(frozen=True)
class JaxChessCFRConstants:
  """Constants for the per-depth
  BFS-traversal of the tree. Also, things like
  isets or actions only for one player, since 
  in chess only one player can ever act at given depth."""

  starting_player: int
  max_depth: int
  players: int

  init_reaches: chex.ArrayTree = ()

  depth_actions: chex.ArrayTree = ()
  
  depth_history_action_utility: chex.ArrayTree = ()
  depth_history_iset: chex.ArrayTree = ()
  depth_history_actions: chex.ArrayTree = ()

  depth_history_next_history: chex.ArrayTree = ()
  depth_history_legal: chex.ArrayTree = ()

  depth_iset_legal: chex.ArrayTree = ()

class JaxDarkChessCFRFast:
  """A version of JaxDarkChessCFR,
  which aims to speed up the tree construction process.
  It operates per depth and constructs the tree using BFS
  instead of DFS. Also has per depth iset maps to help
  against imperfect recall.
  """

  def __init__(
      self,
      fen,
      depth_limit,
      regret_matching_plus=True,
      alternating_updates=True,
      linear_averaging=True,
  ):
    self.game = DarkChessGame(init_fen=fen)
    self._regret_matching_plus = regret_matching_plus
    self._alternating_updates = alternating_updates
    self._linear_averaging = linear_averaging
    self.timestep = 1
    #This is necessary, since chess games are far too long
    self.depth_limit = depth_limit

    self.init()

  def init(self):
    """Constructor."""

    # This implementation will only work for 2 player games !!!
    players = 2
    depth_history_action_utility = []
    depth_history_iset = []
    depth_history_actions = []
    depth_history_next_history = []
    depth_history_legal = []
    depth_iset_legal = []

    depth_iset_map = []
    depth_id_to_action = []
    #For the reconstruction in the original game
    self.full_game_distinct_actions = self.game.action_filter.shape[0]
    #these are just for single depth
    distinct_actions = num_distinct_actions(self.game)

    #Returns (H(D), shape of observation)
    vectorized_get_info = jax.vmap(self.game.observation_tensor,in_axes=(0, None), out_axes=(0))
    #H(D) * distinct_actions will be at most in the next layer
    # Expects (H(D) * |A|, H(D) * |A|) in the first dimension
    # Returns ((H(D) * |A|, state_dim), (H(D)* |A|,full_game_actions next_legals), (H(D) * |A|), (H(D) * |A|))
    # ordered next_states, next_legals, rewards, terminals
    vectorized_next_state = jax.vmap(self.game.apply_action,in_axes=(0, 0), out_axes=(0, 0, 0, 0))

    def create_legals_map(legals):
      true_legal = legals >= 0.5 #(H(D), true |A|)
      num_legal = np.sum(true_legal, axis=-1)
      nonzeros = np.nonzero(true_legal)
      flat_mapped_legals = np.full(legals.shape[0] *  distinct_actions, -1) # (H(D), -1)
      #TODO: Speed this up this is done naively for now
      action_mapped_indices = nonzeros[0].copy()
      for i in range(true_legal.shape[0]):
        action_mapped_indices[nonzeros[0] == i] = np.arange(num_legal[i]) + i * distinct_actions
      action_real_indices = nonzeros[1]
      flat_mapped_legals[action_mapped_indices] = action_real_indices
      mapped_legals = flat_mapped_legals.reshape((legals.shape[0], distinct_actions))
      return mapped_legals, mapped_legals > -1

    def create_iset_map(curr_iset, curr_legal, amount_actions, id_to_action, eps=1e-4):
      isets = []
      iset_map = []
      iset_legal = []
      iset_id_to_action = []
      for i in range(curr_iset.shape[0]): 
        curr_index = -1
        for j in range(0, len(iset_map)):
          if np.sum(np.abs(iset_map[j] - curr_iset[i])) <= eps:
            curr_index = j
            break
        if curr_index < 0:
          curr_index = len(iset_map)
          iset_map.append(curr_iset[i])
          iset_legal.append(curr_legal[i])
          iset_id_to_action.append(id_to_action[i])
        isets.append(curr_index)
          
      isets = np.array(isets)
      actions = isets[..., None] * amount_actions + np.arange(amount_actions)[None, ...]
      iset_map = np.array(iset_map)
      iset_legal = np.array(iset_legal)
      iset_id_to_action = np.array(iset_id_to_action)
      return iset_map, iset_legal, iset_id_to_action, isets, actions
    
    def _traverse_tree(states, legals, player:int, depth:int = 0):
      legals = np.asarray(legals)
      curr_isets = vectorized_get_info(states, player)
      curr_isets = np.asarray(curr_isets)
      history_id_to_action, mapped_legals = create_legals_map(legals)
      iset_map, iset_legal, id_to_action, isets, iset_actions = create_iset_map(curr_isets, mapped_legals, distinct_actions, history_id_to_action)

      actions = np.where(history_id_to_action > -1, history_id_to_action, 0).flatten()
      prev_states = jax.tree_map(lambda x: jnp.repeat(x, distinct_actions, axis=0), states)

      next_states, next_legals, next_utilities, next_terminal = vectorized_next_state(prev_states, actions)

      non_terminal = mapped_legals.flatten() * (~next_terminal)
      #Make sure to put all next_states as terminal if we are at the depth limit
      if depth + 1 == self.depth_limit:
        non_terminal *= 0
      
      action_utility = np.reshape(next_utilities.flatten() * mapped_legals.flatten(), mapped_legals.shape)
      nonzeros = np.nonzero(non_terminal)

      next_states = jax.tree_map(lambda x: x[*nonzeros], next_states)
      next_legals = next_legals[*nonzeros, :]
      # This should be -1 everywhere, except the part where you have next history. Therey ou go by terminal and just add 1
      next_history = np.reshape(((np.cumsum(non_terminal).reshape(non_terminal.shape) * non_terminal) - 1), mapped_legals.shape)

      depth_iset_map.append(iset_map)
      depth_id_to_action.append(id_to_action)

      depth_iset_legal.append(iset_legal)
      depth_history_action_utility.append(action_utility)
      depth_history_iset.append(isets)
      depth_history_actions.append(iset_actions)
      depth_history_legal.append(mapped_legals)
      depth_history_next_history.append(next_history.astype(int))
      if np.all(next_history < 0):
        return
      _traverse_tree(next_states, next_legals, 1 -player, depth + 1)


    root_state, legals = self.game.new_initial_state()
    root_state = jax.tree_map(lambda x: jnp.expand_dims(x, 0), root_state)
    _traverse_tree(
            root_state,
            legals[None, ...],
            int(self.game.init_state.current_player),
        ) 
        

    def convert_to_jax(x):
      return [jnp.asarray(i) for i in x]
    
    

    def convert_to_jax_players(x):
      return [[jnp.asarray(i) for i in x[pl]] for pl in range(self.constants.players)]
    
    depth_history_action_utility = convert_to_jax(depth_history_action_utility)
    depth_history_iset = convert_to_jax(depth_history_iset)
    depth_history_actions = convert_to_jax(depth_history_actions)
    depth_history_next_history = convert_to_jax(depth_history_next_history)
    depth_history_legal = convert_to_jax(depth_history_legal)
    depth_iset_legal = convert_to_jax(depth_iset_legal)

    max_depth = self.depth_limit

    self.constants = JaxChessCFRConstants(
      players = players,
      starting_player = int(self.game.init_state.current_player),
      max_depth = max_depth,

      init_reaches = jnp.ones((2, 1)),
      depth_actions = [distinct_actions] * max_depth,
      
      depth_history_action_utility = depth_history_action_utility,
      depth_history_iset = depth_history_iset,
      depth_history_actions = depth_history_actions,
      depth_history_next_history = depth_history_next_history,
      depth_history_legal = depth_history_legal,
      
      depth_iset_legal = depth_iset_legal,
    )
    #just to mask out turns where player does not play in jnp.where
    self.both_players = jnp.arange(2)
    self.regrets = [jnp.zeros((self.constants.depth_iset_legal[d].shape[0], a)) for d, a in enumerate(self.constants.depth_actions)]
    self.averages = [jnp.zeros((self.constants.depth_iset_legal[d].shape[0], a)) for d, a in enumerate(self.constants.depth_actions)]

    self.regret_matching = jax.vmap(regret_matching, 0, 0)
    if self._regret_matching_plus:
      self.update_regrets = jax.vmap(update_regrets_plus, 0, 0)
    else:
      self.update_regrets = jax.vmap(update_regrets, 0, 0)
    self.depth_iset_map = depth_iset_map
    self.depth_id_to_action = depth_id_to_action


  def multiple_steps(self, iterations: int):
    """Performs several CFR steps.

    Args:
      iterations: Amount of CFR steps, the solver should do.
    """
    for _ in range(iterations):
      self.step()

  def step(self):
    """Wrapper around the jitted function for performing CFR step."""
    averaging_coefficient = self.timestep if self._linear_averaging else 1
    if self._alternating_updates:
      for player in range(self.constants.players):
        self.regrets, self.averages = self.jit_step(
            self.regrets, self.averages, averaging_coefficient, player
        )

    else:
      self.regrets, self.averages = self.jit_step(
          self.regrets,
          self.averages,
          averaging_coefficient,
          JAX_CFR_SIMULTANEOUS_UPDATE,
      )

    self.timestep += 1

  @functools.partial(jax.jit, static_argnums=(0,4))
  def jit_step(
      self, regrets, averages, average_policy_update_coefficient, player
  ):
    """Performs the CFR step.

    This consists of:
    1. Computes the current strategies based on regrets
    2. Computes the realization plan for each action from top of the tree down
    3. Compute the counterfactual regrets from bottom of the tree up
    4. Updates regrets and average stretegies

    Args:
      regrets: Cummulative regrets for all players, list[Float[Isets, Actions]]
      averages: Average strategies for all players, list[Float[Isets, Actions]]
      average_policy_update_coefficient: Weight of the average policy update.
        When enabled linear_averging it is equal to current iteration. Otherwise
        1, int
      player: Player for which the update should be done. When alternating
        updates are distables, it is JAX_CFR_SIMULTANEOUS_UPDATE

    Returns:
      regrets: the regrets.
      averages: the averages.
    """
    current_strategies = [self.regret_matching(regrets[d], self.constants.depth_iset_legal[d]) for d in range(self.constants.max_depth)]

    history_reaches = [self.constants.init_reaches]
    
    history_strategies = []
    # We allow different legal actions in different histories, even if they are in the same infoset.
    # Not relevant yet but it will be once we begin using abstractions
    current_player = self.constants.starting_player
    for d in range(self.constants.max_depth):
      pl_legals = self.constants.depth_history_legal[d]

      pl_strategy = current_strategies[d][self.constants.depth_history_iset[d]]
      
      strategies = pl_strategy * pl_legals
      strategies = jnp.where(jnp.sum(strategies, axis=-1, keepdims=True) > 1e-8, strategies / jnp.sum(strategies, axis=-1, keepdims=True), pl_legals / jnp.sum(pl_legals, axis=-1, keepdims=True))
      history_strategies.append(strategies)
      current_player = 1 - current_player
    
    current_player = self.constants.starting_player
    for d in range(self.constants.max_depth):
      #do not modify the reach 
      strategy_realization = history_reaches[d][current_player][..., None] * history_strategies[d]
      
      strategy_realization_non_masked = history_reaches[d][current_player][..., None] * current_strategies[d][self.constants.depth_history_iset[d]]
      
      pl_iset_realizations = jnp.bincount(self.constants.depth_history_actions[d].ravel(), strategy_realization_non_masked.ravel(), length=self.constants.depth_actions[d] * self.constants.depth_iset_legal[d].shape[0]).reshape(averages[d].shape)
      # TODO: This is dumb 
      if player != 1 and current_player == 0:   
        averages[d] = averages[d] + pl_iset_realizations * average_policy_update_coefficient  
      if player != 0 and current_player == 1:
        averages[d] = averages[d] + pl_iset_realizations * average_policy_update_coefficient
      # We do not compute the next history, since there is none
      if d == self.constants.max_depth - 1:
        break
      next_terminal_mask = self.constants.depth_history_next_history[d] >= 0

      pl_masked_realization = strategy_realization * next_terminal_mask
      not_acting_masked_realization = jnp.repeat(history_reaches[d][1 - current_player], repeats=self.constants.depth_actions[d], axis=0).reshape(pl_masked_realization.shape) * next_terminal_mask
      
      pl_reaches_next = jnp.bincount(self.constants.depth_history_next_history[d].ravel(), pl_masked_realization.ravel(), length=self.constants.depth_history_next_history[d+1].shape[0])
      not_acting_reaches_next = jnp.bincount(self.constants.depth_history_next_history[d].ravel(), not_acting_masked_realization.ravel(), length=self.constants.depth_history_next_history[d+1].shape[0])


      history_reaches.append(jnp.where((current_player == self.both_players)[..., None], jnp.stack([pl_reaches_next, pl_reaches_next], axis=0), jnp.stack([not_acting_reaches_next, not_acting_reaches_next], axis=0)))
      current_player = 1 - current_player
  
    # In last row, there are only terminal, so we start row before it
    depth_utils = [jnp.zeros((1,))]
    #the player in the last depth
    current_player = self.constants.starting_player + ((self.constants.max_depth - 1) % 2) 
    for d in range(self.constants.max_depth - 1, -1, -1):
      action_value = jnp.where(self.constants.depth_history_next_history[d] >= 0, depth_utils[-1][self.constants.depth_history_next_history[d]], self.constants.depth_history_action_utility[d])
      action_probabilities = history_strategies[d]
      history_value = jnp.sum(action_value * action_probabilities, axis=-1)
      depth_utils.append(history_value)
      
      pl_cf_regret = (action_value - history_value[..., None]) * history_reaches[d][1 - current_player][..., None]
      pl_cf_regret = pl_cf_regret * self.constants.depth_history_legal[d]
      pl_bin_regrets = jnp.bincount(self.constants.depth_history_actions[d].ravel(), pl_cf_regret.ravel(), length=self.constants.depth_actions[d] * self.constants.depth_iset_legal[d].shape[0]).reshape(regrets[d].shape) * self.constants.depth_iset_legal[d]
      # TODO: This is dumb
      if player != 1 and current_player == 0:
        regrets[d] = jnp.maximum(regrets[d] + pl_bin_regrets, 0.0)
        
      if player != 0 and current_player == 1: 
        regrets[d] = jnp.maximum(regrets[d] - pl_bin_regrets, 0.0)
        
      current_player = 1 - current_player
      
    return regrets, averages

  def average_policy(self):
    """Extracts the average policy from JAX structures into a JaxPolicy."""
    averages = [np.asarray(self.averages[d]) for d in range(self.constants.max_depth)]
    averages =[np.where(averages[d] >= 0, averages[d], np.ones_like(averages[d])) for d in range(self.constants.max_depth)] 
    averages = [
        averages[d] / np.sum(averages[d], -1, keepdims=True)
        for d in range(self.constants.max_depth)]
    
    avg_strategy = JaxPolicy()
   
    current_player = int(self.game.init_state.current_player)
    #TODO: For now we assume that 
    # the isets appearing in lower depths cannot have
    # appeared in the tree before. Just be careful about this
    # once abstractions are used and also because
    # the chess "isets" are not real isets.
    for d in range(self.constants.max_depth):
      for idx, iset in enumerate(self.depth_iset_map[d]):
        iset_policy = np.zeros(self.full_game_distinct_actions, dtype=float)
        for id, action in enumerate(self.depth_id_to_action[d][idx]):
          iset_policy[action] = averages[d][idx][id]
        #rather renormalize so that numpy does not complain
        total = np.sum(iset_policy)
        iset_policy = np.where(total > 0, iset_policy / total, 0)
        avg_strategy[stringify(iset)] = iset_policy
      current_player = 1 - current_player
    return avg_strategy#, averages, self.regrets, self.constants.depth_iset_legal

