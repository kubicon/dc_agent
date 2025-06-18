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

"""A simultaneous version of JAX-CFR, which is made to
work specifically on the jax implementation of Leduc Poker
"""

# pylint: disable=g-importing-member

from collections import namedtuple
import functools

import chex
import jax
import jax.numpy as jnp
import numpy as np

from cfr.utils import JaxPolicy, stringify
from jax_chess import DarkChessGame, DarkChessGameState, Pieces

JAX_CFR_SIMULTANEOUS_UPDATE = -5
TERMINAL_PLAYER_ID = -4


def regret_matching(regret, mask):
  """Computes current policy based on current regrets.

  Args:
    regret: Current regrets in array Fkiat[Isets, Actions]
    mask: Legal action mask Bool[Isets, Actions]

  Returns:
    policy: the policy.
  """
  regret = jnp.maximum(regret, 0) * mask
  total = jnp.sum(regret, axis=-1, keepdims=True)

  return jnp.where(total > 0.0, regret / total, 1.0 / jnp.sum(mask)) * mask


def update_regrets_plus(regret):
  """Clamps the regrets to be non-negative."""
  return regret * (regret > 0)


def update_regrets(regret):
  """Updates the regrets without CFRPlus."""
  return regret

def num_distinct_actions(game: DarkChessGame):
  """Computes an upper bound on the number of
  actions applicable in the given game instance"""
  board = game.init_state.board
  board_max_dim = max(board.shape)
  #for each type of piece 
  # take the maximum amount across both players.
  # The kings are exception, where we 
  # assume each player has exactly one king
  num_pawns = jnp.maximum(jnp.sum(board == Pieces.WHITE_PAWN), jnp.sum(board == Pieces.BLACK_PAWN))
  # Each pawn can be potentially promoted to either of these pieces
  num_knights = jnp.maximum(jnp.sum(board == Pieces.WHITE_KNIGHT), jnp.sum(board == Pieces.BLACK_KNIGHT)) + num_pawns
  num_rooks = jnp.maximum(jnp.sum(board == Pieces.WHITE_ROOK), jnp.sum(board == Pieces.BLACK_ROOK)) + num_pawns
  num_bishops = jnp.maximum(jnp.sum(board == Pieces.WHITE_BISHOP), jnp.sum(board == Pieces.BLACK_BISHOP)) + num_pawns
  num_queens = jnp.maximum(jnp.sum(board == Pieces.WHITE_QUEEN), jnp.sum(board == Pieces.BLACK_QUEEN)) + num_pawns
  #Castling is allowed only for 8x8 board
  castling_allowed = int(board.shape[0] == 8 and board.shape[1] == 8)
  #Each pawn has in total 4 total possible moves
  #1 forward, 2 forward, and 2 diagonal.
  pawn_actions = 4 * num_pawns
  #knights can jump only to 8 places at max
  knight_actions = 8 * num_knights
  # rooks and bishops can each move in 4 directions,
  # each no longer than the bigger dimension of board - 1
  rook_actions, bishop_actions = num_rooks * (4 * (board_max_dim - 1)),  num_bishops * (4 * (board_max_dim - 1))
  # queens the same except 8 directions
  queen_actions = num_queens * (8 * (board_max_dim - 1))
  king_actions = 8
  #This is a  VERY pessimistic estimate, most of these will not be legal 
  total_actions = int(castling_allowed + pawn_actions + knight_actions + rook_actions + bishop_actions + queen_actions + king_actions)
  return total_actions


@chex.dataclass(frozen=True)
class JaxCFRConstants:
  """Constants for JaxCFR."""

  players: int
  max_depth: int
  max_actions: int

  max_iset_depth: chex.ArrayTree = ()  # Is just a list of integers
  isets: chex.ArrayTree = ()  # Is just a list of integers

  depth_history_utility: chex.ArrayTree = ()
  depth_history_iset: chex.ArrayTree = ()
  depth_history_actions: chex.ArrayTree = ()
  depth_history_previous_iset: chex.ArrayTree = ()
  depth_history_previous_action: chex.ArrayTree = ()

  depth_history_next_history: chex.ArrayTree = ()
  depth_history_player: chex.ArrayTree = ()
  depth_history_previous_history: chex.ArrayTree = ()
  depth_history_action_mask: chex.ArrayTree = ()

  iset_previous_action: chex.ArrayTree = ()
  iset_action_mask: chex.ArrayTree = ()
  iset_action_depth: chex.ArrayTree = ()

def check_constants_shapes(constants: JaxCFRConstants):
  def check_jax_shape(x, label=""):
    if label:
      print(label)
    print("Lenght: ", len(x))
    for i in x:
      print(i.shape)

  def check_jax_shape_players(x, label=""):
    if label:
      print(label)
    for pl in range(2):
      y = x[pl]
      check_jax_shape(y)


  print("Max iset depth: ", constants.max_iset_depth)

  check_jax_shape(constants.depth_history_utility, "Depth history utility")
  check_jax_shape_players(constants.depth_history_previous_iset, "Depth history previous iset")
  check_jax_shape_players(constants.depth_history_actions, "Depth history actions")
  check_jax_shape_players(constants.depth_history_previous_action, "Depth history previous previous action")
  check_jax_shape(constants.depth_history_action_mask, "Depth history action mask")
  check_jax_shape(constants.depth_history_next_history, "Depth history next history")
  check_jax_shape(constants.depth_history_player, "Depth history player")
  check_jax_shape(constants.depth_history_previous_history, "Depth history previous history")
  #check_jax_shape_players(constants.iset_previous_action, "Iset previous action")
  #check_jax_shape_players(constants.iset_action_mask, "Iset action mask")
  #check_jax_shape_players(constants.iset_action_depth, "Iset action depth")

class JaxDarkChessCFR:
  """A JaxCFR as implemented in OpenSpiel, created 
  for full game solving of DarkChessGame. Unlike other JaxGame games,
   DarkChess are treated as a turn-based game inherently.

  First it prepares all the structures in `init`, then it just reuses them
  within jitted function `jit_step`.
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
    depth_history_utility = []
    depth_history_previous_iset = [[] for _ in range(players)]
    depth_history_previous_action = [[] for _ in range(players)]
    depth_history_iset = [[] for _ in range(players)]
    depth_history_actions = [[] for _ in range(players)]
    depth_history_next_history = []
    depth_history_player = []
    depth_history_previous_history = []
    depth_history_action_mask = []
    # Previous action is mapping of both iset and action!
    iset_previous_action = [[] for _ in range(players)]
    iset_action_mask = [[] for _ in range(players)]
    iset_action_depth = [[] for _ in range(players)]
    ids = [0 for _ in range(players)]
    pl_isets = [{} for _ in range(players)]
    #depth_history_action_map = [[] for _ in range(players)]
    #depth_history_action_ids = [[] for _ in range(players)]
    #Per iset. Needed to reconstruct the policy for the original game
    self.ids_to_action = [[] for _ in range(players)]
    #For the reconstruction in the original game
    self.full_game_distinct_actions = self.game.action_filter.shape[0]
    #these are just for single depth
    distinct_actions = num_distinct_actions(self.game)
    self.tree_vertices = 0

    for pl in range(players):
      pl_isets[pl][''] = ids[pl]
      ids[pl] += 1
      am = [0] * distinct_actions
      am[0] = 1
      iset_action_mask[pl].append(am)
      iset_previous_action[pl].append(0)
      iset_action_depth[pl].append(0)
      self.ids_to_action[pl].append([])

    PreviousInfo = namedtuple(
        'PreviousInfo',
        ('actions', 'isets', 'prev_actions', 'history'),
    )

    def _traverse_tree(state: DarkChessGameState, previous_info: PreviousInfo, legal_actions, reward, terminal, depth):
      self.tree_vertices += 1
      legal_actions = np.asarray(legal_actions)
      if len(depth_history_next_history) <= depth:
        for pl in range(players):
          depth_history_previous_iset[pl].append([])
          depth_history_previous_action[pl].append([])
          depth_history_iset[pl].append([])
          depth_history_actions[pl].append([])
          #depth_history_action_map[pl].append([])
          #depth_history_action_ids[pl].append(0)

        depth_history_action_mask.append([])
        depth_history_next_history.append([])
        depth_history_utility.append([])
        depth_history_player.append([])
        depth_history_previous_history.append([])
      history_id = len(depth_history_previous_history[depth])
      next_history_temp = [0] * distinct_actions
      depth_history_next_history[depth].append(next_history_temp)
      #cut off the nodes at depth limit
      if terminal or depth == self.depth_limit:
        current_player = TERMINAL_PLAYER_ID
      else :
        current_player = int(state.current_player)
      depth_history_player[depth].append(current_player)
      depth_history_previous_history[depth].append(previous_info.history)
      actions_mask = [0] * distinct_actions
      actions_to_id = {}
      action_id = 0
      id_to_actions = {}
      for ai, a in enumerate(legal_actions):
        if a < 0.5:
          continue
        if ai not in actions_to_id:
          actions_to_id[ai] = action_id
          action_id += 1
        ai_mapped = actions_to_id[ai]
        id_to_actions[ai_mapped] = ai
        actions_mask[ai_mapped] = 1
      depth_history_action_mask[depth].append(actions_mask)
      for pl in range(players):
        depth_history_previous_iset[pl][depth].append(previous_info.isets[pl])
        depth_history_previous_action[pl][depth].append(
            previous_info.actions[pl]
        )
      
      depth_history_utility[depth].append(reward)
      if current_player >= 0:
        #acting player is the one who does not 
        # have the invalid action as legal
        iset_tensor = self.game.observation_tensor(state, current_player)
        iset = stringify(iset_tensor)
        for pl in range(players):
          if pl == current_player:
            if iset not in pl_isets[pl]:
              pl_isets[pl][iset] = ids[pl]
              ids[pl] += 1
              self.ids_to_action[state.current_player].append(id_to_actions)
              iset_previous_action[pl].append(previous_info.actions[pl])
              iset_action_mask[pl].append(actions_mask)
              iset_action_depth[pl].append(previous_info.prev_actions[pl])
            depth_history_iset[pl][depth].append(pl_isets[pl][iset])
            depth_history_actions[pl][depth].append([
                i + pl_isets[pl][iset] * distinct_actions
                for i in range(distinct_actions)
            ])
          #Give invalid iset to the player that is not acting
          else:
            depth_history_iset[pl][depth].append(0)
            depth_history_actions[pl][depth].append(
                [0 for _ in range(distinct_actions)]
            )
      else:
        for pl in range(players):
          depth_history_iset[pl][depth].append(0)
          depth_history_actions[pl][depth].append(
              [0 for _ in range(distinct_actions)]
          )
      if current_player == TERMINAL_PLAYER_ID:
        return
      for ai, a in enumerate(legal_actions):
        if a < 0.5:
          continue
        ai_mapped = actions_to_id[ai]
        new_actions = tuple(
            pl_isets[pl][iset] * distinct_actions + ai_mapped if pl == current_player else previous_info.actions[pl]
            for pl in range(players)
        )
        new_infosets = tuple(
            pl_isets[pl][iset] if pl == current_player else previous_info.isets[pl]
            for pl in range(players)
        )
        new_prev_actions = tuple(
            previous_info.prev_actions[pl] + int(pl == current_player)
            for pl in range(players)
        )
        new_info = PreviousInfo(
            new_actions,
            new_infosets,
            new_prev_actions,
            history_id,
        )
        new_state,next_legals, next_reward, next_terminal = self.game.apply_action(state, ai)
        next_history_temp[ai_mapped] = (
            len(depth_history_player[depth + 1])
            if len(depth_history_player) > depth + 1
            else 0
        )
        _traverse_tree(new_state, new_info, next_legals, next_reward, next_terminal, depth + 1)
    root_state, legals = self.game.new_initial_state()
    _traverse_tree(
            root_state,
            PreviousInfo(
                tuple(0 for _ in range(players)),
                tuple(0 for _ in range(players)),
                tuple(0 for _ in range(players)),
                0,
            ),
            legals,
            0.0,
            False,
            0,
        )
        

    def convert_to_jax(x):
      return [jnp.asarray(i) for i in x]
    
    

    def convert_to_jax_players(x):
      return [[jnp.asarray(i) for i in x[pl]] for pl in range(players)]
    print("Tree has ", self.tree_vertices, " vertices.")
    
    depth_history_utility = convert_to_jax(depth_history_utility)
    depth_history_iset = convert_to_jax_players(depth_history_iset)
    depth_history_previous_iset = convert_to_jax_players(
        depth_history_previous_iset
    )
    depth_history_actions = convert_to_jax_players(depth_history_actions)
    depth_history_previous_action = convert_to_jax_players(
        depth_history_previous_action
    )
    depth_history_action_mask = convert_to_jax(depth_history_action_mask)

    depth_history_next_history = convert_to_jax(depth_history_next_history)
    depth_history_player = convert_to_jax(depth_history_player)
    depth_history_previous_history = convert_to_jax(
        depth_history_previous_history
    )

    max_iset_depth = [np.max(iset_action_depth[pl]) for pl in range(players)]
    iset_previous_action = convert_to_jax(iset_previous_action)
    iset_action_mask = convert_to_jax(iset_action_mask)
    iset_action_depth = convert_to_jax(iset_action_depth)


    self.constants = JaxCFRConstants(
        players=players,
        max_depth=int(len(depth_history_utility)),
        max_actions=distinct_actions,
        max_iset_depth=max_iset_depth,
        isets=ids,
        depth_history_utility=depth_history_utility,
        depth_history_iset=depth_history_iset,
        depth_history_actions=depth_history_actions,
        depth_history_previous_iset=depth_history_previous_iset,
        depth_history_previous_action=depth_history_previous_action,
        depth_history_next_history=depth_history_next_history,
        depth_history_player=depth_history_player,
        depth_history_previous_history=depth_history_previous_history,
        depth_history_action_mask=depth_history_action_mask,
        iset_previous_action=iset_previous_action,
        iset_action_mask=iset_action_mask,
        iset_action_depth=iset_action_depth,
    )

    self.regrets = [
        jnp.zeros((ids[pl], distinct_actions)) for pl in range(players)
    ]
    self.averages = [
        jnp.zeros((ids[pl], distinct_actions)) for pl in range(players)
    ]

    self.regret_matching = jax.vmap(regret_matching, 0, 0)
    if self._regret_matching_plus:
      self.update_regrets = jax.vmap(update_regrets_plus, 0, 0)
    else:
      self.update_regrets = jax.vmap(update_regrets, 0, 0)

    self.iset_map = pl_isets
    check_constants_shapes(self.constants)


  def multiple_steps(self, iterations: int):
    """Performs several CFR steps.

    Args:
      iterations: Amount of CFR steps, the solver should do.
    """
    for _ in range(iterations):
      self.step()

  def evaluate_and_update_policy(self):
    """Wrapper to step().

    Ensures interchangability with
    open_spiel.python.algorithms.cfr._CFRSolverBase.
    """
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

  def propagate_strategy(self, current_strategies):
      """Propagtes the strategies withing infosets.

      Args:
        current_strategies: Current strategies for all players, list[Float[Isets,
          Actions]]
      Returns:
        realization_plans: the realization plans.
      """
      realization_plans = [
          jnp.ones_like(current_strategies[pl])
          for pl in range(self.constants.players)
      ]

      for pl in range(self.constants.players):
        for i in range(0, self.constants.max_iset_depth[pl] + 1):
          realization_plans[pl] = jnp.where(
              self.constants.iset_action_depth[pl][..., jnp.newaxis] == i,
              current_strategies[pl]
              * realization_plans[pl].ravel()[
                  self.constants.iset_previous_action[pl]
              ][..., jnp.newaxis],
              realization_plans[pl],
          )

      return realization_plans

  @functools.partial(jax.jit, static_argnums=(0,))
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
    current_strategies = [
        self.regret_matching(regrets[pl], self.constants.iset_action_mask[pl])
        for pl in range(self.constants.players)
    ]

    realization_plans = self.propagate_strategy(current_strategies)
    iset_reaches = [
        jnp.sum(realization_plans[pl], -1)
        for pl in range(self.constants.players)
    ]
    # In last row, there are only terminal, so we start row before it
    depth_utils = [
        [self.constants.depth_history_utility[-1] * (1 - 2 * pl)]
        for pl in range(self.constants.players)
    ]
    for i in range(self.constants.max_depth - 2, -1, -1):
      #here was chance original this is why the 1. Dark chess dont have chance.
      each_history_policy = jnp.ones(self.constants.depth_history_player[i][..., jnp.newaxis].shape)
      for pl in range(self.constants.players):
        each_history_policy = each_history_policy * jnp.where(
            self.constants.depth_history_player[i][..., jnp.newaxis] == pl,
            current_strategies[pl][self.constants.depth_history_iset[pl][i]],
            1,
        )

      for pl in range(self.constants.players):
        action_value = jnp.where(
            self.constants.depth_history_player[i][..., jnp.newaxis] == -4,
            self.constants.depth_history_utility[i][..., jnp.newaxis] * (1 - 2 * pl),
            depth_utils[pl][-1][self.constants.depth_history_next_history[i]],
        )
        history_value = jnp.sum(action_value * each_history_policy, -1)
        regret = (
            (action_value - history_value[..., jnp.newaxis])
            * self.constants.depth_history_action_mask[i]
            * (self.constants.depth_history_player[i][..., jnp.newaxis] == pl)
        )
        for pl2 in range(self.constants.players):
          if pl != pl2:
            regret = (
                regret
                * realization_plans[pl2].ravel()[
                    self.constants.depth_history_previous_action[pl2][i]
                ][..., jnp.newaxis]
            )
        bin_regrets = jnp.bincount(
            self.constants.depth_history_actions[pl][i].ravel(),
            regret.ravel(),
            length=self.constants.isets[pl] * self.constants.max_actions,
        )
        bin_regrets = bin_regrets.reshape(-1, self.constants.max_actions)
        regrets[pl] = jnp.where(
            jnp.logical_or(player == pl, player == JAX_CFR_SIMULTANEOUS_UPDATE),
            regrets[pl] + bin_regrets,
            regrets[pl],
        )
        depth_utils[pl][-1] = history_value

    regrets = [
        self.update_regrets(regrets[pl]) for pl in range(self.constants.players)
    ]

    averages = [
        jnp.where(
            jnp.logical_or(player == pl, player == JAX_CFR_SIMULTANEOUS_UPDATE),
            averages[pl]
            + current_strategies[pl]
            * iset_reaches[pl][..., jnp.newaxis]
            * average_policy_update_coefficient,
            averages[pl],
        )
        for pl in range(self.constants.players)
    ]

    return regrets, averages

  def average_policy(self):
    """Extracts the average policy from JAX structures into a JaxPolicy."""
    averages = [
        np.asarray(self.averages[pl]) for pl in range(self.constants.players)
    ]
    averages = [
      np.where(averages[pl] >= 0, averages[pl], np.ones_like(averages[pl])) for pl in range(self.constants.players)
    ]
    averages = [
        averages[pl] / np.sum(averages[pl], -1, keepdims=True)
        for pl in range(self.constants.players)
    ]
    avg_strategy = JaxPolicy()

    for pl in range(2):
      for iset, idx in self.iset_map[pl].items():
        if not iset:
          continue
        state_policy = np.zeros(self.full_game_distinct_actions, dtype=float)
        for id, action in self.ids_to_action[pl][idx].items():
          state_policy[action] = averages[pl][idx][id]
        #rather renormalize so that numpy does not complain
        total = np.sum(state_policy)
        state_policy = np.where(total > 0, state_policy / total, 0)
        avg_strategy[iset] = state_policy
    return avg_strategy

