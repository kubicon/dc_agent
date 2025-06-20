import numpy as np

from cfr.utils import JaxPolicy, stringify, get_policy_with_default, num_distinct_actions
from jax_chess import DarkChessGame, DarkChessGameState
from copy import deepcopy


def expected_value(policy: JaxPolicy, game: DarkChessGame, depth_limit:int) -> float:
    """
    Calculate the expected return of the policy
    
    Args:
        policy (JaxPolicy): The policy to evaluate.
        game (DarkChessGame): The game instance.
        depth_limit(int): The cutoff after which the game ends with a tie. -1 for full game but those are usually intractable.
    
    Returns:
        float: The expected value of the policy at the initial state.
    """
    init_state, init_legals = game.new_initial_state()
    eps = 1e-6 # Cut off trajectories with reach probability less than this

    def get_value(state:DarkChessGameState, legal_actions, reward, terminal, reach, trajectory_return, depth):
        legal_actions = np.asarray(legal_actions)
        trajectory_return += reward
        if terminal or (depth_limit > 0 and depth == depth_limit):
            reach_weighted_value = trajectory_return * reach
            return reach_weighted_value
        iset = stringify(game.observation_tensor(state, int(state.current_player)))
        state_pols, found = get_policy_with_default(policy, iset, legal_actions)
        state_value = 0
        assert len(state_pols) == len(legal_actions)
        for ai, a in enumerate(legal_actions):
            if a < 0.5:
                continue
            action_prob = state_pols[ai]
            new_reach = reach * action_prob
            #cut off the low reaches
            if new_reach < eps:
                continue
            new_state, new_legals, reward, terminal = game.apply_action(state, ai)
            state_value += get_value(new_state, new_legals, reward, terminal, new_reach, trajectory_return, depth + 1)
        return state_value
    init_state_value = get_value(init_state, init_legals, 0, False, 1, 0 ,0)
    return init_state_value


def best_responses(policy: JaxPolicy, game: DarkChessGame, depth_limit:int) -> tuple:
    """
    Get the counterfactual best responses for each player against the given policy
    
    Args:
        policy (JaxPolicy): The policy to evaluate.
        game (DarkChessGame): The game instance.
        depth_limit(int): The cutoff after which the game ends with a tie. -1 for full game but those are usually intractable.
    Returns:
        A tuple of best responses for each player and their respective values,
        ordered as (p2_br, p1_br, p2_br_val, p1_br_val), where pi_br
        means that player i is the best responder.
        Since this is a 2p0s game, exploitability is then (p2_br_val + p1_br_val) / 2
    """
    players = 2
    init_state, init_legals = game.new_initial_state()
    starting_player = init_state.current_player
    depth_reaches = [[] for pl in range(players)] # [Pl, D, H(D), A]
    #The all are NOT per player, because
    # in chess players alternate and there 
    # will always be only 1 player playing at the given depth
    depth_iset = [] # [D, H(D)]
    depth_actions = [] # [D, H(D), A]
    #For now could probably not have to do it
    # per depth because we do not use abstractions
    # yet, but it does not matter
    depth_iset_map = [] # [D, I]
    depth_iset_legals = [] # [D, I, A]
    depth_id_to_action = [] # [Pl, D, I] dictionary for each element, do not convert to numpy
    depth_utilites = [] # [D, H(D), A], only for player 1
    depth_continuations = [] # [D, H(D), A]
    depth_behaviour_policies = [] # [D, H(D), A]
    distinct_actions = num_distinct_actions(game)
    full_game_actions = game.action_filter.shape[0]

    def construct_tree(state: DarkChessGameState, legal_actions, reaches, depth):
        legal_actions = np.asarray(legal_actions)
        if len(depth_utilites) <= depth:
            for pl in range(players):
                depth_reaches[pl].append([])
            depth_iset.append([])
            depth_actions.append([])
            depth_iset_map.append({})
            depth_iset_legals.append([])
            depth_id_to_action.append([])
            depth_utilites.append([])
            depth_continuations.append([])
            depth_behaviour_policies.append([])
        current_player = int(state.current_player)
        state_iset = stringify(game.observation_tensor(state, current_player))
        state_pols, found = get_policy_with_default(policy, state_iset, legal_actions)
        behavior_pols = np.zeros(distinct_actions)
        mapped_legals = np.zeros(distinct_actions)
        continuations = np.full(distinct_actions, -1, dtype=np.int32)
        state_reward = np.zeros(distinct_actions)
        depth_continuations[depth].append(continuations)
        depth_utilites[depth].append(state_reward)
        action_to_id = {}
        id_to_action = {}
        for ai, a in enumerate(legal_actions):
            if a < 0.5:
                continue
            if ai not in action_to_id:
                action_idx = len(action_to_id)
                action_to_id[ai] = action_idx
                id_to_action[action_idx] = ai
            ai_mapped = action_to_id[ai]
            behavior_pols[ai_mapped] = state_pols[ai]
            mapped_legals[ai_mapped] = 1
        depth_behaviour_policies[depth].append(behavior_pols)
        for pl in range(players):
          depth_reaches[pl][depth].append(reaches[pl])
        if state_iset not in depth_iset_map[depth]:
            iset_id = len(depth_iset_map[depth])
            depth_iset_map[depth][state_iset] = iset_id
            depth_iset_legals[depth].append(mapped_legals)
            depth_id_to_action[depth].append(id_to_action)
        depth_iset[depth].append(depth_iset_map[depth][state_iset])
        depth_actions[depth].append([i + depth_iset_map[depth][state_iset] * distinct_actions for i in range(distinct_actions)])
        for ai, a in enumerate(legal_actions):
            if a < 0.5:
                continue
            ai_mapped = action_to_id[ai]
            next_history_id = 0 if len(depth_utilites) <= depth+1 else len(depth_utilites[depth + 1])
            next_state, next_legals, next_reward, next_terminal = game.apply_action(state, ai)
            next_reaches = np.where(current_player == np.arange(2), reaches * behavior_pols[ai_mapped], reaches)
            state_reward[ai_mapped] = next_reward
            if next_terminal or (depth_limit > 0 and depth + 1 == depth_limit):
                continue
            continuations[ai_mapped] = next_history_id
            construct_tree(next_state, next_legals, next_reaches, depth + 1)
    construct_tree(init_state, init_legals, np.ones(2), 0)

    def convert_to_numpy(x, dtype=np.int32):
      return [np.asarray(d, dtype=dtype) for d in x]
    def convert_to_numpy_players(x):
      return [[np.asarray(pl) for pl in d] for d in x]
  
    depth_reaches = convert_to_numpy_players(depth_reaches)
    depth_iset = convert_to_numpy(depth_iset)
    depth_actions = convert_to_numpy(depth_actions)
    depth_iset_legals = convert_to_numpy(depth_iset_legals)
    depth_utilites = convert_to_numpy(depth_utilites)
    depth_continuations = convert_to_numpy(depth_continuations)
    depth_behaviour_policies = convert_to_numpy(depth_behaviour_policies, dtype=float)
    p1_br = deepcopy(policy)
    p2_br = deepcopy(policy)

    state_value = np.zeros(1)
    state_br_value = np.zeros(1)
    max_depth = len(depth_iset)
    #player playing at the last decision node
    player = starting_player + ((max_depth - 1) % 2) 
    for d in range(max_depth -1, -1, -1):
        #for when best_responding now
        pl_action_value = np.where(depth_continuations[d] < 0, depth_utilites[d], state_value[depth_continuations[d]]) #[H(D), A]
        #when best_responder was one depth lower and now we are just playing policy
        pl_next_br_action_value = np.where(depth_continuations[d] < 0, depth_utilites[d], state_br_value[depth_continuations[d]])
        pl_cf_action_value = np.where(player == 0, pl_action_value * depth_reaches[1][d][..., None], pl_action_value * depth_reaches[0][d][..., None]) #[H[D], A]
        pl_iset_action_value = np.bincount(depth_actions[d].ravel(), pl_cf_action_value.ravel()).reshape((-1, depth_actions[d].shape[-1])) #[I, A]
        #Assign the lowest value to illegal actions so that they are never picked
        pl_iset_action_value_masked = np.where(depth_iset_legals[d] == 1, pl_iset_action_value, (np.min(pl_iset_action_value) - 1) * (1 - 2 * player)) #[I, A]
        pl_br_action = np.argmax(pl_iset_action_value_masked * (1 - 2 *(player)), axis=-1) #[I]
        acting_policy = p1_br if player == 0 else p2_br
        for i, iset in enumerate(depth_iset_map[d]):
            #need to remap back to full game actions shape
            iset_policy = np.zeros(full_game_actions)
            real_action = depth_id_to_action[d][i][int(pl_br_action[i])]
            iset_policy[real_action] = 1
            acting_policy[iset] = iset_policy
        pl_history_br = pl_br_action[depth_iset[d]] #H(D)
        #breakpoint()
        state_br_value = np.squeeze(np.take_along_axis(pl_action_value, pl_history_br[..., None], axis=-1)) #H{D}
        state_value = np.sum(pl_next_br_action_value * depth_behaviour_policies[d], axis=-1) #H(D)
        player = 1 - player

    if starting_player == 0:
        p1_br_val, p2_br_val = state_br_value.item(), -state_value.item()
    else:
        p1_br_val, p2_br_val = state_value.item(), -state_br_value.item()

    #breakpoint()

    return p2_br, p1_br, p2_br_val, p1_br_val