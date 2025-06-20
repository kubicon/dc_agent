import numpy as np
import argparse

from cfr.jax_chess_cfr import JaxDarkChessCFR
from cfr.utils import stringify, get_policy_with_default, JaxPolicy
from jax_chess import DarkChessGame
from jax_chess_algorithms import expected_value, best_responses

from pyinstrument import Profiler

parser = argparse.ArgumentParser()
parser.add_argument("--depth_limit", type=int, default=4, help="Maximum CFR tree depth after which it is cut off. The number should be even because there are 2 players.")

def get_cfr_policy(fen:str, depth_limit:int, solve_iterations:int = 3000) ->tuple[JaxPolicy, JaxDarkChessCFR]:
  print("Building CFR tree...")
  cfr = JaxDarkChessCFR(fen=fen, depth_limit=depth_limit)
  print("Building tree done, performing regret updates...")
  cfr.multiple_steps(iterations=solve_iterations)
  print("Extracting policy...")
  acting_policy=cfr.average_policy()
  return acting_policy, cfr

def run_cfr(fen:str, depth_limit:int, steps:int =-1, solve_iterations:int =3000):
  profiler = Profiler()
  profiler.start()
  acting_policy, cfr = get_cfr_policy(fen, depth_limit, solve_iterations)
  profiler.stop()
  print(profiler.output_text(color=True, unicode=True))
  print("Playing game...")
  cur_steps = 0
  game = DarkChessGame(init_fen=fen)
  state, legals = game.new_initial_state()
  terminal = False
  while not terminal:
    legals = np.asarray(legals).astype(bool)
    print("State: ", state)
    if steps > 0 and cur_steps > steps:
      break
    iset = stringify(game.observation_tensor(state, int(state.current_player)))
    iset_policy, found = get_policy_with_default(acting_policy, iset, legals)
    print("Legal policy: ", iset_policy[legals])
    sampled_action = np.random.choice(len(iset_policy), p=iset_policy)
    print("Sampled action: ", sampled_action)
    state, legals, reward, terminal = game.apply_action(state, sampled_action)
    print("Obtained reward: ", reward)
    cur_steps += 1

def cfr_value(fen:str, depth_limit:int,solve_iterations:int =3000):
  acting_policy, cfr = get_cfr_policy(fen, depth_limit, solve_iterations)
  p1_val = expected_value(acting_policy, cfr.game, cfr.depth_limit)
  print("Expected value for p1: ", p1_val)

def cfr_exploitability(fen:str, depth_limit: int, solve_iterations:int = 3000): 
  acting_policy, cfr = get_cfr_policy(fen, depth_limit, solve_iterations)
  p2_br, p1_br, p2_br_val, p1_br_val = best_responses(acting_policy, cfr.game, cfr.depth_limit)
  print("Best response value against p1 policy: ", p2_br_val)
  print("Best response value against p2 policy: ", p1_br_val)



def two_knights_each_test(args):
  """A 4x4 board where each player has
  one king in the corner, surrounded by knights,
  such thath they dont see each other at the start"""
  two_knights_fen = "2nk/3n/N3/KN2 w - - 0 1"
  #run_cfr(two_knights_fen, args.depth_limit, steps=args.depth_limit - 1)
  #cfr_value(two_knights_fen, args.depth_limit)
  cfr_exploitability(two_knights_fen, args.depth_limit)

def early_win_test(args):
  """Test where white has a distinct advantage,
  with two bishops, one queen and black has only a king.
  A rationally playing white should always be able to win in the second move"""

  early_win_fen = "3k/4/B3/KBQ1 w - - 0 1"
  #run_cfr(early_win_fen, args.depth_limit, steps=args.depth_limit - 1)
  #cfr_value(early_win_fen, args.depth_limit)
  cfr_exploitability(early_win_fen, args.depth_limit)

def immediate_win_test(args):
  """Just a sanity check board where white 
  should have only one legal move and that wins him the game."""
  white_win_fen = "3p/3P/2kP/2PK w - - 0 1"
  #run_cfr(white_win_fen, args.depth_limit, steps=args.depth_limit - 1)
  #cfr_value(white_win_fen, args.depth_limit)
  cfr_exploitability(white_win_fen, args.depth_limit)

def main():
  args = parser.parse_args()
  #two_knights_each_test(args)
  early_win_test(args)
  #immediate_win_test(args)


if __name__ == "__main__":
  main()