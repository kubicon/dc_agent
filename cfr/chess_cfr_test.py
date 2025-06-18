import numpy as np
import argparse

from cfr.jax_chess_cfr import JaxDarkChessCFR
from cfr.utils import JaxPolicy, stringify
from jax_chess import DarkChessGame, DarkChessGameState

from pyinstrument import Profiler

parser = argparse.ArgumentParser()
parser.add_argument("--depth_limit", type=int, default=4, help="Maximum CFR tree depth after which it is cut off. The number should be even because there are 2 players.")

def get_policy_with_default(acting_policy: JaxPolicy, iset, legal_actions:np.ndarray):
  if iset in acting_policy.policy:
    return acting_policy[iset]
  print("Not found policy for iset.")
  uniform_pols = legal_actions.astype(float)
  return uniform_pols / np.sum(uniform_pols)

def run_cfr(fen:str, depth_limit, steps:int =-1, solve_iterations:int =3000):
  profiler = Profiler()
  profiler.start()
  print("Building CFR tree...")
  cfr = JaxDarkChessCFR(fen=fen, depth_limit=depth_limit)
  print("Building tree done, performing regret updates...")
  cfr.multiple_steps(iterations=solve_iterations)
  print("Extracting policy...")
  acting_policy=cfr.average_policy()
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
    iset_policy = get_policy_with_default(acting_policy, iset, legals)
    #print(legals)
    print("Legal policy: ", iset_policy[legals])
    sampled_action = np.random.choice(len(iset_policy), p=iset_policy)
    print("Sampled action: ", sampled_action)
    state, legals, reward, terminal = game.apply_action(state, sampled_action)
    print("Obtained reward: ", reward)
    cur_steps += 1



def two_knights_each_test(args):
  """A 4x4 board where each player has
  one king in the corner, surrounded by knights,
  such thath they dont see each other at the start"""
  two_knights_fen = "2nk/3n/N3/KN2 w - - 0 1"
  run_cfr(two_knights_fen, args.depth_limit, steps=args.depth_limit - 1)

def main():
  args = parser.parse_args()
  two_knights_each_test(args)


if __name__ == "__main__":
  main()