import numpy as np
import jax.numpy as jnp
from jax_chess import DarkChessGame, Pieces


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

def get_policy_with_default(acting_policy: JaxPolicy, iset, legal_actions:np.ndarray):
  if iset in acting_policy.policy:
    return np.asarray(acting_policy[iset]), True
  print("Not found policy for iset.")
  uniform_pols = legal_actions.astype(float)
  return uniform_pols / np.sum(uniform_pols), False

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
  num_knights = jnp.maximum(jnp.sum(board == Pieces.WHITE_KNIGHT), jnp.sum(board == Pieces.BLACK_KNIGHT)) 
  num_rooks = jnp.maximum(jnp.sum(board == Pieces.WHITE_ROOK), jnp.sum(board == Pieces.BLACK_ROOK))
  num_bishops = jnp.maximum(jnp.sum(board == Pieces.WHITE_BISHOP), jnp.sum(board == Pieces.BLACK_BISHOP))
  num_queens = jnp.maximum(jnp.sum(board == Pieces.WHITE_QUEEN), jnp.sum(board == Pieces.BLACK_QUEEN))
  #Castling is allowed only for 8x8 board
  castling_allowed = int(board.shape[0] == 8 and board.shape[1] == 8)
  #Each pawn has in total 8 total possible moves
  #1 forward, 2 forward, and 2 diagonal,
  # plus the promotion moves. Each promotion move
  # has 4 options
  pawn_actions = 8 * num_pawns
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
