import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, random
from typing import Tuple, NamedTuple
import chex

from functools import partial
from enum import IntEnum

class Pieces(IntEnum):
  EMPTY = 0
  WHITE_PAWN = 1
  WHITE_ROOK = 2
  WHITE_KNIGHT = 3
  WHITE_BISHOP = 4
  WHITE_QUEEN = 5
  WHITE_KING = 6
  BLACK_PAWN = 7
  BLACK_ROOK = 8
  BLACK_KNIGHT = 9
  BLACK_BISHOP = 10
  BLACK_QUEEN = 11
  BLACK_KING = 12

BOARD = chex.Array


PIECE_TO_ID = {
  "P": 1,
  "R": 2,
  "N": 3,
  "B": 4,
  "Q": 5,
  "K": 6,
  "p": 7,
  "r": 8,
  "n": 9,
  "b": 10,
  "q": 11,
  "k": 12
}

ID_TO_PIECE = {
  1: "P",
  2: "R",
  3: "N",
  4: "B",
  5: "Q",
  6: "K",
  7: "p",
  8: "r",
  9: "n",
  10: "b",
  11: "q",
  12: "k"
}


DISTINCT_PIECES_PER_PLAYER = 6
PROMOTION_PIECES = [Pieces.WHITE_ROOK, Pieces.WHITE_KNIGHT, Pieces.WHITE_BISHOP, Pieces.WHITE_QUEEN]

KNIGHT_DIRS = [[-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1]]
KING_DIRS = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]

@chex.dataclass(frozen=True)
class DarkChessGameState:
  """Complete chess game state that can be JIT compiled."""
  board: BOARD # Board with
  captured_pieces: chex.Array # The same order as Pieces enum
  castling_rights: chex.Array # [white_kingside, white_queenside, black_kingside, black_queenside]
  current_player: int  # 0 for white, 1 for black
  en_passant_target: int  # target square for en passant (64 for none)
  halfmove_clock: int  # for 50-move rule
  fullmove_number: int  # move counter


# TODO: There are many non-sensical actions. Create a filter mask that removes them.
class DarkChessGame:
  """JAX JIT-compatible Dark Chess Game implementation."""
  
  # Direction vectors for piece movement  
  

  def __init__(self, init_fen: str):
    """Initialize a new chess game."""
    self.init_fen = init_fen
    self.init_state = translate_fen(init_fen)
    self.board_shape = self.init_state.board.shape
    self.board_height, self.board_width = self.board_shape
    self.longest_move_length = max(self.board_height, self.board_width) - 1
    assert self.board_height >= 4, "We do not assume smaller boards than with height 4"
     
    self.check_castling = True if self.board_width == 8 and self.board_height == 8 else False
    self.KNIGHT_DIRS = jnp.array(KNIGHT_DIRS)
    self.KING_DIRS = jnp.array(KING_DIRS)
  
    
    self.action_filter, self.action_from, self.action_to, self.action_promotion = self.prepare_filter(self.init_state)
    self.castling_action_ids_from =  self.action_from.shape[0] if self.check_castling else self.action_from.shape[0] - 2
    self.avoid_check = False
    self.use_filter_check = False
  
    assert self.avoid_check == False, "The engine does not handle making illegal actiosn that do not avoid check on king (or those that force you into check)"
   
   
  def new_initial_state(self) -> DarkChessGameState:
    """Create initial game state."""
    state = translate_fen(self.init_fen)
    legal_actions = self.get_legal_actions(state)
    return state, legal_actions
  
  def apply_filter_with_check(self, legal_actions: chex.Array) -> chex.Array:
    filtered_legal_actions = legal_actions[self.action_filter]
    if self.use_filter_check:
      assert jnp.all(jnp.sum(filtered_legal_actions, axis=0) == jnp.sum(legal_actions, axis=0))
    return filtered_legal_actions
  
  def prepare_filter(self, game_state: DarkChessGameState) -> chex.Array:
    legal_actions, from_move_positions, after_move_positions, in_bounds, promotions = self.get_legals_with_details(game_state)
    action_filter = jnp.cumsum(in_bounds)
    action_filter = jnp.where(in_bounds, action_filter, -1)
    action_filter = jnp.where(action_filter >= 0)[0]
    return action_filter, from_move_positions[action_filter], after_move_positions[action_filter], promotions[action_filter]
  
  @partial(jit, static_argnums=(0,))
  def get_legal_actions(self, game_state: DarkChessGameState) -> chex.Array:
    legal_actions_pawns, _, _, _, _ = self.get_legal_actions_pawns(game_state)
    legal_actions_higher_pieces, _, _, _ = self.get_legal_actions_higher_pieces(game_state)
    if self.check_castling:
      legal_actions_castling = self.get_legal_castling(game_state)
    else:
      legal_actions_castling = jnp.zeros((0,), dtype=jnp.int32)
    
    legal_actions = jnp.concatenate([
      legal_actions_pawns,
      legal_actions_higher_pieces,
      legal_actions_castling
    ], axis=0)
    
    legal_actions = self.apply_filter_with_check(legal_actions)
    return legal_actions
  
  @partial(jit, static_argnums=(0,))
  def get_legals_with_details(self, game_state: DarkChessGameState) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]: 
    """Get all legal moves for current player. Returns array of shape (N, 2) where N is max possible moves."""
    legal_actions_pawns, from_move_positions_pawns, after_move_positions_pawns, in_bounds_pawns, promotions_pawns = self.get_legal_actions_pawns(game_state)
    legal_actions_higher_pieces, from_move_positions_higher_pieces, after_move_positions_higher_pieces, in_bounds_higher_pieces = self.get_legal_actions_higher_pieces(game_state)
    if self.check_castling:
      legal_actions_castling = self.get_legal_castling(game_state)
      from_move_positions_castling = jnp.array([[7, 4], [7, 4], [0, 4], [0, 4]], dtype=jnp.int32) # TODO: Change this to a king
      after_move_positions_castling = jnp.array([[7, 6], [7, 2], [0, 6], [0, 2]], dtype=jnp.int32)
      in_bounds_castling = jnp.ones((4,), dtype=bool)
      promotions_castling = jnp.zeros((4,), dtype=jnp.int32)
    else:
      legal_actions_castling = jnp.zeros((0, ), dtype=jnp.int32)
      from_move_positions_castling = jnp.zeros((0, 2), dtype=jnp.int32)
      after_move_positions_castling = jnp.zeros((0, 2), dtype=jnp.int32)
      in_bounds_castling = jnp.zeros((0,), dtype=bool)
      promotions_castling = jnp.zeros((0,), dtype=jnp.int32)
      
    higher_pieces_promotions = jnp.zeros_like(legal_actions_higher_pieces, dtype=jnp.int32) 
    
    legal_actions = jnp.concatenate([
      legal_actions_pawns,
      legal_actions_higher_pieces,
      legal_actions_castling
    ], axis=0)
    
    from_move_positions = jnp.concatenate([
      from_move_positions_pawns,
      from_move_positions_higher_pieces,
      from_move_positions_castling
    ], axis=0)
    
    after_move_positions = jnp.concatenate([
      after_move_positions_pawns,
      after_move_positions_higher_pieces,
      after_move_positions_castling
    ], axis=0)
    
    in_bounds = jnp.concatenate([
      in_bounds_pawns,
      in_bounds_higher_pieces,
      in_bounds_castling
    ], axis=0)
    
    promotions = jnp.concatenate([
      promotions_pawns,
      higher_pieces_promotions,
      promotions_castling
    ], axis=0)
    
    return legal_actions, from_move_positions, after_move_positions, in_bounds, promotions
      
  @partial(jit, static_argnums=(0,))
  def is_tile_attacked(self, game_state: DarkChessGameState, tile: chex.Array, opponent_color: int) -> bool:
    """Check if a tile is attacked by the opponent."""
    
    def scan_possible_move_wrapper(square_pieces: chex.Array, in_bounds: chex.Array, tested_piece: int) -> bool:
      @chex.dataclass(frozen=True)
      class IsAttackedCarry:
        is_attacked: bool
        occluded: bool
      
      def scan_possible_move(carry: IsAttackedCarry, x):
        square_piece, in_bounds = x
        is_attacked = (self.get_piece_color(square_piece) == opponent_color) & ((self.get_piece_type(square_piece)) == tested_piece) & in_bounds & jnp.logical_not(carry.occluded)
        
        new_occluded = carry.occluded | (square_piece != Pieces.EMPTY)
        new_carry = IsAttackedCarry(is_attacked=is_attacked | carry.is_attacked, occluded=new_occluded)
        return new_carry, is_attacked
      
      _, is_attacked = lax.scan(scan_possible_move, IsAttackedCarry(is_attacked=False, occluded=False), (square_pieces, in_bounds))
      return is_attacked
     
    def rook_attacks(tile: chex.Array) -> bool:
      row, col = self.square_to_coords(tile)
      all_moves = jnp.arange(1, self.longest_move_length + 1) 
      left_cols = col - all_moves
      right_cols = col + all_moves
      up_rows = row - all_moves
      down_rows = row + all_moves
      
      left_squares = game_state.board[row, left_cols]
      right_squares = game_state.board[row, right_cols]
      up_squares = game_state.board[up_rows, col]
      down_squares = game_state.board[down_rows, col]
      
      square_pieces = jnp.stack([left_squares, right_squares, up_squares, down_squares], axis=0)
      in_bounds = jnp.stack([left_cols >= 0, right_cols < self.board_width, up_rows >= 0, down_rows < self.board_height], axis=0)
      is_attacked = jnp.any(jax.vmap(scan_possible_move_wrapper, in_axes=(0, 0, None))(square_pieces, in_bounds, Pieces.WHITE_ROOK))
      return is_attacked
    
    def bishop_attacks(tile: chex.Array) -> bool:
      row, col = self.square_to_coords(tile)
      all_moves = jnp.arange(1, self.longest_move_length + 1) 
      left_cols = col - all_moves
      right_cols = col + all_moves
      up_rows = row - all_moves
      down_rows = row + all_moves
      
      left_upper_diagonal = game_state.board[up_rows, left_cols]
      right_upper_diagonal = game_state.board[up_rows, right_cols]
      left_down_diagonal = game_state.board[down_rows, left_cols]
      right_down_diagonal = game_state.board[down_rows, right_cols]
      
      square_pieces = jnp.stack([left_upper_diagonal, right_upper_diagonal, left_down_diagonal, right_down_diagonal], axis=0)
      in_bounds = jnp.stack([ 
        (left_cols >= 0) & (up_rows >= 0),
        (right_cols < self.board_width) & (up_rows >= 0),
        (left_cols >= 0) & (down_rows < self.board_height),
        (right_cols < self.board_width) & (down_rows < self.board_height)
      ], axis=0)
      
      is_attacked = jnp.any(jax.vmap(scan_possible_move_wrapper, in_axes=(0, 0, None))(square_pieces, in_bounds, Pieces.WHITE_BISHOP))
      return is_attacked
    
    def queen_attacks(tile: chex.Array) -> bool:
      row, col = self.square_to_coords(tile)
      all_moves = jnp.arange(1, self.longest_move_length + 1) 
      left_cols = col - all_moves
      right_cols = col + all_moves
      up_rows = row - all_moves
      down_rows = row + all_moves
      
      left_squares = game_state.board[row, left_cols]
      right_squares = game_state.board[row, right_cols]
      up_squares = game_state.board[up_rows, col]
      down_squares = game_state.board[down_rows, col]
      left_upper_diagonal = game_state.board[up_rows, left_cols]
      right_upper_diagonal = game_state.board[up_rows, right_cols]
      left_down_diagonal = game_state.board[down_rows, left_cols]
      right_down_diagonal = game_state.board[down_rows, right_cols]
      
      square_pieces = jnp.stack([left_squares, right_squares, up_squares, down_squares, left_upper_diagonal, right_upper_diagonal, left_down_diagonal, right_down_diagonal], axis=0)
      in_bounds = jnp.stack([ 
        left_cols >= 0, 
        right_cols < self.board_width,
        up_rows >= 0, 
        down_rows < self.board_height,
        (left_cols >= 0) & (up_rows >= 0),
        (right_cols < self.board_width) & (up_rows >= 0),
        (left_cols >= 0) & (down_rows < self.board_height),
        (right_cols < self.board_width) & (down_rows < self.board_height)
      ], axis=0)
      is_attacked = jnp.any(jax.vmap(scan_possible_move_wrapper, in_axes=(0, 0, None))(square_pieces, in_bounds, Pieces.WHITE_QUEEN))
      return is_attacked
     
    def given_piece_attacks(tile: chex.Array, moves: chex.Array, piece: int) -> bool: 
      pos = jnp.array(self.square_to_coords(tile))
      after_move_position = pos + moves
      square_piece = game_state.board[after_move_position[0], after_move_position[1]]
      
      in_bounds = jnp.all(after_move_position >= 0, axis=-1) & jnp.all(after_move_position < jnp.array(self.board_shape), axis=-1)
      is_attacked = (self.get_piece_color(square_piece) == opponent_color) & (self.get_piece_type(square_piece)  == piece) & in_bounds
      return is_attacked
    
    def knight_attacks(tile: chex.Array) -> bool:
      return jnp.any(jax.vmap(given_piece_attacks, in_axes=(None, 0, None))(tile, self.KNIGHT_DIRS, Pieces.WHITE_KNIGHT))
    
    def king_attacks(tile: chex.Array) -> bool:
      return jnp.any(jax.vmap(given_piece_attacks, in_axes=(None, 0, None))(tile, self.KING_DIRS, Pieces.WHITE_KING))
    
    def pawn_attacks(tile: chex.Array) -> bool:
      pawn_dangers = jnp.where(game_state.current_player == 0, jnp.array([[-1, -1], [-1, 1]]), jnp.array([[1, -1], [1, 1]]))
      return jnp.any(jax.vmap(given_piece_attacks, in_axes=(None, 0, None))(tile, pawn_dangers, Pieces.WHITE_PAWN))
    
    
    return rook_attacks(tile) | bishop_attacks(tile) | queen_attacks(tile) | knight_attacks(tile) | king_attacks(tile) | pawn_attacks(tile) 
     
  def get_legal_castling(self, game_state: DarkChessGameState) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Get legal castling actions. Only implemented for 8x8"""
    is_king_castling_allowed = jnp.where(game_state.current_player == 0, game_state.castling_rights[0], game_state.castling_rights[2])
    is_queen_castling_allowed = jnp.where(game_state.current_player == 0, game_state.castling_rights[1], game_state.castling_rights[3])
    
    white_kingside_row = jnp.repeat(7, 3)
    white_queenside_row = jnp.repeat(7, 4) 
    black_kingside_row = jnp.repeat(0, 3)
    black_queenside_row = jnp.repeat(0, 4)
    
    kingside_cols = jnp.array([4, 5, 6])
    queenside_cols = jnp.array([1, 2, 3, 4])
    
    white_kingside_tiles = white_kingside_row * self.board_width + kingside_cols
    white_queenside_tiles = white_queenside_row * self.board_width + queenside_cols
    black_kingside_tiles = black_kingside_row * self.board_width + kingside_cols
    black_queenside_tiles = black_queenside_row * self.board_width + queenside_cols
    
    white_kingside_empty = jnp.all(game_state.board[white_kingside_row[1:], kingside_cols[1:]] == Pieces.EMPTY)
    white_queenside_empty = jnp.all(game_state.board[white_queenside_row[:-1], queenside_cols[:-1]] == Pieces.EMPTY)
    black_kingside_empty = jnp.all(game_state.board[black_kingside_row[1:], kingside_cols[1:]] == Pieces.EMPTY)
    black_queenside_empty = jnp.all(game_state.board[black_queenside_row[:-1], queenside_cols[:-1]] == Pieces.EMPTY)
    
    
    white_kingside_attacked = jax.vmap(self.is_tile_attacked, in_axes=(None, 0, None))(game_state, white_kingside_tiles, 1)
    white_queenside_attacked = jax.vmap(self.is_tile_attacked, in_axes=(None, 0, None))(game_state, white_queenside_tiles[1:], 1)
    black_kingside_attacked = jax.vmap(self.is_tile_attacked, in_axes=(None, 0, None))(game_state, black_kingside_tiles, 0)
    black_queenside_attacked = jax.vmap(self.is_tile_attacked, in_axes=(None, 0, None))(game_state, black_queenside_tiles[1:], 0)
    
    white_kingside_legal = jnp.all(white_kingside_empty & (~white_kingside_attacked) & is_king_castling_allowed) & (game_state.current_player == 0)
    white_queenside_legal = jnp.all(white_queenside_empty & (~white_queenside_attacked) & is_queen_castling_allowed) & (game_state.current_player == 0) 
    black_kingside_legal = jnp.all(black_kingside_empty & (~black_kingside_attacked) & is_king_castling_allowed) & (game_state.current_player == 1) 
    black_queenside_legal = jnp.all(black_queenside_empty & (~black_queenside_attacked) & is_queen_castling_allowed) & (game_state.current_player == 1)
    
    return jnp.stack([white_kingside_legal, white_queenside_legal, black_kingside_legal, black_queenside_legal], axis=0)
  
  def get_legal_actions_higher_pieces(self, game_state: DarkChessGameState) -> chex.Array:
    # This takes into account the current, player, so it may sometime give weird results.
    def scan_possible_movements(square_piece: chex.Array, in_bounds: chex.Array) -> chex.Array:
      
      @chex.dataclass(frozen=True)
      class PossibleMoveCarry:
        legal: bool
        
      def scan_possible_move(carry: PossibleMoveCarry, x):
        square_piece, in_bounds = x
        
        new_legal = carry.legal & in_bounds & (self.get_piece_color(square_piece) != game_state.current_player)
        new_carry = PossibleMoveCarry(legal=new_legal & (self.get_piece_type(square_piece) == 0))
        return new_carry, new_legal
      
      _, legal = lax.scan(scan_possible_move, PossibleMoveCarry(legal=True), (square_piece, in_bounds))
      return legal
    
    def check_legal_rooks(square: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
      row, col = self.square_to_coords(square)
      all_moves = jnp.arange(1, self.longest_move_length + 1) 
      left_cols = col - all_moves
      right_cols = col + all_moves
      up_rows = row - all_moves
      down_rows = row + all_moves
      left_squares = game_state.board[row, left_cols]
      right_squares = game_state.board[row, right_cols]
      up_squares = game_state.board[up_rows, col]
      down_squares = game_state.board[down_rows, col]
      
      square_piece = jnp.stack([left_squares, right_squares, up_squares, down_squares], axis=0)
      after_move_positions = jnp.stack([
        jnp.stack((jnp.repeat(row, self.longest_move_length), left_cols), axis=-1),
        jnp.stack((jnp.repeat(row, self.longest_move_length), right_cols), axis=-1),
        jnp.stack((up_rows, jnp.repeat(col, self.longest_move_length)), axis=-1),
        jnp.stack((down_rows, jnp.repeat(col, self.longest_move_length)), axis=-1)
      ], axis=0)
      in_bounds = jnp.stack([left_cols >= 0, right_cols < self.board_width, up_rows >= 0, down_rows < self.board_height], axis=0)
      legal = jax.vmap(scan_possible_movements, in_axes=(0, 0))(square_piece, in_bounds)
      return legal, after_move_positions, in_bounds
    
    def check_legal_bishop(square: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
      row, col = self.square_to_coords(square)
      all_moves = jnp.arange(1, self.longest_move_length + 1) 
      left_cols = col - all_moves
      right_cols = col + all_moves
      up_rows = row - all_moves
      down_rows = row + all_moves
      left_upper_diagonal = game_state.board[up_rows, left_cols]
      right_upper_diagonal = game_state.board[up_rows, right_cols]
      left_down_diagonal = game_state.board[down_rows, left_cols]
      right_down_diagonal = game_state.board[down_rows, right_cols]
      
      square_piece = jnp.stack([left_upper_diagonal, right_upper_diagonal, left_down_diagonal, right_down_diagonal], axis=0)
      
      after_move_positions = jnp.stack([
        jnp.stack((up_rows, left_cols), axis=-1),
        jnp.stack((up_rows, right_cols), axis=-1),
        jnp.stack((down_rows, left_cols), axis=-1),
        jnp.stack((down_rows, right_cols), axis=-1)
      ], axis=0)
      
      in_bounds = jnp.stack([ 
        (left_cols >= 0) & (up_rows >= 0),
        (right_cols < self.board_width) & (up_rows >= 0),
        (left_cols >= 0) & (down_rows < self.board_height),
        (right_cols < self.board_width) & (down_rows < self.board_height)
      ], axis=0)
      
      legal = jax.vmap(scan_possible_movements, in_axes=(0, 0))(square_piece, in_bounds)
      return legal, after_move_positions, in_bounds
    
    def get_positions_and_squares(square: chex.Array, moves: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
      pos = jnp.array(self.square_to_coords(square))
      after_move_position = pos + moves
      squares = game_state.board[after_move_position[:, 0], after_move_position[:, 1]]
      in_bounds = jnp.all(after_move_position >= 0, axis=1) & jnp.all(after_move_position < jnp.array(self.board_shape), axis=1)
      return after_move_position, squares, in_bounds

    def check_legal_given_moves(square: chex.Array, moves: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
      after_move_position, squares, in_bounds = get_positions_and_squares(square, moves)
      legal = in_bounds * (jax.vmap(self.get_piece_color, in_axes=0)(squares) != game_state.current_player)
      return legal, after_move_position, in_bounds
    
    def check_legal_knight(square: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
      return check_legal_given_moves(square, self.KNIGHT_DIRS)
    
    def check_legal_king(square: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
      return check_legal_given_moves(square, self.KING_DIRS)
    
    def check_legal_higher_pieces(square: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
      
      legal_rooks, after_move_position_rooks, in_bounds_rooks = check_legal_rooks(square) 
      legal_bishop, after_move_position_bishop, in_bounds_bishop = check_legal_bishop(square)
      legal_queen = jnp.concatenate([legal_rooks, legal_bishop], axis=0)
      after_move_position_queen = jnp.concatenate([after_move_position_rooks, after_move_position_bishop], axis=0)
      in_bounds_queen = jnp.concatenate([in_bounds_rooks, in_bounds_bishop], axis=0)
      
      legal_knight, after_move_position_knight, in_bounds_knight = check_legal_knight(square)
      legal_king, after_move_position_king, in_bounds_king = check_legal_king(square)  
      
      coords = self.square_to_coords(square)
      
      return (jnp.concatenate([
        legal_rooks.flatten() * (game_state.board[coords] == Pieces.WHITE_ROOK + game_state.current_player * DISTINCT_PIECES_PER_PLAYER),
        legal_knight.flatten() * (game_state.board[coords] == Pieces.WHITE_KNIGHT + game_state.current_player * DISTINCT_PIECES_PER_PLAYER),
        legal_bishop.flatten() * (game_state.board[coords] == Pieces.WHITE_BISHOP + game_state.current_player * DISTINCT_PIECES_PER_PLAYER),
        legal_queen.flatten() * (game_state.board[coords] == Pieces.WHITE_QUEEN + game_state.current_player * DISTINCT_PIECES_PER_PLAYER),
        legal_king.flatten() * (game_state.board[coords] == Pieces.WHITE_KING + game_state.current_player * DISTINCT_PIECES_PER_PLAYER), 
      ], axis=0), 
      jnp.concatenate([
        after_move_position_rooks.reshape(-1, 2),
        after_move_position_knight.reshape(-1, 2),
        after_move_position_bishop.reshape(-1, 2),
        after_move_position_queen.reshape(-1, 2),
        after_move_position_king.reshape(-1, 2),
      ], axis=0), 
      jnp.concatenate([
        in_bounds_rooks.flatten(),
        in_bounds_knight.flatten(),
        in_bounds_bishop.flatten(),
        in_bounds_queen.flatten(),
        in_bounds_king.flatten(),
      ], axis=0)
    
      )
    
    tiles = jnp.arange(self.board_height * self.board_width) 
    legal_actions, after_move_positions, in_bounds = jax.vmap(check_legal_higher_pieces, in_axes=(0,))(tiles)
    
    from_moves = jnp.repeat(jnp.arange(self.board_height * self.board_width)[..., jnp.newaxis], legal_actions.shape[1], axis=1)
    from_moves_positions = jnp.stack(jax.vmap(self.square_to_coords, in_axes=(0,))(from_moves), axis=-1)
    
    return legal_actions.flatten(), from_moves_positions.reshape(-1, 2), after_move_positions.reshape(-1, 2), in_bounds.flatten()
  
  def get_legal_actions_pawns(self, game_state: DarkChessGameState) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
  
    def get_positions_and_squares(square: chex.Array, moves: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
      pos = jnp.array(self.square_to_coords(square))
      after_move_position = pos + moves
      squares = game_state.board[after_move_position[:, 0], after_move_position[:, 1]]
      in_bounds = jnp.all(after_move_position >= 0, axis=1) & jnp.all(after_move_position < jnp.array(self.board_shape), axis=1)
      return after_move_position, squares, in_bounds
    
    def check_legal_with_capture(square: chex.Array, moves: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
      after_move_position, squares, in_bounds = get_positions_and_squares(square, moves)
      legal = in_bounds * (jax.vmap(self.get_piece_color, in_axes=0)(squares) == (1 - game_state.current_player))
      return legal, after_move_position, in_bounds
    
    def check_legal_without_capture(square: chex.Array, moves: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
      after_move_position, squares, in_bounds = get_positions_and_squares(square, moves)
      legal = in_bounds * (jax.vmap(self.get_piece_type, in_axes=0)(squares) == 0)
      return legal, after_move_position, in_bounds
    
    # We split white and black pawns
    def check_legal_pawn_move(square: chex.Array, player: int) -> tuple[chex.Array, chex.Array, chex.Array]:
      
      # White goes up, which is one row up in the matrix, that is we go to the previous row
      move_for_color = jnp.where(player == 0, -1, 1)
      move = jnp.array([[move_for_color, 0]])
      return check_legal_without_capture(square, move)
    
    # We could do this only for subset of tiles, but I guess it is fast to do it this way.
    def check_legal_pawn_double_move(square: chex.Array, player: int) -> tuple[chex.Array, chex.Array, chex.Array]:
      move_for_color = jnp.where(player == 0, -1, 1)
      move_1 = jnp.array([[move_for_color, 0]])
      move_2 = jnp.array([[2 * move_for_color, 0]])
      
      # You need to be able to do the first move and also the second one
      legal_1, _, in_bounds_1 = check_legal_without_capture(square, move_1)
      legal_2, after_move_position_2, in_bounds_2 = check_legal_without_capture(square, move_2)
      
      return legal_1 & legal_2, after_move_position_2, in_bounds_1 & in_bounds_2
    
  # TODO: Try to make en_passant works 
    def check_legal_pawn_attack(square: chex.Array, player: int) -> tuple[chex.Array, chex.Array, chex.Array]:
      move_for_color = jnp.where(player == 0, -1, 1)
      attack_moves = jnp.array([[move_for_color, -1], [move_for_color, 1]])
      
      return check_legal_with_capture(square, attack_moves)
     

    def check_legal_pawns(square: chex.Array, player: int) -> chex.Array:
      legal_pawn_move, after_move_position_pawn_move, in_bounds_pawn_move = check_legal_pawn_move(square, player)
      legal_pawn_attack, after_move_position_pawn_attack, in_bounds_pawn_attack = check_legal_pawn_attack(square, player)
      coords = self.square_to_coords(square)
      
      return (jnp.concatenate([
        legal_pawn_move.flatten() * (game_state.board[coords] == Pieces.WHITE_PAWN + game_state.current_player * DISTINCT_PIECES_PER_PLAYER),
        legal_pawn_attack.flatten() * (game_state.board[coords] == Pieces.WHITE_PAWN + game_state.current_player * DISTINCT_PIECES_PER_PLAYER)
      ], axis=0),
      jnp.concatenate([
        after_move_position_pawn_move.reshape(-1, 2),
        after_move_position_pawn_attack.reshape(-1, 2),
      ], axis=0),
      jnp.concatenate([
        in_bounds_pawn_move.flatten(),
        in_bounds_pawn_attack.flatten(),
      ], axis=0)
      )
      
    def check_legal_pawns_double_move(square: chex.Array, player: int) -> chex.Array:
      # The white pawns cannot be at the first row (that is black's side) and when they are just 1 row below it, we handle it separately in check_legal_pawns_promote
      pawn_row = jnp.where(player == 0, self.board_height - 2, 1)
      square = pawn_row * self.board_width + square
      coords = self.square_to_coords(square)
      
      legal_pawn_double_move, after_move_position_pawn_double_move, in_bounds_pawn_double_move = check_legal_pawn_double_move(square, player)
      legal_pawn_double_move = legal_pawn_double_move * (game_state.board[coords] == Pieces.WHITE_PAWN + game_state.current_player * DISTINCT_PIECES_PER_PLAYER)
      
      return legal_pawn_double_move, after_move_position_pawn_double_move, in_bounds_pawn_double_move
    
    def check_legal_pawns_promote(square: chex.Array, player: int) -> chex.Array:
      # The input is always in [0, 7] so we convert it to the actual last row where the pieces moves
      pawn_row = jnp.where(player == 0, 1, self.board_height - 2)
      square = pawn_row * self.board_width + square
      
      legal_pawn_move, after_move_position_pawn_move, in_bounds_pawn_move = check_legal_pawn_move(square, player)
      legal_pawn_attack, after_move_position_pawn_attack, in_bounds_pawn_attack = check_legal_pawn_attack(square, player)
      coords = self.square_to_coords(square)
      
      legal_pawn_move_with_promotion = jnp.repeat(legal_pawn_move, len(PROMOTION_PIECES), axis=0) * (game_state.board[coords] == Pieces.WHITE_PAWN + game_state.current_player * DISTINCT_PIECES_PER_PLAYER)
      legal_pawn_attack_with_promotion = jnp.repeat(legal_pawn_attack, len(PROMOTION_PIECES), axis=0) * (game_state.board[coords] == Pieces.WHITE_PAWN + game_state.current_player * DISTINCT_PIECES_PER_PLAYER)
      
      after_move_pawn_move_with_promotion = jnp.repeat(after_move_position_pawn_move, len(PROMOTION_PIECES), axis=0)
      after_move_pawn_attack_with_promotion = jnp.repeat(after_move_position_pawn_attack, len(PROMOTION_PIECES), axis=0)
      
      in_bounds_pawn_move_with_promotion = jnp.repeat(in_bounds_pawn_move, len(PROMOTION_PIECES), axis=0)
      in_bounds_pawn_attack_with_promotion = jnp.repeat(in_bounds_pawn_attack, len(PROMOTION_PIECES), axis=0)
      
      
      
      legal_pawn_move_with_promotion = jnp.concatenate([legal_pawn_move_with_promotion.flatten(), legal_pawn_attack_with_promotion.flatten()], axis=0)
      after_move_pawn_move_with_promotion = jnp.concatenate([after_move_pawn_move_with_promotion.reshape(-1, 2), after_move_pawn_attack_with_promotion.reshape(-1, 2)], axis=0)
      in_bounds_pawn_move_with_promotion = jnp.concatenate([in_bounds_pawn_move_with_promotion.flatten(), in_bounds_pawn_attack_with_promotion.flatten()], axis=0)
      
      return (legal_pawn_move_with_promotion,
      after_move_pawn_move_with_promotion,
      in_bounds_pawn_move_with_promotion)
    
    
    white_pawn_tiles = jnp.arange(2 * self.board_width, self.board_width * (self.board_height - 2) + 2 * self.board_width)
    black_pawn_tiles = jnp.arange((self.board_height  - 2) * self.board_width)
    
    white_pawn_legal_moves, after_move_white_pawns, in_bounds_white_pawns = jax.vmap(check_legal_pawns, in_axes=(0, None))(white_pawn_tiles, 0)
    black_pawn_legal_moves, after_move_black_pawns, in_bounds_black_pawns = jax.vmap(check_legal_pawns, in_axes=(0, None))(black_pawn_tiles, 1)
    
    
    reduced_tiles = jnp.arange(self.board_width)
    # The pawns start here, you can use it for both promotion and the double move
    white_init_tiles = reduced_tiles + (self.board_height - 2) * self.board_width
    black_init_tiles = reduced_tiles + self.board_width
    
    white_pawn_legal_double_move, after_move_white_pawns_double_move, in_bounds_white_pawns_double_move = jax.vmap(check_legal_pawns_double_move, in_axes=(0, None))(reduced_tiles, 0)
    black_pawn_legal_double_move, after_move_black_pawns_double_move, in_bounds_black_pawns_double_move = jax.vmap(check_legal_pawns_double_move, in_axes=(0, None))(reduced_tiles, 1)
    
    white_pawn_legal_promote, after_move_white_pawn_promote, in_bounds_white_pawn_promote = jax.vmap(check_legal_pawns_promote, in_axes=(0, None))(reduced_tiles, 0)
    black_pawn_legal_promote, after_move_black_pawn_promote, in_bounds_black_pawn_promote = jax.vmap(check_legal_pawns_promote, in_axes=(0, None))(reduced_tiles, 1)
    
    # Creates array that looks liek [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    pawn_promotion_types = jnp.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]) + 2
    # pawn_promotion_types = jnp.tile(jnp.arange(len(PROMOTION_PIECES))[..., jnp.newaxis], white_pawn_legal_promote.shape[1], (3, 1)).flatten()
    pawn_promotions = jnp.repeat(pawn_promotion_types[None, ...], self.board_width, axis=0)
    
    # We make actions for opponent illegal
    legal_actions = jnp.concatenate([
      white_pawn_legal_moves.flatten() * (1 -game_state.current_player),
      black_pawn_legal_moves.flatten() * game_state.current_player,
      white_pawn_legal_double_move.flatten() * (1 -game_state.current_player),
      black_pawn_legal_double_move.flatten() * game_state.current_player,
      white_pawn_legal_promote.flatten() * (1 -game_state.current_player),
      black_pawn_legal_promote.flatten() * game_state.current_player,
    ], axis=0)
    
    from_move_pawns = jnp.concatenate([
      jnp.repeat(white_pawn_tiles, 3, axis=0),
      jnp.repeat(black_pawn_tiles, 3, axis=0),
      white_init_tiles,
      black_init_tiles,
      jnp.repeat(black_init_tiles, 3 * len(PROMOTION_PIECES), axis=0), # White promotion
      jnp.repeat(white_init_tiles, 3 * len(PROMOTION_PIECES), axis=0), # Black promotion
    ], axis=0)
    
    after_move_pawns = jnp.concatenate([
      after_move_white_pawns.reshape(-1, 2),
      after_move_black_pawns.reshape(-1, 2),
      after_move_white_pawns_double_move.reshape(-1, 2),
      after_move_black_pawns_double_move.reshape(-1, 2),
      after_move_white_pawn_promote.reshape(-1, 2),
      after_move_black_pawn_promote.reshape(-1, 2),
    ], axis=0)
    
    in_bounds_pawns = jnp.concatenate([
      in_bounds_white_pawns.flatten(),
      in_bounds_black_pawns.flatten(),
      in_bounds_white_pawns_double_move.flatten(),
      in_bounds_black_pawns_double_move.flatten(),
      in_bounds_white_pawn_promote.flatten(),
      in_bounds_black_pawn_promote.flatten(),
    ], axis=0)
    
    promotions = jnp.concatenate([
      jnp.zeros_like(white_pawn_legal_moves, dtype=jnp.int32).flatten(),
      jnp.zeros_like(black_pawn_legal_moves, dtype=jnp.int32).flatten(),
      jnp.zeros_like(white_pawn_legal_double_move, dtype=jnp.int32).flatten(),
      jnp.zeros_like(black_pawn_legal_double_move, dtype=jnp.int32).flatten(),
      pawn_promotions.flatten(),
      pawn_promotions.flatten(),
    ], axis=0)
    
    
    from_move_pawns = jnp.stack(jax.vmap(self.square_to_coords, in_axes=(0,))(from_move_pawns), axis=-1)
    
    return legal_actions, from_move_pawns, after_move_pawns, in_bounds_pawns, promotions
    
  @partial(jit, static_argnums=(0,))
  def apply_action(self, game_state: DarkChessGameState, action: chex.Array) -> Tuple[DarkChessGameState, chex.Array, int, bool]:
    """Apply action to game state. Returns new game state, legal actions, reward, is_terminal."""
    
    # Handle castling separately
    
    
      
    # Get move details from action ID
    from_pos = self.action_from[action]
    to_pos = self.action_to[action]
    promotion_piece = self.action_promotion[action]
    
    # Extract coordinates
    from_row, from_col = from_pos[0], from_pos[1]
    to_row, to_col = to_pos[0], to_pos[1]
    
    # Get the piece being moved
    moving_piece = game_state.board[from_row, from_col]
    captured_piece = game_state.board[to_row, to_col]
    
    # Create new board with basic move
    new_board = game_state.board.at[to_row, to_col].set(moving_piece)
    new_board = new_board.at[from_row, from_col].set(Pieces.EMPTY)
    
    # Handle pawn promotion
    # TODO: Create a test for this.
    is_promotion = promotion_piece > 0
    promoted_piece = lax.cond(
        is_promotion,
        lambda: promotion_piece + game_state.current_player * DISTINCT_PIECES_PER_PLAYER,
        lambda: moving_piece
    )
    new_board = lax.cond(
        is_promotion,
        lambda b: b.at[to_row, to_col].set(promoted_piece),
        lambda b: b,
        new_board
    )
    
    # Handle castling
    is_king = self.get_piece_type(moving_piece) == 6  # King piece type
    is_castling_move = is_king & (jnp.abs(from_col - to_col) == 2)
    
    # Castling rook movement
    rook_from_col = lax.cond(to_col > from_col, lambda: 7, lambda: 0)  # Kingside vs queenside
    rook_to_col = lax.cond(to_col > from_col, lambda: 5, lambda: 3)
    rook_piece = Pieces.WHITE_ROOK + game_state.current_player * DISTINCT_PIECES_PER_PLAYER
    
    new_board = lax.cond(
        is_castling_move,
        lambda b: b.at[from_row, rook_to_col].set(rook_piece).at[from_row, rook_from_col].set(Pieces.EMPTY),
        lambda b: b,
        new_board
    )
    
    # Handle en passant capture
    is_pawn = self.get_piece_type(moving_piece) == 1  # Pawn piece type
    # is_en_passant = is_pawn & (captured_piece == Pieces.EMPTY) & (from_col != to_col) & (game_state.en_passant_target < 64)
    # en_passant_capture_row = lax.cond(game_state.current_player == 0, lambda: to_row + 1, lambda: to_row - 1)
    
    # new_board = lax.cond(
    #     is_en_passant,
    #     lambda b: b.at[en_passant_capture_row, to_col].set(Pieces.EMPTY),
    #     lambda b: b,
    #     new_board
    # )
    # Rook moves or captures disable specific castling rights
    is_rook = self.get_piece_type(moving_piece) == 2  # Rook piece type
    captured_type = self.get_piece_type(captured_piece)
    is_rook_capture = captured_type == Pieces.WHITE_ROOK
    
    # White rook moves
    # Create masks for each castling right condition
    # TODO: Test this
    white_kingside_mask = (is_king & (game_state.current_player == 0)) | (is_rook & (game_state.current_player == 0) & (from_row == 7) & (from_col == 7)) | (is_rook_capture & (to_row == 7) & (to_col == 7))
    white_queenside_mask = (is_king & (game_state.current_player == 0)) | (is_rook & (game_state.current_player == 0) & (from_row == 7) & (from_col == 0)) | (is_rook_capture & (to_row == 7) & (to_col == 0))
    black_kingside_mask = (is_king & (game_state.current_player == 1)) | (is_rook & (game_state.current_player == 1) & (from_row == 0) & (from_col == 7)) | (is_rook_capture & (to_row == 0) & (to_col == 7))
    black_queenside_mask = (is_king & (game_state.current_player == 1)) | (is_rook & (game_state.current_player == 1) & (from_row == 0) & (from_col == 0)) | (is_rook_capture & (to_row == 0) & (to_col == 0))

    new_castling_rights = jnp.array([
      ~white_kingside_mask,
      ~white_queenside_mask,
      ~black_kingside_mask,
      ~black_queenside_mask
    ])
    new_castling_rights = jnp.logical_and(new_castling_rights, game_state.castling_rights) 
    
    # Update en passant target
    # is_pawn_double_move = is_pawn & (jnp.abs(from_row - to_row) == 2)
    # en_passant_row = (from_row + to_row) // 2
    # new_en_passant_target = lax.cond(
    #     is_pawn_double_move,
    #     lambda: en_passant_row * self.board_width + to_col,
    #     lambda: 64  # No en passant
    # )
    
    # Update halfmove clock (reset on pawn move or capture)
    is_capture = captured_piece != Pieces.EMPTY
    new_halfmove_clock = lax.cond(
        is_pawn | is_capture,
        lambda: 0,
        lambda: game_state.halfmove_clock + 1
    )
    
    # Update fullmove number (increment after black's move)
    new_fullmove_number = game_state.fullmove_number + game_state.current_player 
    
    # Switch current player
    new_current_player = 1 - game_state.current_player
    
    # Create new game state
    new_game_state = DarkChessGameState(
        board=new_board,
        castling_rights=new_castling_rights,
        captured_pieces = game_state.captured_pieces + jax.nn.one_hot(captured_piece - 1, len(PIECE_TO_ID)),
        current_player=new_current_player,
        en_passant_target=game_state.en_passant_target,
        halfmove_clock=new_halfmove_clock,
        fullmove_number=new_fullmove_number
    )
    
    # Get new legal actions
    new_legal_actions = self.get_legal_actions(new_game_state)
    
    # Check for game termination
    has_legal_moves = jnp.any(new_legal_actions)
    # is_in_check = self.is_in_check(new_game_state, new_current_player)
    
    # Determine game outcome
    # is_checkmate = ~has_legal_moves & is_in_check
    is_king_capture = self.get_piece_type(captured_piece) == Pieces.WHITE_KING
    is_stalemate = ~has_legal_moves
    is_fifty_move_rule = new_halfmove_clock >= 100  # 50 full moves = 100 halfmoves
    
    is_terminal = is_stalemate | is_fifty_move_rule | is_king_capture
    
    # Calculate reward (from perspective of player who just moved)
    reward = lax.cond(
        is_king_capture,
        lambda: 1 - 2 * game_state.current_player,  # Reward to the first player
        lambda: 0
    )
    
    return new_game_state, new_legal_actions, reward, is_terminal


  def pad_to_last_dimension(self, tensor: chex.Array) -> chex.Array:
    tensor = jnp.pad(tensor, (0, self.board_width - tensor.shape[0]))
    tensor = jnp.tile(tensor, (1, self.board_height, 1))
    return tensor
  
  def move_numbers_vector(self, game_state: DarkChessGameState) -> chex.Array:
    """Convert the game state to a move numbers vector."""
    move_numbers = jnp.array([game_state.halfmove_clock / 100, game_state.fullmove_number / 100], dtype=jnp.float32)
    move_numbers = self.pad_to_last_dimension(move_numbers)
    return move_numbers
  
  def acting_player_vector(self, game_state: DarkChessGameState) -> chex.Array:
    """Convert the game state to a acting player vector."""
    acting_player = jax.nn.one_hot(game_state.current_player, num_classes=2, dtype=jnp.float32)
    acting_player = self.pad_to_last_dimension(acting_player)
    return acting_player
  
  def acting_asking_player_vector(self, game_state: DarkChessGameState, player: int) -> chex.Array:
    """Convert the game state to a acting current player vector."""
    acting_player = jax.nn.one_hot(game_state.current_player, num_classes=2, dtype=jnp.float32)
    asking_player = jax.nn.one_hot(player, num_classes=2, dtype=jnp.float32)
    acting_asking_player = jnp.concatenate([acting_player, asking_player], axis=0)
    acting_asking_player = self.pad_to_last_dimension(acting_asking_player)
    return acting_asking_player
  
  def captured_pieces_player_vector(self, game_state: DarkChessGameState, player: int) -> chex.Array:
    player_offset = player * DISTINCT_PIECES_PER_PLAYER
    captured_pieces = jax.nn.one_hot(game_state.captured_pieces[player_offset:player_offset+DISTINCT_PIECES_PER_PLAYER] - 1, num_classes = self.board_width, dtype=jnp.float32)
    if self.board_height >= DISTINCT_PIECES_PER_PLAYER:
      captured_pieces = jnp.expand_dims(captured_pieces, axis=0)
    else:
      # We split it into two parts, first 3 pieces and last 3 pieces
      captured_pieces = jnp.stack([captured_pieces[:3], captured_pieces[3:]], axis=0) 
      
    captured_pieces = jnp.pad(captured_pieces, ((0, 0),(0, self.board_height - captured_pieces.shape[1]), (0, 0)))
    return captured_pieces
      
  # TODO: Could be optimized by doing the one_hot for each captured pieces, but then we would have to separate the arrays etc.
  def captured_pieces_both_vector(self, game_state: DarkChessGameState) -> chex.Array:
    captured_pieces_white = self.captured_pieces_player_vector(game_state, 0)
    captured_pieces_black = self.captured_pieces_player_vector(game_state, 1)
    captured_pieces = jnp.concatenate([captured_pieces_white, captured_pieces_black], axis=0)
    return captured_pieces
  
  def piece_tensor(self, game_state: DarkChessGameState) -> chex.Array:
    """Convert the game state to a piece tensor."""
    return jax.nn.one_hot(game_state.board, num_classes=2 * DISTINCT_PIECES_PER_PLAYER + 1, dtype=jnp.float32).transpose(2, 0, 1) # Each piece or empty
   
  def generate_player_visibility(self, game_state: DarkChessGameState, player: int) -> chex.Array:
    """Generate the visibility given the game state."""
    player_game_state = DarkChessGameState(
      board = game_state.board,
      current_player = player,
      castling_rights = game_state.castling_rights,
      en_passant_target = game_state.en_passant_target,
      halfmove_clock = game_state.halfmove_clock,
      fullmove_number = game_state.fullmove_number,
      captured_pieces = game_state.captured_pieces
    )
    legal_actions = self.get_legal_actions(player_game_state)
    return self.generate_visibility_given_legals(player_game_state, legal_actions)
 
  def generate_visibility_given_legals(self, game_state: DarkChessGameState, legal_actions: chex.Array) -> chex.Array:
    """Generate the visibility given the legal actions."""
      
    # Get the indices of legal actions
    visible_squares = jnp.where(legal_actions, self.action_to[:, 0] * self.board_width + self.action_to[:, 1], -1)
    is_visible = jnp.isin(jnp.arange(self.board_height * self.board_width), visible_squares).reshape(self.board_height, self.board_width)
    filtered_board = jnp.where(is_visible, game_state.board, 2 * DISTINCT_PIECES_PER_PLAYER + 1) # We set the not visible tiles to the piece with highest value + 1
    
    filtered_board = jnp.where((game_state.board >= (1 + DISTINCT_PIECES_PER_PLAYER * game_state.current_player)) & (game_state.board <= (DISTINCT_PIECES_PER_PLAYER + DISTINCT_PIECES_PER_PLAYER * game_state.current_player)), game_state.board, filtered_board)
    
    visibility = jax.nn.one_hot(filtered_board, num_classes=2 * DISTINCT_PIECES_PER_PLAYER + 2, dtype=jnp.float32).transpose(2, 0, 1) # Each piece or empty or unknown
    return visibility

  @partial(jit, static_argnums=(0,))
  def get_piece_attack_squares(self, game_state: DarkChessGameState, piece_square: int) -> chex.Array:
    """Get all squares that a piece can attack/see (regardless of legality)."""
    row, col = self.square_to_coords(piece_square)
    piece = game_state.board[row, col]
    
    if piece == Pieces.EMPTY:
      return jnp.zeros((0,), dtype=jnp.int32)
    
    piece_type = self.get_piece_type(piece)
    piece_color = self.get_piece_color(piece)
    
    def get_sliding_attacks(directions: chex.Array) -> chex.Array:
      """Get attack squares for sliding pieces (rook, bishop, queen)."""
      def scan_direction(direction: chex.Array) -> chex.Array:
        def scan_step(carry, step):
          pos, blocked = carry
          new_pos = pos + direction
          in_bounds = jnp.all(new_pos >= 0) & jnp.all(new_pos < jnp.array([self.board_height, self.board_width]))
          square_piece = lax.cond(in_bounds, 
                                 lambda: game_state.board[new_pos[0], new_pos[1]], 
                                 lambda: Pieces.EMPTY)
          is_blocked = blocked | (square_piece != Pieces.EMPTY)
          
          # Can attack this square if in bounds and not previously blocked
          can_attack = in_bounds & jnp.logical_not(blocked)
          attack_square = lax.cond(can_attack,
                                  lambda: new_pos[0] * self.board_width + new_pos[1],
                                  lambda: -1)
          
          return (new_pos, is_blocked), attack_square
        
        _, attacks = lax.scan(scan_step, (jnp.array([row, col]), False), 
                             jnp.arange(self.longest_move_length))
        return attacks
      
      all_attacks = jax.vmap(scan_direction)(directions)
      return all_attacks.flatten()
    
    def get_single_step_attacks(moves: chex.Array) -> chex.Array:
      """Get attack squares for non-sliding pieces (king, knight, pawn)."""
      pos = jnp.array([row, col])
      attack_positions = pos + moves
      in_bounds = jnp.all(attack_positions >= 0, axis=1) & jnp.all(attack_positions < jnp.array([self.board_height, self.board_width]), axis=1)
      attack_squares = attack_positions[:, 0] * self.board_width + attack_positions[:, 1]
      return jnp.where(in_bounds, attack_squares, -1)
    
    # Define attack patterns for each piece type
    rook_directions = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    bishop_directions = jnp.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    queen_directions = jnp.concatenate([rook_directions, bishop_directions])
    
    attacks = lax.switch(piece_type - 1, [
      # Pawn attacks
      lambda: get_single_step_attacks(
        jnp.where(piece_color == 0, 
                 jnp.array([[-1, -1], [-1, 1]]),  # White pawn attacks
                 jnp.array([[1, -1], [1, 1]]))    # Black pawn attacks
      ),
      # Rook attacks  
      lambda: get_sliding_attacks(rook_directions),
      # Knight attacks
      lambda: get_single_step_attacks(self.KNIGHT_DIRS),
      # Bishop attacks
      lambda: get_sliding_attacks(bishop_directions),
      # Queen attacks
      lambda: get_sliding_attacks(queen_directions),
      # King attacks
      lambda: get_single_step_attacks(self.KING_DIRS)
    ])
    
    # Filter out invalid squares (-1)
    return jnp.where(attacks >= 0, attacks, self.board_height * self.board_width)

  @partial(jit, static_argnums=(0,))
  def generate_public_visibility(self, game_state: DarkChessGameState) -> chex.Array:
    """
      Generate the public visibility given the game state.
      As a public visibility we take if the piece A sees piece B while piece B sees piece A, then this is publicly visible.    
    """
     
    # For the higher tier pieces we evaluate only for the white player.
    def check_public_visibility_rook(square: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:  
      row, col = self.square_to_coords(square)
      all_moves = jnp.arange(0, self.longest_move_length + 1) 
      left_cols = col - all_moves
      right_cols = col + all_moves
      up_rows = row - all_moves
      down_rows = row + all_moves
      left_squares = game_state.board[row, left_cols]
      right_squares = game_state.board[row, right_cols]
      up_squares = game_state.board[up_rows, col]
      down_squares = game_state.board[down_rows, col]
      
      square_piece = jnp.stack([left_squares, right_squares, up_squares, down_squares], axis=0)
      after_move_positions = jnp.stack([
        jnp.stack((jnp.repeat(row, self.longest_move_length+1), left_cols), axis=-1),
        jnp.stack((jnp.repeat(row, self.longest_move_length+1), right_cols), axis=-1),
        jnp.stack((up_rows, jnp.repeat(col, self.longest_move_length+1)), axis=-1),
        jnp.stack((down_rows, jnp.repeat(col, self.longest_move_length+1)), axis=-1)
      ], axis=0)
      in_bounds = jnp.stack([left_cols >= 0, right_cols < self.board_width, up_rows >= 0, down_rows < self.board_height], axis=0)
      
      def scan_rook_visibility(square_piece: chex.Array, in_bounds: chex.Array) -> chex.Array:
        @chex.dataclass(frozen=True)
        class RookVisibilityCarry:
          first: bool
          could_see: bool # It becomes false if you encounter first piece 
          
        def _scan_rook_visibility(carry: RookVisibilityCarry, x):
          square_piece, in_bounds = x
          new_could_see = carry.first | (in_bounds & carry.could_see & (square_piece == Pieces.EMPTY))
          first_opponent_piece = in_bounds & carry.could_see & ((square_piece == Pieces.BLACK_ROOK) | (square_piece == Pieces.BLACK_QUEEN))
          new_carry = RookVisibilityCarry(first=False, could_see=new_could_see)
          return new_carry, (first_opponent_piece, new_could_see)
        
        _, (encountered_opponent_piece, could_see) = lax.scan(_scan_rook_visibility, RookVisibilityCarry(first=True, could_see=True), (square_piece, in_bounds))
        
        public_visibility = jnp.any(encountered_opponent_piece) & (could_see | encountered_opponent_piece) * (game_state.board[row, col] == Pieces.WHITE_ROOK)
        return public_visibility
      
      public_visibility = jax.vmap(scan_rook_visibility, in_axes=(0, 0))(square_piece, in_bounds)
      return public_visibility, after_move_positions
    
    def check_public_visibility_bishop(square: chex.Array) -> tuple[chex.Array, chex.Array]:
      row, col = self.square_to_coords(square)
      all_moves = jnp.arange(0, self.longest_move_length + 1) 
      left_cols = col - all_moves
      right_cols = col + all_moves
      up_rows = row - all_moves
      down_rows = row + all_moves
      
      left_upper_diagonal = game_state.board[up_rows, left_cols]
      right_upper_diagonal = game_state.board[up_rows, right_cols]
      left_down_diagonal = game_state.board[down_rows, left_cols]
      right_down_diagonal = game_state.board[down_rows, right_cols]
      
      square_piece = jnp.stack([left_upper_diagonal, right_upper_diagonal, left_down_diagonal, right_down_diagonal], axis=0)
      
      after_move_positions = jnp.stack([
        jnp.stack((up_rows, left_cols), axis=-1),
        jnp.stack((up_rows, right_cols), axis=-1),
        jnp.stack((down_rows, left_cols), axis=-1),
        jnp.stack((down_rows, right_cols), axis=-1)
      ], axis=0)
      
      in_bounds = jnp.stack([ 
        (left_cols >= 0) & (up_rows >= 0),
        (right_cols < self.board_width) & (up_rows >= 0),
        (left_cols >= 0) & (down_rows < self.board_height),
        (right_cols < self.board_width) & (down_rows < self.board_height)
      ], axis=0)
      
      def scan_bishop_visibility(square_piece: chex.Array, in_bounds: chex.Array) -> chex.Array:
        @chex.dataclass(frozen=True)
        class BishopVisibilityCarry:
          first: bool
          could_see: bool # It becomes false if you encounter first piece 
          
        def _scan_bishop_visibility(carry: BishopVisibilityCarry, x):
          square_piece, in_bounds = x
          new_could_see = carry.first | (in_bounds & carry.could_see & (square_piece == Pieces.EMPTY))
          first_opponent_piece = in_bounds & carry.could_see & ((square_piece == Pieces.BLACK_BISHOP) | (square_piece == Pieces.BLACK_QUEEN))
          new_carry = BishopVisibilityCarry(first=False, could_see=new_could_see)
          return new_carry, (first_opponent_piece, new_could_see)
        
        _, (encountered_opponent_piece, could_see) = lax.scan(_scan_bishop_visibility, BishopVisibilityCarry(first=True, could_see=True), (square_piece, in_bounds))
        
        public_visibility = jnp.any(encountered_opponent_piece) & (could_see | encountered_opponent_piece) * (game_state.board[row, col] == Pieces.WHITE_BISHOP)
        return public_visibility
      
      public_visibility = jax.vmap(scan_bishop_visibility, in_axes=(0, 0))(square_piece, in_bounds)
      return public_visibility, after_move_positions
    
    def check_public_visibility_queen(square: chex.Array) -> tuple[chex.Array, chex.Array]:
      row, col = self.square_to_coords(square)
      all_moves = jnp.arange(0, self.longest_move_length + 1) 
      left_cols = col - all_moves
      right_cols = col + all_moves
      up_rows = row - all_moves
      down_rows = row + all_moves
      left_squares = game_state.board[row, left_cols]
      right_squares = game_state.board[row, right_cols]
      up_squares = game_state.board[up_rows, col]
      down_squares = game_state.board[down_rows, col]
      
      left_upper_diagonal = game_state.board[up_rows, left_cols]
      right_upper_diagonal = game_state.board[up_rows, right_cols]
      left_down_diagonal = game_state.board[down_rows, left_cols]
      right_down_diagonal = game_state.board[down_rows, right_cols]
      
      
      square_piece = jnp.stack([left_squares, right_squares, up_squares, down_squares, left_upper_diagonal, right_upper_diagonal, left_down_diagonal, right_down_diagonal], axis=0)
      after_move_positions = jnp.stack([
        jnp.stack((jnp.repeat(row, self.longest_move_length+1), left_cols), axis=-1),
        jnp.stack((jnp.repeat(row, self.longest_move_length+1), right_cols), axis=-1),
        jnp.stack((up_rows, jnp.repeat(col, self.longest_move_length+1)), axis=-1),
        jnp.stack((down_rows, jnp.repeat(col, self.longest_move_length+1)), axis=-1),
        jnp.stack((up_rows, left_cols), axis=-1),
        jnp.stack((up_rows, right_cols), axis=-1),
        jnp.stack((down_rows, left_cols), axis=-1),
        jnp.stack((down_rows, right_cols), axis=-1)
      ], axis=0)
      in_bounds = jnp.stack([
        left_cols >= 0,
        right_cols < self.board_width,
        up_rows >= 0,
        down_rows < self.board_height,
        (left_cols >= 0) & (up_rows >= 0),
        (right_cols < self.board_width) & (up_rows >= 0),
        (left_cols >= 0) & (down_rows < self.board_height),
        (right_cols < self.board_width) & (down_rows < self.board_height)
      ], axis=0)
      
      is_diagonal = jnp.array([False, False, False, False, True, True, True, True])
      
      def scan_queen_visibility(square_piece: chex.Array, in_bounds: chex.Array, is_diagonal: chex.Array) -> chex.Array:
        @chex.dataclass(frozen=True)
        class QueenVisibilityCarry:
          first: bool
          could_see: bool # It becomes false if you encounter first piece 
          
        def _scan_queen_visibility(carry: QueenVisibilityCarry, x):
          square_piece, in_bounds = x
          new_could_see = carry.first | (in_bounds & carry.could_see & (square_piece == Pieces.EMPTY))
          first_opponent_piece = in_bounds & carry.could_see & ((square_piece == Pieces.BLACK_QUEEN) | ((square_piece == Pieces.BLACK_ROOK) & ~is_diagonal) | (square_piece == Pieces.BLACK_BISHOP) & is_diagonal)
          new_carry = QueenVisibilityCarry(first=False, could_see=new_could_see)
          return new_carry, (first_opponent_piece, new_could_see)
        
        _, (encountered_opponent_piece, could_see) = lax.scan(_scan_queen_visibility, QueenVisibilityCarry(first=True, could_see=True), (square_piece, in_bounds))
        
        public_visibility = jnp.any(encountered_opponent_piece) & (could_see | encountered_opponent_piece) * (game_state.board[row, col] == Pieces.WHITE_QUEEN)
        return public_visibility
      
      public_visibility = jax.vmap(scan_queen_visibility, in_axes=(0, 0, 0))(square_piece, in_bounds, is_diagonal)
      return public_visibility, after_move_positions
    
    def get_positions_and_squares(square: chex.Array, moves: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
      pos = jnp.array(self.square_to_coords(square))
      after_move_position = pos + moves
      squares = game_state.board[after_move_position[:, 0], after_move_position[:, 1]]
      in_bounds = jnp.all(after_move_position >= 0, axis=1) & jnp.all(after_move_position < jnp.array(self.board_shape), axis=1)
      return after_move_position, squares, in_bounds
     
    
    def check_public_visibility_knight(square: chex.Array) -> tuple[chex.Array, chex.Array]:
      row, col = self.square_to_coords(square)
      after_move_position, squares, in_bounds = get_positions_and_squares(square, self.KNIGHT_DIRS)
      
      # We do not do the same thing as for king and pawns, just because it is slightly less operations this way.
      knight_visibility = (((game_state.board[row, col] == Pieces.WHITE_KNIGHT) & (squares == Pieces.BLACK_KNIGHT)) | ((game_state.board[row, col] == Pieces.BLACK_KNIGHT) & (squares == Pieces.WHITE_KNIGHT))) & in_bounds
      return knight_visibility, after_move_position
    
    def check_public_visibility_king(square: chex.Array) -> tuple[chex.Array, chex.Array]:
      row, col = self.square_to_coords(square)
    
      after_move_position, squares, in_bounds = get_positions_and_squares(square, self.KING_DIRS)
      
      is_diagonal_move = jnp.sum(jnp.abs(self.KING_DIRS), axis=1) == 2
      is_straight_move = jnp.sum(jnp.abs(self.KING_DIRS), axis=1) == 1
      
      # TODO: This is not correct, if king sees queen, it wouldnt see it back.
      white_king_visibility = (game_state.board[row, col] == Pieces.WHITE_KING) & ((squares == Pieces.BLACK_KING) | (squares == Pieces.BLACK_QUEEN) | ((squares == Pieces.BLACK_ROOK) & is_straight_move) | ((squares == Pieces.BLACK_BISHOP) & is_diagonal_move)) & in_bounds
      
      black_king_visibility = (game_state.board[row, col] == Pieces.BLACK_KING) & ((squares == Pieces.WHITE_KING) | (squares == Pieces.WHITE_QUEEN) | ((squares == Pieces.WHITE_ROOK) & is_straight_move) | ((squares == Pieces.WHITE_BISHOP) & is_diagonal_move)) & in_bounds
      
      king_visibility = (white_king_visibility | black_king_visibility) & in_bounds
      
      self_visibility = jnp.any(king_visibility)
      self_position = jnp.array([row, col])
      
      king_visibility = jnp.concatenate([self_visibility[None], king_visibility], axis=0)
      after_move_position = jnp.concatenate([self_position[None, :], after_move_position], axis=0)
      
      return king_visibility, after_move_position
    
    def check_public_visibility_white_pawn(square: chex.Array) -> tuple[chex.Array, chex.Array]:
      row, col = self.square_to_coords(square)
      pawn_dirs = jnp.array([[-1, 1], [-1, -1]])
      after_move_position, squares, in_bounds = get_positions_and_squares(square, pawn_dirs)
      pawn_visibility = (game_state.board[row, col] == Pieces.WHITE_PAWN) & ((squares == Pieces.BLACK_PAWN) | (squares == Pieces.BLACK_QUEEN) | (squares == Pieces.BLACK_BISHOP) | (squares == Pieces.BLACK_KING)) & in_bounds
      
      self_visibility = jnp.any(pawn_visibility)
      self_position = jnp.array([row, col])
      
      pawn_visibility = jnp.concatenate([self_visibility[None], pawn_visibility], axis=0)
      after_move_position = jnp.concatenate([self_position[None, :], after_move_position], axis=0)
      
      return pawn_visibility, after_move_position
    
    def check_public_visibility_black_pawn(square: chex.Array) -> tuple[chex.Array, chex.Array]:
      row, col = self.square_to_coords(square)
      pawn_dirs = jnp.array([[1, 1], [1, -1]])
      after_move_position, squares, in_bounds = get_positions_and_squares(square, pawn_dirs)
      pawn_visibility = (game_state.board[row, col] == Pieces.BLACK_PAWN) & ((squares == Pieces.WHITE_PAWN) | (squares == Pieces.WHITE_QUEEN) | (squares == Pieces.WHITE_BISHOP) | (squares == Pieces.WHITE_KING)) & in_bounds
      
      self_visibility = jnp.any(pawn_visibility)
      self_position = jnp.array([row, col])
      
      pawn_visibility = jnp.concatenate([self_visibility[None], pawn_visibility], axis=0)
      after_move_position = jnp.concatenate([self_position[None, :], after_move_position], axis=0)
      
      return pawn_visibility, after_move_position

    def check_public_visibility_square(square: chex.Array) -> tuple[chex.Array, chex.Array]:
      rook_visibility, rook_after_move_positions = check_public_visibility_rook(square)
      bishop_visibility, bishop_after_move_positions = check_public_visibility_bishop(square)
      queen_visibility, queen_after_move_positions = check_public_visibility_queen(square)
      knight_visibility, knight_after_move_positions = check_public_visibility_knight(square)
      king_visibility, king_after_move_positions = check_public_visibility_king(square)
      white_pawn_visibility, white_pawn_after_move_positions = check_public_visibility_white_pawn(square)
      black_pawn_visibility, black_pawn_after_move_positions = check_public_visibility_black_pawn(square)
      
      visibility = jnp.concatenate([
        rook_visibility.flatten(),
        bishop_visibility.flatten(),
        queen_visibility.flatten(),
        knight_visibility.flatten(),
        king_visibility.flatten(),
        white_pawn_visibility.flatten(),
        black_pawn_visibility.flatten()
        ],axis=0)
      after_move_positions = jnp.concatenate([
        rook_after_move_positions.reshape(-1, 2),
        bishop_after_move_positions.reshape(-1, 2),
        queen_after_move_positions.reshape(-1, 2),
        knight_after_move_positions.reshape(-1, 2),
        king_after_move_positions.reshape(-1, 2),
        white_pawn_after_move_positions.reshape(-1, 2),
        black_pawn_after_move_positions.reshape(-1, 2)
        ], axis=0)
      return visibility, after_move_positions

    tiles = jnp.arange(self.board_height * self.board_width)
    
    pieces_visibility, after_move_positions = jax.vmap(check_public_visibility_square, in_axes=(0,))(tiles) 
    
    visible_squares = jnp.where(pieces_visibility, after_move_positions[..., 0] * self.board_width + after_move_positions[..., 1], -1)
    is_visible = jnp.isin(jnp.arange(self.board_height * self.board_width), visible_squares).reshape(self.board_height, self.board_width)
    
    public_board = jnp.where(is_visible, game_state.board, 2 * DISTINCT_PIECES_PER_PLAYER + 1)
    public_board_oh = jax.nn.one_hot(public_board, num_classes=DISTINCT_PIECES_PER_PLAYER * 2 + 2).transpose(2, 0, 1)
    
    return public_board_oh
  
  @partial(jit, static_argnums=(0,))
  def state_tensor(self, game_state: DarkChessGameState) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Convert the game state to tensors.
    Tensors shape is (15 + C, height, width)
    where C is additional dimension if castling is present (only for 8x8 board)
    The First dimension denotes:
      - oh_board: One-hot encoded board, same order as Pieces (13)
      - castling: Castling rights (only for 8x8 board) (C)
      - move_numbers: Move numbers, where [0, :] is halfomove clock and [1, :], then [2:, :] is padding (1)
      - acting_player: Acting player, where [0, :] == 1 iff player 1 is acting and [1, :] iff player 2 is acting, [2:, :] is padding  (1)
    """
    oh_board = self.piece_tensor(game_state)
    if self.check_castling:
      castling_rights = jax.nn.one_hot(game_state.castling_rights.astype(jnp.int32), num_classes=2, dtype=jnp.float32)
      castling = jnp.tile(castling_rights, (1, 2, 4)) # This is fixed because castling is only for 8x8 board
    else:
      castling = jnp.zeros((0, self.board_height, self.board_width), dtype=jnp.float32) 
    move_numbers = self.move_numbers_vector(game_state) 
    acting_player = self.acting_player_vector(game_state)
      
    state_tensor = jnp.concatenate([
        oh_board,
        castling,
        move_numbers,
        acting_player
      ],
      axis=0
    )
    return state_tensor
  

  @partial(jit, static_argnums=(0, 2)) # Different jit for each player
  def observation_tensor(self, game_state: DarkChessGameState, player: int = -1) -> chex.Array:
    """
    Convert the game state to an observation tensor of a given player.
    Tensors shape is (17 + C, height, width) fr height >= 6 and (18, height, width) otherwise.
    Again C=1 if castling is present (only for 8x8 board)
    The First dimension denotes:
      - visibility: One-hot encoded board, same order as Pieces, where the last dimension is kept for unknown (14)
      - captured_pieces: Captured pieces of the opponent. For boards with height >= 6 it is (1), for smaller it is (2), where [0, 0:3, :] are PAWNS, ROOKS, KNIGHTS, and [1, 0:3, :] are BISHOPS, QUEENS. The rest is padding for board of height 4 or 5.
      - castling: Castling rights of player (only for 8x8 board) (C)
      - move_numbers: Move numbers, where [0, :] is halfomove clock and [1, :], then [2:, :] is padding (1)
      - acting_player: Acting player and asking player, where [0, :] == 1 iff player 1 is acting and [1, :] == 1 iff player 2 is acting, [2, :] == 1 iff player 1 is asking and [3, :] == 1 iff player 2 is asking, [4:, :] is padding  (1)
    """
    player = player if player >= 0 else game_state.current_player
    visibility = self.generate_player_visibility(game_state, player)
    if self.check_castling:
      castling_rights = jax.nn.one_hot(game_state.castling_rights[2 * player: 2 * player + 2].astype(jnp.int32), num_classes=2, dtype=jnp.float32)
      castling = jnp.tile(castling_rights, (1, 4, 4)) # This is fixed because castling is only for 8x8 board
    else:
      castling = jnp.zeros((0, self.board_height, self.board_width), dtype=jnp.float32)
    move_numbers = self.move_numbers_vector(game_state) 
    acting_asking_player = self.acting_asking_player_vector(game_state, player)
    captured_pieces = self.captured_pieces_player_vector(game_state, 1 - player)
    
    iset_tensor = jnp.concatenate([
      visibility, 
      captured_pieces,
      castling, 
      move_numbers, 
      acting_asking_player
      ], axis=0)
    
    return iset_tensor
  
  @partial(jit, static_argnums=(0,))
  def public_observation_tensor(self, game_state: DarkChessGameState) -> chex.Array:
    """
    Convert the game state to an observation tensor of a given player.
    Tensors shape is (18, height, width) for height >= 6 and (20, height, width) otherwise.
    The First dimension denotes:
      - visibility: One-hot encoded board, same order as Pieces, where the last dimension is kept for unknown (14)
      - captured_pieces: Captured pieces of both players. For boards with height >= 6 it is (2), for smaller it is (4), where [0 + 2*Player, 0:3, :] are PAWNS, ROOKS, KNIGHTS, and [1 + 2*Player, 0:3, :] are BISHOPS, QUEENS. The rest is padding for board of height 4 or 5.
      - move_numbers: Move numbers, where [0, :] is halfomove clock and [1, :], then [2:, :] is padding (1)
      - acting_player: Acting player, where [0, :] == 1 iff player 1 is acting and [1, :] == 1 iff player 2 is acting, [2:, :] is padding  (1)
    """
    visibility = self.generate_public_visibility(game_state)
    move_numbers = self.move_numbers_vector(game_state) 
    acting_player = self.acting_player_vector(game_state)
    captured_pieces = self.captured_pieces_both_vector(game_state)
    
    public_state_tensor = jnp.concatenate([
      visibility,
      captured_pieces, 
      move_numbers,
      acting_player
    ], axis=0)
    
    return public_state_tensor




  @partial(jit, static_argnums=(0,))
  def is_in_check(self, game_state: DarkChessGameState, player: int) -> bool:
    """Check if current player is in check."""
    # Find the king of the current player
    king_piece = Pieces.WHITE_KING + player * DISTINCT_PIECES_PER_PLAYER
    king_position = jnp.where(game_state.board == king_piece, size=1, fill_value=-1)
    king_position = self.coords_to_square(king_position[0], king_position[1])[0]
    
    # Check if any opponent piece can attack the king
    opponent_color = 1 - player
    
    return self.is_tile_attacked(game_state, king_position, opponent_color)
     

  @partial(jit, static_argnums=(0,))
  def square_to_coords(self, square: chex.Array) -> Tuple[int, int]:
    """Convert square index (0-63) to row, col coordinates."""
    return square // self.board_width, square % self.board_width
    

  @partial(jit, static_argnums=(0,))
  def coords_to_square(self, row: int, col: int) -> int:
    """Convert row, col coordinates to square index (0-63)."""
    return row * self.board_width + col
  
  @partial(jit, static_argnums=(0,))  
  def is_valid_square(self, row: int, col: int) -> bool:
    """Check if coordinates are within board bounds."""
    return (row >= 0) & (row < self.board_height) & (col >= 0) & (col < self.board_width)

  @partial(jit, static_argnums=(0,))
  def get_piece_color(self, piece: int) -> int:
    """Get color of piece: 0 for white, 1 for black, -1 for empty."""
    return lax.cond(
        piece == 0,
        lambda: jnp.array(-1),
        lambda: jnp.array((piece - 1) // DISTINCT_PIECES_PER_PLAYER)
    )

  @partial(jit, static_argnums=(0,))
  def get_piece_type(self, piece: int) -> int:
    """Get piece type (1-6 for pawn-king), 0 for empty."""
    return lax.cond(
        piece == 0,
        lambda: jnp.array(0),
        lambda: ((piece - 1) % DISTINCT_PIECES_PER_PLAYER) + 1
    )

  @partial(jit, static_argnums=(0,))
  def is_same_color(self, piece1: int, piece2: int) -> bool:
    """Check if two pieces are same color."""
    color1 = self.get_piece_color(piece1)
    color2 = self.get_piece_color(piece2)
    return (color1 != -1) & (color2 != -1) & (color1 == color2)
  
  @partial(jit, static_argnums=(0,))
  def is_different_color(self, piece1: int, piece2: int) -> bool:
    """Check if two pieces are enemies."""
    return jnp.logical_not(self.is_same_color(piece1, piece2))
 
    
    

def translate_fen(fen_str: str) -> DarkChessGameState:
  """Translate FEN string to board array."""
  rows = fen_str.split('/') 
  
  def count_columns(row: str) -> int:
    columns = 0
    for c in row:
      if c.isdigit():
        columns += int(c)
      else:
        columns += 1
    return columns
  board_height = len(rows)
  board_width = count_columns(rows[0])
  # The board does not have to be square
  board = np.zeros((board_height, board_width), dtype=np.int32)
  # This just ensures that the board is valid
  for i, row in enumerate(rows):
    row = row.split(" ")[0]
    if count_columns(row) != board_width:
      raise ValueError("Invalid board, some columns have different lengths")
    curr_column = 0
    for c in row:
      if c.isdigit():
        curr_column += int(c)
      else:
        board[i, curr_column] = PIECE_TO_ID[c]
        curr_column += 1
  additional_data = rows[-1].split(" ")
  
  acting_player = 0 if additional_data[1] == "w" else 1
  castling_rights = [
    "K" in additional_data[2],
    "Q" in additional_data[2],
    "k" in additional_data[2],
    "q" in additional_data[2]
  ]
  
  
  #TODO: en_passant does not work!
  en_passant_target = -1 if additional_data[3] == "-" else int(additional_data[3][1]) * 8 + ord(additional_data[3][0]) - ord("a")
  halfmove_clock = int(additional_data[4])
  fullmove_number = int(additional_data[5])
  
  return DarkChessGameState(
    board = jnp.array(board),
    captured_pieces = jnp.zeros(len(PIECE_TO_ID), dtype=jnp.int32),
    current_player = acting_player,
    castling_rights = jnp.array(castling_rights),
    en_passant_target = en_passant_target,
    halfmove_clock = halfmove_clock,
    fullmove_number = fullmove_number
  )

# Example usage and testing
if __name__ == "__main__":
  
  # White is at the bottom
  init_board = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
  # init_4_board = "rqkb/pppp/PPPP/RQKB w - - 0 1"
  game = DarkChessGame(init_board)  
  state, legal_actions = game.new_initial_state()
  vis = game.generate_visibility_given_legals(state, legal_actions)
  print(vis)
  # state, legal_actions, reward, is_terminal = game.apply_action(state, jnp.argmax(legal_actions))
  # print(state)
  # print(legal_actions)
  # print(reward)
  # print(is_terminal)
    # game.create_legal_filter_mask()
    # print(state)
  #   print(state.current_player)
