import jax
import jax.numpy as jnp
import numpy as np
from jax_chess import DarkChessGame, DarkChessGameState, translate_fen, Pieces, PIECE_TO_ID, ID_TO_PIECE, KNIGHT_DIRS, KING_DIRS, PROMOTION_PIECES, DISTINCT_PIECES_PER_PLAYER

def find_action_with_destination(game, state, legal_actions, piece_coords, destination_coords): 
  piece_moves, piece_to_positions = find_moves_for_piece(game, state, legal_actions, piece_coords)
  for i, dest in enumerate(piece_to_positions):
    if dest[0] == destination_coords[0] and dest[1] == destination_coords[1]:  # b6 position
      return piece_moves[i]
  return None
  
def find_moves_for_piece(game, state, legal_actions, piece_coords):
  """Helper function to find all legal moves for a piece at given coordinates."""
  legal_move_indices = jnp.where(legal_actions)[0]
  legal_from_positions = game.action_from[legal_move_indices]
  # piece_cords are often send as list not a array
  piece_moves = legal_move_indices[jnp.all(legal_from_positions == jnp.array(piece_coords), axis=1)]
  piece_to_positions = game.action_to[piece_moves]
  return piece_moves, piece_to_positions

def evaluate_legal_actions(game, fen, piece_cords, expected_moves, all_legals: int):
  state = translate_fen(fen)
  legal_actions = game.get_legal_actions(state) 
  piece_moves, piece_to_positions = find_moves_for_piece(game, state, legal_actions, piece_cords)
  
  assert jnp.sum(legal_actions) == all_legals, f"Expected {all_legals} legal actions, got {jnp.sum(legal_actions)} for {fen} at {piece_cords}"
  assert piece_moves.shape[0] == len(expected_moves), f"Expected {len(expected_moves)} moves, got {piece_moves.shape[0]} for {fen} at {piece_cords}"
  for move in piece_to_positions:
    assert move.tolist() in expected_moves, f"Unexpected move {move} for {fen} at {piece_cords}"
  
def create_fen_for_single_piece(piece_cords, piece_name, moving_player):
  y, x = piece_cords
  fen = "8/" * y + f"{"" if x == 0 else x}" + piece_name + f"{"" if x == 7 else (7-x)}" + "/8" * (7 - y)  + f" {moving_player} - - 0 1"
  return fen
 

def generate_rook_actions(chess_size, piece_position):
  y, x = piece_position
  expected_y_moves = [[y2, x] for y2 in range(chess_size) if y2 != y]
  expected_x_moves = [[y, x2] for x2 in range(chess_size) if x2 != x]
  expected_moves = expected_y_moves + expected_x_moves
  return expected_moves

def generate_bishop_actions(chess_size, piece_position):
  y, x = piece_position
  expected_descending_diagonal_moves = [[y + offset, x + offset] for offset in range(-chess_size,chess_size) if y + offset >= 0 and y + offset < chess_size and x + offset >= 0 and x + offset < chess_size and offset != 0]
  expected_ascending_diagonal_moves = [[y + offset, x - offset] for offset in range(-chess_size,chess_size) if y + offset >= 0 and y + offset < chess_size and x - offset >= 0 and x - offset < chess_size and offset != 0]
  expected_moves = expected_descending_diagonal_moves + expected_ascending_diagonal_moves
  return expected_moves

def generate_queen_actions(chess_size, piece_position):
  return generate_rook_actions(chess_size, piece_position) + generate_bishop_actions(chess_size, piece_position)

def generate_king_actions(chess_size, piece_position):
  y, x = piece_position
  expected_moves = [[y + y_offset, x + x_offset] for y_offset, x_offset in KING_DIRS if y + y_offset >= 0 and y + y_offset < chess_size and x + x_offset >= 0 and x + x_offset < chess_size]
  return expected_moves

def generate_knight_actions(chess_size, piece_position):
  y, x = piece_position
  
  expected_moves = [[y + y_offset, x + x_offset] for y_offset, x_offset in KNIGHT_DIRS if y + y_offset >= 0 and y + y_offset < chess_size and x + x_offset >= 0 and x + x_offset < chess_size]
  return expected_moves

def generate_promote_actions(move):
  return [move] * len(PROMOTION_PIECES)

# This generates even the attack actions!
def generate_pawn_actions(chess_size, piece_position, player):
  y, x = piece_position
  expected_moves= []
  
  y_offset =  (2 * player) - 1
  pawn_default_moves = [[y + y_offset, x + x_offset] for x_offset in [-1, 0, 1] if y + y_offset >= 0 and y + y_offset < chess_size and x + x_offset >= 0 and x + x_offset < chess_size]
  if (y == 1 and player == 0) or (y == chess_size - 2 and player == 1):
    for move in pawn_default_moves:
      expected_moves.extend(generate_promote_actions(move))
  elif (y == 1 and player == 1) or (y == chess_size - 2 and player == 0):
    expected_moves = pawn_default_moves
    expected_moves.append([y + 2 * y_offset, x])
  elif not (y == 0 and player == 0) or (y == chess_size - 1 and player == 1):
    expected_moves = pawn_default_moves
  return expected_moves


def replace_in_fen(fen, piece, piece_position):
  y, x = piece_position
  new_fen = ""
  for y_row, row in enumerate(fen.split("/")):
    if y != y_row:
      new_fen += row
      new_fen += "" if len(fen.split("/")) == y_row + 1 else "/"
      continue
    splitted_row = row.strip().split(" ")
    
    row_pieces = []
    
    for char in splitted_row[0]:
      if char.isdigit():
        row_pieces += [0] * int(char)
      else:
        row_pieces.append(PIECE_TO_ID[char])
    row_pieces[x] = PIECE_TO_ID[piece]
    
    empty_count = 0
    for p in row_pieces:
      if p == 0:
        empty_count += 1
      else:
        new_fen += str(empty_count) if empty_count > 0 else ""
        new_fen += ID_TO_PIECE[p]
        empty_count = 0
    new_fen += str(empty_count) if empty_count > 0 else ""
    new_fen += "/" if len(splitted_row) == 1 else (" " + " ".join(splitted_row[1:]))
  return new_fen


class DarkChessTest:
  def __init__(self):
    self.init_fen4 = "rqkb/pppp/PPPP/RQKB w - - 0 1"
    self.init_fen6 = "rnqknr/pppppp/6/6/PPPPPP/RNQKNR w - - 0 1"
    self.init_fen8 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    
    self.game4 = DarkChessGame(self.init_fen4)
    self.game6 = DarkChessGame(self.init_fen6)
    self.game8 = DarkChessGame(self.init_fen8)
    
    
  def run_all_tests(self):
    print("Skipping some tests")
    self.test_fen_parsing_standard_position() 
    self.test_rook_moveset()
    self.test_bishop_moveset()
    self.test_queen_moveset()
    self.test_king_moveset()
    self.test_knight_moveset()
    self.test_pawn_moveset()
    self.test_castling_moveset()
    self.test_check_detection()
    self.test_move_application()
    self.test_piece_capture_mechanics()
    self.test_king_capture_game_termination()
    self.test_state_tensor()
    self.test_observation_tensor()
    self.test_observation_rook()
    self.test_observation_bishop()
    self.test_public_observation_rook()
    self.test_public_observation_bishop()
    self.test_public_observation_queen()
    self.test_public_observation_knight()
    self.test_public_observation_king()
    # self.test_public_observation_tensor()
  
  def test_fen_parsing_standard_position(self):
    """Test FEN string parsing for standard starting position.""" 
    state = translate_fen(self.init_fen8)
    
    # Check board setup for standard position
    assert state.board[0, 0] == Pieces.BLACK_ROOK
    assert state.board[0, 1] == Pieces.BLACK_KNIGHT
    assert state.board[0, 2] == Pieces.BLACK_BISHOP
    assert state.board[0, 3] == Pieces.BLACK_QUEEN
    assert state.board[0, 4] == Pieces.BLACK_KING
    assert state.board[0, 5] == Pieces.BLACK_BISHOP
    assert state.board[0, 6] == Pieces.BLACK_KNIGHT
    assert state.board[0, 7] == Pieces.BLACK_ROOK
    
    
    assert np.all(state.board[1, :] == Pieces.BLACK_PAWN)
    assert np.all(state.board[2:6, :] == Pieces.EMPTY) 
    assert np.all(state.board[6, :] == Pieces.WHITE_PAWN)
        
    # Check white pieces
    assert state.board[7, 0] == Pieces.WHITE_ROOK
    assert state.board[7, 1] == Pieces.WHITE_KNIGHT
    assert state.board[7, 2] == Pieces.WHITE_BISHOP
    assert state.board[7, 3] == Pieces.WHITE_QUEEN
    assert state.board[7, 4] == Pieces.WHITE_KING
    assert state.board[7, 5] == Pieces.WHITE_BISHOP
    assert state.board[7, 6] == Pieces.WHITE_KNIGHT
    assert state.board[7, 7] == Pieces.WHITE_ROOK
    
    assert state.current_player == 0, "White should move first"
    assert jnp.all(state.castling_rights) , "All castling rights should be available"
    assert state.en_passant_target == -1, "No en passant target"
    assert state.halfmove_clock == 0, "Halfmove clock should be 0"
    assert state.fullmove_number == 1, "Fullmove number should be 1"
    
    fen_4 = self.init_fen4
    state_4 = translate_fen(fen_4)
    
    assert state_4.board[0, 0] == Pieces.BLACK_ROOK
    assert state_4.board[0, 1] == Pieces.BLACK_QUEEN
    assert state_4.board[0, 2] == Pieces.BLACK_KING
    assert state_4.board[0, 3] == Pieces.BLACK_BISHOP
    assert np.all(state_4.board[1, :] == Pieces.BLACK_PAWN)
    assert np.all(state_4.board[2, :] == Pieces.WHITE_PAWN)
    assert state_4.board[3, 0] == Pieces.WHITE_ROOK
    assert state_4.board[3, 1] == Pieces.WHITE_QUEEN
    assert state_4.board[3, 2] == Pieces.WHITE_KING
    assert state_4.board[3, 3] == Pieces.WHITE_BISHOP
    assert state_4.current_player == 0, "White should move first"
    assert np.all(state_4.castling_rights == False), "No castling rights"
    assert state_4.en_passant_target == -1, "No en passant target"
    assert state_4.halfmove_clock == 0, "Halfmove clock should be 0"
    assert state_4.fullmove_number == 1, "Fullmove number should be 1"
    
    fen_6 = self.init_fen6
    state_6 = translate_fen(fen_6)
    assert state_6.board[0, 0] == Pieces.BLACK_ROOK
    assert state_6.board[0, 1] == Pieces.BLACK_KNIGHT
    assert state_6.board[0, 2] == Pieces.BLACK_QUEEN
    assert state_6.board[0, 3] == Pieces.BLACK_KING
    assert state_6.board[0, 4] == Pieces.BLACK_KNIGHT
    assert state_6.board[0, 5] == Pieces.BLACK_ROOK
    assert np.all(state_6.board[1, :] == Pieces.BLACK_PAWN)
    assert np.all(state_6.board[2:4, :] == Pieces.EMPTY)
    assert np.all(state_6.board[4, :] == Pieces.WHITE_PAWN)
    assert state_6.board[5, 0] == Pieces.WHITE_ROOK
    assert state_6.board[5, 1] == Pieces.WHITE_KNIGHT
    assert state_6.board[5, 2] == Pieces.WHITE_QUEEN
    assert state_6.board[5, 3] == Pieces.WHITE_KING
    assert state_6.board[5, 4] == Pieces.WHITE_KNIGHT
    assert state_6.board[5, 5] == Pieces.WHITE_ROOK 
    assert state_6.current_player == 0, "White should move first"
    assert np.all(state_6.castling_rights == False), "No castling rights"
    assert state_6.en_passant_target == -1, "No en passant target"
    assert state_6.halfmove_clock == 0, "Halfmove clock should be 0"
    assert state_6.fullmove_number == 1, "Fullmove number should be 1"
    
    
    midgame_fen = "rn1q1b2/p3k1p1/bp1p1n1r/1Bp1pp1p/1P2P1P1/B1NP1N2/P1P1QP1P/2R1K2R b K - 1 11"
    state_midgame = translate_fen(midgame_fen)
    assert state_midgame.board[0, 0] == Pieces.BLACK_ROOK
    assert state_midgame.board[0, 1] == Pieces.BLACK_KNIGHT
    assert state_midgame.board[0, 2] == Pieces.EMPTY
    assert state_midgame.board[0, 3] == Pieces.BLACK_QUEEN
    assert state_midgame.board[0, 4] == Pieces.EMPTY
    assert state_midgame.board[0, 5] == Pieces.BLACK_BISHOP
    assert state_midgame.board[0, 6] == Pieces.EMPTY
    assert state_midgame.board[0, 7] == Pieces.EMPTY
    assert state_midgame.board[1, 0] == Pieces.BLACK_PAWN
    assert state_midgame.board[1, 1] == Pieces.EMPTY
    assert state_midgame.board[1, 2] == Pieces.EMPTY
    assert state_midgame.board[1, 3] == Pieces.EMPTY
    assert state_midgame.board[1, 4] == Pieces.BLACK_KING
    assert state_midgame.board[1, 5] == Pieces.EMPTY
    assert state_midgame.board[1, 6] == Pieces.BLACK_PAWN
    assert state_midgame.board[1, 7] == Pieces.EMPTY 
    assert state_midgame.board[2, 0] == Pieces.BLACK_BISHOP
    assert state_midgame.board[2, 1] == Pieces.BLACK_PAWN
    assert state_midgame.board[2, 2] == Pieces.EMPTY
    assert state_midgame.board[2, 3] == Pieces.BLACK_PAWN
    assert state_midgame.board[2, 4] == Pieces.EMPTY
    assert state_midgame.board[2, 5] == Pieces.BLACK_KNIGHT
    assert state_midgame.board[2, 6] == Pieces.EMPTY
    assert state_midgame.board[2, 7] == Pieces.BLACK_ROOK
    assert state_midgame.board[3, 0] == Pieces.EMPTY
    assert state_midgame.board[3, 1] == Pieces.WHITE_BISHOP
    assert state_midgame.board[3, 2] == Pieces.BLACK_PAWN
    assert state_midgame.board[3, 3] == Pieces.EMPTY
    assert state_midgame.board[3, 4] == Pieces.BLACK_PAWN
    assert state_midgame.board[3, 5] == Pieces.BLACK_PAWN
    assert state_midgame.board[3, 6] == Pieces.EMPTY
    assert state_midgame.board[3, 7] == Pieces.BLACK_PAWN
    assert state_midgame.board[4, 0] == Pieces.EMPTY
    assert state_midgame.board[4, 1] == Pieces.WHITE_PAWN
    assert state_midgame.board[4, 2] == Pieces.EMPTY
    assert state_midgame.board[4, 3] == Pieces.EMPTY
    assert state_midgame.board[4, 4] == Pieces.WHITE_PAWN
    assert state_midgame.board[4, 5] == Pieces.EMPTY
    assert state_midgame.board[4, 6] == Pieces.WHITE_PAWN
    assert state_midgame.board[4, 7] == Pieces.EMPTY
    assert state_midgame.board[5, 0] == Pieces.WHITE_BISHOP
    assert state_midgame.board[5, 1] == Pieces.EMPTY
    assert state_midgame.board[5, 2] == Pieces.WHITE_KNIGHT
    assert state_midgame.board[5, 3] == Pieces.WHITE_PAWN
    assert state_midgame.board[5, 4] == Pieces.EMPTY
    assert state_midgame.board[5, 5] == Pieces.WHITE_KNIGHT
    assert state_midgame.board[5, 6] == Pieces.EMPTY
    assert state_midgame.board[5, 7] == Pieces.EMPTY
    assert state_midgame.board[6, 0] == Pieces.WHITE_PAWN
    assert state_midgame.board[6, 1] == Pieces.EMPTY
    assert state_midgame.board[6, 2] == Pieces.WHITE_PAWN
    assert state_midgame.board[6, 3] == Pieces.EMPTY
    assert state_midgame.board[6, 4] == Pieces.WHITE_QUEEN
    assert state_midgame.board[6, 5] == Pieces.WHITE_PAWN
    assert state_midgame.board[6, 6] == Pieces.EMPTY
    assert state_midgame.board[6, 7] == Pieces.WHITE_PAWN
    assert state_midgame.board[7, 0] == Pieces.EMPTY 
    assert state_midgame.board[7, 1] == Pieces.EMPTY
    assert state_midgame.board[7, 2] == Pieces.WHITE_ROOK
    assert state_midgame.board[7, 3] == Pieces.EMPTY
    assert state_midgame.board[7, 4] == Pieces.WHITE_KING
    assert state_midgame.board[7, 5] == Pieces.EMPTY
    assert state_midgame.board[7, 6] == Pieces.EMPTY
    assert state_midgame.board[7, 7] == Pieces.WHITE_ROOK
    
    assert state_midgame.current_player == 1, "Black should move next"
    assert np.all(state_midgame.castling_rights == np.array([True, False, False, False])), "Different castling right"
    assert state_midgame.en_passant_target == -1, "No en passant target"
    assert state_midgame.halfmove_clock == 1, "Halfmove clock should be 1"
    assert state_midgame.fullmove_number == 11, "Fullmove number should be 11"
    
    print("✓ FEN parsing test passed") 


  def test_rook_moveset(self):
    """Test rook move generation.""" 
    
    # Test whether each tile has correct moveset
    chess_size = 8
    for pl in range(2):
      piece_name = "R" if pl == 0 else "r"
      moving_player = "w" if pl == 0 else "b"
      for y in range(8):
        for x in range(8): 
          piece_position = [y, x]
          fen = create_fen_for_single_piece(piece_position, piece_name, moving_player)
          expected_moves = generate_rook_actions(chess_size, piece_position)
          evaluate_legal_actions(self.game8, fen, piece_position, expected_moves, len(expected_moves))
    
    # TODO: Test every blocking position?
    # Test whether blocking by own pieces is handled correctly
      y, x = 4, 4
      piece_position = [y, x]
      blocking_piece = "P" if pl == 0 else "p" 
      
      fen = create_fen_for_single_piece(piece_position, piece_name, moving_player)
      left_fen = replace_in_fen(fen, blocking_piece, [y, x-1])
      right_fen = replace_in_fen(fen, blocking_piece, [y, x+1])
      up_fen = replace_in_fen(fen, blocking_piece, [y-1, x])
      down_fen = replace_in_fen(fen, blocking_piece, [y+1, x])
      expected_moves = generate_rook_actions(chess_size, piece_position)
      left_expected_moves = list(filter(lambda move: move[1] >= x, expected_moves))
      right_expected_moves = list(filter(lambda move: move[1] <= x, expected_moves))
      up_expected_moves = list(filter(lambda move: move[0] >= y, expected_moves))
      down_expected_moves = list(filter(lambda move: move[0] <= y, expected_moves))
    
      
      evaluate_legal_actions(self.game8, left_fen, piece_position, left_expected_moves, len(left_expected_moves) + 1)
      evaluate_legal_actions(self.game8, right_fen, piece_position, right_expected_moves, len(right_expected_moves) + 1)
      evaluate_legal_actions(self.game8, up_fen, piece_position, up_expected_moves, len(up_expected_moves) + (1 - pl)) # Only if the pawn is not blocked by the rook
      evaluate_legal_actions(self.game8, down_fen, piece_position, down_expected_moves, len(down_expected_moves) + pl)
    
    # Test whether blocking by opponents piece is handled correctly
      blocking_piece = "p" if pl == 0 else "P"
      left_fen = replace_in_fen(fen, blocking_piece, [y, x-1])
      right_fen = replace_in_fen(fen, blocking_piece, [y, x+1])
      up_fen = replace_in_fen(fen, blocking_piece, [y-1, x])
      down_fen = replace_in_fen(fen, blocking_piece, [y+1, x])
      expected_moves = generate_rook_actions(chess_size, piece_position)
      left_expected_moves = list(filter(lambda move: move[1] >= x or move[1] == x - 1, expected_moves))
      right_expected_moves = list(filter(lambda move: move[1] <= x or move[1] == x + 1, expected_moves))
      up_expected_moves = list(filter(lambda move: move[0] >= y or move[0] == y - 1, expected_moves))
      down_expected_moves = list(filter(lambda move: move[0] <= y or move[0] == y + 1, expected_moves)) 
      evaluate_legal_actions(self.game8, left_fen, jnp.array([y, x]), left_expected_moves, len(left_expected_moves))
      evaluate_legal_actions(self.game8, right_fen, jnp.array([y, x]), right_expected_moves, len(right_expected_moves))
      evaluate_legal_actions(self.game8, up_fen, jnp.array([y, x]), up_expected_moves, len(up_expected_moves))
      evaluate_legal_actions(self.game8, down_fen, jnp.array([y, x]), down_expected_moves, len(down_expected_moves))
    
    print("✓ Rook moveset test passed")

  def test_bishop_moveset(self):
    """Test bishop move generation.""" 
    
    # Test whether each tile has correct moveset
    chess_size = 8
    for pl in range(2):
      piece_name = "B" if pl == 0 else "b"
      moving_player = "w" if pl == 0 else "b"
      for y in range(8):
        for x in range(8): 
          piece_position = [y, x]
          fen = create_fen_for_single_piece(piece_position, piece_name, moving_player)
          expected_moves = generate_bishop_actions(chess_size, piece_position)
          evaluate_legal_actions(self.game8, fen, piece_position, expected_moves, len(expected_moves))
        
    # Test whether blocking by own pieces is handled correctly 
      y, x = 4, 4
      piece_position = [y, x]
      blocking_piece = "P" if pl == 0 else "p"
      fen = create_fen_for_single_piece(piece_position, piece_name, moving_player)
      left_up_fen = replace_in_fen(fen, blocking_piece, [y-1, x-1])
      right_up_fen = replace_in_fen(fen, blocking_piece, [y-1, x+1])
      left_down_fen = replace_in_fen(fen, blocking_piece, [y+1, x-1])
      right_down_fen = replace_in_fen(fen, blocking_piece, [y+1, x+1])
      expected_moves = generate_bishop_actions(chess_size, piece_position)
      left_up_expected_moves = list(filter(lambda move: move[0] > y or move[1] > x, expected_moves))
      right_up_expected_moves = list(filter(lambda move: move[0] > y or move[1] < x, expected_moves))
      left_down_expected_moves = list(filter(lambda move: move[0] < y or move[1] > x, expected_moves))
      right_down_expected_moves = list(filter(lambda move: move[0] < y or move[1] < x, expected_moves))
      
      evaluate_legal_actions(self.game8, left_up_fen, piece_position, left_up_expected_moves, len(left_up_expected_moves) + 1 ) # The additional move of pawn up
      evaluate_legal_actions(self.game8, right_up_fen, piece_position, right_up_expected_moves, len(right_up_expected_moves) + 1)
      evaluate_legal_actions(self.game8, left_down_fen, piece_position, left_down_expected_moves, len(left_down_expected_moves) + 1)
      evaluate_legal_actions(self.game8, right_down_fen, piece_position, right_down_expected_moves, len(right_down_expected_moves) + 1)
      
    # Test whether blocking by opponents piece is handled correctly
      blocking_piece = "p" if pl == 0 else "P"
      left_up_fen = replace_in_fen(fen, blocking_piece, [y-1, x-1])
      right_up_fen = replace_in_fen(fen, blocking_piece, [y-1, x+1])
      left_down_fen = replace_in_fen(fen, blocking_piece, [y+1, x-1])
      right_down_fen = replace_in_fen(fen, blocking_piece, [y+1, x+1])
      expected_moves = generate_bishop_actions(chess_size, piece_position)
      left_up_expected_moves = list(filter(lambda move: move[0] > y or move[1] > x or move[0] == y-1, expected_moves))
      right_up_expected_moves = list(filter(lambda move: move[0] > y or move[1] < x or move[0] == y-1, expected_moves))
      left_down_expected_moves = list(filter(lambda move: move[0] < y or move[1] > x or move[0] == y+1, expected_moves))
      right_down_expected_moves = list(filter(lambda move: move[0] < y or move[1] < x or move[0] == y+1, expected_moves))
      
      evaluate_legal_actions(self.game8, left_up_fen, piece_position, left_up_expected_moves, len(left_up_expected_moves))
      evaluate_legal_actions(self.game8, right_up_fen, piece_position, right_up_expected_moves, len(right_up_expected_moves))
      evaluate_legal_actions(self.game8, left_down_fen, piece_position, left_down_expected_moves, len(left_down_expected_moves))
      evaluate_legal_actions(self.game8, right_down_fen, piece_position, right_down_expected_moves, len(right_down_expected_moves))
      
    
    print("✓ Bishop moveset test passed")

  def test_queen_moveset(self):
    """Test queen move generation.""" 
    
    # Test whether each tile has correct moveset
    chess_size = 8
    for pl in range(2):
      piece_name = "Q" if pl == 0 else "q"
      moving_player = "w" if pl == 0 else "b"
      for y in range(8):
        for x in range(8): 
          piece_position = [y, x]
          fen = create_fen_for_single_piece(piece_position, piece_name, moving_player)
          expected_moves = generate_queen_actions(chess_size, piece_position)
          evaluate_legal_actions(self.game8, fen, piece_position, expected_moves, len(expected_moves))
          
    # Test whether blocking by own pieces is handled correctly 
      y, x = 4, 4
      piece_position = [y, x]
      blocking_piece = "P" if pl == 0 else "p" 
      fen = create_fen_for_single_piece(piece_position, piece_name, moving_player)
      left_fen = replace_in_fen(fen, blocking_piece, [y, x-1])
      right_fen = replace_in_fen(fen, blocking_piece, [y, x+1])
      up_fen = replace_in_fen(fen, blocking_piece, [y-1, x])
      down_fen = replace_in_fen(fen, blocking_piece, [y+1, x])
      left_up_fen = replace_in_fen(fen, blocking_piece, [y-1, x-1])
      right_up_fen = replace_in_fen(fen, blocking_piece, [y-1, x+1])
      left_down_fen = replace_in_fen(fen, blocking_piece, [y+1, x-1])
      right_down_fen = replace_in_fen(fen, blocking_piece, [y+1, x+1])
      expected_moves = generate_queen_actions(chess_size, piece_position)
      left_expected_moves = list(filter(lambda move: not (move[1] < x and move[0] == y), expected_moves))
      right_expected_moves = list(filter(lambda move: not (move[1] > x and move[0] == y), expected_moves))
      up_expected_moves = list(filter(lambda move: not (move[0] < y and move[1] == x), expected_moves))
      down_expected_moves = list(filter(lambda move: not (move[0] > y and move[1] == x), expected_moves))
      left_up_expected_moves = list(filter(lambda move: not (move[1] < x and move[0] < y), expected_moves))
      right_up_expected_moves = list(filter(lambda move: not (move[1] > x and move[0] < y), expected_moves))
      left_down_expected_moves = list(filter(lambda move: not (move[1] < x and move[0] > y), expected_moves))
      right_down_expected_moves = list(filter(lambda move: not (move[1] > x and move[0] > y), expected_moves))
      
      evaluate_legal_actions(self.game8, left_fen, piece_position, left_expected_moves, len(left_expected_moves) + 1)
      evaluate_legal_actions(self.game8, right_fen, piece_position, right_expected_moves, len(right_expected_moves) + 1)
      evaluate_legal_actions(self.game8, up_fen, piece_position, up_expected_moves, len(up_expected_moves) + 1 - pl )
      evaluate_legal_actions(self.game8, down_fen, piece_position, down_expected_moves, len(down_expected_moves) + pl)
      evaluate_legal_actions(self.game8, left_up_fen, piece_position, left_up_expected_moves, len(left_up_expected_moves) + 1)
      evaluate_legal_actions(self.game8, right_up_fen, piece_position, right_up_expected_moves, len(right_up_expected_moves) + 1)
      evaluate_legal_actions(self.game8, left_down_fen, piece_position, left_down_expected_moves, len(left_down_expected_moves) + 1)
      evaluate_legal_actions(self.game8, right_down_fen, piece_position, right_down_expected_moves, len(right_down_expected_moves) + 1)  
    
    # Test whether blocking by opponents pieces is handled correctly       
      blocking_piece = "p" if pl == 0 else "P" 
      fen = create_fen_for_single_piece(piece_position, piece_name, moving_player)
      left_fen = replace_in_fen(fen, blocking_piece, [y, x-1])
      right_fen = replace_in_fen(fen, blocking_piece, [y, x+1])
      up_fen = replace_in_fen(fen, blocking_piece, [y-1, x])
      down_fen = replace_in_fen(fen, blocking_piece, [y+1, x])
      left_up_fen = replace_in_fen(fen, blocking_piece, [y-1, x-1])
      right_up_fen = replace_in_fen(fen, blocking_piece, [y-1, x+1])
      left_down_fen = replace_in_fen(fen, blocking_piece, [y+1, x-1])
      right_down_fen = replace_in_fen(fen, blocking_piece, [y+1, x+1])
      expected_moves = generate_queen_actions(chess_size, piece_position)
      left_expected_moves = list(filter(lambda move: (not (move[1] < x and move[0] == y)) or (move[1] == x-1 and move[0] == y), expected_moves))
      right_expected_moves = list(filter(lambda move: (not (move[1] > x and move[0] == y)) or (move[1] == x+1 and move[0] == y), expected_moves))
      up_expected_moves = list(filter(lambda move: (not (move[0] < y and move[1] == x)) or (move[0] == y-1 and move[1] == x), expected_moves))
      down_expected_moves = list(filter(lambda move: (not (move[0] > y and move[1] == x)) or (move[0] == y+1 and move[1] == x), expected_moves))
      left_up_expected_moves = list(filter(lambda move: (not (move[1] < x and move[0] < y)) or (move[1] == x-1 and move[0] == y-1), expected_moves))
      right_up_expected_moves = list(filter(lambda move: (not (move[1] > x and move[0] < y)) or (move[1] == x+1 and move[0] == y-1), expected_moves))
      left_down_expected_moves = list(filter(lambda move: (not (move[1] < x and move[0] > y)) or (move[1] == x-1 and move[0] == y+1), expected_moves))
      right_down_expected_moves = list(filter(lambda move: (not (move[1] > x and move[0] > y)) or (move[1] == x+1 and move[0] == y+1), expected_moves))
      
      evaluate_legal_actions(self.game8, left_fen, piece_position, left_expected_moves, len(left_expected_moves))
      evaluate_legal_actions(self.game8, right_fen, piece_position, right_expected_moves, len(right_expected_moves))
      evaluate_legal_actions(self.game8, up_fen, piece_position, up_expected_moves, len(up_expected_moves))
      evaluate_legal_actions(self.game8, down_fen, piece_position, down_expected_moves, len(down_expected_moves))
      evaluate_legal_actions(self.game8, left_up_fen, piece_position, left_up_expected_moves, len(left_up_expected_moves))
      evaluate_legal_actions(self.game8, right_up_fen, piece_position, right_up_expected_moves, len(right_up_expected_moves))
      evaluate_legal_actions(self.game8, left_down_fen, piece_position, left_down_expected_moves, len(left_down_expected_moves))
      evaluate_legal_actions(self.game8, right_down_fen, piece_position, right_down_expected_moves, len(right_down_expected_moves))  
    
    print("✓ Queen moveset test passed")

  def test_king_moveset(self):
    """Test king move generation."""
    
    # Test whether each tile has correct moveset
    chess_size = 8
    for pl in range(2):
      piece_name = "K" if pl == 0 else "k"
      moving_player = "w" if pl == 0 else "b"
      for y in range(8):
        for x in range(8): 
          piece_position = [y, x]
          fen = create_fen_for_single_piece(piece_position, piece_name, moving_player)
          expected_moves = generate_king_actions(chess_size, piece_position)
          evaluate_legal_actions(self.game8, fen, piece_position, expected_moves, len(expected_moves))
        
    # Test whether blocking by own pieces is handled correctly 
      y, x = 4, 4
      piece_position = [y, x]
      blocking_piece = "P" if pl == 0 else "p"
      fen = create_fen_for_single_piece(piece_position, piece_name, moving_player)
      expected_moves = generate_king_actions(chess_size, piece_position)
      for y_offset, x_offset in KING_DIRS:
        new_fen = replace_in_fen(fen, blocking_piece, [y + y_offset, x + x_offset])
        new_expected_moves = list(filter(lambda move: move[0] != y + y_offset or move[1] != x + x_offset, expected_moves))
        
        pawn_move_action = 0 if pl == 0 and y_offset == 1 and x_offset == 0 else (0 if pl == 1 and y_offset == -1 and x_offset == 0 else 1)
        evaluate_legal_actions(self.game8, new_fen, piece_position, new_expected_moves, len(new_expected_moves) + pawn_move_action) 
      
    # Test whether blocking by opponents pieces is handled correctly
      blocking_piece = "p" if pl == 0 else "P"
      for y_offset, x_offset in KING_DIRS:
        new_fen = replace_in_fen(fen, blocking_piece, [y + y_offset, x + x_offset]) 
        evaluate_legal_actions(self.game8, new_fen, piece_position, expected_moves, len(expected_moves)) 
        
    print("✓ King moveset test passed")

  def test_knight_moveset(self):
    """Test knight move generation."""
    
    # Test whether each tile has correct moveset
    chess_size = 8
    for pl in range(2):
      piece_name = "N" if pl == 0 else "n"
      moving_player = "w" if pl == 0 else "b"
      for y in range(8):
        for x in range(8): 
          piece_position = [y, x]
          fen = create_fen_for_single_piece(piece_position, piece_name, moving_player)
          expected_moves = generate_knight_actions(chess_size, piece_position)
          evaluate_legal_actions(self.game8, fen, piece_position, expected_moves, len(expected_moves))
        
    # Test whether blocking by own pieces is handled correctly 
      y, x = 3, 4 # We do this, to avoid problems with double move of pawns
      piece_position = [y, x]
      blocking_piece = "K" if pl == 0 else "k"
      fen = create_fen_for_single_piece(piece_position, piece_name, moving_player)
      expected_moves = generate_knight_actions(chess_size, piece_position)
      for y_offset, x_offset in KNIGHT_DIRS:
        new_fen = replace_in_fen(fen, blocking_piece, [y + y_offset, x + x_offset])
        new_expected_moves = list(filter(lambda move: move[0] != y + y_offset or move[1] != x + x_offset, expected_moves))
        evaluate_legal_actions(self.game8, new_fen, piece_position, new_expected_moves, len(new_expected_moves) + 8) 
      
    # Test whether blocking by opponents pieces is handled correctly
      blocking_piece = "k" if pl == 0 else "K" 
      for y_offset, x_offset in KNIGHT_DIRS:
        new_fen = replace_in_fen(fen, blocking_piece, [y + y_offset, x + x_offset])
        evaluate_legal_actions(self.game8, new_fen, piece_position, expected_moves, len(expected_moves)) 
      
    print("✓ Knight moveset test passed")

  def test_pawn_moveset(self):
    """Test pawn move generation."""
    # Position with white pawn that can move forward and capture
    chess_size = 8
    for pl in range(2):
      piece_name = "P" if pl == 0 else "p"
      moving_player = "w" if pl == 0 else "b"
      for y in range(chess_size):
        for x in range(chess_size): 
          piece_position = [y, x]
          fen = create_fen_for_single_piece(piece_position, piece_name, moving_player)
          expected_moves = generate_pawn_actions(chess_size, piece_position, pl)
          expected_moves = list(filter(lambda move: move[1] == x, expected_moves)) # We filter out the attack actions
          evaluate_legal_actions(self.game8, fen, piece_position, expected_moves, len(expected_moves))
          
    # Test whether blocking by own pieces is handled correctly 
      y_pl_offset = (2 * pl) - 1
      blocking_piece = "K" if pl == 0 else "k"
      for y in range(1 - pl, chess_size - pl):
        for x in range(chess_size):
          piece_position = [y, x]
          fen = create_fen_for_single_piece(piece_position, piece_name, moving_player)
          expected_moves = generate_pawn_actions(chess_size, piece_position, pl)
          expected_moves = list(filter(lambda move: move[1] == x, expected_moves)) # We filter out the attack actions 
          if x != 0:
            left_fen = replace_in_fen(fen, blocking_piece, [y + y_pl_offset, x-1]) 
            evaluate_legal_actions(self.game8, left_fen, piece_position, expected_moves, len(expected_moves) + len(generate_king_actions(chess_size, [y + y_pl_offset, x-1])) - 1) # The king is blocked by the pawn in one direction
          if x != chess_size - 1:
            right_fen = replace_in_fen(fen, blocking_piece, [y + y_pl_offset, x+1])
            evaluate_legal_actions(self.game8, right_fen, piece_position, expected_moves, len(expected_moves) + len(generate_king_actions(chess_size, [y + y_pl_offset, x+1])) - 1) # The king is blocked by the pawn in one direction
          if x != 0 and x != chess_size - 1:
            both_fen = replace_in_fen(left_fen, blocking_piece, [y + y_pl_offset, x+1])
            evaluate_legal_actions(self.game8, both_fen, piece_position, expected_moves, len(expected_moves) + len(generate_king_actions(chess_size, [y + y_pl_offset, x-1])) + len(generate_king_actions(chess_size, [y + y_pl_offset, x+1])) - 2) # Two kings are blocked by the pawn in two directions
          up_fen = replace_in_fen(fen, blocking_piece, [y + y_pl_offset, x])
          
          evaluate_legal_actions(self.game8, up_fen, piece_position, [], len(generate_king_actions(chess_size, [y + y_pl_offset, x])) - 1)  # No legal action if blocked
          
      blocking_piece = "k" if pl == 0 else "K"
      for y in range(1 - pl, chess_size - pl):
        for x in range(chess_size):
          piece_position = [y, x]
          fen = create_fen_for_single_piece(piece_position, piece_name, moving_player)
          expected_moves = generate_pawn_actions(chess_size, piece_position, pl)
          if x != 0:
            left_fen = replace_in_fen(fen, blocking_piece, [y + y_pl_offset, x-1])
            left_expected_moves = list(filter(lambda move: move[1] == x or move[1] == x-1, expected_moves)) # We filter out the attack actions
            evaluate_legal_actions(self.game8, left_fen, piece_position, left_expected_moves, len(left_expected_moves))
          if x != chess_size - 1:
            right_fen = replace_in_fen(fen, blocking_piece, [y + y_pl_offset, x+1])
            right_expected_moves = list(filter(lambda move: move[1] == x or move[1] == x+1, expected_moves)) # We filter out the attack actions
            evaluate_legal_actions(self.game8, right_fen, piece_position, right_expected_moves, len(right_expected_moves))
          if x != 0 and x != chess_size - 1:
            both_fen = replace_in_fen(left_fen, blocking_piece, [y + y_pl_offset, x+1]) 
            evaluate_legal_actions(self.game8, both_fen, piece_position, expected_moves, len(expected_moves))  # No legal action if blocked
            
          up_fen = replace_in_fen(fen, blocking_piece, [y + y_pl_offset, x])
          evaluate_legal_actions(self.game8, up_fen, piece_position, [], 0)  # No legal action if blocked
      
      # Test blocking double move
      y = 6 if pl == 0 else 1
      for x in range(chess_size):
        blocking_piece = ["K", "k"]
        piece_position = [y, x] 
        fen = create_fen_for_single_piece(piece_position, piece_name, moving_player)
        blocking_own_fen = replace_in_fen(fen, blocking_piece[pl], [y + 2*y_pl_offset, x])  
        blocking_opponent_fen = replace_in_fen(fen, blocking_piece[1-pl], [y + 2*y_pl_offset, x])  
        blocking_moves = list(filter(lambda move: move[1] == x and move[0] == y + y_pl_offset, generate_pawn_actions(chess_size, piece_position, pl))) 
        king_moves = generate_king_actions(chess_size, [y + 2*y_pl_offset, x])
        evaluate_legal_actions(self.game8, blocking_own_fen, piece_position, blocking_moves, len(blocking_moves) + len(king_moves))
        evaluate_legal_actions(self.game8, blocking_opponent_fen, piece_position, blocking_moves, len(blocking_moves))
    # Test whether blocking move is handled correctly
    
    print("✓ Pawn moveset test passed")

  def test_castling_moveset(self):
    """Test castling move generation."""
    chess_size = 8
    p1_king_castling_fen = "8/8/8/8/8/8/8/4K2R w K - 0 1"
    p1_queen_castling_fen = "8/8/8/8/8/8/8/R3K3 w Q - 0 1"
    p2_king_castling_fen = "4k2r/8/8/8/8/8/8/8 b k - 0 1"
    p2_queen_castling_fen = "r3k3/8/8/8/8/8/8/8 b q - 0 1"
    p1_both_castling_fen = "8/8/8/8/8/8/8/R3K2R w KQ - 0 1"
    p2_both_castling_fen = "r3k2r/8/8/8/8/8/8/8 b kq - 0 1"
    
    p1_king_legals = generate_king_actions(chess_size, [chess_size-1, 4])
    p2_king_legals = generate_king_actions(chess_size, [0, 4])
    p1_left_rook_legals = list(filter(lambda move: move[1] < 4, generate_rook_actions(chess_size, [chess_size-1, 0])))
    p1_right_rook_legals = list(filter(lambda move: move[1] > 4, generate_rook_actions(chess_size, [chess_size-1, 7])))
    p2_left_rook_legals = list(filter(lambda move: move[1] < 4, generate_rook_actions(chess_size, [0, 0])))
    p2_right_rook_legals = list(filter(lambda move: move[1] > 4, generate_rook_actions(chess_size, [0, 7])))
    
    p1_king_castling_state = translate_fen(p1_king_castling_fen)
    p1_queen_castling_state = translate_fen(p1_queen_castling_fen)
    p2_king_castling_state = translate_fen(p2_king_castling_fen)
    p2_queen_castling_state = translate_fen(p2_queen_castling_fen)
    p1_both_castling_state = translate_fen(p1_both_castling_fen)
    p2_both_castling_state = translate_fen(p2_both_castling_fen)
    
    p1_king_castling_legal_actions = self.game8.get_legal_actions(p1_king_castling_state)
    p1_queen_castling_legal_actions = self.game8.get_legal_actions(p1_queen_castling_state)
    p2_king_castling_legal_actions = self.game8.get_legal_actions(p2_king_castling_state)
    p2_queen_castling_legal_actions = self.game8.get_legal_actions(p2_queen_castling_state)
    p1_both_castling_legal_actions = self.game8.get_legal_actions(p1_both_castling_state)
    p2_both_castling_legal_actions = self.game8.get_legal_actions(p2_both_castling_state)
    
    # Checking whether castling is available in FEN
    assert np.sum(p1_king_castling_legal_actions) == len(p1_king_legals) + len(p1_right_rook_legals) + 1
    assert np.sum(p1_queen_castling_legal_actions) == len(p1_king_legals) + len(p1_left_rook_legals) + 1
    assert np.sum(p2_king_castling_legal_actions) == len(p2_king_legals) + len(p2_right_rook_legals) + 1
    assert np.sum(p2_queen_castling_legal_actions) == len(p2_king_legals) + len(p2_left_rook_legals) + 1
    assert np.sum(p1_both_castling_legal_actions) == len(p1_king_legals) + len(p1_left_rook_legals) + len(p1_right_rook_legals) + 2
    assert np.sum(p2_both_castling_legal_actions) == len(p2_king_legals) + len(p2_left_rook_legals) + len(p2_right_rook_legals) + 2
    
    assert p1_king_castling_legal_actions[-4] and not np.any(p1_king_castling_legal_actions[-3:]), "King castling should be available"
    assert p1_queen_castling_legal_actions[-3] and not np.any(p1_queen_castling_legal_actions[-2:]) and not p1_queen_castling_legal_actions[-4], "Queen castling should be available"
    assert p2_king_castling_legal_actions[-2] and not np.any(p2_king_castling_legal_actions[-4:-2]) and not p2_king_castling_legal_actions[-1], "King castling should be available"
    assert p2_queen_castling_legal_actions[-1] and not np.any(p2_queen_castling_legal_actions[-4:-1]), "Queen castling should be available"
    assert np.all(p1_both_castling_legal_actions[-4:2]), "All castling should be available"
    assert np.all(p2_both_castling_legal_actions[-2:]), "All castling should be available"
    
    
    # Castling not available in FEN
    p1_unavailable_castling_fen = "8/8/8/8/8/8/8/R3K2R w - - 0 1"
    p2_unavailable_castling_fen = "r3k2r/8/8/8/8/8/8/8 b - - 0 1"
    p1_unavailable_castling_state = translate_fen(p1_unavailable_castling_fen)
    p2_unavailable_castling_state = translate_fen(p2_unavailable_castling_fen)
    
    p1_unavailable_castling_legal_actions = self.game8.get_legal_actions(p1_unavailable_castling_state)
    p2_unavailable_castling_legal_actions = self.game8.get_legal_actions(p2_unavailable_castling_state)
    
    assert np.sum(p1_unavailable_castling_legal_actions) == len(p1_king_legals) + len(p1_left_rook_legals) + len(p1_right_rook_legals)
    assert np.sum(p2_unavailable_castling_legal_actions) == len(p2_king_legals) + len(p2_left_rook_legals) + len(p2_right_rook_legals)
    
    assert not np.any(p1_unavailable_castling_legal_actions[-4:]), "All castling should be unavailable"
    assert not np.any(p2_unavailable_castling_legal_actions[-4:]), "All castling should be unavailable"
    
    # Checking whether players rook do not block castlign, which should not be possible
    for x in range(chess_size):
      p1_attacking_own_piece_fen = replace_in_fen(p1_both_castling_fen, "R", [0, x])
      p2_attacking_own_piece_fen = replace_in_fen(p2_both_castling_fen, "r", [7, x])
      p1_attacking_own_piece_state = translate_fen(p1_attacking_own_piece_fen)
      p2_attacking_own_piece_state = translate_fen(p2_attacking_own_piece_fen)
      p1_attacking_own_piece_legal_actions = self.game8.get_legal_actions(p1_attacking_own_piece_state)
      p2_attacking_own_piece_legal_actions = self.game8.get_legal_actions(p2_attacking_own_piece_state)
      
      assert np.all(p1_attacking_own_piece_legal_actions[-4:-2]), "Both castling should be available"
      assert np.all(p2_attacking_own_piece_legal_actions[-2:]), "Both castling should be available"
    
    # Checking whether opponent rook blocks castling
    for x in range(chess_size):
      p1_attacking_opponent_piece_fen = replace_in_fen(p1_both_castling_fen, "r", [0, x])
      p2_attacking_opponent_piece_fen = replace_in_fen(p2_both_castling_fen, "R", [7, x])
      p1_attacking_opponent_piece_state = translate_fen(p1_attacking_opponent_piece_fen)
      p2_attacking_opponent_piece_state = translate_fen(p2_attacking_opponent_piece_fen)
      p1_attacking_opponent_piece_legal_actions = self.game8.get_legal_actions(p1_attacking_opponent_piece_state)
      p2_attacking_opponent_piece_legal_actions = self.game8.get_legal_actions(p2_attacking_opponent_piece_state)
      
      if x == 0 or x == 1 or x == 7:
        assert np.all(p1_attacking_opponent_piece_legal_actions[-4:-2]), "Both castling should be available"
        assert np.all(p2_attacking_opponent_piece_legal_actions[-2:]), "Both castling should be available"
      elif x == 2 or x == 3:
        assert p1_attacking_opponent_piece_legal_actions[-4] and not p1_attacking_opponent_piece_legal_actions[-3] and not p1_attacking_opponent_piece_legal_actions[-2] and not p1_attacking_opponent_piece_legal_actions[-1], "King castling should be available"
        assert p2_attacking_opponent_piece_legal_actions[-2] and not p2_attacking_opponent_piece_legal_actions[-1] and not p2_attacking_opponent_piece_legal_actions[-3] and not p2_attacking_opponent_piece_legal_actions[-4], "King castling should be available"
      elif x == 5 or x == 6:
        assert p1_attacking_opponent_piece_legal_actions[-3] and not p1_attacking_opponent_piece_legal_actions[-4] and not p1_attacking_opponent_piece_legal_actions[-2] and not p1_attacking_opponent_piece_legal_actions[-1], "Queen castling should be available"
        assert p2_attacking_opponent_piece_legal_actions[-1] and not p2_attacking_opponent_piece_legal_actions[-2] and not p2_attacking_opponent_piece_legal_actions[-4] and not p2_attacking_opponent_piece_legal_actions[-3], "Queen castling should be available"
      else:
        assert not np.any(p1_attacking_opponent_piece_legal_actions[-4:]), "Both castling should be unavailable"
        assert not np.any(p2_attacking_opponent_piece_legal_actions[-4:]), "Both castling should be unavailable"

    # Checking whether opponent pawn attack blocks castling
    for x in range(chess_size):
      p1_opponent_pawn_fen = replace_in_fen(p1_both_castling_fen, "p", [6, x])
      p2_opponent_pawn_fen = replace_in_fen(p2_both_castling_fen, "P", [1, x])
      p1_opponent_pawn_state = translate_fen(p1_opponent_pawn_fen)
      p2_opponent_pawn_state = translate_fen(p2_opponent_pawn_fen)
      p1_opponent_pawn_legal_actions = self.game8.get_legal_actions(p1_opponent_pawn_state)
      p2_opponent_pawn_legal_actions = self.game8.get_legal_actions(p2_opponent_pawn_state)
      if x == 0:
        assert np.all(p1_opponent_pawn_legal_actions[-4:-2]), "Both castling should be available"
        assert np.all(p2_opponent_pawn_legal_actions[-2:]), "Both castling should be available"
      elif x < 3:
        assert p1_opponent_pawn_legal_actions[-4] and not p1_opponent_pawn_legal_actions[-3] and not p1_opponent_pawn_legal_actions[-2] and not p1_opponent_pawn_legal_actions[-1], "King castling should be available"
        assert p2_opponent_pawn_legal_actions[-2] and not p2_opponent_pawn_legal_actions[-1] and not p2_opponent_pawn_legal_actions[-3] and not p2_opponent_pawn_legal_actions[-4], "King castling should be available"
      elif x > 5:
        assert p1_opponent_pawn_legal_actions[-3] and not p1_opponent_pawn_legal_actions[-4] and not p1_opponent_pawn_legal_actions[-2] and not p1_opponent_pawn_legal_actions[-1], "Queen castling should be available"
        assert p2_opponent_pawn_legal_actions[-1] and not p2_opponent_pawn_legal_actions[-2] and not p2_opponent_pawn_legal_actions[-4] and not p2_opponent_pawn_legal_actions[-3], "Queen castling should be available"
      else:
        assert not np.any(p1_opponent_pawn_legal_actions[-4:]), "Both castling should be unavailable"
        assert not np.any(p2_opponent_pawn_legal_actions[-4:]), "Both castling should be unavailable"
    
    # Checking whether piece in the way blocks castling
    for x in range(1, chess_size - 1):
      if x == 4:
        continue
      p1_block_fen = replace_in_fen(p1_both_castling_fen, "P", [7, x])
      p2_block_fen = replace_in_fen(p2_both_castling_fen, "p", [0, x])
      p1_block_state = translate_fen(p1_block_fen)
      p2_block_state = translate_fen(p2_block_fen)
      p1_block_legal_actions = self.game8.get_legal_actions(p1_block_state)
      p2_block_legal_actions = self.game8.get_legal_actions(p2_block_state)
      if x < 4:
        assert p1_block_legal_actions[-4] and not p1_block_legal_actions[-3] and not p1_block_legal_actions[-2] and not p1_block_legal_actions[-1], "King castling should be available"
        assert p2_block_legal_actions[-2] and not p2_block_legal_actions[-1] and not p2_block_legal_actions[-3] and not p2_block_legal_actions[-4], "King castling should be available"
      else:
        assert p1_block_legal_actions[-3] and not p1_block_legal_actions[-4] and not p1_block_legal_actions[-2] and not p1_block_legal_actions[-1], "Queen castling should be available"
        assert p2_block_legal_actions[-1] and not p2_block_legal_actions[-2] and not p2_block_legal_actions[-4] and not p2_block_legal_actions[-3], "Queen castling should be available"
    
    print("✓ Castling moveset test passed")
    
  def test_check_detection(self):
    """Test check detection in various scenarios."""
    
    # Test 1: King not in check (safe position)
    safe_fen = "8/8/8/8/4K3/8/8/8 w - - 0 1"
    safe_state = translate_fen(safe_fen)
    assert not self.game8.is_in_check(safe_state, 0), "King should not be in check in safe position"
    
    # Test 2: Rook check - horizontal
    rook_check_horizontal_fen = "8/8/8/8/4K2r/8/8/8 w - - 0 1"
    rook_check_horizontal_state = translate_fen(rook_check_horizontal_fen)
    assert self.game8.is_in_check(rook_check_horizontal_state, 0), "King should be in check from horizontal rook"
    
    # Test 3: Rook check - vertical
    rook_check_vertical_fen = "8/8/8/4r3/8/8/8/4K3 w - - 0 1"
    rook_check_vertical_state = translate_fen(rook_check_vertical_fen)
    assert self.game8.is_in_check(rook_check_vertical_state, 0), "King should be in check from vertical rook"
    
    # Test 4: Bishop check - diagonal
    bishop_check_fen = "8/8/8/8/4K3/8/8/7b w - - 0 1"
    bishop_check_state = translate_fen(bishop_check_fen)
    assert self.game8.is_in_check(bishop_check_state, 0), "King should be in check from diagonal bishop"
    
    # Test 5: Queen check - horizontal
    queen_check_horizontal_fen = "8/8/8/8/4K2q/8/8/8 w - - 0 1"
    queen_check_horizontal_state = translate_fen(queen_check_horizontal_fen)
    assert self.game8.is_in_check(queen_check_horizontal_state, 0), "King should be in check from horizontal queen"
    
    # Test 6: Queen check - diagonal
    queen_check_diagonal_fen = "8/8/8/8/4K3/8/8/7q w - - 0 1"
    queen_check_diagonal_state = translate_fen(queen_check_diagonal_fen)
    assert self.game8.is_in_check(queen_check_diagonal_state, 0), "King should be in check from diagonal queen"
    
    # Test 7: Knight check
    knight_check_fen = "8/8/3n4/8/4K3/8/8/8 w - - 0 1"
    knight_check_state = translate_fen(knight_check_fen)
    assert self.game8.is_in_check(knight_check_state, 0), "King should be in check from knight"
    
    # Test 8: Pawn check - white king attacked by black pawn
    pawn_check_fen = "8/8/8/3p4/4K3/8/8/8 w - - 0 1"
    pawn_check_state = translate_fen(pawn_check_fen)
    assert self.game8.is_in_check(pawn_check_state, 0), "King should be in check from pawn"
    
    # Test 9: Blocked rook check (piece in the way)
    blocked_rook_fen = "8/8/8/8/4KP1r/8/8/8 w - - 0 1"
    blocked_rook_state = translate_fen(blocked_rook_fen)
    assert not self.game8.is_in_check(blocked_rook_state, 0), "King should not be in check when rook is blocked"
    
    # Test 10: Blocked bishop check (piece in the way)
    blocked_bishop_fen = "8/8/8/8/4K3/5P2/6b1/8 w - - 0 1"
    blocked_bishop_state = translate_fen(blocked_bishop_fen)
    assert not self.game8.is_in_check(blocked_bishop_state, 0), "King should not be in check when bishop is blocked"
    
    # Test 11: Black king in check from white rook
    black_king_check_fen = "8/8/8/8/4k2R/8/8/8 b - - 0 1"
    black_king_check_state = translate_fen(black_king_check_fen)
    assert self.game8.is_in_check(black_king_check_state, 1), "Black king should be in check from white rook"
    
    # Test 12: Black king not in check
    black_king_safe_fen = "8/8/8/8/4k3/8/8/8 b - - 0 1"
    black_king_safe_state = translate_fen(black_king_safe_fen)
    assert not self.game8.is_in_check(black_king_safe_state, 1), "Black king should not be in check in safe position"
    
    # Test 13: Double check (multiple pieces attacking)
    double_check_fen = "8/8/8/8/4K2r/8/8/7b w - - 0 1"
    double_check_state = translate_fen(double_check_fen)
    assert self.game8.is_in_check(double_check_state, 0), "King should be in check from multiple pieces"
    
    # Test 14: King adjacent to enemy king (should be in check)
    king_vs_king_fen = "8/8/8/8/4Kk2/8/8/8 w - - 0 1"
    king_vs_king_state = translate_fen(king_vs_king_fen)
    assert self.game8.is_in_check(king_vs_king_state, 0), "King should be in check from enemy king"
    
    # Test 15: Pawn check on black king from white pawn
    black_pawn_check_fen = "8/8/8/3k4/4P3/8/8/8 b - - 0 1"
    black_pawn_check_state = translate_fen(black_pawn_check_fen)
    assert self.game8.is_in_check(black_pawn_check_state, 1), "Black king should be in check from white pawn"
    
    # Test 16: Knight check with multiple knights around
    multiple_knights_fen = "8/8/3n4/8/4K3/8/2n5/8 w - - 0 1"
    multiple_knights_state = translate_fen(multiple_knights_fen)
    assert self.game8.is_in_check(multiple_knights_state, 0), "King should be in check from one of the knights"
    
    # Test 17: Complex position - king not in check despite many pieces
    complex_safe_fen = "rn1q1rk1/ppp2pbp/3p1np1/4p3/2PPP3/2N2N2/PP3PPP/R1BQKB1R w KQ - 0 8"
    complex_safe_state = translate_fen(complex_safe_fen)
    assert not self.game8.is_in_check(complex_safe_state, 0), "White king should not be in check in complex position"
    assert not self.game8.is_in_check(complex_safe_state, 1), "Black king should not be in check in complex position"
    
    print("✓ Check detection test passed")


  def test_move_application(self):
    """Test that moves are correctly applied."""
    
    # Test basic piece movements on different board sizes
    for game, board_size in [(self.game8, 8), (self.game6, 6), (self.game4, 4)]:
      # Test with initial state
      state, legal_actions = game.new_initial_state()
      
      # Verify initial state properties
      assert state.current_player == 0, "White should move first"
      assert state.halfmove_clock == 0, "Halfmove clock should start at 0"
      assert state.fullmove_number == 1, "Fullmove number should start at 1"
      assert jnp.all(state.captured_pieces == 0), "No captured pieces in the beginning"
      
      # Test first move application 
      assert jnp.sum(legal_actions) > 0, f"Should have legal moves in initial position for {board_size}x{board_size}"
      
      # Find first legal action and apply it
      first_legal_idx = jnp.argmax(legal_actions)
      original_board = state.board.copy()
      
      new_state, legal_actions_black, reward, terminal = game.apply_action(state, first_legal_idx)
      
      # Verify state transitions
      assert new_state.current_player == 1, "Player should switch to black after white's move"
      assert new_state.fullmove_number == 1, "Fullmove number should still be 1 after white's first move"
      assert not terminal, "Game should not be terminal after first move"
      assert reward == 0, "Reward should be 0 for non-terminal moves"
      
      # Verify board changed
      board_changed = not jnp.array_equal(original_board, new_state.board)
      assert board_changed, "Board should change after applying a move"
      
      # Test that the move actually moved a piece from source to destination
      from_pos = game.action_from[first_legal_idx]
      to_pos = game.action_to[first_legal_idx]
      
      original_piece = original_board[from_pos[0], from_pos[1]]
      new_piece_at_dest = new_state.board[to_pos[0], to_pos[1]]
      new_piece_at_source = new_state.board[from_pos[0], from_pos[1]]
      
      assert original_piece != Pieces.EMPTY, "Source square should have had a piece"
      assert new_piece_at_dest == original_piece, "Piece should have moved to destination"
      assert new_piece_at_source == Pieces.EMPTY, "Source square should now be empty"
      
      # Test black's move 
      assert jnp.sum(legal_actions_black) > 0, "Should have legal moves in initial position for {board_size}x{board_size}" 
      first_black_move = jnp.argmax(legal_actions_black)
      newer_state, newer_legal_actions, reward2, terminal2 = game.apply_action(new_state, first_black_move)
      
      assert newer_state.current_player == 0, "Player should switch back to white"
      assert newer_state.fullmove_number == 2, "Fullmove number should increment after black's move"
      assert not terminal2, "Game should not be terminal after second move"
    
    # Test castling moves specifically
    
    # Test white kingside castling
    white_kingside_fen = "8/8/8/8/8/8/8/4K2R w K - 0 1"
    castling_state = translate_fen(white_kingside_fen)
    legal_actions = self.game8.get_legal_actions(castling_state)
    
    # Castling moves should be at indices -2 (kingside) and -1 (queenside)
    assert legal_actions[-4], "Kingside castling should be legal"
    
    # Apply kingside castling
    original_king_pos = jnp.where(castling_state.board == Pieces.WHITE_KING)
    original_rook_pos = jnp.where(castling_state.board == Pieces.WHITE_ROOK)
    
    assert original_king_pos == (7, 4), "King should be at e1"
    assert original_rook_pos == (7, 7), "Rook should be at h1"
    
    new_state, _, reward, terminal = self.game8.apply_action(castling_state, len(legal_actions) - 4)
    
    # Check that king and rook moved correctly for kingside castling
    new_king_pos = jnp.where(new_state.board == Pieces.WHITE_KING)
    new_rook_pos = jnp.where(new_state.board == Pieces.WHITE_ROOK)
    
    # King should be at g1 (row 7, col 6), rook should be at f1 (row 7, col 5)
    assert new_state.board[7, 6] == Pieces.WHITE_KING, "King should be at g1 after kingside castling"
    assert new_state.board[7, 5] == Pieces.WHITE_ROOK, "Rook should be at f1 after kingside castling"
    assert new_state.board[7, 4] == Pieces.EMPTY, "e1 should be empty after castling"
    assert new_state.board[7, 7] == Pieces.EMPTY, "h1 should be empty after castling"
    assert new_king_pos == (7, 6), "King should be at g1 after kingside castling"
    assert new_rook_pos == (7, 5), "Rook should be at f1 after kingside castling"
    
    # Test white queenside castling
    white_queenside_fen = "8/8/8/8/8/8/8/R3K3 w Q - 0 1"
    castling_state = translate_fen(white_queenside_fen)
    legal_actions = self.game8.get_legal_actions(castling_state)
    
    assert legal_actions[-3], "Queenside castling should be legal"
    
    new_state, _, reward, terminal = self.game8.apply_action(castling_state, len(legal_actions) - 3)
    
    # King should be at c1 (row 7, col 2), rook should be at d1 (row 7, col 3)
    assert new_state.board[7, 2] == Pieces.WHITE_KING, "King should be at c1 after queenside castling"
    assert new_state.board[7, 3] == Pieces.WHITE_ROOK, "Rook should be at d1 after queenside castling"
    assert new_state.board[7, 4] == Pieces.EMPTY, "e1 should be empty after castling"
    assert new_state.board[7, 0] == Pieces.EMPTY, "a1 should be empty after castling"
    
    # # Test black castling
    black_kingside_fen = "4k2r/8/8/8/8/8/8/8 b k - 0 1"
    castling_state = translate_fen(black_kingside_fen)
    legal_actions = self.game8.get_legal_actions(castling_state)
    
    assert legal_actions[-2], "Black kingside castling should be legal"
    
    new_state, _, reward, terminal = self.game8.apply_action(castling_state, len(legal_actions) - 2)
    
    # King should be at g8 (row 0, col 6), rook should be at f8 (row 0, col 5)
    assert new_state.board[0, 6] == Pieces.BLACK_KING, "Black king should be at g8 after kingside castling"
    assert new_state.board[0, 5] == Pieces.BLACK_ROOK, "Black rook should be at f8 after kingside castling"
    assert new_state.board[0, 4] == Pieces.EMPTY, "e8 should be empty after castling"
    assert new_state.board[0, 7] == Pieces.EMPTY, "h8 should be empty after castling"
    
    black_queenside_fen = "r3k3/8/8/8/8/8/8/8 b q - 0 1"
    castling_state = translate_fen(black_queenside_fen)
    legal_actions = self.game8.get_legal_actions(castling_state)
    
    assert legal_actions[-1], "Black queenside castling should be legal"
    
    new_state, _, reward, terminal = self.game8.apply_action(castling_state, len(legal_actions) - 1)
    
    # King should be at g8 (row 0, col 6), rook should be at f8 (row 0, col 5)
    assert new_state.board[0, 2] == Pieces.BLACK_KING, "Black king should be at c8 after queenside castling"
    assert new_state.board[0, 3] == Pieces.BLACK_ROOK, "Black rook should be at d8 after queenside castling"
    assert new_state.board[0, 4] == Pieces.EMPTY, "e8 should be empty after castling"
    assert new_state.board[0, 1] == Pieces.EMPTY, "b8 should be empty after castling"
    assert new_state.board[0, 0] == Pieces.EMPTY, "a8 should be empty after castling"
    
    # Test pawn promotion (if we can find a position) 
    white_promotion_fen = "k7/4P3/8/8/8/8/8/4K3 w - - 0 1"
    white_promotion_state = translate_fen(white_promotion_fen)
    legal_actions = self.game8.get_legal_actions(white_promotion_state)
    assert jnp.sum(legal_actions) > 0, f"Should have legal moves in initial position for {board_size}x{board_size}"
    
    # Find a pawn promotion move
    pawn_moves, _ = find_moves_for_piece(self.game8, white_promotion_state, legal_actions, [1, 4])
    assert len(pawn_moves) > 0, "Should have pawn promotion moves"
    # Apply first promotion move
    for promote_index, pawn_move in enumerate(pawn_moves):
      new_state, _, reward, terminal = self.game8.apply_action(white_promotion_state, pawn_move)
      
      # Check that pawn is gone and a promoted piece is on the 8th rank
      assert new_state.board[1, 4] == Pieces.EMPTY, "Pawn should be gone from source square" 
      assert new_state.board[0, 4] == PROMOTION_PIECES[promote_index], "Should have promoted to the correct piece" 
      assert reward == 0, "Reward should be 0 for non-terminal moves"
      assert not terminal, "Game should not be terminal after promotion"
      
    black_promotion_fen = "4k3/8/8/8/8/8/4p3/K7 b - - 0 1"
    black_promotion_state = translate_fen(black_promotion_fen)
    legal_actions = self.game8.get_legal_actions(black_promotion_state)
    assert jnp.sum(legal_actions) > 0, f"Should have legal moves in initial position for {board_size}x{board_size}"
    
    # Find a pawn promotion move
    pawn_moves, _ = find_moves_for_piece(self.game8, black_promotion_state, legal_actions, [6, 4])
    assert len(pawn_moves) > 0, "Should have pawn promotion moves"
    # Apply first promotion move
    for promote_index, pawn_move in enumerate(pawn_moves):
      new_state, _, reward, terminal = self.game8.apply_action(black_promotion_state, pawn_move)
      
      # Check that pawn is gone and a promoted piece is on the 8th rank
      assert new_state.board[6, 4] == Pieces.EMPTY, "Pawn should be gone from source square"
      assert new_state.board[7, 4] == PROMOTION_PIECES[promote_index] + DISTINCT_PIECES_PER_PLAYER, "Should have promoted to the correct piece" 
    
    # Test that halfmove clock increments correctly
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    test_state = translate_fen(test_fen)
    legal_actions = self.game8.get_legal_actions(test_state)
    
    # Find a pawn move (should reset halfmove clock)
    pawn_moves, _ = find_moves_for_piece(self.game8, test_state, legal_actions, [6, 4])  # e2 pawn 
    new_state, _, _, _ = self.game8.apply_action(test_state, pawn_moves[0])
    assert new_state.halfmove_clock == 0, "Pawn move should reset halfmove clock"
    
    # Test different board sizes have working move application
    for game, board_size in [(self.game6, 6), (self.game4, 4)]:
      state, _ = game.new_initial_state()
      legal_actions = game.get_legal_actions(state)
  
      for move_idx in jnp.where(legal_actions)[0]:
        original_board = state.board.copy()
        new_state, _, reward, terminal = game.apply_action(state, move_idx)
        
        # Basic checks
        assert new_state.current_player != state.current_player, f"Player should switch on {board_size}x{board_size}"
        assert not jnp.array_equal(original_board, new_state.board), f"Board should change on {board_size}x{board_size}"
        assert not terminal, f"Game should not be terminal after first move on {board_size}x{board_size}"
        assert reward == 0, "Reward should be 0 for non-terminal moves"
        # Test that piece actually moved
        from_pos = game.action_from[move_idx]
        to_pos = game.action_to[move_idx]
        
        original_piece = original_board[from_pos[0], from_pos[1]]
        assert new_state.board[to_pos[0], to_pos[1]] == original_piece, f"Piece should move to destination on {board_size}x{board_size}"
        assert new_state.board[from_pos[0], from_pos[1]] == Pieces.EMPTY, f"Source should be empty after move on {board_size}x{board_size}"
    
    print("✓ Move application test passed")

  def test_piece_capture_mechanics(self):
    """Test piece capture mechanics and captured_pieces tracking."""
    
    # Test case 1: Simple piece capture - pawn takes pawn
    capture_fen = "7k/8/8/3p4/4P3/8/8/7K w - - 0 1"
    capture_state = translate_fen(capture_fen)
    legal_actions = self.game8.get_legal_actions(capture_state)
     
    # Find the capture move (should be to d5)
    capture_move_idx = find_action_with_destination(self.game8, capture_state, legal_actions, [4, 4], [3, 3]) 

    assert capture_move_idx is not None, "Should find capture move"
    
    # Before capture - verify initial state
    assert capture_state.board[3, 3] == Pieces.BLACK_PAWN, "Black pawn should be at d4"
    assert capture_state.board[4, 4] == Pieces.WHITE_PAWN, "White pawn should be at e4"
    assert jnp.sum(capture_state.captured_pieces) == 0, "No pieces should be captured initially"
    
    # Apply the capture
    new_state, _, reward, terminal = self.game8.apply_action(capture_state, capture_move_idx)
    
    # Verify piece was captured and removed from board
    assert new_state.board[3, 3] == Pieces.WHITE_PAWN, "White pawn should now be at d4"
    assert new_state.board[4, 4] == Pieces.EMPTY, "e4 should now be empty"
    assert terminal == False, "Game should not be terminal"
    assert reward == 0, "Reward should be 0 for non-terminal moves"
    
    # Verify captured_pieces array is updated correctly
    # Black pawn has piece ID 7, so index 6 in captured_pieces array (7-1)
    expected_captured = jnp.zeros(len(PIECE_TO_ID), dtype=jnp.int32)
    expected_captured = expected_captured.at[Pieces.BLACK_PAWN - 1].set(1)
    assert jnp.array_equal(new_state.captured_pieces, expected_captured), "Captured pieces should be tracked correctly"
    
    # Test case 2: Multiple piece captures
    multi_capture_fen = "7k/8/2r5/3P4/8/8/8/7K w - - 0 1"
    multi_capture_fen = replace_in_fen(multi_capture_fen, "n", [0, 1])
    multi_capture_fen = replace_in_fen(multi_capture_fen, "Q", [2, 7])
    multi_state = translate_fen(multi_capture_fen)
    legal_actions = self.game8.get_legal_actions(multi_state)
    
    # White pawn at d5 can capture black rook at c6
    pawn_moves, pawn_destinations = find_moves_for_piece(self.game8, multi_state, legal_actions, [3, 3])  # d5
    
    # Find the capture move (should be to c6)
    rook_capture_idx = find_action_with_destination(self.game8, multi_state, legal_actions, [3, 3], [2, 2])
    
    
    assert rook_capture_idx is not None, "Should find rook capture move"
    
    # Apply the rook capture
    new_state, _, reward, terminal = self.game8.apply_action(multi_state, rook_capture_idx)
    
    # Verify rook was captured
    assert new_state.board[2, 2] == Pieces.WHITE_PAWN, "White pawn should now be at c6"
    assert new_state.board[3, 3] == Pieces.EMPTY, "d5 should now be empty"
    assert terminal == False, "Game should not be terminal"
    assert reward == 0, "Reward should be 0 for non-terminal moves"
    
    # Verify captured_pieces array shows captured rook
    # Black rook has piece ID 8, so index 7 in captured_pieces array
    expected_captured = jnp.zeros(len(PIECE_TO_ID), dtype=jnp.int32)
    expected_captured = expected_captured.at[Pieces.BLACK_ROOK - 1].set(1)
    assert jnp.array_equal(new_state.captured_pieces, expected_captured), "Captured rook should be tracked"
    
    # Test case 3: Capture accumulation - apply multiple captures to same state
    # Start with a state that already has one captured piece 

    # Knight captures the moved pawn to c3
    
    legal_actions = self.game8.get_legal_actions(new_state)
    knight_moves, knight_destinations = find_moves_for_piece(self.game8, new_state, legal_actions, [0, 1])  # a5
    
    # Queen captures knight at b6
    pawn_capture_idx = find_action_with_destination(self.game8, new_state, legal_actions, [0, 1], [2, 2])
    
    assert pawn_capture_idx is not None, "Should find knight capture move"
    
    new_state, _, reward, terminal = self.game8.apply_action(new_state, pawn_capture_idx)
    
    # Verify knight was captured and queen moved
    assert new_state.board[2, 2] == Pieces.BLACK_KNIGHT, "White queen should now be at b6"
    assert new_state.board[3, 0] == Pieces.EMPTY, "a5 should now be empty"
    assert terminal == False, "Game should not be terminal"
    assert reward == 0, "Reward should be 0 for non-terminal moves"
    
    # Verify captured_pieces array shows both captured pieces  
    expected_captured = expected_captured.at[Pieces.WHITE_PAWN - 1].set(1)  # New capture
    assert jnp.array_equal(new_state.captured_pieces, expected_captured), "Both captured pieces should be tracked"
    
    
    legal_actions = self.game8.get_legal_actions(new_state)
    queen_moves, queen_destinations = find_moves_for_piece(self.game8, new_state, legal_actions, [2, 7])  # a5
    
    # Queen captures knight at b6
    knight_capture_idx = find_action_with_destination(self.game8, new_state, legal_actions, [2, 7], [2, 2])  
    assert knight_capture_idx is not None, "Should find knight capture move"
    
    new_state, _, reward, terminal = self.game8.apply_action(new_state, knight_capture_idx)
    
    # Verify knight was captured and queen moved
    assert new_state.board[2, 2] == Pieces.WHITE_QUEEN, "White queen should now be at b6"
    assert new_state.board[3, 0] == Pieces.EMPTY, "a5 should now be empty"
    assert terminal == False, "Game should not be terminal"
    assert reward == 0, "Reward should be 0 for non-terminal moves"
    
    # Verify captured_pieces array shows both captured pieces  
    expected_captured = expected_captured.at[Pieces.BLACK_KNIGHT - 1].set(1)  # New capture
    assert jnp.array_equal(new_state.captured_pieces, expected_captured), "Both captured pieces should be tracked"
    
    print("✓ Piece capture mechanics test passed")

  def test_king_capture_game_termination(self):
    """Test that the game ends when a king is captured."""
    
    # Test case 1: White captures black king
    white_wins_fen = "K7/8/8/8/8/3Q4/8/5k2 w - - 0 1"
    white_wins_state = translate_fen(white_wins_fen)
    legal_actions = self.game8.get_legal_actions(white_wins_state)
     
    
    # Find the king capture move
    king_capture_idx = find_action_with_destination(self.game8, white_wins_state, legal_actions, [5, 3], [7, 5])
     
    # Verify king is actually there before capture
    assert white_wins_state.board[7, 5] == Pieces.BLACK_KING, "Black king should be at e1"
    
    # Apply the king capture
    new_state, new_legal_actions, reward, terminal = self.game8.apply_action(white_wins_state, king_capture_idx)
    
    # Verify game termination
    assert terminal == True, "Game should be terminal after king capture"
    assert reward == 1, "White should get reward of 1 for capturing black king"
    
    # Verify king was captured and removed from board
    assert new_state.board[7, 5] == Pieces.WHITE_QUEEN, "White queen should now be at e1"
    assert new_state.board[5, 3] == Pieces.EMPTY, "d3 should now be empty"
    
    # Verify captured king is tracked in captured_pieces
    expected_captured = jnp.zeros(len(PIECE_TO_ID), dtype=jnp.int32)
    expected_captured = expected_captured.at[Pieces.BLACK_KING - 1].set(1)
    assert jnp.array_equal(new_state.captured_pieces, expected_captured), "Captured black king should be tracked"
    
    # Test case 2: Black captures white king
    black_wins_fen = "4r2k/8/8/8/8/8/8/4K3 b - - 0 1"
    black_wins_state = translate_fen(black_wins_fen)
    legal_actions = self.game8.get_legal_actions(black_wins_state)
     
    # Find the king capture move
    king_capture_idx = find_action_with_destination(self.game8, black_wins_state, legal_actions, [0, 4], [7, 4])
    
    assert king_capture_idx is not None, "Should find white king capture move"
    
    # Verify white king is there
    assert black_wins_state.board[7, 4] == Pieces.WHITE_KING, "White king should be at e1"
    
    # Apply the king capture
    new_state, new_legal_actions, reward, terminal = self.game8.apply_action(black_wins_state, king_capture_idx)
    
    # Verify game termination
    assert terminal == True, "Game should be terminal after white king capture"
    assert reward == -1, "Black should get reward of -1 (from white's perspective, since reward is calculated from the perspective of the player who just moved)"
    
    # Verify king was captured
    assert new_state.board[7, 4] == Pieces.BLACK_ROOK, "Black rook should now be at e1"
    assert new_state.board[0, 4] == Pieces.EMPTY, "e8 should now be empty"
    
    # Verify captured white king is tracked
    expected_captured = jnp.zeros(len(PIECE_TO_ID), dtype=jnp.int32)
    expected_captured = expected_captured.at[Pieces.WHITE_KING - 1].set(1)
    assert jnp.array_equal(new_state.captured_pieces, expected_captured), "Captured white king should be tracked"
    
    # Test case 3: Verify that no legal moves are generated after king capture
    # (The game should be terminal, so this is more of a consistency check)
    assert jnp.sum(new_legal_actions) == 0 or terminal, "No moves should be legal after game ends or game should be terminal"
    
    # Test case 4: Multiple pieces on board, but king capture still ends game
    complex_board_fen = "r1bqk2r/pppp1ppp/2n2n2/2b5/2B5/3P1N2/PPP2PPP/RNBKQ2R w KQkq - 0 1"
    complex_state = translate_fen(complex_board_fen)
     
    
    legal_actions = self.game8.get_legal_actions(complex_state)
    king_capture_idx = find_action_with_destination(self.game8, complex_state, legal_actions, [7, 4], [0, 4]) 
    
    assert king_capture_idx is not None, "Should find king capture move" 
    new_state, _, reward, terminal = self.game8.apply_action(complex_state, king_capture_idx)
    assert terminal == True, "Game should end even with many pieces on board when king is captured"
    assert reward == 1, "Reward should be 1 for white capturing black king"
    
    print("✓ King capture game termination test passed")

  def test_state_tensor(self): 
    init_state8, _ = self.game8.new_initial_state()
    state8 = self.game8.state_tensor(init_state8) 
    assert state8.shape == (16, 8, 8)
    assert jnp.sum(state8[0, :, :]) == 32, "There should 32 empty tiles in 8x8 initial state"
    for pl in range(2):
      pl_offset = pl * DISTINCT_PIECES_PER_PLAYER
      assert jnp.sum(state8[1 + pl_offset, :, :]) == 8, "There should be 8 pawns in 8x8 initial state for player " + str(pl)
      assert jnp.sum(state8[2 + pl_offset, :, :]) == 2, "There should be 2 rooks in 8x8 initial state for player " + str(pl)
      assert jnp.sum(state8[3 + pl_offset, :, :]) == 2, "There should be 2 knights in 8x8 initial state for player " + str(pl)
      assert jnp.sum(state8[4 + pl_offset, :, :]) == 2, "There should be 2 bishops in 8x8 initial state for player " + str(pl)
      assert jnp.sum(state8[5 + pl_offset, :, :]) == 1, "There should be 1 queen in 8x8 initial state for player " + str(pl)
      assert jnp.sum(state8[6 + pl_offset, :, :]) == 1, "There should be 1 king in 8x8 initial state for player " + str(pl)
    assert jnp.all(state8[13, :, ::2] == 0), "Initial Castling rights should be available"
    assert jnp.all(state8[13, :, 1::2] == 1), "Initial Castling rights should be available"
    assert jnp.all(state8[14, :, 0] == 0), "Half moves are 0 at the beginning"
    assert jnp.allclose(state8[14, :, 1], 1/100), "Full moves are 1 at the beginning"
    assert jnp.all(state8[14, :, 2:] == 0), "Padding should be 0"
    assert jnp.all(state8[15, :, 0] == 1), "Current player should be 0"
    assert jnp.all(state8[15, :, 1:] == 0), "Current player should be 0"
    
    init_state6, _ = self.game6.new_initial_state()
    state6 = self.game6.state_tensor(init_state6)
    assert state6.shape == (15, 6, 6)
    assert jnp.sum(state6[0, :, :]) == 12, "There should 12 empty tiles in 6x6 initial state"
    for pl in range(2):
      pl_offset = pl * DISTINCT_PIECES_PER_PLAYER
      assert jnp.sum(state6[1 + pl_offset, :, :]) == 6, "There should be 6 pawns in 6x6 initial state for player " + str(pl)
      assert jnp.sum(state6[2 + pl_offset, :, :]) == 2, "There should be 2 rooks in 6x6 initial state for player " + str(pl)
      assert jnp.sum(state6[3 + pl_offset, :, :]) == 2, "There should be 2 knights in 6x6 initial state for player " + str(pl)
      assert jnp.sum(state6[4 + pl_offset, :, :]) == 0, "There should be 0 bishops in 6x6 initial state for player " + str(pl)
      assert jnp.sum(state6[5 + pl_offset, :, :]) == 1, "There should be 1 queen in 6x6 initial state for player " + str(pl)
      assert jnp.sum(state6[6 + pl_offset, :, :]) == 1, "There should be 1 king in 6x6 initial state for player " + str(pl) 
    assert jnp.all(state6[13, :, 0] == 0), "Half moves are 0 at the beginning"
    assert jnp.allclose(state6[13, :, 1], 1/100), "Full moves are 1 at the beginning"
    assert jnp.all(state6[13, :, 2:] == 0), "Padding should be 0"
    assert jnp.all(state6[14, :, 0] == 1), "Current player should be 0"
    assert jnp.all(state6[14, :, 1:] == 0), "Current player should be 0"
    
    init_state4, _ = self.game4.new_initial_state()
    state4 = self.game4.state_tensor(init_state4)
    assert state4.shape == (15, 4, 4)
    assert jnp.sum(state4[0, :, :]) == 0, "There should be no empty tiles in 4x4 initial state"
    for pl in range(2):
      pl_offset = pl * DISTINCT_PIECES_PER_PLAYER
      assert jnp.sum(state4[1 + pl_offset, :, :]) == 4, "There should be 4 pawns in 4x4 initial state for player " + str(pl)
      assert jnp.sum(state4[2 + pl_offset, :, :]) == 1, "There should be 1 rooks in 4x4 initial state for player " + str(pl)
      assert jnp.sum(state4[3 + pl_offset, :, :]) == 0, "There should be 0 knights in 4x4 initial state for player " + str(pl)
      assert jnp.sum(state4[4 + pl_offset, :, :]) == 1, "There should be 1 bishops in 4x4 initial state for player " + str(pl)
      assert jnp.sum(state4[5 + pl_offset, :, :]) == 1, "There should be 1 queens in 4x4 initial state for player " + str(pl)
      assert jnp.sum(state4[6 + pl_offset, :, :]) == 1, "There should be 1 kings in 4x4 initial state for player " + str(pl)
    assert jnp.all(state4[13, :, 0] == 0), "Half moves are 0 at the beginning"
    assert jnp.allclose(state4[13, :, 1], 1/100), "Full moves are 1 at the beginning"
    assert jnp.all(state4[13, :, 2:] == 0), "Padding should be 0"
    assert jnp.all(state4[14, :, 0] == 1), "Current player should be 0"
    assert jnp.all(state4[14, :, 1:] == 0), "Current player should be 0"
    
    
    for castling_index, castling_option in enumerate("KQkq"):
      init_fen_castling = f"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w {castling_option} - 0 1"
      init_state_castling = translate_fen(init_fen_castling)
      state_castling = self.game8.state_tensor(init_state_castling)
      assert jnp.all(state_castling[13, castling_index, ::2] == 0), f"Castling {castling_option} should be available"
      assert jnp.all(state_castling[13, castling_index + 4, ::2] == 0), f"Castling {castling_option} should be available"
      assert jnp.all(state_castling[13, castling_index, 1::2] == 1), f"Castling {castling_option} should be available"
      assert jnp.all(state_castling[13, castling_index + 4, 1::2] == 1), f"Castling {castling_option} should be available"
      for i in range(8):
        if i % 4 == castling_index:
          continue
        assert jnp.all(state_castling[13, i, ::2] == 1), f"Other than castling {castling_option} should be unavailable"
        assert jnp.all(state_castling[13, i, 1::2] == 0), f"Other than castling {castling_option} should be unavailable" 
    
    print("✓ State tensor test passed")

  def test_observation_tensor(self):
    init_state8, _ = self.game8.new_initial_state()
    for player in range(2):
      player_offset = player * DISTINCT_PIECES_PER_PLAYER
      opponent_offset = (1 - player) * DISTINCT_PIECES_PER_PLAYER
      state8 = self.game8.observation_tensor(init_state8, player) 
      assert state8.shape == (18, 8, 8)
      assert jnp.sum(state8[0, :, :]) == 16, "There should 16 empty visible tiles in 8x8 initial state"  
      assert jnp.sum(state8[1 + player_offset, :, :]) == 8, f"There should be 8 pawns in 8x8 initial state for player {player}"
      assert jnp.sum(state8[2 + player_offset, :, :]) == 2, f"There should be 2 rooks in 8x8 initial state for player {player}"
      assert jnp.sum(state8[3 + player_offset, :, :]) == 2, f"There should be 2 knights in 8x8 initial state for player {player}" 
      assert jnp.sum(state8[4 + player_offset, :, :]) == 2, f"There should be 2 bishops in 8x8 initial state for player {player}"
      assert jnp.sum(state8[5 + player_offset, :, :]) == 1, f"There should be 1 queen in 8x8 initial state for player {player}"
      assert jnp.sum(state8[6 + player_offset, :, :]) == 1, f"There should be 1 king in 8x8 initial state for player {player}"
      assert jnp.sum(state8[1+opponent_offset:7+opponent_offset, :, :]) == 0, f"There should be no visible pieces of the opponent in 8x8 initial state for player {player}"
      assert jnp.sum(state8[13, :, :]) == 32, f"There should be 32 non-visible tiles for player {player}"
      assert jnp.all(state8[14, :, :] == 0), "No pieces captured initially"
      assert jnp.all(state8[15, :, ::2] == 0), "Castling rights should be available"
      assert jnp.all(state8[15, :, 1::2] == 1), "Castling rights should be available"
      assert jnp.all(state8[16, :, 0] == 0), "Half moves are 0 at the beginning"
      assert jnp.allclose(state8[16, :, 1], 1/100), "Full moves are 1 at the beginning"
      assert jnp.all(state8[17, :, 0] == 1), f"Current player should be 0"
      assert jnp.all(state8[17, :, 1] == 0), f"Current player should be 0"
      assert jnp.all(state8[17, :, 2+player] == 1), f"Asking player should be {player}" 
      assert jnp.all(state8[17, :, 2+(1 - player)] == 0), f"Asking player should be {player}"
      assert jnp.all(state8[17, :, 4:] == 0), "Padding"
       
    init_state6, _ = self.game6.new_initial_state()
    for player in range(2):
      player_offset = player * DISTINCT_PIECES_PER_PLAYER
      opponent_offset = (1 - player) * DISTINCT_PIECES_PER_PLAYER
      state6 = self.game6.observation_tensor(init_state6, player) 
      assert state6.shape == (17, 6, 6)
      assert jnp.sum(state6[0, :, :]) == 12, "There should 12 empty visible tiles in 6x6 initial state"  
      assert jnp.sum(state6[1 + player_offset, :, :]) == 6, f"There should be 6 pawns in 6x6 initial state for player {player}"
      assert jnp.sum(state6[2 + player_offset, :, :]) == 2, f"There should be 2 rooks in 6x6 initial state for player {player}"
      assert jnp.sum(state6[3 + player_offset, :, :]) == 2, f"There should be 2 knights in 6x6 initial state for player {player}" 
      assert jnp.sum(state6[4 + player_offset, :, :]) == 0, f"There should be 0 bishops in 6x6 initial state for player {player}"
      assert jnp.sum(state6[5 + player_offset, :, :]) == 1, f"There should be 1 queen in 6x6 initial state for player {player}"
      assert jnp.sum(state6[6 + player_offset, :, :]) == 1, f"There should be 1 king in 6x6 initial state for player {player}"
      assert jnp.sum(state6[1+opponent_offset:7+opponent_offset, :, :]) == 0, f"There should be no visible pieces of the opponent in 6x6 initial state for player {player}"
      assert jnp.sum(state6[13, :, :]) == 12, f"There should be 12 non-visible tiles for player {player}"
      assert jnp.all(state6[14, :, :] == 0), "No pieces captured initially"
      assert jnp.all(state6[15, :, 0] == 0), "Half moves are 0 at the beginning"
      assert jnp.allclose(state6[15, :, 1], 1/100), "Full moves are 1 at the beginning"
      assert jnp.all(state6[16, :, 0] == 1), f"Current player should be 0"
      assert jnp.all(state6[16, :, 1] == 0), f"Current player should be 0"
      assert jnp.all(state6[16, :, 2+player] == 1), f"Asking player should be {player}"
      assert jnp.all(state6[16, :, 2+(1 - player)] == 0), f"Asking player should be {player}"
      assert jnp.all(state6[16, :, 4:] == 0), "Padding"
      
    init_state4, _ = self.game4.new_initial_state()
    for player in range(2):
      player_offset = player * DISTINCT_PIECES_PER_PLAYER
      opponent_offset = (1 - player) * DISTINCT_PIECES_PER_PLAYER
      state4 = self.game4.observation_tensor(init_state4, player) 
      assert state4.shape == (18, 4, 4)
      assert jnp.sum(state4[0, :, :]) == 0, "There should be no empty visible tiles in 4x4 initial state"
      assert jnp.sum(state4[1 + player_offset, :, :]) == 4, f"There should be 4 pawns in 4x4 initial state for player {player}"
      assert jnp.sum(state4[2 + player_offset, :, :]) == 1, f"There should be 1 rook in 4x4 initial state for player {player}"
      assert jnp.sum(state4[3 + player_offset, :, :]) == 0, f"There should be 0 knights in 4x4 initial state for player {player}"
      assert jnp.sum(state4[4 + player_offset, :, :]) == 1, f"There should be 1 bishop in 4x4 initial state for player {player}"
      assert jnp.sum(state4[5 + player_offset, :, :]) == 1, f"There should be 1 queen in 4x4 initial state for player {player}"
      assert jnp.sum(state4[6 + player_offset, :, :]) == 1, f"There should be 1 king in 4x4 initial state for player {player}"
      assert jnp.sum(state4[1 + opponent_offset, :, :]) == 4, f"There should be 4 pawns in 4x4 initial state for opponent {1 - player}"
      assert jnp.sum(state4[2+opponent_offset:7+opponent_offset, :, :]) == 0, f"There should be no visible pieces of the opponent in 4x4 initial state for player {player}"
      assert jnp.sum(state4[13, :, :]) == 4, f"There should be 12 non-visible tiles for player {player}"
      assert jnp.all(state4[14:16, :, :] == 0), "No pieces captured initially" 
      assert jnp.all(state4[16, :, 0] == 0), "Half moves are 0 at the beginning"
      assert jnp.allclose(state4[16, :, 1], 1/100), "Full moves are 1 at the beginning"
      assert jnp.all(state4[17, :, 0] == 1), f"Current player should be 0"
      assert jnp.all(state4[17, :, 1] == 0), f"Current player should be 0"
      assert jnp.all(state4[17, :, 2+player] == 1), f"Asking player should be {player}"
      assert jnp.all(state4[17, :, 2+(1 - player)] == 0), f"Asking player should be {player}"
    
    
    print("✓ Observation tensor test passed")

  def test_observation_rook(self):
    white_rook_visibility_fen = "4p3/4P3/8/8/4Rpp1/8/8/kqrn1bnr w - - 0 1"
    white_rook_visibility_state = translate_fen(white_rook_visibility_fen) 
    white_rook_visibility_iset = self.game8.observation_tensor(white_rook_visibility_state, 0)
    assert white_rook_visibility_iset[1, 1, 4] == 1, "We see our pawn"
    assert white_rook_visibility_iset[2, 4, 4] == 1, "We see our rook"
    assert jnp.all(white_rook_visibility_iset[0, 4, :4] == 1), "Rook should see left"
    assert jnp.all(white_rook_visibility_iset[0, 5:, 4] == 1), "Rook should see down"
    assert jnp.all(white_rook_visibility_iset[0, 2:4, 4] == 1), "Rook should partially see up"

    assert white_rook_visibility_iset[7, 4, 5] == 1, "Rook should see opponent pawn"
    assert white_rook_visibility_iset[13, 0, 4] == 1, "Rook should not see opponent pawn"
    assert jnp.all(white_rook_visibility_iset[13, 4, 6:] == 1), "Rook should not see beyond opponent pawn"
        
    assert jnp.sum(white_rook_visibility_iset[0]) == 9, "Rook should see 9 tiles"
    assert jnp.sum(white_rook_visibility_iset[1]) == 1, "Player has 1 pawn"
    assert jnp.sum(white_rook_visibility_iset[2]) == 1, "Player has 1 rook"
    assert jnp.sum(white_rook_visibility_iset[3:7]) == 0, "Player has 0 other pieces"
    assert jnp.sum(white_rook_visibility_iset[7]) == 1, "We see one opponent pawn"
    assert jnp.sum(white_rook_visibility_iset[8:13]) == 0, "No other seen pieces"
    assert jnp.sum(white_rook_visibility_iset[13]) == (64 - 12), "There should be 64 - 12 non-visible tiles"
    
    black_rook_visibility_fen = "3P4/3p4/8/8/1PPr4/8/8/KQR1NBNR b - - 0 1"
    black_rook_visibility_state = translate_fen(black_rook_visibility_fen)
    black_rook_visibility_iset = self.game8.observation_tensor(black_rook_visibility_state, 1)
    
    assert black_rook_visibility_iset[7, 1, 3] == 1, "We see our pawn"
    assert black_rook_visibility_iset[8, 4, 3] == 1, "We see our rook"
    assert jnp.all(black_rook_visibility_iset[0, 4, 4:] == 1), "Rook should see right"
    assert jnp.all(black_rook_visibility_iset[0, 5:, 3] == 1), "Rook should see down"
    assert jnp.all(black_rook_visibility_iset[0, 2:4, 3] == 1), "Rook should partially see up" 
    assert black_rook_visibility_iset[1, 4, 2] == 1, "Rook should see opponent pawn"
    assert black_rook_visibility_iset[13, 4, 1] == 1, "Rook should not see opponent pawn"
    assert jnp.all(black_rook_visibility_iset[13, 4, 0] == 1), "Rook should not see beyond opponent pawn"
    assert jnp.sum(black_rook_visibility_iset[0]) == 9, "Rook should see 9 tiles"
    assert jnp.sum(black_rook_visibility_iset[1]) == 1, "Player has 1 pawn"
    assert jnp.sum(black_rook_visibility_iset[2:7]) == 0, "Player has 0 other pieces"
    assert jnp.sum(black_rook_visibility_iset[7]) == 1, "Player has 1 rook"
    assert jnp.sum(black_rook_visibility_iset[8]) == 1, "No other seen pieces"
    assert jnp.sum(black_rook_visibility_iset[9:13]) == 0, "There should be 64 - 12 non-visible tiles"
    assert jnp.sum(black_rook_visibility_iset[13]) == (64 - 12), "There should be 64 - 12 non-visible tiles"
        
    print("✓ Observation rook test passed")
  
  def test_observation_bishop(self): 
    white_bishop_visibility_fen = "8/8/8/8/8/8/8/knrqbpr1 w - - 0 1"
    white_bishop_visibility_fen = replace_in_fen(white_bishop_visibility_fen, "B", [4, 4])
    white_bishop_visibility_fen = replace_in_fen(white_bishop_visibility_fen, "P", [1, 1])
    white_bishop_visibility_fen = replace_in_fen(white_bishop_visibility_fen, "p", [0, 1])
    white_bishop_visibility_fen = replace_in_fen(white_bishop_visibility_fen, "p", [5, 3])
    white_bishop_visibility_fen = replace_in_fen(white_bishop_visibility_fen, "p", [6, 2])
    
    white_bishop_visibility_state = translate_fen(white_bishop_visibility_fen)
    white_bishop_visibility_iset = self.game8.observation_tensor(white_bishop_visibility_state, 0)
    
    assert white_bishop_visibility_iset[1, 1, 1] == 1, "We see our pawn"
    assert white_bishop_visibility_iset[4, 4, 4] == 1, "We see our bishop"
    assert white_bishop_visibility_iset[7, 5, 3] == 1, "Bishop sees opponents pawn"
    assert white_bishop_visibility_iset[13, 0, 0] == 1, "We see not see beyond pawn"
    assert white_bishop_visibility_iset[0, 2, 2] == 1, "Bishop should see left up"
    assert white_bishop_visibility_iset[0, 3, 3] == 1, "Bishop should see left up"
    assert white_bishop_visibility_iset[0, 5, 5] == 1, "Bishop should see right down"
    assert white_bishop_visibility_iset[0, 6, 6] == 1, "Bishop should see right down"
    assert white_bishop_visibility_iset[0, 7, 7] == 1, "Bishop should see right down" 
    assert white_bishop_visibility_iset[0, 1, 7] == 1, "Bishop should see right up"
    assert white_bishop_visibility_iset[0, 2, 6] == 1, "Bishop should see right up"
    assert white_bishop_visibility_iset[0, 3, 5] == 1, "Bishop should see right up"
    assert white_bishop_visibility_iset[13, 6, 2] == 1, "Bishop should see left down"
    assert white_bishop_visibility_iset[13, 7, 1] == 1, "Bishop should see left down"
    
    assert jnp.sum(white_bishop_visibility_iset[0]) == 8, "Bishop should see 8 tiles"
    assert jnp.sum(white_bishop_visibility_iset[1]) == 1, "Bishop should see 1 pawn"
    assert jnp.sum(white_bishop_visibility_iset[4]) == 1, "Bishop should see 1 bishop"
    assert jnp.sum(white_bishop_visibility_iset[2:4]) == 0, "Bishop should see 0 other pieces"
    assert jnp.sum(white_bishop_visibility_iset[5:7]) == 0, "Bishop should see 0 other pieces"
    assert jnp.sum(white_bishop_visibility_iset[7]) == 1, "Bishop should see 1 opponent pawn"
    assert jnp.sum(white_bishop_visibility_iset[8:13]) == 0, "Bishop should see 0 other pieces"
    assert jnp.sum(white_bishop_visibility_iset[13]) == (64 - 11), "Bishop should see 64 - 11 non-visible tiles"
    
    black_bishop_visibility_fen = "1KNRQBPR/8/8/8/8/8/8/8 b - - 0 1"
    black_bishop_visibility_fen = replace_in_fen(black_bishop_visibility_fen, "b", [4, 4])
    black_bishop_visibility_fen = replace_in_fen(black_bishop_visibility_fen, "p", [6, 6])
    black_bishop_visibility_fen = replace_in_fen(black_bishop_visibility_fen, "P", [7, 6])
    black_bishop_visibility_fen = replace_in_fen(black_bishop_visibility_fen, "P", [5, 3])
    black_bishop_visibility_fen = replace_in_fen(black_bishop_visibility_fen, "P", [6, 2])
    
    black_bishop_visibility_state = translate_fen(black_bishop_visibility_fen)
    black_bishop_visibility_iset = self.game8.observation_tensor(black_bishop_visibility_state, 1)
    
    assert black_bishop_visibility_iset[7, 6, 6] == 1, "We see our pawn"
    assert black_bishop_visibility_iset[10, 4, 4] == 1, "We see our bishop"
    assert black_bishop_visibility_iset[1, 5, 3] == 1, "Bishop sees opponents pawn"
    assert black_bishop_visibility_iset[13, 7, 7] == 1, "We see not see beyond pawn"
    assert black_bishop_visibility_iset[0, 0, 0] == 1, "Bishop should see left up"
    assert black_bishop_visibility_iset[0, 1, 1] == 1, "Bishop should see left up"
    assert black_bishop_visibility_iset[0, 2, 2] == 1, "Bishop should see left up"
    assert black_bishop_visibility_iset[0, 3, 3] == 1, "Bishop should see left up"
    assert black_bishop_visibility_iset[0, 5, 5] == 1, "Bishop should see right down"
    assert black_bishop_visibility_iset[0, 1, 7] == 1, "Bishop should see right up"
    assert black_bishop_visibility_iset[0, 2, 6] == 1, "Bishop should see right up"
    assert black_bishop_visibility_iset[0, 3, 5] == 1, "Bishop should see right up"
    assert black_bishop_visibility_iset[13, 6, 2] == 1, "Bishop should see left down"
    assert black_bishop_visibility_iset[13, 7, 1] == 1, "Bishop should see left down"
    
    assert jnp.sum(black_bishop_visibility_iset[0]) == 8, "Bishop should see 8 tiles"
    assert jnp.sum(black_bishop_visibility_iset[1]) == 1, "Bishop should see 1 pawn"
    assert jnp.sum(black_bishop_visibility_iset[2:7]) == 0, "Bishop should see 0 other pieces"
    assert jnp.sum(black_bishop_visibility_iset[7]) == 1, "Bishop should see 1 pawn"
    assert jnp.sum(black_bishop_visibility_iset[8:10]) == 0, "Bishop should see 0 other pieces"
    assert jnp.sum(black_bishop_visibility_iset[10]) == 1, "Bishop should see 1 bishop"
    assert jnp.sum(black_bishop_visibility_iset[11:13]) == 0, "Bishop should see 0 other pieces"
    assert jnp.sum(black_bishop_visibility_iset[13]) == (64 - 11), "Bishop should see 64 - 11 non-visible tiles" 
    
    print("✓ Observation bishop test passed")

  def test_public_observation_rook(self):
    empty_fen = "8/8/8/8/8/8/8/8 w - - 0 1"
    two_rooks_fen = replace_in_fen(empty_fen, "R", [4, 4])
    two_rooks_fen = replace_in_fen(two_rooks_fen, "r", [2, 4])
    two_rooks_fen = replace_in_fen(two_rooks_fen, "P", [2, 6])
    two_rooks_fen = replace_in_fen(two_rooks_fen, "p", [4, 2])
    two_rooks_state = translate_fen(two_rooks_fen)
    two_rooks = self.game8.public_observation_tensor(two_rooks_state)
    
    assert two_rooks[0, 3, 4] == 1, "Rooks should see empty space"
    assert two_rooks[2, 4, 4] == 1, "Rook should see our rook"
    assert two_rooks[8, 2, 4] == 1, "Rook should see opponent rook"
    assert jnp.sum(two_rooks[0]) == 1, "The observable part is only between rooks"
    assert jnp.sum(two_rooks[1]) == 0, "No other pieces"
    assert jnp.sum(two_rooks[2]) == 1, "No other pieces"
    assert jnp.sum(two_rooks[3:8]) == 0, "No other pieces"
    assert jnp.sum(two_rooks[8]) == 1, "No other pieces"
    assert jnp.sum(two_rooks[9:13]) == 0, "No other pieces"
    assert jnp.sum(two_rooks[13]) == 64 - 3, "There should be 64 - 3 non-visible tiles"
     
    three_rooks_fen = replace_in_fen(empty_fen, "R", [4, 4])
    three_rooks_fen = replace_in_fen(three_rooks_fen, "r", [4, 5])
    three_rooks_fen = replace_in_fen(three_rooks_fen, "r", [4, 6]) 
    three_rooks_state = translate_fen(three_rooks_fen)
    three_rooks = self.game8.public_observation_tensor(three_rooks_state)
    
    assert three_rooks[2, 4, 4] == 1, "Rook should see our rook"
    assert three_rooks[8, 4, 5] == 1, "Rook should see our rook"
    assert jnp.sum(three_rooks[0:2]) == 0, "The observable part is only between rooks" 
    assert jnp.sum(three_rooks[2]) == 1, "No other pieces"
    assert jnp.sum(three_rooks[3:8]) == 0, "No other pieces"
    assert jnp.sum(three_rooks[8]) == 1, "No other pieces"
    assert jnp.sum(three_rooks[9:13]) == 0, "No other pieces"
    assert jnp.sum(three_rooks[13]) == 64 - 2, "There should be 64 - 2 non-visible tiles"
    
    
    several_rooks_fen = replace_in_fen(empty_fen, "R", [1, 1])
    several_rooks_fen = replace_in_fen(several_rooks_fen, "r", [1, 6])
    several_rooks_fen = replace_in_fen(several_rooks_fen, "R", [6, 6])
    several_rooks_fen = replace_in_fen(several_rooks_fen, "r", [2, 2])
    several_rooks_fen = replace_in_fen(several_rooks_fen, "r", [3, 3])
    several_rooks_fen = replace_in_fen(several_rooks_fen, "r", [4, 4])
    several_rooks_fen = replace_in_fen(several_rooks_fen, "r", [5, 5]) 
    several_rooks_fen = replace_in_fen(several_rooks_fen, "R", [7, 7])
    several_rooks_fen = replace_in_fen(several_rooks_fen, "R", [0, 0])
    several_rooks_fen = replace_in_fen(several_rooks_fen, "R", [7, 0])
    several_rooks_fen = replace_in_fen(several_rooks_fen, "R", [0, 7])

    several_rooks_state = translate_fen(several_rooks_fen)
    several_rooks = self.game8.public_observation_tensor(several_rooks_state)
    
    assert jnp.all(several_rooks[0, 1, 2:6] == 1), "Rook should see our rook"
    assert jnp.all(several_rooks[0, 2:6, 6] == 1), "Rook should see our rook"
    assert several_rooks[2, 1, 1] == 1, "Rook should see our rook"
    assert several_rooks[2, 6, 6] == 1, "Rook should see our rook"
    assert several_rooks[8, 1, 6] == 1, "Rook should see our rook"
    assert jnp.sum(several_rooks[0]) == 8, "Rook should see 8 tiles"
    assert jnp.sum(several_rooks[1]) == 0, "The observable part is only between rooks" 
    assert jnp.sum(several_rooks[2]) == 2, "Two rooks"
    assert jnp.sum(several_rooks[3:8]) == 0, "No other pieces"
    assert jnp.sum(several_rooks[8]) == 1, "No other pieces"
    assert jnp.sum(several_rooks[9:13]) == 0, "No other pieces"
    assert jnp.sum(several_rooks[13]) == 64 - 11, "There should be 64 - 11 non-visible tiles"
    
    filled_fen = "1RrRrRRR/R7/r7/R7/r7/r7/r7/r7 w - - 0 1"
    filled_state = translate_fen(filled_fen)
    filled = self.game8.public_observation_tensor(filled_state)
    
    assert filled[2, 0, 1] == 1, "Rook should be visible"
    assert filled[2, 0, 3] == 1, "Rook should be visible"
    assert filled[2, 0, 5] == 1, "Rook should be visible"
    assert filled[2, 1, 0] == 1, "Rook should be visible"
    assert filled[2, 3, 0] == 1, "Rook should be visible"
    assert filled[8, 0, 2] == 1, "Rook should be visible"
    assert filled[8, 0, 4] == 1, "Rook should be visible"
    assert filled[8, 2, 0] == 1, "Rook should be visible"
    assert filled[8, 4, 0] == 1, "Rook should be visible"
    
    
    assert jnp.sum(filled[0:2]) == 0, "No empty spaces visible"
    assert jnp.sum(filled[2]) == 5, "One black rook"
    assert jnp.sum(filled[3:8]) == 0, "No other pieces"
    assert jnp.sum(filled[8]) == 4, "No other pieces"
    assert jnp.sum(filled[9:13]) == 0, "No other pieces"
    assert jnp.sum(filled[13]) == 64 - 9, "There should be 64 - 11 non-visible tiles"
    
    
    print("✓ Public observation tensor test passed")
    
  def test_public_observation_bishop(self):
    # Test public observation for bishop pieces
    empty_fen = "8/8/8/8/8/8/8/8 w - - 0 1"
    two_bishops_fen = replace_in_fen(empty_fen, "B", [4, 4])
    two_bishops_fen = replace_in_fen(two_bishops_fen, "b", [2, 2])
    two_bishops_fen = replace_in_fen(two_bishops_fen, "p", [2, 6])
    two_bishops_fen = replace_in_fen(two_bishops_fen, "P", [0, 0])
    two_bishops_state = translate_fen(two_bishops_fen)
    two_bishops = self.game8.public_observation_tensor(two_bishops_state)
    
    assert two_bishops[0, 3, 3] == 1, "Should see empty space between bishops"
    assert two_bishops[4, 4, 4] == 1, "Should see white bishop"
    assert two_bishops[10, 2, 2] == 1, "Should see black bishop"
    assert jnp.sum(two_bishops[0]) == 1, "Only the path between bishops should be visible"
    assert jnp.sum(two_bishops[1:4]) == 0, "No other pieces"
    assert jnp.sum(two_bishops[4]) == 1, "One white bishop"
    assert jnp.sum(two_bishops[5:10]) == 0, "No other pieces"
    assert jnp.sum(two_bishops[10]) == 1, "One black bishop"
    assert jnp.sum(two_bishops[11:13]) == 0, "No other pieces"
    assert jnp.sum(two_bishops[13]) == 64 - 3, "Should be 61 non-visible tiles"
    
    blocked_bishops_fen = replace_in_fen(empty_fen, "B", [5, 5])
    blocked_bishops_fen = replace_in_fen(blocked_bishops_fen, "b", [2, 2])
    blocked_bishops_fen = replace_in_fen(blocked_bishops_fen, "P", [0, 0])
    blocked_bishops_fen = replace_in_fen(blocked_bishops_fen, "p", [7, 7])
    blocked_bishops_state = translate_fen(blocked_bishops_fen)
    blocked_bishops = self.game8.public_observation_tensor(blocked_bishops_state)
    
    assert blocked_bishops[0, 3, 3] == 1, "Should see white bishop"
    assert blocked_bishops[0, 4, 4] == 1, "Should see white bishop"
    assert blocked_bishops[4, 5, 5] == 1, "Should see blocking pawn"
    assert blocked_bishops[10, 2, 2] == 1, "Should see blocking pawn"
    assert jnp.sum(blocked_bishops[0]) == 2, "No empty spaces visible"
    assert jnp.sum(blocked_bishops[1:4]) == 0, "No other pieces"
    assert jnp.sum(blocked_bishops[4]) == 1, "One white bishop"
    assert jnp.sum(blocked_bishops[5:10]) == 0, "No other pieces"
    assert jnp.sum(blocked_bishops[10]) == 1, "One black bishop"
    assert jnp.sum(blocked_bishops[11:13]) == 0, "No other pieces"
    assert jnp.sum(blocked_bishops[13]) == 64 - 4, "Should be 60 non-visible tiles"
    
  
    filled_fen = replace_in_fen(empty_fen, "B", [1, 1])
    filled_fen = replace_in_fen(filled_fen, "b", [2, 2])
    filled_fen = replace_in_fen(filled_fen, "B", [3, 3])
    filled_fen = replace_in_fen(filled_fen, "b", [4, 4])
    filled_fen = replace_in_fen(filled_fen, "B", [5, 5])
    filled_fen = replace_in_fen(filled_fen, "B", [6, 6])
    filled_fen = replace_in_fen(filled_fen, "B", [3, 5])
    filled_fen = replace_in_fen(filled_fen, "b", [2, 6])
    filled_fen = replace_in_fen(filled_fen, "b", [1, 7])
    
    filled_state = translate_fen(filled_fen)
    filled = self.game8.public_observation_tensor(filled_state)
    
    
    assert filled[4, 1, 1] == 1, "Should see white bishop"
    assert filled[10, 2, 2] == 1, "Should see white bishop"
    assert filled[4, 3, 3] == 1, "Should see white bishop"
    assert filled[10, 4, 4] == 1, "Should see white bishop"
    assert filled[4, 5, 5] == 1, "Should see white bishop"
    assert filled[4, 3, 5] == 1, "Should see white bishop"
    assert filled[10, 2, 6] == 1, "Should see white bishop"
    assert jnp.sum(filled[0:4]) == 0, "Should see 6 empty spaces"
    assert jnp.sum(filled[4]) == 4, "Should see 4 white bishops"
    assert jnp.sum(filled[5:10]) == 0, "Should see 0 black bishops"
    assert jnp.sum(filled[10]) == 3, "Should see 4 black bishops"
    assert jnp.sum(filled[11:13]) == 0, "Should see 0 other pieces"
    assert jnp.sum(filled[13]) == 64 - 7, "Should be 57 non-visible tiles"
    
    print("✓ Public observation bishop test passed")
  
  def   test_public_observation_queen(self):
    """Test public observation for queen pieces."""
    empty_fen = "8/8/8/8/8/8/8/8 w - - 0 1"
    
    # Test queens seeing each other on the same diagonal
    two_queens_diagonal_fen = replace_in_fen(empty_fen, "Q", [4, 4])
    two_queens_diagonal_fen = replace_in_fen(two_queens_diagonal_fen, "q", [2, 2])
    two_queens_diagonal_state = translate_fen(two_queens_diagonal_fen)
    two_queens_diagonal = self.game8.public_observation_tensor(two_queens_diagonal_state)
    
    assert two_queens_diagonal[5, 4, 4] == 1, "Should see white queen"
    assert two_queens_diagonal[11, 2, 2] == 1, "Should see black queen"
    assert two_queens_diagonal[0, 3, 3] == 1, "Should see empty space between queens"
    assert jnp.sum(two_queens_diagonal[0]) == 1, "Only one empty space should be visible"
    assert jnp.sum(two_queens_diagonal[1:4]) == 0, "No other pieces"
    assert jnp.sum(two_queens_diagonal[5]) == 1, "One white queen"
    assert jnp.sum(two_queens_diagonal[6:10]) == 0, "No other pieces"
    assert jnp.sum(two_queens_diagonal[11]) == 1, "One black queen"
    assert jnp.sum(two_queens_diagonal[12]) == 0, "No other pieces"
    assert jnp.sum(two_queens_diagonal[13]) == 64 - 3, "Should be 61 non-visible tiles"
    
    # Test queens seeing each other on the same file
    two_queens_file_fen = replace_in_fen(empty_fen, "Q", [4, 3])
    two_queens_file_fen = replace_in_fen(two_queens_file_fen, "q", [1, 3])
    two_queens_file_state = translate_fen(two_queens_file_fen)
    two_queens_file = self.game8.public_observation_tensor(two_queens_file_state)
    
    assert two_queens_file[5, 4, 3] == 1, "Should see white queen"
    assert two_queens_file[11, 1, 3] == 1, "Should see black queen"
    assert two_queens_file[0, 2, 3] == 1, "Should see empty space between queens"
    assert two_queens_file[0, 3, 3] == 1, "Should see empty space between queens"
    assert jnp.sum(two_queens_file[0]) == 2, "Two empty spaces should be visible"
    assert jnp.sum(two_queens_file[1:4]) == 0, "No other pieces"
    assert jnp.sum(two_queens_file[5]) == 1, "One white queen"
    assert jnp.sum(two_queens_file[6:10]) == 0, "No other pieces"
    assert jnp.sum(two_queens_file[11]) == 1, "One black queen"
    assert jnp.sum(two_queens_file[12]) == 0, "No other pieces"
    assert jnp.sum(two_queens_file[13]) == 64 - 4, "Should be 60 non-visible tiles"
    
    # Test blocked queen vision
    blocked_queen_fen = replace_in_fen(empty_fen, "Q", [5, 5])
    blocked_queen_fen = replace_in_fen(blocked_queen_fen, "q", [1, 1])
    blocked_queen_fen = replace_in_fen(blocked_queen_fen, "P", [3, 3]) # Blocking pawn
    blocked_queen_state = translate_fen(blocked_queen_fen)
    blocked_queen = self.game8.public_observation_tensor(blocked_queen_state)
    
    assert jnp.sum(blocked_queen[0:13]) == 0, "No other pieces"
    assert jnp.sum(blocked_queen[13]) == 64, "Should be 62 non-visible tiles"
    
    
    for pl in range(2):
      pl_offset = 0 if pl == 0 else 6
      pl_type = "Q" if pl == 0 else "q"
      opp_type = "q" if pl == 0 else "Q"
      multiple_queens_fen = replace_in_fen(empty_fen, pl_type, [4, 4])
      for i in range(2, 7):
        multiple_queens_fen = replace_in_fen(multiple_queens_fen, opp_type, [i, 2])
        multiple_queens_fen = replace_in_fen(multiple_queens_fen, opp_type, [i, 6])
        multiple_queens_fen = replace_in_fen(multiple_queens_fen, opp_type, [2, i])
        multiple_queens_fen = replace_in_fen(multiple_queens_fen, opp_type, [6, i])
        

      multiple_queens_state = translate_fen(multiple_queens_fen)
      multiple_queens = self.game8.public_observation_tensor(multiple_queens_state)
      
      assert jnp.sum(multiple_queens[0]) == 8, "No empty spaces visible"
      assert jnp.all(multiple_queens[0, 3:6, 3] == 1), "Visible empty spaces"
      assert jnp.all(multiple_queens[0, 3:6, 5] == 1), "Visible empty spaces"
      assert jnp.all(multiple_queens[0, 3, 3:6] == 1), "Visible empty spaces"
      assert jnp.all(multiple_queens[0, 5, 3:6] == 1), "Visible empty spaces"
      assert jnp.sum(multiple_queens[1:4]) == 0, "No other pieces"
      assert multiple_queens[5 + pl_offset, 4, 4] == 1, "One white queen"
      assert jnp.sum(multiple_queens[5 + pl_offset]) == 1, "One white queen" 
      assert jnp.sum(multiple_queens[6:10]) == 0, "No other pieces"
      assert jnp.sum(multiple_queens[11 - pl_offset]) == 8, "16 black queens"
      assert jnp.sum(multiple_queens[12]) == 0, "No other pieces"
      assert jnp.sum(multiple_queens[13]) == 64 - 17, "Should be 39 non-visible tiles"
    
    print("✓ Public observation queen test passed")
  
  def test_public_observation_knight(self):
    empty_fen = "8/8/8/8/8/8/8/8 w - - 0 1"
    two_knights_fen = replace_in_fen(empty_fen, "N", [3, 3])
    two_knights_fen = replace_in_fen(two_knights_fen, "n", [5, 4])
    two_knights_fen = replace_in_fen(two_knights_fen, "N", [3, 5])
    two_knights_fen = replace_in_fen(two_knights_fen, "n", [4, 7])
    two_knights_fen = replace_in_fen(two_knights_fen, "N", [0, 0])
    two_knights_fen = replace_in_fen(two_knights_fen, "n", [0, 1])
    two_knights_state = translate_fen(two_knights_fen)
     
    two_knights = self.game8.public_observation_tensor(two_knights_state)
    
    assert two_knights[3, 3, 3] == 1, "Should see white knight"
    assert two_knights[9, 5, 4] == 1, "Should see black knight"
    assert two_knights[3, 3, 5] == 1, "Should see white knight"
    assert two_knights[9, 4, 7] == 1, "Should see black knight"
    assert jnp.sum(two_knights[0:3]) == 0, "Knights don't create visible paths"
    assert jnp.sum(two_knights[3]) == 2, "Two white knights"
    assert jnp.sum(two_knights[4:9]) == 0, "No other pieces"
    assert jnp.sum(two_knights[9]) == 2, "Two black knights"
    assert jnp.sum(two_knights[10:13]) == 0, "No other pieces"
    assert jnp.sum(two_knights[13]) == 64 - 4, "Should be 58 non-visible tiles"
    
    for pl in range(2):
      pl_offset = 0 if pl == 0 else 6
      pl_type = "N" if pl == 0 else "n"
      opp_type = "n" if pl == 0 else "N"
      full_fen = (opp_type * 8 + "/") * 7 + (opp_type * 8 ) + " w - - 0 1"
      for y in range(8):
        for x in range(8):
          full_knights_fen = replace_in_fen(full_fen, pl_type, [y, x]) 
          full_knights_state = translate_fen(full_knights_fen)
          full_knights = self.game8.public_observation_tensor(full_knights_state)
          assert full_knights[3 + pl_offset, y, x] == 1, "One white knight" 
          black_knights = 0
          for i, move in enumerate(KNIGHT_DIRS):
            if y + move[0] >= 0 and y + move[0] < 8 and x + move[1] >= 0 and x + move[1] < 8:
              black_knights += 1
              assert full_knights[9 - pl_offset, y + move[0], x + move[1]] == 1, "Should see black knight" 
          assert jnp.all(full_knights[0:3] == 0), "No black knights"  
          assert jnp.sum(full_knights[3 + pl_offset]) == 1, "One white knight"
          assert jnp.all(full_knights[5:9] == 0), "No black knights"
          assert jnp.sum(full_knights[9 - pl_offset]) == black_knights, "Should see black knights"
          assert jnp.all(full_knights[10:13] == 0), "No black knights"
          assert jnp.sum(full_knights[13]) == 64 - (black_knights + 1), "Should be 64 non-visible tiles"
    
    for pl in range(2):
      pl_offset = 0 if pl == 0 else 6
      pl_type = "N" if pl == 0 else "n"
      opp_type = ["p", "k", "q", "b", "r"] if pl == 0 else ["P", "K", "Q", "B", "R"]
      multiple_knights_fen = replace_in_fen(empty_fen, pl_type, [3, 3]) 
      for i, move in enumerate(KNIGHT_DIRS):
        multiple_knights_fen = replace_in_fen(multiple_knights_fen, opp_type[i % len(opp_type)], [3 + move[0], 3 + move[1]])
          
      multiple_knights_state = translate_fen(multiple_knights_fen)
      multiple_knights = self.game8.public_observation_tensor(multiple_knights_state)
       
      assert jnp.all(multiple_knights[0:13] == 0), "Nothing visible spaces" 
      assert jnp.all(multiple_knights[13] == 1), "No other pieces"
    
    for pl in range(2):
      pl_offset = 0 if pl == 0 else 6
      pl_type = "N" if pl == 0 else "n"
        
      knight_pos = [4, 4]
      same_knights_fen = replace_in_fen(empty_fen, pl_type, [4, 4])
      for move in KNIGHT_DIRS:
        same_knights_fen = replace_in_fen(same_knights_fen, pl_type, [knight_pos[0] + move[0], knight_pos[1] + move[1]])
        knight_pos = [knight_pos[0] + move[0], knight_pos[1] + move[1]]
        
      same_knights_state = translate_fen(same_knights_fen)
      same_knights = self.game8.public_observation_tensor(same_knights_state)
      
      assert jnp.all(same_knights[0:13] == 0), "Nothing visible spaces" 
      assert jnp.all(same_knights[13] == 1), "All invisible"
        
    print("✓ Public observation knight test passed")
  
  
  def test_public_observation_king(self):
    empty_fen = "8/8/8/8/8/8/8/8 w - - 0 1" 
    for pl in range(2):
      pl_offset = 0 if pl == 0 else 6
      pl_type = "K" if pl == 0 else "k"
      opp_type = "k" if pl == 0 else "K"
      all_opp_types = ["k", "q", "b", "r"] if pl == 0 else ["K", "Q", "B", "R"]
      for y in range(8):
        for x in range(8):
          single_king_fen = replace_in_fen(empty_fen, pl_type, [y, x])
          single_king_state = translate_fen(single_king_fen)
          single_king = self.game8.public_observation_tensor(single_king_state) 
          
          
          assert jnp.all(single_king[0:13] == 0), "No seen pieces" 
          assert jnp.all(single_king[13] == 1), "Should be 64 non-visible tiles"
          
          for o_type in all_opp_types:
            type_id = PIECE_TO_ID[o_type]
            surrounded_pieces = 0 
            surrounded_king_fen = replace_in_fen(empty_fen, pl_type, [y, x])
            
            full_fen = (o_type * 8 + "/") * 7 + (o_type * 8 ) + " w - - 0 1"
            full_fen = replace_in_fen(full_fen, pl_type, [y, x])
            
            
            for move in KING_DIRS:
              if y + move[0] >= 0 and y + move[0] < 8 and x + move[1] >= 0 and x + move[1] < 8:
                surrounded_king_fen = replace_in_fen(surrounded_king_fen, o_type, [y + move[0], x + move[1]]) 
                if (o_type == "r" or o_type == "R") and (move[0] != 0 and move[1] != 0):
                  continue
                if (o_type == "b" or o_type == "B") and (move[0] == 0 or move[1] == 0):
                  continue
                surrounded_pieces += 1
            surrounded_king_state = translate_fen(surrounded_king_fen)
            surrounded_king = self.game8.public_observation_tensor(surrounded_king_state) 
            
            full_king_state = translate_fen(full_fen)
            full_king = self.game8.public_observation_tensor(full_king_state) 
            
            
            for move in KING_DIRS:
              if y + move[0] >= 0 and y + move[0] < 8 and x + move[1] >= 0 and x + move[1] < 8:
                if (o_type == "r" or o_type == "R") and (move[0] != 0 and move[1] != 0):
                  continue
                if (o_type == "b" or o_type == "B") and (move[0] == 0 or move[1] == 0):
                  continue
                assert surrounded_king[type_id, y + move[0], x + move[1]] == 1, "Should see piece"
                assert full_king[type_id, y + move[0], x + move[1]] == 1, "Should see piece"
              
            
            assert jnp.sum(surrounded_king[type_id]) == surrounded_pieces, "Should see surrounded pieces"
            assert jnp.sum(full_king[type_id]) == surrounded_pieces, "Should see surrounded pieces"
            for i in range(13):
              if i == 6 + pl_offset or i == type_id:
                continue
              assert jnp.all(surrounded_king[i] == 0), "Should not see any other piece"
              assert jnp.all(full_king[i] == 0), "Should not see any other piece"
            assert jnp.sum(surrounded_king[13]) == 64 - (surrounded_pieces + 1)
            assert jnp.sum(full_king[13]) == 64 - 1 - surrounded_pieces, "Should be 63 non-visible tiles"
            
            
          surrounded_king_fen = replace_in_fen(empty_fen, pl_type, [y, x])
          opp_knight = "n" if pl == 0 else "N"
          for move in KING_DIRS:
            if y + move[0] >= 0 and y + move[0] < 8 and x + move[1] >= 0 and x + move[1] < 8:
              surrounded_king_fen = replace_in_fen(surrounded_king_fen, opp_knight, [y + move[0], x + move[1]])
          surrounded_king_state = translate_fen(surrounded_king_fen)
          surrounded_king = self.game8.public_observation_tensor(surrounded_king_state) 
          assert jnp.all(surrounded_king[0:13] == 0), "No other pieces"
          assert jnp.sum(surrounded_king[13]) == 64, "Should be 63 non-visible tiles"
          
          for o_type in all_opp_types:
            type_id = PIECE_TO_ID[o_type]
            for move in KING_DIRS:
              if y + move[0] >= 0 and y + move[0] < 8 and x + move[1] >= 0 and x + move[1] < 8:
                single_attack_fen = replace_in_fen(single_king_fen, o_type, [y + move[0], x + move[1]])
                single_attack_state = translate_fen(single_attack_fen)
                single_attack = self.game8.public_observation_tensor(single_attack_state) 
                attacking_pieces = 1
                if (o_type == "R" or o_type == "r") and (move[0] != 0 and move[1] != 0):
                  attacking_pieces = 0
                if (o_type == "B" or o_type == "b") and (move[0] == 0 or move[1] == 0):
                  attacking_pieces = 0
                  
                assert single_attack[6 + pl_offset, y, x] == attacking_pieces, "No other pieces"
                assert single_attack[type_id, y + move[0], x + move[1]] == attacking_pieces, "Seeing attacking piece"
                assert jnp.sum(single_attack[6 + pl_offset]) == attacking_pieces, "Should see piece"
                assert jnp.sum(single_attack[type_id]) == attacking_pieces, "Should not see own piece"
                for i in range(13):
                  if i == 6 + pl_offset or i == type_id:
                    continue
                  assert jnp.all(single_attack[i] == 0), "Should not see any other piece"
                assert jnp.sum(single_attack[13]) == 64 - (2 *attacking_pieces), "Should be 62 non-visible tiles"
              
              
          for o_type in all_opp_types:
            type_id = PIECE_TO_ID[o_type]
            non_threat_fen = replace_in_fen(empty_fen, pl_type, [y, x])
            for y2 in range(8):
              for x2 in range(8):
                if abs(y2 - y) <= 1 and abs(x2 - x) <= 1:
                  continue
                non_threat_fen = replace_in_fen(non_threat_fen, o_type, [y2, x2])
            non_threat_state = translate_fen(non_threat_fen)
            non_threat = self.game8.public_observation_tensor(non_threat_state)
            assert jnp.all(non_threat[0:13] == 0), "No other pieces"
            assert jnp.sum(non_threat[13]) == 64, "Should be 63 non-visible tiles"
              
              
              
              
          
    print("✓ Public observation king test passed")

if __name__ == "__main__":
  test = DarkChessTest()
  test.run_all_tests()
