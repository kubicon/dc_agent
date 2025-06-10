# JAX Chess Engine

A fully functional chess engine implemented in JAX with JIT compilation support. This implementation provides a complete chess game with all standard rules, optimized for high-performance computation using JAX's just-in-time compilation.

## Features

### Complete Chess Implementation
- **All piece movements**: Pawns, Rooks, Knights, Bishops, Queens, Kings
- **Special moves**: Castling (kingside and queenside), En passant, Pawn promotion
- **Game rules**: Check detection, Checkmate, Stalemate, 50-move rule
- **Game state tracking**: Turn management, castling rights, move counters

### JAX JIT Compatibility
- **Pure functional design**: All functions are side-effect free
- **JAX array operations**: Uses `jax.numpy` instead of regular numpy
- **Vectorized operations**: Efficient move generation and validation
- **No Python loops**: Uses `jax.lax.scan`, `jax.vmap`, and `jax.lax.cond` for control flow
- **Static shapes**: All arrays have compile-time known shapes

### Performance Optimizations
- **JIT compiled functions**: All core game logic is decorated with `@jit`
- **Efficient board representation**: 8x8 integer array with piece encoding
- **Vectorized move checking**: Simultaneous validation of multiple moves
- **Minimal memory allocation**: Reuses arrays where possible

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import jax.numpy as jnp
from jax_chess import initial_game_state, make_move, print_board, parse_move

# Create a new game
game = initial_game_state()

# Print the initial board
print_board(game.board)

# Make a move (e2 to e4)
from_sq, to_sq = parse_move("e2e4")
new_game = make_move(game, jnp.array(from_sq), jnp.array(to_sq))

# Print the board after the move
print_board(new_game.board)
```

## Core API

### Game State

The `GameState` is a JAX-compatible NamedTuple containing:
- `board`: 8x8 array representing the chess board
- `turn`: Current player (0 for white, 1 for black)
- `castling_rights`: Array of 4 booleans for castling availability
- `en_passant_target`: Square index for en passant capture (64 if none)
- `halfmove_clock`: Counter for 50-move rule
- `fullmove_number`: Move counter
- `game_over`: Game status (0: ongoing, 1: white wins, 2: black wins, 3: draw)

### Key Functions

#### Game Management
```python
initial_game_state() -> GameState
make_move(game_state, from_square, to_square) -> GameState
is_valid_move(game_state, from_square, to_square) -> bool
get_legal_moves(game_state) -> Array  # All legal moves for current player
```

#### Game Analysis
```python
is_in_check(board, color) -> bool
is_checkmate(game_state) -> bool
is_stalemate(game_state) -> bool
count_legal_moves(game_state) -> int
material_count(board) -> Tuple[int, int]  # (white_material, black_material)
```

#### Utilities
```python
print_board(board)  # Human-readable board display
print_game_info(game_state)  # Comprehensive game information
parse_move(move_str) -> Tuple[int, int]  # Convert "e2e4" to square indices
square_name(square_index) -> str  # Convert square index to "e4" format
```

## Board Representation

### Piece Encoding
- `0`: Empty square
- `1-6`: White pieces (Pawn, Rook, Knight, Bishop, Queen, King)
- `7-12`: Black pieces (Pawn, Rook, Knight, Bishop, Queen, King)

### Square Indexing
Squares are numbered 0-63, where:
- `0` = a8 (top-left)
- `7` = h8 (top-right)
- `56` = a1 (bottom-left)  
- `63` = h1 (bottom-right)

## Usage Examples

### Basic Game Loop
```python
import jax.numpy as jnp
from jax_chess import *

# Initialize game
game = initial_game_state()

# Game loop
while game.game_over == 0:
    print_board(game.board)
    print_game_info(game)
    
    # Get legal moves
    legal_moves = get_legal_moves(game)
    num_moves = count_legal_moves(game)
    print(f"Legal moves available: {int(num_moves)}")
    
    # Make a move (example: first legal move)
    if num_moves > 0:
        move = legal_moves[0]  # [from_square, to_square]
        if move[0] >= 0:  # Valid move
            game = make_move(game, move[0], move[1])
            game = update_game_status(game)
    else:
        break

print("Game finished!")
print_game_info(game)
```

### Performance Testing
```python
import jax
import time

# JIT compile the move function
compiled_make_move = jax.jit(make_move)

# Warm up JIT
game = initial_game_state()
_ = compiled_make_move(game, jnp.array(52), jnp.array(36))  # e2e4

# Benchmark
start_time = time.time()
for _ in range(1000):
    new_game = compiled_make_move(game, jnp.array(52), jnp.array(36))
end_time = time.time()

print(f"1000 moves took {end_time - start_time:.4f} seconds")
```

### Custom Position Setup
```python
# Create custom position
board = jnp.zeros((8, 8), dtype=jnp.int32)
board = board.at[0, 4].set(BLACK_KING)  # Black king on e8
board = board.at[7, 4].set(WHITE_KING)  # White king on e1
board = board.at[1, 0].set(BLACK_QUEEN) # Black queen on a7

custom_game = GameState(
    board=board,
    turn=jnp.array(0),  # White to move
    castling_rights=jnp.array([0, 0, 0, 0]),  # No castling
    en_passant_target=jnp.array(64),
    halfmove_clock=jnp.array(0),
    fullmove_number=jnp.array(1),
    game_over=jnp.array(0)
)

print_board(custom_game.board)
```

## Architecture

### JAX Compatibility
This implementation is designed to be fully compatible with JAX's functional programming model:

1. **Pure Functions**: All functions are deterministic and side-effect free
2. **Immutable Data**: Game state is never modified in-place
3. **JAX Arrays**: All data structures use `jax.numpy` arrays
4. **Control Flow**: Uses `jax.lax.cond`, `jax.lax.scan`, etc. instead of Python control flow
5. **Vectorization**: Leverages `jax.vmap` for efficient parallel operations

### Performance Considerations
- Move validation is vectorized for efficiency
- Attack detection uses scan operations for sliding pieces
- Legal move generation checks all 4096 possible moves in parallel
- All critical functions are JIT-compiled for maximum performance

## Limitations and Future Improvements

### Current Limitations
- No opening book or endgame tablebase
- Basic material counting only (no positional evaluation)
- Move generation could be optimized further for specific piece types

### Potential Enhancements
- Add position evaluation function
- Implement minimax search with alpha-beta pruning
- Add support for PGN format import/export
- Implement repetition detection for threefold repetition rule
- Add UCI protocol support for chess engine integration

## Contributing

This is a complete, working chess implementation optimized for JAX. Feel free to extend it with additional features like AI players, position evaluation, or search algorithms.

## License

This code is provided as-is for educational and research purposes. 