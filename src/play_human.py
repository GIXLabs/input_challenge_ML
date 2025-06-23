import cv2
from tetris import Tetris
from time import sleep
import time

KEY_LEFT = 2   # Left arrow (user system)
KEY_RIGHT = 3  # Right arrow (user system)
KEY_DOWN = 1   # Down arrow (user system)
KEY_UP = 0     # Up arrow (user system)
KEY_ESC = 27   # Escape
KEY_R = ord('r')

# Tunable gravity delay in milliseconds
GRAVITY_DELAY = 500  # Lower is faster gravity (e.g., 500ms)

def main():
    env = Tetris()
    done = False
    print('Controls: ← (left), → (right), ↓ (down), ↑ (rotate), r (reset), esc (quit)')
    print(f'Gravity: piece moves down every {GRAVITY_DELAY} ms')

    last_gravity_time = time.time()
    env.render()  # Initial render

    while True:
        key = cv2.waitKey(1)  # Fast polling for input
        # print(f"Key pressed: {key}")  # Config: print the key code

        now = time.time()
        gravity_applied = False
        if not done and (now - last_gravity_time) * 1000 >= GRAVITY_DELAY:
            # Gravity: move piece down
            env.current_pos[1] += 1
            if env._check_collision(env._get_rotated_piece(), env.current_pos):
                env.current_pos[1] -= 1
                # Place piece
                env.board = env._add_piece_to_board(env._get_rotated_piece(), env.current_pos)
                lines_cleared, env.board = env._clear_lines(env.board)
                env.score += 1 + (lines_cleared ** 2) * Tetris.BOARD_WIDTH
                env._new_round()
                if env.game_over:
                    print('Game Over! Press r to reset or esc to quit.')
                    done = True
            last_gravity_time = now
            env.render()
            gravity_applied = True

        if key == -1:
            continue

        if key == KEY_ESC:
            print('Exiting...')
            break
        if key == KEY_R:
            print('Resetting game...')
            env.reset()
            done = False
            last_gravity_time = time.time()
            env.render()
            continue
        if done:
            continue

        # Copy current state
        x, rotation = env.current_pos[0], env.current_rotation
        state_changed = False

        if key == KEY_LEFT:
            x = max(0, x - 1)
            state_changed = True
        elif key == KEY_RIGHT:
            x = min(Tetris.BOARD_WIDTH - 1, x + 1)
            state_changed = True
        elif key == KEY_DOWN:
            # Drop piece by one
            env.current_pos[1] += 1
            if env._check_collision(env._get_rotated_piece(), env.current_pos):
                env.current_pos[1] -= 1
            state_changed = True
        elif key == KEY_UP:
            # Rotate piece
            rotation = (rotation + 90) % 360
            state_changed = True
        else:
            continue

        # Play move if not down
        if key in [KEY_LEFT, KEY_RIGHT, KEY_UP]:
            # Try to move/rotate, check collision
            env.current_pos[0] = x
            env.current_rotation = rotation
            if env._check_collision(env._get_rotated_piece(), env.current_pos):
                # Undo move if collision
                if key == KEY_LEFT:
                    env.current_pos[0] += 1
                elif key == KEY_RIGHT:
                    env.current_pos[0] -= 1
                elif key == KEY_UP:
                    env.current_rotation = (rotation - 90) % 360
        # If down, try to move down, if can't, place piece
        elif key == KEY_DOWN:
            if env._check_collision(env._get_rotated_piece(), env.current_pos):
                env.current_pos[1] -= 1
                # Place piece
                env.board = env._add_piece_to_board(env._get_rotated_piece(), env.current_pos)
                lines_cleared, env.board = env._clear_lines(env.board)
                env.score += 1 + (lines_cleared ** 2) * Tetris.BOARD_WIDTH
                env._new_round()
                if env.game_over:
                    print('Game Over! Press r to reset or esc to quit.')
                    done = True
            # else: piece moved down by one
        if state_changed:
            env.render()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 