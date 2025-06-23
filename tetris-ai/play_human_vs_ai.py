import cv2
import numpy as np
import time
from tetris import Tetris
from dqn_agent import DQNAgent

# --- CONFIG ---
MODEL_PATH = 'best.keras'  # Path to your trained model
GRAVITY_DELAY = 500  # ms

# Key codes for your system
KEY_LEFT = 2
KEY_RIGHT = 3
KEY_DOWN = 1
KEY_UP = 0
KEY_ESC = 27
KEY_R = ord('r')

CELL_SIZE = 25


def render_dual(human_env, ai_env):
    # Render both boards and stack side by side
    def render_board(env, label, color):
        img = [Tetris.COLORS[p] for row in env._get_complete_board() for p in row]
        img = np.array(img).reshape(Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH, 3).astype(np.uint8)
        img = img[..., ::-1]
        img = cv2.resize(img, (Tetris.BOARD_WIDTH * CELL_SIZE, Tetris.BOARD_HEIGHT * CELL_SIZE), interpolation=cv2.INTER_NEAREST)
        # Draw grid
        for x in range(0, Tetris.BOARD_WIDTH * CELL_SIZE + 1, CELL_SIZE):
            cv2.line(img, (x, 0), (x, Tetris.BOARD_HEIGHT * CELL_SIZE), (200, 200, 200), 1)
        for y in range(0, Tetris.BOARD_HEIGHT * CELL_SIZE + 1, CELL_SIZE):
            cv2.line(img, (0, y), (Tetris.BOARD_WIDTH * CELL_SIZE, y), (200, 200, 200), 1)
        # Label and score
        cv2.putText(img, f'{label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, f'Score: {env.score}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if env.game_over:
            cv2.putText(img, 'GAME OVER', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return img
    human_img = render_board(human_env, 'Human', (0, 167, 247))
    ai_img = render_board(ai_env, 'AI', (247, 64, 99))
    # Stack horizontally
    combined = np.hstack((human_img, ai_img))
    # Draw a vertical separator
    sep_x = human_img.shape[1]
    cv2.rectangle(combined, (sep_x-2, 0), (sep_x+2, combined.shape[0]), (0,0,0), thickness=-1)
    cv2.imshow('Tetris: Human vs AI', combined)


def main():
    print('Controls: ← (left), → (right), ↓ (down), ↑ (rotate), r (reset), esc (quit)')
    print(f'Gravity: piece moves down every {GRAVITY_DELAY} ms')
    print(f'AI model: {MODEL_PATH}')

    human_env = Tetris()
    ai_env = Tetris()
    ai_agent = DQNAgent(ai_env.get_state_size(), modelFile=MODEL_PATH)
    done_human = False
    done_ai = False
    last_gravity_time = time.time()
    render_dual(human_env, ai_env)

    while True:
        key = cv2.waitKey(1)
        now = time.time()
        gravity_trigger = (now - last_gravity_time) * 1000 >= GRAVITY_DELAY

        # --- Human turn ---
        if not done_human:
            state_changed = False
            x, rotation = human_env.current_pos[0], human_env.current_rotation
            if key == KEY_LEFT:
                x = max(0, x - 1)
                state_changed = True
            elif key == KEY_RIGHT:
                x = min(Tetris.BOARD_WIDTH - 1, x + 1)
                state_changed = True
            elif key == KEY_DOWN:
                human_env.current_pos[1] += 1
                if human_env._check_collision(human_env._get_rotated_piece(), human_env.current_pos):
                    human_env.current_pos[1] -= 1
                state_changed = True
            elif key == KEY_UP:
                rotation = (rotation + 90) % 360
                state_changed = True
            elif key == KEY_R:
                print('Resetting game...')
                human_env.reset()
                ai_env.reset()
                done_human = False
                done_ai = False
                last_gravity_time = time.time()
                render_dual(human_env, ai_env)
                continue
            elif key == KEY_ESC:
                print('Exiting...')
                break
            if state_changed:
                human_env.current_pos[0] = x
                human_env.current_rotation = rotation
                if human_env._check_collision(human_env._get_rotated_piece(), human_env.current_pos):
                    if key == KEY_LEFT:
                        human_env.current_pos[0] += 1
                    elif key == KEY_RIGHT:
                        human_env.current_pos[0] -= 1
                    elif key == KEY_UP:
                        human_env.current_rotation = (rotation - 90) % 360
                render_dual(human_env, ai_env)

        # --- Gravity for both ---
        if gravity_trigger:
            # Human gravity
            if not done_human:
                human_env.current_pos[1] += 1
                if human_env._check_collision(human_env._get_rotated_piece(), human_env.current_pos):
                    human_env.current_pos[1] -= 1
                    human_env.board = human_env._add_piece_to_board(human_env._get_rotated_piece(), human_env.current_pos)
                    lines_cleared, human_env.board = human_env._clear_lines(human_env.board)
                    human_env.score += 1 + (lines_cleared ** 2) * Tetris.BOARD_WIDTH
                    human_env._new_round()
                    if human_env.game_over:
                        print('Human: Game Over!')
                        done_human = True
            # AI gravity and move
            if not done_ai:
                # AI chooses best action
                next_states = {tuple(v):k for k, v in ai_env.get_next_states().items()}
                if next_states:
                    best_state = ai_agent.best_state(next_states.keys())
                    best_action = next_states[best_state]
                    ai_env.play(best_action[0], best_action[1])
                    if ai_env.game_over:
                        print('AI: Game Over!')
                        done_ai = True
            last_gravity_time = now
            render_dual(human_env, ai_env)

        # End if both are done
        if done_human and done_ai:
            print('Both players have finished! Press r to restart or esc to quit.')
            key = cv2.waitKey(0)
            if key == KEY_R:
                human_env.reset()
                ai_env.reset()
                done_human = False
                done_ai = False
                last_gravity_time = time.time()
                render_dual(human_env, ai_env)
            elif key == KEY_ESC:
                break
            else:
                continue

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 