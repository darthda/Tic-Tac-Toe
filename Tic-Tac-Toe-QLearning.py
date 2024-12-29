import random
import matplotlib.pyplot as plt

# Spielfeld erstellen
def create_board():
    return [" " for _ in range(9)]

# Gewinner überprüfen
def is_winner(board, marker):
    win_positions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], # Horizontal
        [0, 3, 6], [1, 4, 7], [2, 5, 8], # Vertikal
        [0, 4, 8], [2, 4, 6]             # Diagonal
    ]
    return any(all(board[pos] == marker for pos in win) for win in win_positions)

# Unentschieden überprüfen
def is_draw(board):
    return " " not in board

# Zufälligen Zug wählen
def random_move(board):
    return random.choice([i for i in range(9) if board[i] == " "])

# Q-Learning-basierte KI
class QLearningBot:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay_win=0.995, exploration_boost_loss=0.02):
        self.q_table = {}  # Q-Werte: {(state, action): q_value}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_win = exploration_decay_win
        self.exploration_boost_loss = exploration_boost_loss

    def get_state(self, board):
        return tuple(board)  # Zustand als Tupel des Bretts

    def choose_action(self, board):
        state = self.get_state(board)
        if random.random() < self.exploration_rate:
            return random_move(board)  # Exploration: Zufällige Aktion
        # Exploitation: Beste bekannte Aktion
        q_values = {a: self.q_table.get((state, a), 0) for a in range(9) if board[a] == " "}
        return max(q_values, key=q_values.get, default=random_move(board))

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.q_table.get((state, action), 0)
        max_future_q = max(
            [self.q_table.get((next_state, a), 0) for a in range(9) if next_state[a] == " "],
            default=0
        )
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[(state, action)] = new_q

    def adjust_exploration(self, won):
        if won:
            self.exploration_rate *= self.exploration_decay_win
        else:
            self.exploration_rate = min(self.exploration_rate + self.exploration_boost_loss, 1.0)

# Ein Spiel simulieren
def play_game(starting_player, bot1, bot2, use_bot1=True, use_bot2=False):
    board = create_board()
    players = ["X", "O"] if starting_player == "X" else ["O", "X"]
    first_moves = {players[0]: None, players[1]: None}
    winner = None

    state_action_pairs_bot1 = []
    state_action_pairs_bot2 = []

    for turn in range(9):
        current_player = players[turn % 2]
        if current_player == "X" and use_bot1:
            action = bot1.choose_action(board)
        elif current_player == "O" and use_bot2:
            action = bot2.choose_action(board)
        else:
            action = random_move(board)

        if turn == 0:
            first_moves[players[0]] = action + 1  # 1-basiert machen
        elif turn == 1:
            first_moves[players[1]] = action + 1

        state = bot1.get_state(board) if current_player == "X" else bot2.get_state(board)
        board[action] = current_player

        if current_player == "X" and use_bot1:
            state_action_pairs_bot1.append((state, action))
        elif current_player == "O" and use_bot2:
            state_action_pairs_bot2.append((state, action))

        if is_winner(board, current_player):
            winner = current_player
            break

        if is_draw(board):
            break

    # Belohnungen verteilen
    reward_bot1 = 1 if winner == "X" else -1 if winner == "O" else 0
    reward_bot2 = 1 if winner == "O" else -1 if winner == "X" else 0

    for state, action in reversed(state_action_pairs_bot1):
        next_state = bot1.get_state(board)
        bot1.update_q_value(state, action, reward_bot1, next_state)
        reward_bot1 = 0

    for state, action in reversed(state_action_pairs_bot2):
        next_state = bot2.get_state(board)
        bot2.update_q_value(state, action, reward_bot2, next_state)
        reward_bot2 = 0

    bot1.adjust_exploration(winner == "X")
    bot2.adjust_exploration(winner == "O")

    return first_moves, winner

# Hauptfunktion

def main():
    # Hyperparameter
    num_games = 10000
    learning_rate = 0.1
    discount_factor = 0.9
    exploration_rate = 1.0
    exploration_decay_win = 0.9
    exploration_boost_loss = 0.01

    first_moves_per_game = {"X": [], "O": []}
    win_counts = {"X": 0, "O": 0, "Draw": 0}
    cumulative_wins = {"X": [], "O": []}
    episode_results = []
    starting_player = "X"

    bot1 = QLearningBot(learning_rate, discount_factor, exploration_rate, exploration_decay_win, exploration_boost_loss)
    bot2 = QLearningBot(learning_rate, discount_factor, exploration_rate, exploration_decay_win, exploration_boost_loss)

    # Wählen, welche Spieler die KI verwenden
    use_bot1 = True  # Spieler X nutzt die KI
    use_bot2 = False # Spieler O nutzt die KI (True, wenn gewünscht)

    for game_num in range(1, num_games + 1):
        first_moves, winner = play_game(starting_player, bot1, bot2, use_bot1, use_bot2)
        first_moves_per_game["X"].append(first_moves["X"])
        first_moves_per_game["O"].append(first_moves["O"])

        if winner == "X":
            win_counts[winner] += 1
            episode_results.append(1)  # Sieg
        elif winner == "O":
            win_counts[winner] += 1
            episode_results.append(-1)  # Niederlage
        else:
            win_counts["Draw"] += 1
            episode_results.append(0)  # Unentschieden

        # Gewinnrate berechnen
        cumulative_wins["X"].append(win_counts["X"] / game_num)
        cumulative_wins["O"].append(win_counts["O"] / game_num)

        # Startspieler wechselt
        starting_player = "O" if starting_player == "X" else "X"

    # Ausgabe der Ergebnisse
    print("Erste Felder pro Partie:")
    for i, (x_move, o_move) in enumerate(zip(first_moves_per_game["X"], first_moves_per_game["O"]), start=1):
        print(f"Partie {i}: Spieler X: Feld {x_move}, Spieler O: Feld {o_move}")


    # Visualisierung: Gewinnwahrscheinlichkeiten als Liniendiagramm
    plt.plot(range(1, num_games + 1), cumulative_wins["X"], label="Spieler X", linestyle="-", marker="o")
    plt.plot(range(1, num_games + 1), cumulative_wins["O"], label="Spieler O", linestyle="--", marker="o")
    plt.title("Gewinnwahrscheinlichkeiten")
    plt.xlabel("Partien")
    plt.ylabel("Gewinnwahrscheinlichkeit")
    plt.ylim(0, 1)
    plt.grid()
    plt.legend()
    plt.show()

    # Visualisierung: Episodenergebnisse für Spieler X
    plt.plot(range(1, num_games + 1), episode_results, label="Ergebnis Spieler X", linestyle="-", marker="o")
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title("Episodenergebnisse für Spieler X")
    plt.xlabel("Episoden")
    plt.ylabel("Ergebnis (1=Sieg, 0=Unentschieden, -1=Niederlage)")
    plt.yticks([-1, 0, 1], labels=["Niederlage", "Unentschieden", "Sieg"])
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
