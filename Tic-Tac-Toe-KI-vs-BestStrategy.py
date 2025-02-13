import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Spielfeld erstellen
def create_board():
    return [" " for _ in range(9)]

# Gewinner überprüfen
def is_winner(board, marker):
    win_positions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], # Horizontal
        [0, 3, 6], [1, 4, 7], [2, 5, 8]  # Diagonal
    ]
    return any(all(board[pos] == marker for pos in win) for win in win_positions)

# Unentschieden überprüfen
def is_draw(board):
    return " " not in board

# Zufälligen Zug wählen
def random_move(board):
    return random.choice([i for i in range(9) if board[i] == " "])

# Minimax-Algorithmus
class MinimaxBot:
    def __init__(self):
        pass

    def choose_action(self, board):
        best_score = float('-inf')
        best_move = None

        for move in [i for i in range(9) if board[i] == " "]:
            board[move] = "X"
            score = self.minimax(board, False)
            board[move] = " "
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def minimax(self, board, is_maximizing):
        if is_winner(board, "X"):
            return 1
        if is_winner(board, "O"):
            return -1
        if is_draw(board):
            return 0

        if is_maximizing:
            best_score = float('-inf')
            for move in [i for i in range(9) if board[i] == " "]:
                board[move] = "X"
                score = self.minimax(board, False)
                board[move] = " "
                best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for move in [i for i in range(9) if board[i] == " "]:
                board[move] = "O"
                score = self.minimax(board, True)
                board[move] = " "
                best_score = min(score, best_score)
            return best_score

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

# AlphaZero-basiertes KI-Modell
class AlphaZeroBot:
    def __init__(self, simulations=200):
        self.simulations = simulations
        self.q_values = defaultdict(float)  # Q(s, a): Q-Wert für Zustand und Aktion
        self.visit_counts = defaultdict(int)  # N(s, a): Besuchszähler für Zustand und Aktion
        self.prior_probabilities = defaultdict(float)  # P(s, a): Priorwahrscheinlichkeit für Aktionen

    def get_state(self, board):
        return tuple(board)

    def policy(self, board):
        state = self.get_state(board)
        valid_moves = [i for i in range(9) if board[i] == " "]
        priorities = [3 if i == 4 else 2 if i in [0, 2, 6, 8] else 1 for i in valid_moves]
        probabilities = np.array(priorities) / sum(priorities)
        return dict(zip(valid_moves, probabilities))

    def mcts(self, board, player):
        for _ in range(self.simulations):
            self.simulate(board, player)

    def simulate(self, board, player):
        path = []
        state = self.get_state(board)
        while True:
            valid_moves = [i for i in range(9) if board[i] == " "]
            if not valid_moves or is_winner(board, "X") or is_winner(board, "O"):
                break
            action = self.select_action(state, valid_moves)
            path.append((state, action))
            board[action] = player
            state = self.get_state(board)
            player = "O" if player == "X" else "X"

        reward = self.evaluate_game(board)
        for state, action in path:
            self.visit_counts[(state, action)] += 1
            self.q_values[(state, action)] += (reward - self.q_values[(state, action)]) / self.visit_counts[(state, action)]

    def select_action(self, state, valid_moves):
        if state not in self.prior_probabilities:
            probabilities = self.policy([" " if s == " " else s for s in state])
            self.prior_probabilities.update(probabilities)
        total_counts = sum(self.visit_counts[(state, a)] for a in valid_moves)
        ucb_values = {
            a: self.q_values[(state, a)] + 1 * np.sqrt(np.log(total_counts + 1) / (self.visit_counts[(state, a)] + 1))
            for a in valid_moves
        }
        return max(valid_moves, key=lambda a: ucb_values[a])

    def evaluate_game(self, board):
        if is_winner(board, "X"):
            return 1  # Angepasste Belohnung für Sieg
        elif is_winner(board, "O"):
            return -3  # Angepasste Bestrafung für Verlust
        elif is_draw(board):
            return 0.1  # Leichte Belohnung für Unentschieden
        return 0

    def choose_action(self, board):
        self.mcts(board[:], "X")
        state = self.get_state(board)
        valid_moves = [i for i in range(9) if board[i] == " "]
        action = max(valid_moves, key=lambda a: self.visit_counts[(state, a)])
        return action

# Ein Spiel simulieren
def play_game(starting_player, bot1, bot2, use_bot1=True, use_bot2=True):
    board = create_board()
    players = ["X", "O"] if starting_player == "X" else ["O", "X"]
    first_moves = {players[0]: None, players[1]: None}
    winner = None

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

        board[action] = current_player

        if is_winner(board, current_player):
            winner = current_player
            break

        if is_draw(board):
            break

    return first_moves, winner

# Hauptfunktion

def main():
    # Hyperparameter
    num_games = 100
    first_moves_per_game = {"X": [], "O": []}
    win_counts = {"X": 0, "O": 0, "Draw": 0}
    cumulative_wins = {"X": [], "O": []}
    episode_results = []
    starting_player = "X"

    # Konfiguration der Bots
    bot1 = MinimaxBot()  # Spieler X nutzt Minimax
    bot2_type = "q"  # Wähle zwischen "random", "q", "alpha_zero"

    if bot2_type == "random":
        bot2 = None  # Spieler O ist Random
    elif bot2_type == "q":
        bot2 = QLearningBot(learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0)
    elif bot2_type == "alpha_zero":
        bot2 = AlphaZeroBot(simulations=200)
    else:
        raise ValueError("Ungültiger Bot-Typ für Spieler O")

    # Wählen, welche Spieler die KI verwenden
    use_bot1 = True  # Spieler X nutzt die Minimax KI
    use_bot2 = bot2_type != "random"  # Spieler O nutzt eine KI, falls nicht Random

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
