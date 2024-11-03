import sys

from search_engine import SearchEngine
from tools import (
    make_move,
    StoneMove,
    init_board,
    log_to_file,
    msg2move,
    print_board,
    is_win_by_premove,
    move2msg,
    time,
    unmake_move,
)
from defines import Defines, StonePosition


class GameEngine:
    def __init__(self, name=Defines.ENGINE_NAME):
        if name and len(name) > 0:
            if len(name) < Defines.MSG_LENGTH:
                self.m_engine_name = name
            else:
                print(f"Too long Engine Name: {name}, should be less than: {Defines.MSG_LENGTH}")
        self.m_alphabeta_depth = 6
        self.m_board = t = [[0] * Defines.GRID_NUM for i in range(Defines.GRID_NUM)]
        self.init_game()
        self.m_search_engine = SearchEngine()
        self.m_best_move = StoneMove()

    def init_game(self):
        init_board(self.m_board)

    def on_help(self):
        print(
            f"On help for GameEngine {self.m_engine_name}\n"
            " name        - print the name of the Game Engine.\n"
            " print       - print the board.\n"
            " exit/quit   - quit the game.\n"
            " black XXXX  - place the black stone on the position XXXX on the board.\n"
            " white XXXX  - place the white stone on the position XXXX on the board, X is from A to S.\n"
            " next        - the engine will search the move for the next step.\n"
            " move XXXX   - tell the engine that the opponent made the move XXXX,\n"
            "              and the engine will search the move for the next step.\n"
            " new black   - start a new game and set the engine to black player.\n"
            " new white   - start a new game and set it to white.\n"
            " depth d     - set the alpha beta search depth, default is 6.\n"
            " vcf         - set vcf search.\n"
            " unvcf       - set none vcf search.\n"
            " help        - print this help.\n")

    def run(self):
        msg = ""
        self.on_help()
        while True:
            msg = input().strip()
            log_to_file(msg)
            if msg == "name":
                print(f"avb")

            elif msg == "exit" or msg == "quit":
                break

            elif msg == "print":
                print(f"previous move was: ")
                for move in self.m_best_move.positions:
                    print(move.x, move.y)
                print_board(self.m_board, self.m_best_move)

            elif msg == "vcf":
                self.m_vcf = True

            elif msg == "unvcf":
                self.m_vcf = False

            elif msg.startswith("black"):
                self.m_best_move = msg2move(msg[6:])
                make_move(self.m_board, self.m_best_move, Defines.BLACK)
                self.m_chess_type = Defines.BLACK
            elif msg.startswith("white"):
                self.m_best_move = msg2move(msg[6:])
                make_move(self.m_board, self.m_best_move, Defines.WHITE)
                self.m_chess_type = Defines.WHITE
            elif msg == "next":
                # Change the color of the stone of the next move using XOR
                self.m_chess_type = self.m_chess_type ^ 3

                result, move = self.mini_max(2, True)
                print(move2msg(move))
                self.m_best_move = move
                make_move(self.m_board, move, self.m_chess_type)

            elif msg.startswith("new"):
                self.init_game()
                if msg[4:] == "black":
                    self.m_best_move = msg2move("JJ")
                    make_move(self.m_board, self.m_best_move, Defines.BLACK)
                    self.m_chess_type = Defines.BLACK
                    msg = "move JJ"
                    print(msg)
                    # flush_output()
                else:
                    self.m_chess_type = Defines.WHITE
            elif msg.startswith("move"):
                self.m_best_move = msg2move(msg[5:])
                make_move(self.m_board, self.m_best_move, self.m_chess_type ^ 3)
                if is_win_by_premove(self.m_board, self.m_best_move):
                    print("We lost!")
                print("After checking game state")
                for move in self.m_best_move.positions:
                    print(move.x, move.y)
                if self.search_a_move(self.m_chess_type, self.m_best_move):
                    msg = f"move {move2msg(self.m_best_move)}"
                    make_move(self.m_board, self.m_best_move, self.m_chess_type)
                    print(msg)
                    # flush_output()
            elif msg.startswith("depth"):
                d = int(msg[6:])
                if 0 < d < 10:
                    self.m_alphabeta_depth = d
                print(f"Set the search depth to {self.m_alphabeta_depth}.\n")
            elif msg == "help":
                self.on_help()
        return 0

    def search_a_move(self, ourColor, bestMove):
        score = 0
        start = 0
        end = 0

        start = time.perf_counter()
        self.m_search_engine.before_search(self.m_board, self.m_chess_type, self.m_alphabeta_depth)
        score = self.m_search_engine.alpha_beta_search(
            self.m_alphabeta_depth, Defines.MININT, Defines.MAXINT, ourColor, bestMove, bestMove
        )
        end = time.perf_counter()

        print(f"AB Time:\t{end - start:.3f}")
        print(f"Node:\t{self.m_search_engine.m_total_nodes}\n")
        print(f"Score:\t{score:.3f}")
        return True

    def mini_max(self, depth, maxPlayer):
        result_move = None

        # If game has not ended or depth is 0
        # then evaluate the node and return the value
        if depth == 0 or is_win_by_premove(self.m_board, self.m_best_move):
            return self.naive_static_evaluation(), None

        # Now start the search
        # if maxPlayer is True, then we are looking for the best move
        # else we are looking for the worst move
        if maxPlayer:
            q = Defines.MININT
            for move in self.propose_naive_moves():
                move_str = move2msg(move)
                print(move_str)
                aux = self.m_best_move
                self.m_best_move = move
                make_move(self.m_board, move, Defines.BLACK)
                result, _ = self.mini_max(depth - 1, False)
                unmake_move(self.m_board, move)
                self.m_best_move = aux

                if result > q:
                    q = result
                    result_move = move
                    print("Best move: ", move2msg(move))

        else:
            q = Defines.MAXINT
            for move in self.propose_naive_moves():
                move_str = move2msg(move)
                print(move_str)
                aux = self.m_best_move
                self.m_best_move = move
                make_move(self.m_board, move, Defines.WHITE)
                result, _ = self.mini_max(depth - 1, True)
                unmake_move(self.m_board, move)
                self.m_best_move = aux

                if result < q:
                    q = result
                    result_move = move
                    print("Best move: ", move2msg(move))

        return result, result_move

    def propose_naive_moves(self, limit=10, max_radius=2):
        """
        Proposes a list of possible next moves by checking empty spaces around the latest two moves.
        The search is limited to a radius around the previous moves to avoid scanning the entire board.
        Limits the number of moves returned to avoid combinatorial explosion.

        Parameters:
        - limit: Maximum number of moves to return.
        - radius: How far around the previous moves to search for potential moves.
        """
        # Get the two latest moves
        last_move_1 = self.m_best_move.positions[-1]
        last_move_2 = self.m_best_move.positions[-2]
        print(move2msg(self.m_best_move))
        last_positions = [(last_move_1.x, last_move_1.y), (last_move_2.x, last_move_2.y)]

        possible_moves = set()  # Use a set to avoid duplicates
        for radius in range(1, max_radius):
            for last_x, last_y in last_positions:
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        # Skip positions that are outside the current radius
                        if abs(dx) != radius and abs(dy) != radius:
                            continue

                        x = last_x + dx
                        y = last_y + dy

                        # Check if the position is within bounds and is an empty space
                        if (
                                0 <= x < Defines.GRID_NUM and
                                0 <= y < Defines.GRID_NUM and
                                self.m_board[x][y] == Defines.NOSTONE
                        ):
                            stone_position = StonePosition(x, y)
                            possible_moves.add(
                                StoneMove(positions=[stone_position, stone_position])  # Add the move twice
                            )

                        if len(possible_moves) >= limit:
                            return list(possible_moves)  # Return early if the limit is reached

        return list(possible_moves)

    def naive_static_evaluation(self):
        """
        Evaluates the current state of the game.
        Returns MAX if player 1 wins, MIN if player 2 wins, 0 for a draw.
        If the game is ongoing, returns a heuristic value indicating board advantage.
        """
        if is_win_by_premove(self.m_board, self.m_best_move):
            # Check who won based on the current move
            if self.m_chess_type == 1:  # Assuming player 1 is represented by 1
                return float('inf')  # MAX value for player 1 victory
            else:
                return float('-inf')  # MIN value for player 2 victory
        elif self.is_board_full():
            return 0  # Draw
        else:
            # Heuristic: count potential winning opportunities (number of "live" stones)
            return self.count_living_stones()

    def count_living_stones(self):
        """
        Counts the number of living stones (stones that can potentially form a line)
        as a heuristic for board advantage.
        """
        living_stones_p1 = 0
        living_stones_p2 = 0

        for x in range(Defines.GRID_NUM):
            for y in range(Defines.GRID_NUM):
                if self.m_board[x][y] == 1:  # Player 1 stones
                    living_stones_p1 += self.check_potential_lines(x, y, 1)
                elif self.m_board[x][y] == 2:  # Player 2 stones
                    living_stones_p2 += self.check_potential_lines(x, y, 2)

        return living_stones_p1 - living_stones_p2  # Advantage for player 1

    def check_potential_lines(self, x, y, player):
        """
        Check if the current stone at (x, y) can form part of a potential line for the given player.

        Parameters:
        - x, y: Coordinates of the stone to check.
        - player: The player (1 or 2) whose stones we are checking.

        Returns:
        - 1 if the stone can potentially form part of a live line, 0 otherwise.
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Horizontal, Vertical, Diagonal1, Diagonal2
        opponent = 2 if player == 1 else 1

        for direction in directions:
            count = 1  # Start with the current stone
            blocked = False

            # Check in the positive direction
            count += self.count_consecutive_stones(x, y, direction[0], direction[1], player, opponent)
            # Check in the negative direction
            count += self.count_consecutive_stones(x, y, -direction[0], -direction[1], player, opponent)

            # Check if this line could be extended to 6 stones without being blocked
            if count >= 6:
                return 1  # Potential line found

        return 0  # No potential line found

    def count_consecutive_stones(self, x, y, dx, dy, player, opponent):
        """
        Counts consecutive stones of the same player in a specific direction.
        Stops counting if an opponent's stone is encountered.

        Parameters:
        - x, y: Starting coordinates.
        - dx, dy: The direction to move in (e.g., (0, 1) for horizontal).
        - player: The player whose stones are being checked.
        - opponent: The opponent's stone type.

        Returns:
        - The count of consecutive stones in the given direction.
        """
        count = 0
        x += dx
        y += dy

        while 0 <= x < Defines.GRID_NUM and 0 <= y < Defines.GRID_NUM:
            if self.m_board[x][y] == player:
                count += 1
            elif self.m_board[x][y] == opponent:
                break  # Stop counting if we encounter an opponent's stone
            else:
                break  # Stop if we reach an empty space

            x += dx
            y += dy

        return count

    def is_board_full(self):
        """
        Check if the board is full (no more empty spaces).
        """
        for row in self.m_board:
            if Defines.NOSTONE in row:
                return False
        return True


def flush_output():
    sys.stdout.flush()


# Create an instance of GameEngine and run the game
if __name__ == "__main__":
    game_engine = GameEngine()
    game_engine.run()
