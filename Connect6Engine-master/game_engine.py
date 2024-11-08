import sys
from itertools import combinations

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
        self.stones_placed = []
        self.num_of_total_stones = 0
        self.expanded_nodes = 0
        self.pruned_nodes = 0

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
                for stone in self.m_best_move.positions:
                    self.num_of_total_stones += 1
                    self.stones_placed.append((stone.x, stone.y, self.m_chess_type))
            elif msg.startswith("white"):
                self.m_best_move = msg2move(msg[6:])
                make_move(self.m_board, self.m_best_move, Defines.WHITE)
                self.m_chess_type = Defines.WHITE
                for stone in self.m_best_move.positions:
                    self.num_of_total_stones += 1
                    self.stones_placed.append((stone.x, stone.y, self.m_chess_type))
            elif msg == "next":
                # Change the color of the stone of the next move using XOR
                self.m_chess_type = self.m_chess_type ^ 3
                # Start the timer
                start_time = time.time()

                # Run the MiniMax search
                # result, move = self.mini_max(2, True)

                # Run the Alpha-Beta search
                _, move = self.alpha_beta(4, Defines.MININT, Defines.MAXINT, True)

                # End the timer
                end_time = time.time()

                # Calculate decision time
                decision_time = end_time - start_time

                # Log decision time and explored/pruned nodes
                print(f"Decision Time: {decision_time:.4f} seconds")
                print(f"Explored nodes: {self.expanded_nodes}, Pruned nodes: {self.pruned_nodes}")

                self.m_best_move = move
                make_move(self.m_board, move, self.m_chess_type)
                for stone in self.m_best_move.positions:
                    self.num_of_total_stones += 1
                    self.stones_placed.append((stone.x, stone.y, self.m_chess_type))
                msg = f"move {move2msg(self.m_best_move)}"
                print(msg)

            elif msg.startswith("new"):
                self.init_game()
                if msg[4:] == "black":
                    self.m_best_move = msg2move("JJ")
                    make_move(self.m_board, self.m_best_move, Defines.BLACK)
                    self.m_chess_type = Defines.BLACK
                    for stone in self.m_best_move.positions:
                        self.num_of_total_stones += 1
                        self.stones_placed.append((stone.x, stone.y, self.m_chess_type))
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
            total_stones = sum(
                1 for x in range(Defines.GRID_NUM) for y in range(Defines.GRID_NUM)
                if self.m_board[x][y] != Defines.NOSTONE
            )
            if total_stones >= 4:
                moves = self.propose_smart_moves()
            else:
                moves = self.propose_naive_moves()
            for move in moves:
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

            return result, result_move

        else:
            q = Defines.MAXINT
            total_stones = sum(
                1 for x in range(Defines.GRID_NUM) for y in range(Defines.GRID_NUM)
                if self.m_board[x][y] != Defines.NOSTONE
            )
            if total_stones >= 4:
                moves = self.propose_smart_moves()
            else:
                moves = self.propose_naive_moves()
            for move in moves:
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

    def alpha_beta(self, depth, alpha, beta, maxPlayer):
        if depth == 0 or is_win_by_premove(self.m_board, self.m_best_move):
            return self.evaluate_board(), None

        if maxPlayer:
            max_eval = Defines.MININT
            best_move = None
            if self.num_of_total_stones == 0:
                stone = StonePosition(10, 10)
                move = StoneMove(positions=[stone, stone])
                return None, move
            if self.num_of_total_stones <= 3:
                stone1 = StonePosition(9, 10)
                stone2 = StonePosition(10, 11)
                move = StoneMove(positions=[stone1, stone2])
                return None, move
            moves = self.propose_smart_moves_2()
            for move in moves:
                self.expanded_nodes += 1
                aux = self.m_best_move
                self.m_best_move = move
                make_move(self.m_board, move, Defines.BLACK)
                eval, _ = self.alpha_beta(depth - 1, alpha, beta, False)
                unmake_move(self.m_board, move)
                self.m_best_move = aux

                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)

                if beta <= alpha:
                    self.pruned_nodes += 1
                    break  # Beta cut-off

            return max_eval, best_move
        else:
            min_eval = Defines.MAXINT
            best_move = None
            if self.num_of_total_stones == 0:
                stone = StonePosition(10, 10)
                move = StoneMove(positions=[stone, stone])
                return None, move
            if self.num_of_total_stones <= 3:
                stone1 = StonePosition(9, 10)
                stone2 = StonePosition(10, 11)
                move = StoneMove(positions=[stone1, stone2])
                return None, move
            moves = self.propose_smart_moves_2()
            for move in moves:
                self.expanded_nodes += 1
                aux = self.m_best_move
                self.m_best_move = move
                make_move(self.m_board, move, Defines.WHITE)
                eval, _ = self.alpha_beta(depth - 1, alpha, beta, True)
                unmake_move(self.m_board, move)
                self.m_best_move = aux

                if eval < min_eval:
                    min_eval = eval
                    best_move = move

                beta = min(beta, eval)

                if beta <= alpha:
                    self.pruned_nodes += 1
                    break

            return min_eval, best_move

    def propose_naive_moves(self, limit=10, max_radius=2):
        """
        Proposes a list of possible next moves by checking empty spaces around the latest two moves.
        Each move consists of two stone placements. The search starts from radius 1 and increases
        up to max_radius to prioritize closer moves first.

        Parameters:
        - limit: Maximum number of move pairs to return.
        - max_radius: The maximum radius to search around the previous moves.
        """
        # Get the two latest moves
        last_move_1 = self.m_best_move.positions[-1]
        last_move_2 = self.m_best_move.positions[-2]
        last_positions = [(last_move_1.x, last_move_1.y), (last_move_2.x, last_move_2.y)]

        # Collect all potential single moves in a set to avoid duplicates
        potential_moves = set()
        for radius in range(1, max_radius + 1):  # Increment the radius from 1 to max_radius
            for last_x, last_y in last_positions:
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        # Skip positions that are outside the current radius
                        if abs(dx) != radius and abs(dy) != radius:
                            continue

                        x = last_x + dx
                        y = last_y + dy

                        # Check if the position is within bounds and is an empty space
                        if 0 <= x < Defines.GRID_NUM and 0 <= y < Defines.GRID_NUM and self.m_board[x][
                            y] == Defines.NOSTONE:
                            potential_moves.add((x, y))  # Add the empty position to the set

                        if len(potential_moves) >= limit * 2:  # Collect enough positions for move pairs
                            break

        # Generate all combinations of two moves from the potential moves
        move_pairs = []
        for pos1, pos2 in combinations(potential_moves, 2):
            stone_position1 = StonePosition(pos1[0], pos1[1])
            stone_position2 = StonePosition(pos2[0], pos2[1])
            move = StoneMove(positions=[stone_position1, stone_position2])
            move_pairs.append(move)
            if len(move_pairs) >= limit:  # Stop if we have enough move pairs
                break

        return move_pairs

    def propose_smart_moves(self, limit=10, max_radius=2):
        """
        Proposes a list of smart next moves, focusing on attack and defense threats.
        Generates pairs of moves for the Connect6 two-stone placement rule.

        Parameters:
        - limit: Maximum number of move pairs to return.
        - max_radius: The maximum radius to search around 'alive' stones.
        """
        threat_moves = []
        defensive_moves = []

        # Iterate over the board to detect attack and defense threats
        for x in range(Defines.GRID_NUM):
            for y in range(Defines.GRID_NUM):
                if self.m_board[x][y] == Defines.NOSTONE:
                    # Analyze potential attack and defense benefits of placing stones here
                    score = self.analyze_threats(x, y)
                    if score > 0:
                        threat_moves.append((score, StonePosition(x, y)))
                    elif score < 0:
                        defensive_moves.append((score, StonePosition(x, y)))

        # Sort moves by their scores
        threat_moves.sort(reverse=True, key=lambda x: x[0])  # High scores first
        defensive_moves.sort(key=lambda x: x[0])  # Low scores first

        # Combine attack and defense moves into pairs
        combined_moves = []

        # Generate pairs of moves from the top-ranked threat and defensive moves
        for i in range(min(limit // 2, len(threat_moves))):
            for j in range(min(limit // 2, len(defensive_moves))):
                pos1 = threat_moves[i][1]
                pos2 = defensive_moves[j][1]
                if (pos1.x, pos1.y) != (pos2.x, pos2.y):  # Ensure the two positions are not the same
                    combined_moves.append(StoneMove(positions=[pos1, pos2]))
                    if len(combined_moves) >= limit:
                        return combined_moves  # Return early if limit is reached

        return combined_moves[:limit]  # Return the combined moves, limited to the requested number

    def propose_smart_moves_2(self, limit=15, max_radius=2):
        """
        Proposes a list of smart next moves based on existing stones, focusing on attack and defense
        around the most recently placed stones and living stones.
        Generates pairs of moves for Connect6.

        Parameters:
        - limit: Maximum number of move pairs to return.
        - max_radius: The maximum radius to search around critical stones.
        """
        scored_moves = []

        # Start with the last placed stones as focal points
        last_moves = self.m_best_move.positions[-2:]  # The two most recent stones placed
        critical_stones = [(move.x, move.y) for move in last_moves]

        # Add any other "living stones" that could potentially form a winning line
        # (This assumes we have a list of living stones or stones that can form lines)
        for stone in self.stones_placed:
            x, y, player = stone
            if self.is_living_stone(x, y, player):  # Method to check if a stone is "alive"
                critical_stones.append((x, y))

        # Analyze moves around each critical stone
        for cx, cy in critical_stones:
            for dx in range(-max_radius, max_radius + 1):
                for dy in range(-max_radius, max_radius + 1):
                    if dx == 0 and dy == 0:
                        continue  # Skip the stone's own position

                    x, y = cx + dx, cy + dy
                    if 0 <= x < Defines.GRID_NUM and 0 <= y < Defines.GRID_NUM and self.m_board[x][
                        y] == Defines.NOSTONE:
                        score = self.analyze_threats(x, y)  # Evaluate the impact of placing a stone here
                        scored_moves.append((score, StonePosition(x, y)))

        # Sort moves by their scores (highest scores first for offensive and defensive threats)
        scored_moves.sort(reverse=True, key=lambda item: item[0])

        # Generate pairs of moves from the top-ranked positions
        combined_moves = []
        for i in range(len(scored_moves)):
            for j in range(i + 1, len(scored_moves)):
                pos1 = scored_moves[i][1]
                pos2 = scored_moves[j][1]
                if (pos1.x, pos1.y) != (pos2.x, pos2.y):  # Ensure different positions
                    combined_moves.append(StoneMove(positions=[pos1, pos2]))
                    if len(combined_moves) >= limit:
                        return combined_moves  # Return early if the limit is reached

        return combined_moves[:limit]  # Return the best move pairs up to the limit

    def is_living_stone(self, x, y, player):
        """
        Determines if a stone at (x, y) for the given player is "alive," meaning it has
        potential to form a line of 6 with open ends.

        Returns:
        - True if the stone is part of an open-ended sequence, False otherwise.
        """
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for direction in directions:
            consecutive_count, open_ends = self.count_consecutive_stones_with_open_ends(x, y, direction, player)
            if consecutive_count >= 2 and open_ends > 0:
                return True  # The stone is "alive" as it has open-ended potential
        return False

    def count_consecutive_stones_with_open_ends(self, x, y, direction, player):
        """
        Counts consecutive stones in a given direction and checks for open ends.

        Parameters:
        - x, y: Position of the stone.
        - direction: The direction to check (e.g., (1, 0) for horizontal).
        - player: The player whose stones we are counting.

        Returns:
        - consecutive_count: Number of consecutive stones in this direction.
        - open_ends: Number of open ends (empty spaces around the sequence).
        """
        dx, dy = direction
        consecutive_count = 1
        open_ends = 0

        # Check in the positive direction
        nx, ny = x + dx, y + dy
        while 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM:
            if self.m_board[nx][ny] == player:
                consecutive_count += 1
            elif self.m_board[nx][ny] == Defines.NOSTONE:
                open_ends += 1
                break
            else:
                break
            nx += dx
            ny += dy

        # Check in the negative direction
        nx, ny = x - dx, y - dy
        while 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM:
            if self.m_board[nx][ny] == player:
                consecutive_count += 1
            elif self.m_board[nx][ny] == Defines.NOSTONE:
                open_ends += 1
                break
            else:
                break
            nx -= dx
            ny -= dy

        return consecutive_count, open_ends

    def analyze_threats(self, x, y):
        """
        Analyzes the potential threats and opportunities of placing a stone at (x, y).

        Returns:
        - A positive score if the move is good for the current player (offensive value).
        - A negative score if the move is necessary to block the opponent (defensive value).
        """
        score = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for direction in directions:
            # Evaluate both offensive and defensive potential
            score += self.evaluate_line_threat2(x, y, direction, player=1)  # Offensive
            score -= self.evaluate_line_threat2(x, y, direction, player=2)  # Defensive (block opponent)

        return score

    def evaluate_line_threat(self, x, y, direction, player):
        """
        Evaluates the threat or opportunity of placing a stone at (x, y) in a given direction.

        Parameters:
        - direction: Tuple indicating the direction to check (e.g., (1, 0) for horizontal).
        - player: The player for whom the threat is evaluated.

        Returns:
        - A score representing the strength of the move in that direction.
        """
        dx, dy = direction
        count = 0
        open_ends = 0

        # Check forward in the given direction
        nx, ny = x + dx, y + dy
        while 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM and self.m_board[nx][ny] == player:
            count += 1
            nx += dx
            ny += dy

        # Check if the end is open (empty space)
        if 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM and self.m_board[nx][ny] == Defines.NOSTONE:
            open_ends += 1

        # Check backward in the opposite direction
        nx, ny = x - dx, y - dy
        while 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM and self.m_board[nx][ny] == player:
            count += 1
            nx -= dx
            ny -= dy

        # Check if the end is open (empty space)
        if 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM and self.m_board[nx][ny] == Defines.NOSTONE:
            open_ends += 1

        # Scoring based on the length of the line and open ends
        if count >= 5 and open_ends > 0:
            return 1000  # Winning threat
        elif count == 4 and open_ends > 0:
            return 100  # Very strong
        elif count == 3 and open_ends > 0:
            return 50  # Strong
        elif count == 2 and open_ends > 0:
            return 10  # Weak
        else:
            return 0  # No significant threat

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
        - A score based on the number of consecutive stones found in each direction.
          The score increases as the number of consecutive stones increases.
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Horizontal, Vertical, Diagonal1, Diagonal2
        n = Defines.GRID_NUM
        opponent = 2 if player == 1 else 1
        score = 0

        # Define score weights for different numbers of consecutive stones
        score_weights = {2: 5, 3: 10, 4: 20, 5: 50}

        for direction in directions:
            count = 1  # Start with the current stone

            # Count consecutive stones in the positive direction
            count += self.count_consecutive_stones(x, y, direction[0], direction[1], player, opponent)
            # Count consecutive stones in the negative direction
            count += self.count_consecutive_stones(x, y, -direction[0], -direction[1], player, opponent)

            # If we have 2 or more stones in a row, assign a score based on the count
            if 2 <= count <= 5:
                score += score_weights[count]  # Use the score weights for different counts
            elif count >= 6:
                score += 100  # A very high score if a line of 6 is completed

        return score

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

    def evaluate_board(self):
        """
        Evaluates the board state, considering both offensive and defensive factors.
        """
        score = 0

        for x in range(Defines.GRID_NUM):
            for y in range(Defines.GRID_NUM):
                if self.m_board[x][y] == 1:  # Player 1's stone
                    score += self.evaluate_position(x, y, 1)
                elif self.m_board[x][y] == 2:  # Player 2's stone
                    score -= self.evaluate_position(x, y, 2)

        return score

    def evaluate_position(self, x, y, player):
        """
        Evaluates a position for the given player, assigning higher scores for longer sequences.
        """
        score = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for direction in directions:
            consecutive_stones = self.count_consecutive_stones(x, y, direction, player)
            if consecutive_stones >= 6:
                score += 1000  # Winning line
            elif consecutive_stones == 5:
                score += 100  # Very strong
            elif consecutive_stones == 4:
                score += 50  # Strong
            elif consecutive_stones == 3:
                score += 10  # Decent
            elif consecutive_stones == 2:
                score += 3  # Weak

        return score

    def evaluate_line_threat2(self, x, y, direction, player):
        """
        Evaluates the threat or opportunity of placing a stone at (x, y) in a given direction.

        Parameters:
        - direction: Tuple indicating the direction to check (e.g., (1, 0) for horizontal).
        - player: The player for whom the threat or opportunity is evaluated.

        Returns:
        - A score representing the strength of the move in that direction.
        """
        dx, dy = direction
        count = 0
        open_ends = 0

        # Check forward in the given direction
        nx, ny = x + dx, y + dy
        while 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM and self.m_board[nx][ny] == player:
            count += 1
            nx += dx
            ny += dy

        # Check if the end is open (empty space)
        if 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM and self.m_board[nx][ny] == Defines.NOSTONE:
            open_ends += 1

        # Check backward in the opposite direction
        nx, ny = x - dx, y - dy
        while 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM and self.m_board[nx][ny] == player:
            count += 1
            nx -= dx
            ny -= dy

        # Check if the end is open (empty space)
        if 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM and self.m_board[nx][ny] == Defines.NOSTONE:
            open_ends += 1

        # Scoring based on offensive opportunities and defensive needs
        if count >= 5 and open_ends > 0:
            return 1000  # Winning opportunity for offense
        elif count == 4 and open_ends > 0:
            if player == 1:  # Offensive score
                return 100  # Strong offensive move
            else:  # Defensive score
                return 200  # Even stronger defense to block the opponent
        elif count == 3 and open_ends > 0:
            return 50  # Moderate offensive or defensive value
        elif count == 2 and open_ends > 0:
            return 10  # Weak offensive or defensive value
        else:
            return 0  # No significant impact

    def count_consecutive_stones(self, x, y, direction, player):
        """
        Counts consecutive stones in a given direction for the player.
        """
        count = 0
        dx, dy = direction
        while 0 <= x < Defines.GRID_NUM and 0 <= y < Defines.GRID_NUM and self.m_board[x][y] == player:
            count += 1
            x += dx
            y += dy
        return count


def flush_output():
    sys.stdout.flush()


# Create an instance of GameEngine and run the game
if __name__ == "__main__":
    game_engine = GameEngine()
    game_engine.run()
