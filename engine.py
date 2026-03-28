"""
MIT Pokerbots game engine (TACS 2026 variant).
"""

from collections import namedtuple
from queue import Queue
from threading import Thread
import json
import math
import os
import socket
import subprocess
import sys
import time

import pkrbot

sys.path.append(os.getcwd())
from config import *  # noqa: E402,F403


RedrawAction = namedtuple("RedrawAction", ["target_type", "target_index", "action"])
FoldAction = namedtuple("FoldAction", [])
CallAction = namedtuple("CallAction", [])
CheckAction = namedtuple("CheckAction", [])
RaiseAction = namedtuple("RaiseAction", ["amount"])

TerminalState = namedtuple("TerminalState", ["deltas", "previous_state"])

STREET_NAMES = {0: "Preflop", 3: "Flop", 4: "Turn", 5: "River"}
CCARDS = lambda cards: ",".join(map(str, cards))
PCARDS = lambda cards: "[{}]".format(" ".join(map(str, cards)))
PVALUE = lambda name, value: ", {} ({})".format(name, value)
STATUS = lambda players: "".join([PVALUE(p.name, p.bankroll) for p in players])


def _resolve_output_path(filename):
    """
    Resolves output files into RESULTS_DIR unless a directory is already provided.
    """
    if os.path.isabs(filename) or os.path.dirname(filename):
        return filename
    return os.path.join(RESULTS_DIR, filename)


class RoundState(
    namedtuple(
        "_RoundState",
        [
            "button",
            "street",
            "pips",
            "stacks",
            "hands",
            "deck",
            "board",
            "redraws_used",
            "previous_state",
        ],
    )
):
    """Encodes one round of poker."""

    def _pot_size(self):
        return 2 * STARTING_STACK - self.stacks[0] - self.stacks[1]

    def get_delta(self, winner_index):
        """Returns the bankroll delta for player 0."""
        assert winner_index in (0, 1, 2)
        pot = self._pot_size()
        p0_contribution = STARTING_STACK - self.stacks[0]
        if winner_index == 0:
            delta = pot - p0_contribution
        elif winner_index == 1:
            delta = -p0_contribution
        else:
            delta = (pot / 2.0) - p0_contribution
        if abs(delta - round(delta)) > 1e-9:
            delta = math.floor(delta) if self.button % 2 == 0 else math.ceil(delta)
        return int(delta)

    def showdown(self):
        """Compares both 7-card hands via pkrbot."""
        score0 = pkrbot.evaluate(self.hands[0] + self.board)
        score1 = pkrbot.evaluate(self.hands[1] + self.board)
        if score0 > score1:
            delta = self.get_delta(0)
        elif score1 > score0:
            delta = self.get_delta(1)
        else:
            delta = self.get_delta(2)
        return TerminalState([delta, -delta], self)

    def _board_target_limit(self):
        if self.street < 3:
            return -1
        if self.street == 3:
            return 2
        if self.street == 4:
            return 3
        return 4

    def _is_valid_redraw_target(self, active, target_type, target_index):
        if self.redraws_used[active] or self.street >= 5:
            return False
        if target_type == "hole":
            return 0 <= target_index <= 1
        if target_type == "board":
            return 0 <= target_index <= self._board_target_limit()
        return False

    def peek_redraw_old_card(self, active, target_type, target_index):
        if not self._is_valid_redraw_target(active, target_type, target_index):
            return None
        if target_type == "hole":
            return self.hands[active][target_index]
        return self.board[target_index]

    def legal_actions(self):
        """Returns a set of legal action classes for the active player."""
        active = self.button % 2
        continue_cost = self.pips[1 - active] - self.pips[active]

        actions = {FoldAction}
        if continue_cost == 0:
            actions.add(CheckAction)
            bets_forbidden = self.stacks[active] == 0 or self.stacks[1 - active] == 0
            if not bets_forbidden:
                actions.add(RaiseAction)
        else:
            actions.add(CallAction)
            raises_forbidden = (
                continue_cost >= self.stacks[active] or self.stacks[1 - active] == 0
            )
            if not raises_forbidden:
                actions.add(RaiseAction)

        if self.street < 5 and not self.redraws_used[active]:
            actions.add(RedrawAction)
        return actions

    def raise_bounds(self):
        """Returns (min_raise_to, max_raise_to) for the active player."""
        active = self.button % 2
        continue_cost = self.pips[1 - active] - self.pips[active]
        max_contribution = min(
            self.stacks[active],
            self.stacks[1 - active] + continue_cost,
        )
        min_contribution = min(
            max_contribution,
            continue_cost + max(continue_cost, BIG_BLIND),
        )
        return (
            self.pips[active] + min_contribution,
            self.pips[active] + max_contribution,
        )

    def _advance_street_no_showdown(self):
        """Advances to the next street and deals community cards."""
        new_board = list(self.board)
        if self.street == 0:
            new_street = 3
            new_board.extend(self.deck.deal(3))
        elif self.street == 3:
            new_street = 4
            new_board.extend(self.deck.deal(1))
        elif self.street == 4:
            new_street = 5
            new_board.extend(self.deck.deal(1))
        else:
            return self
        return RoundState(
            1,  # BB acts first post-flop
            new_street,
            [0, 0],
            list(self.stacks),
            [list(self.hands[0]), list(self.hands[1])],
            self.deck,
            new_board,
            list(self.redraws_used),
            self,
        )

    def proceed_street(self):
        """Closes a betting street and advances game progression."""
        if self.street == 5:
            return self.showdown()

        state = self._advance_street_no_showdown()

        # If anyone is all-in after a closed street, run out to river then showdown.
        if 0 in state.stacks:
            while state.street < 5:
                state = state._advance_street_no_showdown()
            return state.showdown()

        return state

    def _proceed_betting_action(self, action):
        active = self.button % 2

        if isinstance(action, FoldAction):
            delta = self.get_delta(1 - active)
            return TerminalState([delta, -delta], self)

        if isinstance(action, CallAction):
            if self.street == 0 and self.button == 0:
                # SB completes to BB; BB acts next.
                return RoundState(
                    1,
                    0,
                    [BIG_BLIND, BIG_BLIND],
                    [STARTING_STACK - BIG_BLIND, STARTING_STACK - BIG_BLIND],
                    [list(self.hands[0]), list(self.hands[1])],
                    self.deck,
                    list(self.board),
                    list(self.redraws_used),
                    self,
                )

            new_pips = list(self.pips)
            new_stacks = list(self.stacks)
            contribution = new_pips[1 - active] - new_pips[active]
            new_stacks[active] -= contribution
            new_pips[active] += contribution
            state = RoundState(
                self.button + 1,
                self.street,
                new_pips,
                new_stacks,
                [list(self.hands[0]), list(self.hands[1])],
                self.deck,
                list(self.board),
                list(self.redraws_used),
                self,
            )
            return state.proceed_street()

        if isinstance(action, CheckAction):
            both_acted = (self.street == 0 and self.button > 0) or (
                self.street > 0 and self.button > 1
            )
            if both_acted:
                return self.proceed_street()
            return RoundState(
                self.button + 1,
                self.street,
                list(self.pips),
                list(self.stacks),
                [list(self.hands[0]), list(self.hands[1])],
                self.deck,
                list(self.board),
                list(self.redraws_used),
                self,
            )

        # RaiseAction
        new_pips = list(self.pips)
        new_stacks = list(self.stacks)
        contribution = action.amount - new_pips[active]
        new_stacks[active] -= contribution
        new_pips[active] += contribution
        return RoundState(
            self.button + 1,
            self.street,
            new_pips,
            new_stacks,
            [list(self.hands[0]), list(self.hands[1])],
            self.deck,
            list(self.board),
            list(self.redraws_used),
            self,
        )

    def proceed(self, action):
        """Advances game tree by one action."""
        active = self.button % 2
        if isinstance(action, RedrawAction):
            inner_action = action.action
            target_type = action.target_type
            target_index = action.target_index

            if self._is_valid_redraw_target(active, target_type, target_index):
                hands = [list(self.hands[0]), list(self.hands[1])]
                board = list(self.board)
                redraws_used = list(self.redraws_used)
                new_card = self.deck.deal(1)[0]
                if target_type == "hole":
                    hands[active][target_index] = new_card
                else:
                    board[target_index] = new_card
                redraws_used[active] = True
                state_after_redraw = RoundState(
                    self.button,
                    self.street,
                    list(self.pips),
                    list(self.stacks),
                    hands,
                    self.deck,
                    board,
                    redraws_used,
                    self,
                )
                return state_after_redraw._proceed_betting_action(inner_action)

            # Invalid redraw target: ignore redraw, only process inner betting action.
            return self._proceed_betting_action(inner_action)

        return self._proceed_betting_action(action)


class Player:
    """Handles subprocess and socket interactions with one pokerbot."""

    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.game_clock = STARTING_GAME_CLOCK
        self.bankroll = 0
        self.commands = None
        self.bot_subprocess = None
        self.socketfile = None
        self.bytes_queue = Queue()

    def build(self):
        try:
            with open(self.path + "/commands.json", "r") as json_file:
                commands = json.load(json_file)
            if (
                "build" in commands
                and "run" in commands
                and isinstance(commands["build"], list)
                and isinstance(commands["run"], list)
            ):
                self.commands = commands
            else:
                print(self.name, "commands.json missing command")
        except FileNotFoundError:
            print(self.name, "commands.json not found - check PLAYER_PATH")
        except json.decoder.JSONDecodeError:
            print(self.name, "commands.json misformatted")

        if self.commands is not None and len(self.commands["build"]) > 0:
            try:
                proc = subprocess.run(
                    self.commands["build"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=self.path,
                    timeout=BUILD_TIMEOUT,
                    check=False,
                )
                self.bytes_queue.put(proc.stdout)
            except subprocess.TimeoutExpired as timeout_expired:
                error_message = "Timed out waiting for " + self.name + " to build"
                print(error_message)
                self.bytes_queue.put(timeout_expired.stdout)
                self.bytes_queue.put(error_message.encode())
            except (TypeError, ValueError):
                print(self.name, "build command misformatted")
            except OSError:
                print(self.name, 'build failed - check "build" in commands.json')

    def run(self):
        if self.commands is not None and len(self.commands["run"]) > 0:
            try:
                server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                with server_socket:
                    server_socket.bind(("", 0))
                    server_socket.settimeout(CONNECT_TIMEOUT)
                    server_socket.listen()
                    port = server_socket.getsockname()[1]
                    proc = subprocess.Popen(
                        self.commands["run"] + [str(port)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        cwd=self.path,
                    )
                    self.bot_subprocess = proc

                    def enqueue_output(out, queue):
                        try:
                            for line in out:
                                if self.path == r"./player_chatbot":
                                    print(line.strip().decode("utf-8"))
                                else:
                                    queue.put(line)
                        except ValueError:
                            pass

                    Thread(
                        target=enqueue_output,
                        args=(proc.stdout, self.bytes_queue),
                        daemon=True,
                    ).start()
                    client_socket, _ = server_socket.accept()
                    with client_socket:
                        if self.path == r"./player_chatbot":
                            client_socket.settimeout(PLAYER_TIMEOUT)
                        else:
                            client_socket.settimeout(CONNECT_TIMEOUT)
                        self.socketfile = client_socket.makefile("rw")
                        print(self.name, "connected successfully")
            except (TypeError, ValueError):
                print(self.name, "run command misformatted")
            except OSError:
                print(self.name, 'run failed - check "run" in commands.json')
            except socket.timeout:
                print("Timed out waiting for", self.name, "to connect")

    def stop(self):
        if self.socketfile is not None:
            try:
                self.socketfile.write("Q\n")
                self.socketfile.close()
            except socket.timeout:
                print("Timed out waiting for", self.name, "to disconnect")
            except OSError:
                print("Could not close socket connection with", self.name)
        if self.bot_subprocess is not None:
            try:
                if self.path == r"./player_chatbot":
                    outs, _ = self.bot_subprocess.communicate(timeout=PLAYER_TIMEOUT)
                else:
                    outs, _ = self.bot_subprocess.communicate(timeout=CONNECT_TIMEOUT)
                self.bytes_queue.put(outs)
            except subprocess.TimeoutExpired:
                print("Timed out waiting for", self.name, "to quit")
                self.bot_subprocess.kill()
                outs, _ = self.bot_subprocess.communicate()
                self.bytes_queue.put(outs)
        player_log_path = _resolve_output_path(self.name + ".txt")
        os.makedirs(os.path.dirname(player_log_path) or ".", exist_ok=True)
        with open(player_log_path, "wb") as log_file:
            bytes_written = 0
            for output in self.bytes_queue.queue:
                try:
                    bytes_written += log_file.write(output)
                    if bytes_written >= PLAYER_LOG_SIZE_LIMIT:
                        break
                except TypeError:
                    pass

    @staticmethod
    def _parse_basic_action(clause):
        if not clause:
            raise ValueError("empty clause")
        code = clause[0]
        if code == "F":
            return FoldAction()
        if code == "C":
            return CallAction()
        if code == "K":
            return CheckAction()
        if code == "R":
            return RaiseAction(int(clause[1:]))
        raise ValueError("unknown action")

    def query(self, round_state, player_message, game_log):
        legal_actions = (
            round_state.legal_actions() if isinstance(round_state, RoundState) else {CheckAction}
        )
        if self.socketfile is not None and self.game_clock > 0.0:
            clause = ""
            try:
                player_message[0] = "T{:.3f}".format(self.game_clock)
                message = " ".join(player_message) + "\n"
                del player_message[1:]
                start_time = time.perf_counter()
                self.socketfile.write(message)
                self.socketfile.flush()
                clause = self.socketfile.readline().strip()
                end_time = time.perf_counter()

                if ENFORCE_GAME_CLOCK and self.path != r"./player_chatbot":
                    self.game_clock -= end_time - start_time
                if self.game_clock <= 0.0:
                    raise socket.timeout

                redraw_available = RedrawAction in legal_actions
                basic_legal = set(legal_actions) - {RedrawAction}

                if clause.startswith("W"):
                    if not redraw_available:
                        raise ValueError("redraw not legal")
                    target_code = clause[1:2]
                    index_part = clause[2:3]
                    if target_code not in ("H", "B") or not index_part.isdigit():
                        raise ValueError("malformed redraw action")
                    target_index = int(index_part)
                    inner_clause = clause[3:]
                    inner_action = self._parse_basic_action(inner_clause)
                    if type(inner_action) not in basic_legal:
                        raise ValueError("illegal redraw inner action")
                    if isinstance(inner_action, RaiseAction):
                        min_raise, max_raise = round_state.raise_bounds()
                        if not (min_raise <= inner_action.amount <= max_raise):
                            raise ValueError("raise amount out of bounds")
                    target_type = "hole" if target_code == "H" else "board"
                    if not round_state._is_valid_redraw_target(
                        round_state.button % 2, target_type, target_index
                    ):
                        raise ValueError("invalid redraw target")
                    return RedrawAction(target_type, target_index, inner_action)

                action = self._parse_basic_action(clause)
                if type(action) not in basic_legal:
                    raise ValueError("illegal action")
                if isinstance(action, RaiseAction):
                    min_raise, max_raise = round_state.raise_bounds()
                    if not (min_raise <= action.amount <= max_raise):
                        raise ValueError("raise amount out of bounds")
                return action

            except socket.timeout:
                error_message = self.name + " ran out of time"
                game_log.append(error_message)
                print(error_message)
                self.game_clock = 0.0
            except OSError:
                error_message = self.name + " disconnected"
                game_log.append(error_message)
                print(error_message)
                self.game_clock = 0.0
            except (IndexError, KeyError, ValueError):
                game_log.append(self.name + " response misformatted: " + str(clause))

        return CheckAction() if CheckAction in legal_actions else FoldAction()


class Game:
    """Manages logs and high-level game flow."""

    def __init__(self):
        self.log = ["TACS Pokerbots - " + PLAYER_1_NAME + " vs " + PLAYER_2_NAME]
        self.player_messages = [[], []]

    def log_round_state(self, players, round_state):
        if round_state.street == 0 and round_state.button == 0:
            self.log.append("{} posts blind {}".format(players[0].name, SMALL_BLIND))
            self.log.append("{} posts blind {}".format(players[1].name, BIG_BLIND))
            self.log.append("{} dealt {}".format(players[0].name, PCARDS(round_state.hands[0])))
            self.log.append("{} dealt {}".format(players[1].name, PCARDS(round_state.hands[1])))
            self.player_messages[0] = ["T0.", "P0", "H" + CCARDS(round_state.hands[0]), "G"]
            self.player_messages[1] = ["T0.", "P1", "H" + CCARDS(round_state.hands[1]), "G"]
            return

        if (
            round_state.previous_state is not None
            and isinstance(round_state.previous_state, RoundState)
            and round_state.street != round_state.previous_state.street
        ):
            street_name = STREET_NAMES[round_state.street]
            self.log.append(
                "{} {}{}{}".format(
                    street_name,
                    PCARDS(round_state.board),
                    PVALUE(players[0].name, STARTING_STACK - round_state.stacks[0]),
                    PVALUE(players[1].name, STARTING_STACK - round_state.stacks[1]),
                )
            )
            compressed_board = "B" + CCARDS(round_state.board)
            self.player_messages[0].append(compressed_board)
            self.player_messages[1].append(compressed_board)

    def log_action(self, name, action, bet_override):
        redraw_note = ""
        if isinstance(action, RedrawAction):
            target = "hole" if action.target_type == "hole" else "board"
            redraw_note = " redraws {} card {} and".format(target, action.target_index)
            action = action.action

        if isinstance(action, FoldAction):
            phrasing = " folds"
            code = "F"
        elif isinstance(action, CallAction):
            phrasing = " calls"
            code = "C"
        elif isinstance(action, CheckAction):
            phrasing = " checks"
            code = "K"
        else:
            phrasing = (" bets " if bet_override else " raises to ") + str(action.amount)
            code = "R" + str(action.amount)

        if redraw_note:
            self.log.append(name + redraw_note + phrasing)
        else:
            self.log.append(name + phrasing)
        self.player_messages[0].append(code)
        self.player_messages[1].append(code)

    def log_redraw_reveal(self, actor_idx, target_type, target_index, old_card):
        if old_card is None:
            return
        target_code = "H" if target_type == "hole" else "B"
        redraw_clause = "W{}{}".format(target_code, target_index)
        reveal_clause = "X" + str(old_card)
        opponent = 1 - actor_idx
        self.player_messages[opponent].append(redraw_clause)
        self.player_messages[opponent].append(reveal_clause)
        self.log.append(
            "Redraw reveal for {}: {} {}".format(
                opponent,
                redraw_clause,
                reveal_clause,
            )
        )

    def log_terminal_state(self, players, terminal_state):
        previous_state = terminal_state.previous_state
        if not self.log[-1].endswith(" folds"):
            self.log.append("{} shows {}".format(players[0].name, PCARDS(previous_state.hands[0])))
            self.log.append("{} shows {}".format(players[1].name, PCARDS(previous_state.hands[1])))
            self.player_messages[0].append("O" + CCARDS(previous_state.hands[1]))
            self.player_messages[1].append("O" + CCARDS(previous_state.hands[0]))
        self.log.append("{} awarded {}".format(players[0].name, terminal_state.deltas[0]))
        self.log.append("{} awarded {}".format(players[1].name, terminal_state.deltas[1]))
        self.player_messages[0].append("A" + str(terminal_state.deltas[0]))
        self.player_messages[1].append("A" + str(terminal_state.deltas[1]))

    def run_round(self, players):
        deck = pkrbot.Deck()
        deck.shuffle()
        hands = [deck.deal(2), deck.deal(2)]
        pips = [SMALL_BLIND, BIG_BLIND]
        stacks = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]
        round_state = RoundState(0, 0, pips, stacks, hands, deck, [], [False, False], None)

        while not isinstance(round_state, TerminalState):
            self.log_round_state(players, round_state)
            active = round_state.button % 2
            player = players[active]
            action = player.query(round_state, self.player_messages[active], self.log)
            bet_override = round_state.pips == [0, 0]

            if isinstance(action, RedrawAction):
                old_card = round_state.peek_redraw_old_card(
                    active,
                    action.target_type,
                    action.target_index,
                )
                self.log_redraw_reveal(active, action.target_type, action.target_index, old_card)

            self.log_action(player.name, action, bet_override)
            round_state = round_state.proceed(action)

        self.log_terminal_state(players, round_state)
        for player, player_message, delta in zip(players, self.player_messages, round_state.deltas):
            player.query(round_state, player_message, self.log)
            player.bankroll += delta

    def run(self):
        print("Starting Pokerbots engine...")
        players = [Player(PLAYER_1_NAME, PLAYER_1_PATH), Player(PLAYER_2_NAME, PLAYER_2_PATH)]

        for player in players:
            player.build()
        for player in players:
            player.run()

        for round_num in range(1, NUM_ROUNDS + 1):
            self.log.append("")
            self.log.append("Round #" + str(round_num) + STATUS(players))
            self.run_round(players)
            players = players[::-1]

        self.log.append("")
        self.log.append("Final" + STATUS(players))

        for player in players:
            player.stop()

        name = _resolve_output_path(GAME_LOG_FILENAME + ".txt")
        os.makedirs(os.path.dirname(name) or ".", exist_ok=True)
        print("Writing", name)
        with open(name, "w") as log_file:
            log_file.write("\n".join(self.log))


if __name__ == "__main__":
    Game().run()
