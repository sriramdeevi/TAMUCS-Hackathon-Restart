"""
TACS PokerBots 2026 — Advanced Bot
====================================
Architecture (priority order):
  1. HAND EVALUATION   — Self-contained 5/6/7-card evaluator. No external libs.
  2. EQUITY ENGINE     — Preflop lookup table (O(1)) + postflop Monte Carlo.
                         Time-budget guard ensures we never time out.
  3. BET SIZING        — Street-aware, equity-proportional, SPR-aware.
  4. OPPONENT MODEL    — Tracks VPIP, PFR, AF, fold-to-cbet. Adjusts thresholds
                         after ~25 hands of data.
  5. REDRAW STRATEGY   — Evaluates EV gain; only redraws if improvement > 4pp.
                         Skipped when time budget is tight.

Card format: 2-char strings, e.g. 'Ah', 'Td', '2c'
  Ranks: 2 3 4 5 6 7 8 9 T J Q K A
  Suits: c d h s
"""

import random
import time as _time
from collections import Counter
from itertools import combinations
import numpy as np

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, RedrawAction
from skeleton.bot import Bot
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND

# ═══════════════════════════════════════════════════════════════════════════════
# §1  CARD REPRESENTATION
# ═══════════════════════════════════════════════════════════════════════════════

RANKS   = '23456789TJQKA'
SUITS   = 'cdhs'
RANK_IX = {r: i for i, r in enumerate(RANKS)}   # '2'→0, 'A'→12
SUIT_IX = {s: i for i, s in enumerate(SUITS)}   # 'c'→0, 's'→3
FULL_DECK = [r + s for r in RANKS for s in SUITS]  # 52 cards

def _r(c): return RANK_IX[c[0]]
def _s(c): return SUIT_IX[c[1]]


# ═══════════════════════════════════════════════════════════════════════════════
# §2  HAND EVALUATOR
#     Returns (category 0-8, tiebreak_list). Higher = better.
#     0=high card, 1=pair, 2=two-pair, 3=trips, 4=straight,
#     5=flush, 6=full house, 7=quads, 8=straight flush.
# ═══════════════════════════════════════════════════════════════════════════════

def _eval5(cards):
    ranks = sorted([_r(c) for c in cards], reverse=True)
    suits = [_s(c) for c in cards]

    is_flush = (len(set(suits)) == 1)

    # Straight check
    unique = sorted(set(ranks))
    is_straight = False
    if len(unique) == 5:
        if unique[-1] - unique[0] == 4:
            is_straight = True
        elif unique == [0, 1, 2, 3, 12]:   # Wheel: A-2-3-4-5
            is_straight = True
            ranks = [3, 2, 1, 0, -1]       # Ace plays low

    cnt    = Counter(ranks)
    groups = sorted(cnt.items(), key=lambda x: (x[1], x[0]), reverse=True)
    gvals  = [g[0] for g in groups]
    gsizes = [g[1] for g in groups]

    if is_straight and is_flush: return (8, ranks)
    if gsizes[0] == 4:           return (7, gvals)
    if gsizes[:2] == [3, 2]:     return (6, gvals)
    if is_flush:                 return (5, ranks)
    if is_straight:              return (4, ranks)
    if gsizes[0] == 3:           return (3, gvals)
    if gsizes[:2] == [2, 2]:     return (2, gvals)
    if gsizes[0] == 2:           return (1, gvals)
    return (0, ranks)

def best_hand_rank(cards):
    """Evaluate best 5-card hand out of 5-7 cards. Returns comparable tuple."""
    return max(_eval5(list(combo)) for combo in combinations(cards, 5))


# ═══════════════════════════════════════════════════════════════════════════════
# §3  PREFLOP EQUITY TABLE (169 canonical hands → ~win equity vs random)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_preflop_table():
    t = {}

    # Pocket pairs (AA→22)
    for rank, eq in zip(range(12, -1, -1),
                        [.852,.823,.798,.772,.746,.721,.695,.670,.645,.620,.598,.577,.557]):
        t[(rank, rank, False)] = eq

    # Suited hands  (hi, lo, eq)
    for hi, lo, eq in [
        (12,11,.670),(12,10,.660),(12, 9,.639),(12, 8,.621),(12, 7,.607),
        (12, 6,.594),(12, 5,.582),(12, 4,.567),(12, 3,.556),(12, 2,.544),
        (11,10,.641),(11, 9,.622),(11, 8,.604),(11, 7,.587),(11, 6,.572),
        (10, 9,.614),(10, 8,.596),(10, 7,.577),( 9, 8,.586),( 9, 7,.567),
        ( 8, 7,.558),( 7, 6,.548),( 6, 5,.540),( 5, 4,.532),( 4, 3,.523),
        ( 3, 2,.514),
    ]:
        t[(hi, lo, True)] = eq

    # Off-suit hands
    for hi, lo, eq in [
        (12,11,.647),(12,10,.636),(12, 9,.614),(12, 8,.596),(12, 7,.582),
        (12, 6,.569),(12, 5,.556),(12, 4,.541),(12, 3,.530),(12, 2,.519),
        (11,10,.618),(11, 9,.598),(11, 8,.580),(11, 7,.562),(10, 9,.591),
        (10, 8,.572),( 9, 8,.562),( 9, 7,.543),( 8, 7,.534),( 7, 6,.524),
        ( 6, 5,.515),( 5, 4,.506),( 4, 3,.498),( 3, 2,.489),
    ]:
        t[(hi, lo, False)] = eq

    return t

PREFLOP_EQ = _build_preflop_table()

def preflop_equity(hole):
    """O(1) preflop equity lookup for our 2 hole cards vs random opponent."""
    r0, r1  = _r(hole[0]), _r(hole[1])
    hi, lo  = max(r0, r1), min(r0, r1)
    suited  = (_s(hole[0]) == _s(hole[1]))
    key     = (hi, lo, suited if hi != lo else False)
    if key in PREFLOP_EQ:
        return PREFLOP_EQ[key]
    # Linear fallback for unlisted combos (shouldn't happen often)
    return min(max(0.45 + 0.007*hi + 0.003*lo + (0.02 if suited else 0.0), 0.42), 0.88)


# ═══════════════════════════════════════════════════════════════════════════════
# §4  MONTE CARLO EQUITY
# ═══════════════════════════════════════════════════════════════════════════════

def monte_carlo_equity(hole, board, n=300):
    """
    Estimate win probability via random rollout.
    ~1ms per 50 samples on typical hardware.
    """
    known      = set(hole + board)
    deck       = [c for c in FULL_DECK if c not in known]
    board_need = 5 - len(board)

    wins = 0.0
    for _ in range(n):
        sample   = random.sample(deck, 2 + board_need)
        opp      = sample[:2]
        runout   = board + sample[2:]
        my_rank  = best_hand_rank(hole + runout)
        opp_rank = best_hand_rank(opp  + runout)
        if   my_rank > opp_rank: wins += 1.0
        elif my_rank == opp_rank: wins += 0.5

    return wins / n


def calc_equity(hole, board, street, fast=False):
    """
    Dispatch: preflop table on street 0, MC otherwise.
    fast=True → use minimal samples (time-constrained path).
    """
    if street == 0:
        return preflop_equity(hole)
    if fast:
        return monte_carlo_equity(hole, board, n=60)
    # Normal sample counts — tuned so 300 hands comfortably fits 180s.
    # Each MC call: ~50-80ms. ~3 streets × 300 hands = ~54-72s total.
    samples = {3: 250, 4: 300, 5: 350}.get(street, 250)
    return monte_carlo_equity(hole, board, n=samples)


# ═══════════════════════════════════════════════════════════════════════════════
# §5  REDRAW EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

MIN_REDRAW_GAIN = 0.04   # Only redraw if equity improves by 4+ percentage points

def evaluate_redraw(hole, board, street, base_eq):
    """
    Determine whether to redraw and which card to swap.
    Samples candidate replacements and estimates expected equity gain.
    Returns (should_redraw: bool, target_type: str|None, target_index: int|None).
    """
    known       = set(hole + board)
    deck        = [c for c in FULL_DECK if c not in known]
    sample_pool = random.sample(deck, min(18, len(deck)))

    best_gain = MIN_REDRAW_GAIN
    best      = (False, None, None)

    # --- Hole card swaps ---
    for idx in range(2):
        gains = []
        for new_card in sample_pool:
            new_hole = hole[:]
            new_hole[idx] = new_card
            eq = monte_carlo_equity(new_hole, board, n=80)
            gains.append(eq)
        gain = float(np.mean(gains)) - base_eq
        if gain > best_gain:
            best_gain = gain
            best      = (True, 'hole', idx)

    # --- Board card swaps (flop only: 3 revealed cards at indices 0-2) ---
    if street == 3 and len(board) == 3:
        for bidx in range(3):
            gains = []
            for new_card in sample_pool[:10]:
                new_board       = board[:]
                new_board[bidx] = new_card
                eq = monte_carlo_equity(hole, new_board, n=50)
                gains.append(eq)
            gain = float(np.mean(gains)) - base_eq
            if gain > best_gain:
                best_gain = gain
                best      = (True, 'board', bidx)

    return best


# ═══════════════════════════════════════════════════════════════════════════════
# §6  OPPONENT MODEL
#     Stats: VPIP, PFR, AF (aggression factor), fold-to-cbet.
# ═══════════════════════════════════════════════════════════════════════════════

class OpponentModel:

    def __init__(self):
        self.hands        = 0
        self.vpip_n       = 0     # voluntarily put money in preflop
        self.pfr_n        = 0     # preflop raises
        self.postflop_bets = 0    # bets + raises postflop
        self.postflop_calls = 0   # calls postflop
        self.total_folds  = 0
        self.cbet_faced   = 0
        self.cbet_folded  = 0
        self._pf_acted    = False  # guard: count preflop action once per hand

    def new_hand(self):
        self.hands   += 1
        self._pf_acted = False

    def record_preflop(self, action):
        """action: 'fold' | 'call' | 'raise'"""
        if self._pf_acted:
            return
        self._pf_acted = True
        if action in ('call', 'raise'):
            self.vpip_n += 1
        if action == 'raise':
            self.pfr_n  += 1
        if action == 'fold':
            self.total_folds += 1

    def record_postflop(self, action, facing_cbet=False):
        """action: 'bet' | 'call' | 'fold'"""
        if action == 'bet':
            self.postflop_bets  += 1
        elif action == 'call':
            self.postflop_calls += 1
        elif action == 'fold':
            self.total_folds += 1
            if facing_cbet:
                self.cbet_folded += 1
        if facing_cbet:
            self.cbet_faced += 1

    # ── Derived stats ──────────────────────────────────────────────────────────

    @property
    def vpip(self):   return self.vpip_n  / max(1, self.hands)
    @property
    def pfr(self):    return self.pfr_n   / max(1, self.hands)
    @property
    def af(self):     return self.postflop_bets / max(1, self.postflop_calls)
    @property
    def ftcb(self):   return self.cbet_folded   / max(1, self.cbet_faced)

    def reliable(self): return self.hands >= 25

    # ── Profile classification ─────────────────────────────────────────────────

    def is_nit(self):
        return self.reliable() and self.vpip < 0.22 and self.af < 1.5

    def is_station(self):
        return self.reliable() and self.vpip > 0.45 and self.af < 1.2

    def is_lag(self):
        return self.reliable() and self.vpip > 0.40 and self.af > 2.5

    def is_tag(self):
        return self.reliable() and self.vpip < 0.30 and self.af > 2.0

    def folds_to_pressure(self):
        return self.reliable() and self.ftcb > 0.55

    # ── Threshold adjustments ─────────────────────────────────────────────────

    def adjustments(self):
        """
        Returns (value_adj, call_adj, bluff_mult).
          value_adj  < 0 → bet thinner  (lower threshold)
          call_adj   > 0 → call wider
          bluff_mult     → multiply base bluff frequency
        """
        if not self.reliable():
            return 0.0, 0.0, 1.0

        if self.is_station():
            return -0.05, 0.03, 0.2    # thin value, no bluffs
        if self.is_nit():
            return  0.03, -0.02, 2.0   # need strong hand, bluff liberally
        if self.is_lag():
            return  0.05, 0.04, 0.6    # tighten up, trap more
        if self.is_tag():
            return  0.04, -0.03, 0.5   # respect their range
        return 0.0, 0.0, 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# §7  BET SIZING
# ═══════════════════════════════════════════════════════════════════════════════

def size_raise(equity_val, pot, my_stack, opp_stack, street, is_bluff=False):
    """
    Kelly-inspired bet sizing.
    Value bets scale with equity edge and street; bluffs are ~55% pot.
    Low SPR (<2) biases toward jams.
    """
    spr  = my_stack / max(1, pot)
    edge = max(0.0, equity_val - 0.5)

    if is_bluff:
        fraction = 0.50 + random.uniform(-0.05, 0.10)   # 45-60% pot
    else:
        base = {5: 0.75, 4: 0.62, 3: 0.50, 0: 0.40}.get(street, 0.50)
        mult = {5: 1.40, 4: 1.10, 3: 0.80, 0: 0.60}.get(street, 0.80)
        fraction = base + mult * edge

    if spr < 2.0 and not is_bluff:
        fraction = max(fraction, 0.90)   # low SPR: lean toward shove

    amount = int(pot * fraction)
    amount = max(amount, BIG_BLIND * 2)
    amount = min(amount, my_stack)           # can't bet more than we have
    amount = min(amount, my_stack + opp_stack)  # can't over-shove
    return amount


# ═══════════════════════════════════════════════════════════════════════════════
# §8  MAIN BOT
# ═══════════════════════════════════════════════════════════════════════════════

_MATCH_BUDGET    = 172.0   # seconds (8s safety buffer below 180)
_HANDS_PER_MATCH = 300


class Player(Bot):

    def __init__(self):
        self.opp           = OpponentModel()
        self._eq_cache     = {}     # board_tuple → equity float
        self._used_redraw  = False
        self._hand_num     = 0
        self._match_start  = _time.time()

        # Per-hand state for opp model inference
        self._i_raised_pf  = False   # did I raise preflop (for cbet detection)
        self._opp_prev_pip = 0
        self._prev_street  = -1

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def handle_new_round(self, game_state, round_state, active):
        self._hand_num    += 1
        self._used_redraw  = False
        self._eq_cache     = {}
        self._i_raised_pf  = False
        self._opp_prev_pip = round_state.pips[1 - active]
        self._prev_street  = round_state.street
        self.opp.new_hand()

    def handle_round_over(self, game_state, terminal_state, active):
        # We could inspect terminal_state.deltas here for P&L tracking.
        pass

    # ── Main decision ──────────────────────────────────────────────────────────

    def get_action(self, game_state, round_state, active):
        legal     = round_state.legal_actions()
        street    = round_state.street
        hole      = list(round_state.hands[active])
        board     = list(round_state.deck[:street])

        my_pip    = round_state.pips[active]
        opp_pip   = round_state.pips[1 - active]
        my_stack  = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]
        pot       = my_pip + opp_pip

        # ── Opponent model: infer action from pip delta ────────────────────────
        opp_delta = opp_pip - self._opp_prev_pip
        if opp_delta > 0:
            if street == 0:
                self.opp.record_preflop('raise')
            else:
                facing_cbet = self._i_raised_pf and street == 3
                self.opp.record_postflop('bet', facing_cbet=facing_cbet)
        elif self._prev_street != street and street > 0:
            # Street changed without pip increase → opponent checked/called
            pass
        self._opp_prev_pip = opp_pip
        self._prev_street  = street

        # ── Time budget ────────────────────────────────────────────────────────
        elapsed      = _time.time() - self._match_start
        hands_left   = max(1, _HANDS_PER_MATCH - self._hand_num)
        budget_left  = _MATCH_BUDGET - elapsed
        time_ok      = (budget_left / hands_left) > 0.20   # >200ms/hand remaining

        # ── Equity ─────────────────────────────────────────────────────────────
        board_key = tuple(board)
        if board_key not in self._eq_cache:
            self._eq_cache[board_key] = calc_equity(hole, board, street, fast=not time_ok)
        eq = self._eq_cache[board_key]

        # ── Pot odds ───────────────────────────────────────────────────────────
        call_amount = opp_pip - my_pip
        pot_odds    = call_amount / (pot + call_amount) if call_amount > 0 else 0.0

        # ── Opponent-adjusted thresholds ───────────────────────────────────────
        val_adj, call_adj, bluff_mult = self.opp.adjustments()

        VALUE_THRESH = 0.60 + val_adj    # raise/bet threshold
        CALL_THRESH  = 0.44 + call_adj   # pure call threshold
        BLUFF_FREQ   = 0.10 * bluff_mult  # frequency to bluff in bluff range

        # ── Redraw ─────────────────────────────────────────────────────────────
        if (not self._used_redraw
                and street < 5
                and time_ok
                and RedrawAction in legal):
            should, ttype, tidx = evaluate_redraw(hole, board, street, eq)
            if should:
                self._used_redraw = True
                self._eq_cache    = {}   # invalidate; hand changed
                embed = self._decide_bet(
                    eq, pot_odds, pot, my_stack, opp_stack, street,
                    VALUE_THRESH, CALL_THRESH, BLUFF_FREQ,
                    legal, call_amount, allow_bluff=False
                )
                return RedrawAction(ttype, tidx, embed)

        # ── Normal betting ──────────────────────────────────────────────────────
        action = self._decide_bet(
            eq, pot_odds, pot, my_stack, opp_stack, street,
            VALUE_THRESH, CALL_THRESH, BLUFF_FREQ,
            legal, call_amount
        )

        if street == 0 and isinstance(action, RaiseAction):
            self._i_raised_pf = True

        return action

    # ── Betting logic ──────────────────────────────────────────────────────────

    def _decide_bet(self, eq, pot_odds, pot, my_stack, opp_stack, street,
                    value_thresh, call_thresh, bluff_freq,
                    legal, call_amount, allow_bluff=True):
        """
        Decision tree. Priority:
          1. Value raise/bet  (eq ≥ value_thresh)
          2. Call with equity (eq ≥ call_thresh AND beats pot odds)
          3. Semi-bluff raise (30-40% equity, some frequency)
          4. Pure bluff raise (< 35% equity, lower frequency)
          5. Marginal call    (eq barely beats pot odds, cheap)
          6. Check            (take free card)
          7. Fold
        """

        # 1. Value raise
        if eq >= value_thresh and RaiseAction in legal:
            amt = size_raise(eq, pot, my_stack, opp_stack, street, is_bluff=False)
            return RaiseAction(amt)

        # 2. Call with positive EV
        if eq >= call_thresh and eq > pot_odds and CallAction in legal:
            return CallAction()

        # 3. Semi-bluff (good draw + fold equity)
        if allow_bluff and RaiseAction in legal and 0.30 <= eq < value_thresh:
            semi_freq = bluff_freq * 2.0   # semi-bluffs are twice as valuable
            if random.random() < semi_freq:
                amt = size_raise(eq, pot, my_stack, opp_stack, street, is_bluff=True)
                return RaiseAction(amt)

        # 4. Pure bluff (air)
        if allow_bluff and RaiseAction in legal and eq < 0.35:
            if random.random() < bluff_freq:
                amt = size_raise(eq, pot, my_stack, opp_stack, street, is_bluff=True)
                return RaiseAction(amt)

        # 5. Marginal call (getting a great price)
        if call_amount > 0 and eq >= pot_odds and eq > 0.33 and CallAction in legal:
            return CallAction()

        # 6. Check
        if CheckAction in legal:
            return CheckAction()

        # 7. Fold
        if FoldAction in legal:
            return FoldAction()

        # Unreachable in a valid game state, but satisfy the type checker
        return CallAction() if CallAction in legal else CheckAction()