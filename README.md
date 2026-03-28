# Build4Good PokerBots 2026 Challenge

Welcome to the Build4Good PokerBots challenge. In this competition, competitors are challenged to build autonomous poker bots in Python. These poker bots will then compete in head-to-head matches in a tournament bracket.

## Overview

- Two-player head-to-head poker competition
- All poker bots **MUST** be written in Python and strictly adhere to the provided API.
- External Python libraries other than those listed below and in [requirements.txt](requirements.txt) are **NOT ALLOWED**
- Time-limited decision making per match
- Team submissions will compete in a single-elimination tournament

## Game Rules (Hold'em + Redraw)

This year’s game is based on no-limit Texas Hold'em, with one major twist: **redraw**.

- Each player gets **2 hole cards**
- Community cards are dealt as:
  - Flop: 3 cards
  - Turn: 1 card
  - River: 1 card
- Streets are represented as:
  - `0` = preflop
  - `3` = flop betting
  - `4` = turn betting
  - `5` = river betting
- Each player may redraw **once per hand**, only **before river** (`street < 5`)
- You may redraw:
  - one of your hole cards (`hole`, index `0` or `1`), OR
  - one revealed board card (`board`, index depends on street)
- Redraw is combined with a betting action in a single move:
  - check/call/raise/fold + redraw in one response
- The old redrawn card is revealed to the opponent

That is, the competition rules are the standard Texas Hold'em rules with the option to redraw one card per hand. 

Your poker bot will have a total time limit of 180 seconds per **match**. Each match will consist of 300 hands. At the start of each hand you will have 400 chips. These chips do not persist between hands, i.e. at the start of every hand you will always have 400 chips regardless of what happened in previous rounds. You chip wins/losses in each round will then be summed at the end of the 300 hands to determine the winner. The small/big blinds are 1 and 2, respectively. 

## Repository Layout

- `engine.py` - core game engine and socket protocol implementation
- `config.py` - local head-to-head match configuration
- `python_skeleton/` - reference bot scaffold
- `results_directory/` - tournament output JSON/log artifacts

## What to Submit

You will fill in your poker bot strategy in the [python_skeleton/skeleton/bot.py](python_skeleton/skeleton/bot.py) file. At a minimum, you need to fill in the three methods in the `Bot` class. **DO NOT** change the method signatures in the `Bot` class or modify  any of the other existing files in the `python_skeleton/` folder. You can add new files as you like.

At the end of the contest, you will submit a `.zip` file of the `python_skeleton/` folder. That is, you should ZIP the entire folder, including your `Bot` class, all of the current code, and any other files that you create. 

## Getting Started

### 1) Create environment and install dependencies

After cloning the repository, create a virtual environment and install dependencies with `pip`:

Mac:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

Alternatively, if you have conda/miniconda:
```bash
conda create -y -n pkrbot python=3.10 
conda activate pkrbot 
pip install -r requirements.txt
```

The dependency set is intentionally lightweight. You are **NOT ALLOWED** to use any external Python libraries other than `numpy`, `numba`, `cython`, and `pkrbot`.

### 2) Run a local match

```bash
python3 engine.py
```

Configure player paths and match parameters in `config.py`.

There are two example strategies in the [all_in_bot/](all_in_bot) and [check_call_bot/](check_call_bot) folders. At a minimum, you should ensure your poker bot can beat these basic strategies. 

## Building Your Bot

Each bot lives in its own directory and must include:

- `player.py` (your strategy)
- `commands.json` (how to run your bot process)
- `skeleton/` (copied from `python_skeleton/skeleton/`)

`commands.json` should follow:

```json
{
  "build": [],
  "run": ["python3", "player.py"]
}
```

Implement these methods in your `Player` class:

- `handle_new_round(game_state, round_state, active)`
- `handle_round_over(game_state, terminal_state, active)`
- `get_action(game_state, round_state, active)`

Always validate with `round_state.legal_actions()` before returning an action.

## Actions API

Available actions in `python_skeleton/skeleton/actions.py`:

- `FoldAction()`
- `CallAction()`
- `CheckAction()`
- `RaiseAction(amount)`
- `RedrawAction(target_type, target_index, action)`

Where:

- `target_type` is `'hole'` or `'board'`
- `target_index` is hole index (`0..1`) or board index (`0..4`, street-dependent)
- `action` is the embedded betting action

## Acknowledgements

This challenge is sponsored by **Jane Street** and **Hudson River Trading**, alongside broader TACS/competition support.

This repository is forked and adapted from the [MIT PokerBots](https://github.com/mitpokerbots/engine-2025) codebase