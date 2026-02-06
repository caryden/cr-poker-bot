#!/usr/bin/env python3
"""Test a complete game: multiple hands until one player busts."""
import sys
sys.path.insert(0, '.')
import time
import re
import os
import anthropic

from src.core.game_state import create_6max_game
from src.core.primitives import Position, HoleCards, Card, Action, ActionType, Board
from src.core import Street
from src.game.runner import Deck
from src.tools.equity import calculate_equity
from src.tools.hand_eval import evaluate_hand

client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

def parse_action(response, to_call, stack):
    response = response.strip().lower()
    if 'fold' in response:
        return Action.fold()
    elif 'check' in response:
        return Action.check()
    elif 'call' in response:
        return Action.call(to_call)
    elif 'raise' in response or 'bet' in response:
        match = re.search(r'(\d+)', response)
        if match:
            return Action.raise_to(float(match.group(1)))
        return Action.raise_to(to_call * 2 if to_call > 0 else 10)
    elif 'all' in response:
        return Action.all_in(stack)
    return Action.check() if to_call == 0 else Action.call(to_call)

def get_llm_decision(game, hero_cards):
    pot_odds = game.to_call / (game.pot.total + game.to_call) if game.to_call > 0 else 0
    eq = calculate_equity(hero_cards, game.board, num_opponents=1, simulations=300)
    board_str = str(game.board) if game.board and game.board.cards else 'none'
    prompt = f'Poker. ONLY: fold/check/call/bet N/raise N/all-in\nBoard: {board_str}, Hand: {hero_cards}\nPot: {game.pot.total:.0f}, To call: {game.to_call:.0f}, Stack: {game.hero.stack:.0f}, Equity: {eq.equity:.0%}'
    start = time.time()
    resp = client.messages.create(model='claude-sonnet-4-5', max_tokens=15, messages=[{'role': 'user', 'content': prompt}])
    return resp.content[0].text.strip(), time.time() - start

def run_hand(hero_stack, villain_stack, sb, bb):
    """Run one hand. Returns (hero_profit, decisions, time)."""
    deck = Deck()
    deck.shuffle()

    hero_cards = HoleCards(deck.draw(1)[0], deck.draw(1)[0])
    villain_cards = HoleCards(deck.draw(1)[0], deck.draw(1)[0])

    game = create_6max_game(hand_id='h', hero_id='hero', hero_position=Position.BTN,
                            hero_cards=hero_cards, stacks={Position.BTN: hero_stack, Position.BB: villain_stack},
                            small_blind=sb, big_blind=bb)
    game.players = {k: v for k, v in game.players.items() if k in ['hero', 'villain_BB']}
    villain = game.players['villain_BB']

    initial_hero = game.hero.stack
    total_time = 0
    decisions = 0
    hand_over = False
    hero_folded = False

    board_cards = [deck.draw(1)[0] for _ in range(5)]

    for i, street in enumerate([Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER]):
        if hand_over:
            break
        game.street = street
        if street == Street.FLOP:
            game.board = Board(cards=board_cards[:3])
        elif street == Street.TURN:
            game.board.cards.append(board_cards[3])
        elif street == Street.RIVER:
            game.board.cards.append(board_cards[4])

        if street == Street.PREFLOP:
            game.current_bet = bb
            game.hero.bet_this_street = 0
            villain.bet_this_street = bb
        else:
            game.current_bet = 0
            game.hero.bet_this_street = 0
            villain.bet_this_street = 0

        response, elapsed = get_llm_decision(game, hero_cards)
        total_time += elapsed
        decisions += 1
        action = parse_action(response, game.to_call, game.hero.stack)

        if action.action_type == ActionType.FOLD:
            hero_folded = True
            hand_over = True
        elif action.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            amt = min(action.amount, game.hero.stack + game.hero.bet_this_street)
            add = amt - game.hero.bet_this_street
            game.hero.stack -= add
            game.pot.add(add)
            v_call = min(amt - villain.bet_this_street, villain.stack)
            villain.stack -= v_call
            game.pot.add(v_call)
            if game.hero.stack == 0 or villain.stack == 0:
                hand_over = True
        elif action.action_type == ActionType.CALL and game.to_call > 0:
            call_amt = min(game.to_call, game.hero.stack)
            game.hero.stack -= call_amt
            game.pot.add(call_amt)

    # Determine winner
    if hero_folded:
        hero_profit = initial_hero - game.hero.stack
        hero_profit = -hero_profit  # Lost what we put in
    else:
        hero_rank = evaluate_hand(hero_cards, game.board)
        villain_rank = evaluate_hand(villain_cards, game.board)
        if hero_rank > villain_rank:
            hero_profit = game.pot.total - (initial_hero - game.hero.stack)
        elif villain_rank > hero_rank:
            hero_profit = -(initial_hero - game.hero.stack)
        else:
            hero_profit = game.pot.total / 2 - (initial_hero - game.hero.stack)

    return hero_profit, decisions, total_time, hero_cards, villain_cards

if __name__ == '__main__':
    print('=== COMPLETE GAME (until bust) ===', flush=True)
    hero_stack = 200.0
    villain_stack = 200.0
    sb, bb = 5, 10

    hand_num = 0
    total_decisions = 0
    total_time = 0
    max_hands = 20  # Safety limit

    while hero_stack > 0 and villain_stack > 0 and hand_num < max_hands:
        hand_num += 1
        profit, decisions, elapsed, h_cards, v_cards = run_hand(hero_stack, villain_stack, sb, bb)
        total_decisions += decisions
        total_time += elapsed

        hero_stack += profit
        villain_stack -= profit

        result = f'+{profit:.0f}' if profit >= 0 else f'{profit:.0f}'
        print(f'Hand {hand_num}: {h_cards} vs {v_cards} | {result} | Hero: {hero_stack:.0f} Villain: {villain_stack:.0f} | {decisions}d {elapsed:.1f}s', flush=True)

    print('', flush=True)
    if hero_stack <= 0:
        print('Villain wins the game!', flush=True)
    elif villain_stack <= 0:
        print('Hero wins the game!', flush=True)
    else:
        print(f'Max hands reached. Hero: {hero_stack:.0f}, Villain: {villain_stack:.0f}', flush=True)

    print(f'=== {hand_num} hands, {total_decisions} decisions, {total_time:.1f}s ({total_time/total_decisions:.2f}s avg) ===', flush=True)
