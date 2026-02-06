#!/usr/bin/env python3
"""Test a complete hand with LLM agent."""
import sys
sys.path.insert(0, '.')
import time
import re
import os
import anthropic

from src.core.game_state import create_6max_game
from src.core.primitives import Position, HoleCards, Card, Action, ActionType, Board
from src.core import Street
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
    return Action.check() if to_call == 0 else Action.call(to_call)

def get_llm_decision(game, hero_cards):
    pot_odds = game.to_call / (game.pot.total + game.to_call) if game.to_call > 0 else 0
    eq = calculate_equity(hero_cards, game.board, num_opponents=1, simulations=500)
    board_str = str(game.board) if game.board and game.board.cards else 'none'
    prompt = f'Poker. ONLY respond: fold/check/call/bet N/raise N\nStreet: {game.street}, Board: {board_str}\nHand: {hero_cards}, Pot: {game.pot.total:.0f}, To call: {game.to_call:.0f}, Stack: {game.hero.stack:.0f}, Equity: {eq.equity:.0%}'
    start = time.time()
    resp = client.messages.create(model='claude-sonnet-4-5', max_tokens=15, messages=[{'role': 'user', 'content': prompt}])
    return resp.content[0].text.strip(), time.time() - start

if __name__ == '__main__':
    print('=== COMPLETE HAND TO SHOWDOWN ===', flush=True)
    hero_cards = HoleCards(Card.from_str('As'), Card.from_str('Ks'))
    villain_cards = HoleCards(Card.from_str('Qh'), Card.from_str('Jh'))
    print(f'Hero: {hero_cards}, Villain: {villain_cards}', flush=True)

    game = create_6max_game(hand_id='h1', hero_id='hero', hero_position=Position.BTN,
                            hero_cards=hero_cards, stacks={Position.BTN: 500, Position.BB: 500},
                            small_blind=5, big_blind=10)
    game.players = {k: v for k, v in game.players.items() if k in ['hero', 'villain_BB']}
    villain = game.players['villain_BB']

    total_time = 0
    decisions = 0
    hand_over = False

    board_cards = [Card.from_str('Qs'), Card.from_str('7h'), Card.from_str('2d'),
                   Card.from_str('Kc'), Card.from_str('3s')]

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
            game.current_bet = 10
            game.hero.bet_this_street = 0
            villain.bet_this_street = 10
        else:
            game.current_bet = 0
            game.hero.bet_this_street = 0
            villain.bet_this_street = 0

        board_str = str(game.board) if game.board and game.board.cards else ''
        print(f'--- {street.name} {board_str} ---', flush=True)
        if street != Street.PREFLOP:
            print('Villain checks', flush=True)

        response, elapsed = get_llm_decision(game, hero_cards)
        total_time += elapsed
        decisions += 1
        action = parse_action(response, game.to_call, game.hero.stack)
        print(f'LLM: "{response}" -> {action} ({elapsed:.2f}s)', flush=True)

        if action.action_type == ActionType.FOLD:
            print('Hero folds. Villain wins.', flush=True)
            hand_over = True
        elif action.action_type in (ActionType.BET, ActionType.RAISE):
            add = action.amount - game.hero.bet_this_street
            game.hero.stack -= add
            game.pot.add(add)
            v_call = action.amount - villain.bet_this_street
            villain.stack -= v_call
            game.pot.add(v_call)
            print(f'Villain calls. Pot: {game.pot.total:.0f}', flush=True)
        elif action.action_type == ActionType.CALL and game.to_call > 0:
            game.hero.stack -= game.to_call
            game.pot.add(game.to_call)
            print(f'Pot: {game.pot.total:.0f}', flush=True)
        else:
            print(f'Pot: {game.pot.total:.0f}', flush=True)

    if not hand_over:
        print('--- SHOWDOWN ---', flush=True)
        hero_rank = evaluate_hand(hero_cards, game.board)
        villain_rank = evaluate_hand(villain_cards, game.board)
        print(f'Hero: {hero_cards} = {hero_rank}', flush=True)
        print(f'Villain: {villain_cards} = {villain_rank}', flush=True)
        winner = 'Hero' if hero_rank > villain_rank else ('Villain' if villain_rank > hero_rank else 'Split')
        print(f'{winner} wins {game.pot.total:.0f}!', flush=True)

    print(f'=== {decisions} decisions, {total_time:.2f}s total, {total_time/decisions:.2f}s avg ===', flush=True)
