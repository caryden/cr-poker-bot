#!/usr/bin/env python3
"""
Test 6-player table: Hero (LLM) vs 5 villains with different play styles.

Proves:
1. 6-player game works
2. Beliefs update for each villain based on their observed actions
3. LLM uses villain-specific beliefs
4. Different villain styles produce different belief profiles
"""
import sys
sys.path.insert(0, '.')
import time
import re
import os
import anthropic

from src.core.game_state import create_6max_game
from src.core.primitives import Position, HoleCards, Card, Action, ActionType, Board
from src.core import Street
from src.core.beliefs import (
    BeliefState, BeliefRevisionEngine, Observation, ObservationType,
    GameContext, create_default_villain_beliefs
)
from src.game.runner import Deck
from src.game.opponents import CallingStation, TightPassive, LooseAggressive, TagBot, RandomPlayer
from src.tools.equity import calculate_equity
from src.tools.hand_eval import evaluate_hand

client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

# Villain configurations
VILLAINS = {
    'villain_UTG': ('CallingStation', CallingStation('villain_UTG')),
    'villain_HJ': ('TightPassive', TightPassive('villain_HJ')),
    'villain_CO': ('LooseAggressive', LooseAggressive('villain_CO')),
    'villain_SB': ('TagBot', TagBot('villain_SB')),
    'villain_BB': ('Random', RandomPlayer('villain_BB')),
}


def parse_action(response: str, to_call: float, stack: float) -> tuple[Action, str]:
    """Parse LLM response into Action and extract reasoning."""
    lines = response.strip().split('\n')
    action_line = ""
    reasoning_lines = []

    for line in lines:
        line_lower = line.lower().strip()
        if any(x in line_lower for x in ['action:', 'decision:', 'fold', 'check', 'call', 'bet', 'raise', 'all-in']):
            if not action_line:
                action_line = line_lower
        else:
            if line.strip():
                reasoning_lines.append(line.strip())

    reasoning = ' '.join(reasoning_lines)

    if 'fold' in action_line:
        return Action.fold(), reasoning
    elif 'check' in action_line:
        return Action.check(), reasoning
    elif 'call' in action_line:
        return Action.call(to_call), reasoning
    elif 'raise' in action_line or 'bet' in action_line:
        match = re.search(r'(\d+)', action_line)
        if match:
            return Action.raise_to(float(match.group(1))), reasoning
        return Action.raise_to(to_call * 2 if to_call > 0 else 10), reasoning
    elif 'all' in action_line:
        return Action.all_in(stack), reasoning

    if to_call == 0:
        return Action.check(), reasoning
    return Action.call(to_call), reasoning


def get_llm_decision(game, hero_cards, belief_state: BeliefState, villains_in_hand: list) -> tuple[str, float]:
    """Get LLM decision with belief state context."""
    pot_odds = game.to_call / (game.pot.total + game.to_call) if game.to_call > 0 else 0
    num_opponents = len(villains_in_hand)
    eq = calculate_equity(hero_cards, game.board, num_opponents=max(1, num_opponents), simulations=500)
    board_str = str(game.board) if game.board and game.board.cards else 'none'

    # Format belief state - only for villains still in hand
    belief_lines = ["=== VILLAIN BELIEFS ==="]
    for vid in villains_in_hand:
        if vid in belief_state.villain_beliefs:
            vb = belief_state.villain_beliefs[vid]
            best_type, prob = vb.get_most_likely_player_type()
            belief_lines.append(f"{vid}: likely {best_type} ({prob:.0%})")
            # Show top tendency
            sorted_beliefs = vb.sorted_by_knowledge()
            if sorted_beliefs and not sorted_beliefs[0].opinion.is_vacuous:
                top = sorted_beliefs[0]
                belief_lines.append(f"  - {top.label}: b={top.opinion.belief:.2f}")
    belief_str = '\n'.join(belief_lines)

    prompt = f'''Poker AI: Analyze using villain beliefs and decide.

=== GAME STATE ===
Street: {game.street}, Board: {board_str}
Your hand: {hero_cards}
Pot: {game.pot.total:.0f}, To call: {game.to_call:.0f}, Stack: {game.hero.stack:.0f}
Opponents in hand: {len(villains_in_hand)}
Pot odds: {pot_odds:.0%}, Equity: {eq.equity:.0%}

{belief_str}

REASONING: [brief analysis using villain beliefs]
ACTION: [fold/check/call/bet N/raise N]'''

    start = time.time()
    resp = client.messages.create(
        model='claude-sonnet-4-5',
        max_tokens=200,
        messages=[{'role': 'user', 'content': prompt}]
    )
    return resp.content[0].text.strip(), time.time() - start


def create_observation(player_id: str, action: Action, street: Street,
                       position: Position, pot: float, facing_raise: bool = False) -> Observation:
    """Create observation from villain action."""
    return Observation(
        obs_id=f"obs_{player_id}_{street.name}",
        obs_type=ObservationType.ACTION,
        player_id=player_id,
        action=action,
        context=GameContext(
            street=street, position=position, pot_size=pot,
            to_call=0, num_players=6, is_heads_up=False,
            facing_raise=facing_raise
        )
    )


def run_table_hand(belief_state: BeliefState, revision_engine: BeliefRevisionEngine,
                   hand_num: int) -> tuple[BeliefState, dict]:
    """Run one hand at 6-max table. Returns updated beliefs and hand result."""

    deck = Deck()
    deck.shuffle()

    # Deal cards
    hero_cards = HoleCards(deck.draw(1)[0], deck.draw(1)[0])
    villain_cards = {vid: HoleCards(deck.draw(1)[0], deck.draw(1)[0]) for vid in VILLAINS}
    board_cards = [deck.draw(1)[0] for _ in range(5)]

    # Create game state
    all_positions = [Position.BTN, Position.SB, Position.BB, Position.UTG, Position.HJ, Position.CO]
    hero_pos = all_positions[hand_num % 6]  # Rotate position

    stacks = {pos: 500 for pos in all_positions}

    game = create_6max_game(
        hand_id=f'table_hand_{hand_num}',
        hero_id='hero',
        hero_position=hero_pos,
        hero_cards=hero_cards,
        stacks=stacks,
        small_blind=5,
        big_blind=10
    )

    # Map positions to villain IDs
    pos_to_villain = {}
    villain_positions = [p for p in all_positions if p != hero_pos]
    villain_ids = list(VILLAINS.keys())
    for i, pos in enumerate(villain_positions):
        if i < len(villain_ids):
            pos_to_villain[pos] = villain_ids[i]

    print(f'', flush=True)
    print(f'=== HAND {hand_num} ===', flush=True)
    print(f'Hero: {hero_cards} at {hero_pos}', flush=True)

    villains_in_hand = list(VILLAINS.keys())
    total_llm_time = 0
    hero_invested = 0
    hand_over = False
    hero_folded = False

    for street in [Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER]:
        if hand_over or len(villains_in_hand) == 0:
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
        else:
            game.current_bet = 0
            game.hero.bet_this_street = 0

        board_str = str(game.board) if game.board and game.board.cards else ''
        print(f'--- {street.name} {board_str} ---', flush=True)

        # Villains act first (simplified)
        for vid, (style, bot) in VILLAINS.items():
            if vid not in villains_in_hand:
                continue

            # Create a mock game state for the bot
            villain_action = bot.decide(game)

            # Record observation and update beliefs
            obs = create_observation(
                vid, villain_action, street,
                Position.BB,  # Simplified
                game.pot.total,
                facing_raise=(game.current_bet > 0)
            )
            belief_state = revision_engine.process_observation(obs, belief_state)

            # Apply action
            if villain_action.action_type == ActionType.FOLD:
                villains_in_hand.remove(vid)
                print(f'  {vid} ({style}): folds', flush=True)
            elif villain_action.action_type == ActionType.CALL:
                game.pot.add(game.current_bet)
                print(f'  {vid} ({style}): calls', flush=True)
            elif villain_action.action_type in (ActionType.BET, ActionType.RAISE):
                amt = villain_action.amount
                game.pot.add(amt)
                game.current_bet = max(game.current_bet, amt)
                print(f'  {vid} ({style}): raises to {amt}', flush=True)
            else:
                print(f'  {vid} ({style}): checks', flush=True)

        if len(villains_in_hand) == 0:
            print(f'All villains folded!', flush=True)
            break

        # Hero acts
        response, elapsed = get_llm_decision(game, hero_cards, belief_state, villains_in_hand)
        total_llm_time += elapsed
        action, reasoning = parse_action(response, game.to_call, game.hero.stack)

        print(f'  Hero: {action} ({elapsed:.1f}s)', flush=True)
        print(f'    Reasoning: {reasoning[:100]}...' if len(reasoning) > 100 else f'    Reasoning: {reasoning}', flush=True)

        if action.action_type == ActionType.FOLD:
            hero_folded = True
            hand_over = True
        elif action.action_type in (ActionType.BET, ActionType.RAISE):
            hero_invested += action.amount
            game.pot.add(action.amount)
            game.current_bet = action.amount

    # Show belief updates
    print(f'', flush=True)
    print(f'Belief updates after hand:', flush=True)
    for vid in VILLAINS:
        if vid in belief_state.villain_beliefs:
            vb = belief_state.villain_beliefs[vid]
            best_type, prob = vb.get_most_likely_player_type()
            print(f'  {vid}: {best_type} ({prob:.0%})', flush=True)

    return belief_state, {
        'llm_time': total_llm_time,
        'hero_folded': hero_folded,
        'villains_remaining': len(villains_in_hand)
    }


if __name__ == '__main__':
    print('=== 6-PLAYER TABLE TEST ===', flush=True)
    print('Hero (LLM) vs 5 villains:', flush=True)
    for vid, (style, _) in VILLAINS.items():
        print(f'  {vid}: {style}', flush=True)
    print('', flush=True)

    # Initialize belief system
    belief_state = BeliefState()
    revision_engine = BeliefRevisionEngine(trust_discount=0.9)

    # Create beliefs for each villain
    for vid in VILLAINS:
        belief_state.villain_beliefs[vid] = create_default_villain_beliefs(vid)

    print('Initial beliefs: All villains at vacuous (no information)', flush=True)

    # Run 3 hands to accumulate beliefs
    total_time = 0
    for hand_num in range(1, 4):
        belief_state, result = run_table_hand(belief_state, revision_engine, hand_num)
        total_time += result['llm_time']

    print('', flush=True)
    print('=== FINAL BELIEF STATE ===', flush=True)
    for vid, (expected_style, _) in VILLAINS.items():
        vb = belief_state.villain_beliefs[vid]
        best_type, prob = vb.get_most_likely_player_type()
        dist = vb.get_player_type_distribution()

        # Check if belief matches expected style
        expected_map = {
            'CallingStation': 'Fish',
            'TightPassive': 'NIT',
            'LooseAggressive': 'LAG',
            'TagBot': 'TAG',
            'Random': 'Fish',  # Random appears fish-like
        }
        expected = expected_map.get(expected_style, '?')
        match = '✓' if best_type == expected else '?'

        print(f'{vid} ({expected_style}):', flush=True)
        print(f'  Detected: {best_type} ({prob:.0%}) {match}', flush=True)
        print(f'  Distribution: TAG={dist["TAG"]:.0%} LAG={dist["LAG"]:.0%} NIT={dist["NIT"]:.0%} Fish={dist["Fish"]:.0%}', flush=True)

    print('', flush=True)
    print(f'=== Total LLM time: {total_time:.1f}s ===', flush=True)
