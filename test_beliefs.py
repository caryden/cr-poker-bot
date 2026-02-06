#!/usr/bin/env python3
"""
Test belief state integration with LLM agent.

Proves:
1. Belief state is created for each villain
2. Beliefs update after observing villain actions
3. Belief info is passed to LLM in prompt
4. LLM returns action + reasoning
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
from src.tools.equity import calculate_equity
from src.tools.hand_eval import evaluate_hand

client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])


def parse_action(response: str, to_call: float, stack: float) -> tuple[Action, str]:
    """Parse LLM response into Action and extract reasoning."""
    lines = response.strip().split('\n')

    # Find action line and reasoning
    action_line = ""
    reasoning_lines = []

    for line in lines:
        line_lower = line.lower().strip()
        if any(x in line_lower for x in ['action:', 'decision:', 'fold', 'check', 'call', 'bet', 'raise', 'all-in']):
            if not action_line:  # Take first action line
                action_line = line_lower
        else:
            if line.strip():
                reasoning_lines.append(line.strip())

    reasoning = ' '.join(reasoning_lines)

    # Parse action
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

    # Default
    if to_call == 0:
        return Action.check(), reasoning
    return Action.call(to_call), reasoning


def get_llm_decision(game, hero_cards, belief_state: BeliefState) -> tuple[str, float]:
    """Get LLM decision with belief state context."""
    pot_odds = game.to_call / (game.pot.total + game.to_call) if game.to_call > 0 else 0
    eq = calculate_equity(hero_cards, game.board, num_opponents=len(game.villains), simulations=500)
    board_str = str(game.board) if game.board and game.board.cards else 'none'

    # Format belief state for prompt
    belief_str = belief_state.to_agent_format() if belief_state.villain_beliefs else "No beliefs yet."

    prompt = f'''You are a poker AI agent. Analyze and decide.

=== GAME STATE ===
Street: {game.street}
Board: {board_str}
Your hand: {hero_cards}
Pot: {game.pot.total:.0f}
To call: {game.to_call:.0f}
Your stack: {game.hero.stack:.0f}
Pot odds: {pot_odds:.0%}
Your equity: {eq.equity:.0%}

{belief_str}

=== INSTRUCTIONS ===
1. First, explain your reasoning (which beliefs/stats you're using)
2. Then state your action

Format:
REASONING: [your analysis using the belief state and game info]
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
    """Create an observation from a villain action."""
    return Observation(
        obs_id=f"obs_{player_id}_{street.name}",
        obs_type=ObservationType.ACTION,
        player_id=player_id,
        action=action,
        context=GameContext(
            street=street,
            position=position,
            pot_size=pot,
            to_call=0,
            num_players=2,
            is_heads_up=True,
            facing_raise=facing_raise
        )
    )


if __name__ == '__main__':
    print('=== BELIEF STATE INTEGRATION TEST ===', flush=True)
    print('', flush=True)

    # Initialize belief system
    belief_state = BeliefState()
    revision_engine = BeliefRevisionEngine(trust_discount=0.9)

    # Create villain with default beliefs
    villain_id = "villain_BB"
    villain_beliefs = create_default_villain_beliefs(villain_id)
    belief_state.villain_beliefs[villain_id] = villain_beliefs

    print('=== INITIAL BELIEFS ===', flush=True)
    print(belief_state.to_agent_format(), flush=True)
    print('', flush=True)

    # Simulate some observed villain actions to build beliefs
    print('=== SIMULATING VILLAIN ACTIONS ===', flush=True)

    # Villain calls preflop (suggests Fish)
    obs1 = create_observation(villain_id, Action.call(10), Street.PREFLOP, Position.BB, 25)
    belief_state = revision_engine.process_observation(obs1, belief_state)
    print(f'Observed: {obs1}', flush=True)

    # Villain calls flop bet (suggests Fish/passive)
    obs2 = create_observation(villain_id, Action.call(20), Street.FLOP, Position.BB, 50)
    belief_state = revision_engine.process_observation(obs2, belief_state)
    print(f'Observed: {obs2}', flush=True)

    # Villain calls turn bet (more Fish evidence)
    obs3 = create_observation(villain_id, Action.call(40), Street.TURN, Position.BB, 100)
    belief_state = revision_engine.process_observation(obs3, belief_state)
    print(f'Observed: {obs3}', flush=True)

    print('', flush=True)
    print('=== UPDATED BELIEFS ===', flush=True)
    print(belief_state.to_agent_format(), flush=True)
    print('', flush=True)

    # Now run a hand with these beliefs
    print('=== RUNNING HAND WITH BELIEF STATE ===', flush=True)

    hero_cards = HoleCards(Card.from_str('As'), Card.from_str('Kd'))
    print(f'Hero: {hero_cards}', flush=True)

    game = create_6max_game(
        hand_id='belief_test', hero_id='hero', hero_position=Position.BTN,
        hero_cards=hero_cards,
        stacks={Position.BTN: 500, Position.BB: 500},
        small_blind=5, big_blind=10
    )
    game.players = {k: v for k, v in game.players.items() if k in ['hero', 'villain_BB']}

    total_time = 0
    decisions = []

    board_cards = [Card.from_str('Qs'), Card.from_str('7h'), Card.from_str('2d'),
                   Card.from_str('Kc'), Card.from_str('3s')]

    for i, street in enumerate([Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER]):
        game.street = street
        if street == Street.FLOP:
            game.board = Board(cards=board_cards[:3])
        elif street == Street.TURN:
            game.board.cards.append(board_cards[3])
        elif street == Street.RIVER:
            game.board.cards.append(board_cards[4])

        if street == Street.PREFLOP:
            game.current_bet = 10
            game.hero.bet_this_street = 0  # to_call will be 10
        else:
            game.current_bet = 0
            game.hero.bet_this_street = 0

        board_str = str(game.board) if game.board and game.board.cards else ''
        print(f'', flush=True)
        print(f'--- {street.name} {board_str} ---', flush=True)

        response, elapsed = get_llm_decision(game, hero_cards, belief_state)
        total_time += elapsed

        action, reasoning = parse_action(response, game.to_call, game.hero.stack)

        print(f'LLM Response ({elapsed:.2f}s):', flush=True)
        print(f'  Reasoning: {reasoning[:200]}...' if len(reasoning) > 200 else f'  Reasoning: {reasoning}', flush=True)
        print(f'  Action: {action}', flush=True)

        decisions.append({
            'street': street.name,
            'action': str(action),
            'reasoning': reasoning,
            'time': elapsed
        })

    print('', flush=True)
    print('=== SUMMARY ===', flush=True)
    print(f'Total decisions: {len(decisions)}', flush=True)
    print(f'Total time: {total_time:.2f}s', flush=True)
    print(f'Avg per decision: {total_time/len(decisions):.2f}s', flush=True)
    print('', flush=True)

    # Show final belief state
    vb = belief_state.villain_beliefs[villain_id]
    best_type, prob = vb.get_most_likely_player_type()
    print(f'Final villain type assessment: {best_type} ({prob:.0%})', flush=True)
    print(f'Type distribution: {vb.get_player_type_distribution()}', flush=True)
