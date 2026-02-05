"""
Agent module for poker bot.

Provides the ReAct agent, tool interface, and TRT integration.
"""

from .tools import (
    ToolName, ToolCall, ToolResult, Tool, ToolContext, ToolRegistry,
    HandEvalTool, EquityCalcTool, PotOddsTool, EVCalcTool,
    GTOAdvisorTool, BeliefQueryTool
)

from .react import (
    AgentPhase, ReasoningStep, AgentDecision, ReActAgent
)

from .trt.strategy import (
    Strategy, StrategyEvaluation, RolloutResult, TRTDecision, TRTEngine
)

__all__ = [
    # Tools
    'ToolName', 'ToolCall', 'ToolResult', 'Tool', 'ToolContext', 'ToolRegistry',
    'HandEvalTool', 'EquityCalcTool', 'PotOddsTool', 'EVCalcTool',
    'GTOAdvisorTool', 'BeliefQueryTool',
    # ReAct
    'AgentPhase', 'ReasoningStep', 'AgentDecision', 'ReActAgent',
    # TRT
    'Strategy', 'StrategyEvaluation', 'RolloutResult', 'TRTDecision', 'TRTEngine',
]
