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

from .llm_client import (
    create_claude_client, create_mock_client, check_api_available
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
    # LLM Client
    'create_claude_client', 'create_mock_client', 'check_api_available',
]
