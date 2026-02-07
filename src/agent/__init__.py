"""
Agent module for poker bot.

Provides tool interfaces and LLM client utilities.
"""

from .tools import (
    ToolName, ToolCall, ToolResult, Tool, ToolContext, ToolRegistry,
    HandEvalTool, EquityCalcTool, PotOddsTool, EVCalcTool,
    GTOAdvisorTool, BeliefQueryTool
)

from .llm_client import (
    create_claude_client, check_api_available
)

__all__ = [
    # Tools
    'ToolName', 'ToolCall', 'ToolResult', 'Tool', 'ToolContext', 'ToolRegistry',
    'HandEvalTool', 'EquityCalcTool', 'PotOddsTool', 'EVCalcTool',
    'GTOAdvisorTool', 'BeliefQueryTool',
    # LLM Client
    'create_claude_client', 'check_api_available',
]
