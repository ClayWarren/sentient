"""
Function Calling and Tool Use Framework for Sentient AI
Enables the AI to call external functions and use tools
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import inspect
import traceback
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import subprocess
import requests
import os
import math
import datetime

class ToolType(Enum):
    COMPUTATION = "computation"
    INFORMATION = "information"
    ACTION = "action"
    ANALYSIS = "analysis"
    COMMUNICATION = "communication"

@dataclass
class Tool:
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Dict[str, Any]]
    tool_type: ToolType
    required_params: List[str]
    examples: List[Dict[str, Any]]

@dataclass
class FunctionCall:
    tool_name: str
    parameters: Dict[str, Any]
    confidence: float
    reasoning: str

@dataclass
class FunctionResult:
    success: bool
    result: Any
    error_message: Optional[str] = None
    execution_time: float = 0.0
    tokens_used: int = 0

class FunctionCallingModule(nn.Module):
    """Neural module for function call detection and parameter extraction"""
    
    def __init__(self, d_model: int = 768, num_tools: int = 50):
        super().__init__()
        self.d_model = d_model
        self.num_tools = num_tools
        
        # Tool selection layer
        self.tool_selector = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_tools),
            nn.Softmax(dim=-1)
        )
        
        # Intent detection
        self.intent_classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 4),  # [no_tool, single_tool, multi_tool, clarification_needed]
            nn.Softmax(dim=-1)
        )
        
        # Parameter extraction attention
        self.param_attention = nn.MultiheadAttention(d_model, num_heads=8)
        
        # Confidence predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input_embedding: torch.Tensor, available_tools: List[str]) -> Dict[str, torch.Tensor]:
        batch_size = input_embedding.size(0)
        
        # Predict tool selection probabilities
        tool_probs = self.tool_selector(input_embedding)
        
        # Predict intent
        intent_probs = self.intent_classifier(input_embedding)
        
        # Extract parameters using attention
        param_weights, _ = self.param_attention(input_embedding, input_embedding, input_embedding)
        
        # Predict confidence
        confidence = self.confidence_predictor(input_embedding).squeeze(-1)
        
        return {
            'tool_probabilities': tool_probs,
            'intent_probabilities': intent_probs,
            'parameter_attention': param_weights,
            'confidence': confidence
        }

class ToolRegistry:
    """Registry for managing available tools and functions"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_builtin_tools()
        
    def register_tool(self, tool: Tool):
        """Register a new tool"""
        self.tools[tool.name] = tool
        
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name)
        
    def list_tools(self, tool_type: Optional[ToolType] = None) -> List[Tool]:
        """List all available tools, optionally filtered by type"""
        if tool_type:
            return [tool for tool in self.tools.values() if tool.tool_type == tool_type]
        return list(self.tools.values())
        
    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all tools"""
        descriptions = "🛠️ **Available Tools:**\n\n"
        
        for tool_type in ToolType:
            type_tools = self.list_tools(tool_type)
            if type_tools:
                descriptions += f"**{tool_type.value.title()} Tools:**\n"
                for tool in type_tools:
                    descriptions += f"• `{tool.name}`: {tool.description}\n"
                descriptions += "\n"
                
        return descriptions
        
    def _register_builtin_tools(self):
        """Register built-in tools"""
        
        # Mathematical computation tools
        self.register_tool(Tool(
            name="calculate",
            description="Perform mathematical calculations with Python expressions",
            function=self._safe_calculate,
            parameters={
                "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
            },
            tool_type=ToolType.COMPUTATION,
            required_params=["expression"],
            examples=[
                {"expression": "2 + 3 * 4", "result": 14},
                {"expression": "math.sqrt(16)", "result": 4.0}
            ]
        ))
        
        # Web search (mock implementation)
        self.register_tool(Tool(
            name="web_search",
            description="Search the web for information",
            function=self._mock_web_search,
            parameters={
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Maximum number of results", "default": 5}
            },
            tool_type=ToolType.INFORMATION,
            required_params=["query"],
            examples=[
                {"query": "Python programming", "max_results": 3}
            ]
        ))
        
        # File operations
        self.register_tool(Tool(
            name="read_file",
            description="Read contents of a text file",
            function=self._read_file,
            parameters={
                "file_path": {"type": "string", "description": "Path to the file to read"},
                "max_lines": {"type": "integer", "description": "Maximum lines to read", "default": 100}
            },
            tool_type=ToolType.ACTION,
            required_params=["file_path"],
            examples=[
                {"file_path": "example.txt", "max_lines": 50}
            ]
        ))
        
        # System information
        self.register_tool(Tool(
            name="get_system_info",
            description="Get current system information (time, date, etc.)",
            function=self._get_system_info,
            parameters={
                "info_type": {"type": "string", "description": "Type of info: time, date, system", "default": "all"}
            },
            tool_type=ToolType.INFORMATION,
            required_params=[],
            examples=[
                {"info_type": "time"},
                {"info_type": "date"}
            ]
        ))
        
        # Text analysis
        self.register_tool(Tool(
            name="analyze_text",
            description="Analyze text for various properties (length, words, etc.)",
            function=self._analyze_text,
            parameters={
                "text": {"type": "string", "description": "Text to analyze"},
                "analysis_type": {"type": "string", "description": "Type: basic, detailed", "default": "basic"}
            },
            tool_type=ToolType.ANALYSIS,
            required_params=["text"],
            examples=[
                {"text": "Hello world", "analysis_type": "basic"}
            ]
        ))
        
    def _safe_calculate(self, expression: str) -> Dict[str, Any]:
        """Safely evaluate mathematical expressions"""
        try:
            # Allowed functions and constants
            safe_dict = {
                "__builtins__": {},
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "len": len, "pow": pow,
                "math": math, "pi": math.pi, "e": math.e,
                "sin": math.sin, "cos": math.cos, "tan": math.tan,
                "sqrt": math.sqrt, "log": math.log, "exp": math.exp
            }
            
            result = eval(expression, safe_dict)
            return {"success": True, "result": result, "type": type(result).__name__}
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    def _mock_web_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Mock web search implementation"""
        # In a real implementation, this would use a search API
        mock_results = [
            {
                "title": f"Result for '{query}' - Documentation",
                "url": f"https://docs.example.com/search?q={query.replace(' ', '+')}",
                "snippet": f"Comprehensive documentation about {query} with examples and tutorials."
            },
            {
                "title": f"{query} - Tutorial and Guide",
                "url": f"https://tutorial.example.com/{query.replace(' ', '-')}",
                "snippet": f"Step-by-step guide to learning {query} from beginner to advanced level."
            },
            {
                "title": f"{query} Best Practices",
                "url": f"https://bestpractices.example.com/{query.replace(' ', '_')}",
                "snippet": f"Industry best practices and common patterns for {query}."
            }
        ]
        
        return {
            "success": True,
            "query": query,
            "results": mock_results[:max_results],
            "total_results": len(mock_results)
        }
        
    def _read_file(self, file_path: str, max_lines: int = 100) -> Dict[str, Any]:
        """Read contents of a text file"""
        try:
            if not os.path.exists(file_path):
                return {"success": False, "error": f"File not found: {file_path}"}
                
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.rstrip('\n'))
                    
            return {
                "success": True,
                "content": lines,
                "lines_read": len(lines),
                "file_path": file_path
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    def _get_system_info(self, info_type: str = "all") -> Dict[str, Any]:
        """Get current system information"""
        import platform
        
        info = {}
        
        if info_type in ["all", "time"]:
            now = datetime.datetime.now()
            info["current_time"] = now.strftime("%H:%M:%S")
            info["current_date"] = now.strftime("%Y-%m-%d")
            info["timestamp"] = now.timestamp()
            
        if info_type in ["all", "system"]:
            info["platform"] = platform.system()
            info["platform_version"] = platform.version()
            info["python_version"] = platform.python_version()
            
        return {"success": True, "info": info}
        
    def _analyze_text(self, text: str, analysis_type: str = "basic") -> Dict[str, Any]:
        """Analyze text for various properties"""
        analysis = {
            "character_count": len(text),
            "word_count": len(text.split()),
            "line_count": len(text.split('\n')),
            "average_word_length": sum(len(word) for word in text.split()) / max(1, len(text.split()))
        }
        
        if analysis_type == "detailed":
            words = text.split()
            analysis.update({
                "unique_words": len(set(words)),
                "longest_word": max(words, key=len) if words else "",
                "shortest_word": min(words, key=len) if words else "",
                "uppercase_chars": sum(1 for c in text if c.isupper()),
                "lowercase_chars": sum(1 for c in text if c.islower()),
                "digit_chars": sum(1 for c in text if c.isdigit())
            })
            
        return {"success": True, "analysis": analysis}

class FunctionCallExecutor:
    """Executes function calls and manages results"""
    
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.execution_history: List[Dict[str, Any]] = []
        
    def execute_function_call(self, function_call: FunctionCall) -> FunctionResult:
        """Execute a function call and return the result"""
        start_time = datetime.datetime.now()
        
        try:
            # Get the tool
            tool = self.tool_registry.get_tool(function_call.tool_name)
            if not tool:
                return FunctionResult(
                    success=False,
                    result=None,
                    error_message=f"Tool '{function_call.tool_name}' not found"
                )
            
            # Validate required parameters
            missing_params = [param for param in tool.required_params 
                            if param not in function_call.parameters]
            if missing_params:
                return FunctionResult(
                    success=False,
                    result=None,
                    error_message=f"Missing required parameters: {missing_params}"
                )
            
            # Execute the function
            result = tool.function(**function_call.parameters)
            
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            
            # Log execution
            self.execution_history.append({
                'timestamp': start_time.isoformat(),
                'tool_name': function_call.tool_name,
                'parameters': function_call.parameters,
                'success': True,
                'execution_time': execution_time,
                'confidence': function_call.confidence
            })
            
            return FunctionResult(
                success=True,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            error_msg = f"Error executing {function_call.tool_name}: {str(e)}"
            
            # Log failed execution
            self.execution_history.append({
                'timestamp': start_time.isoformat(),
                'tool_name': function_call.tool_name,
                'parameters': function_call.parameters,
                'success': False,
                'error': error_msg,
                'execution_time': execution_time,
                'confidence': function_call.confidence
            })
            
            return FunctionResult(
                success=False,
                result=None,
                error_message=error_msg,
                execution_time=execution_time
            )

class FunctionCallingSystem:
    """Main function calling system that integrates with consciousness"""
    
    def __init__(self, consciousness_system=None):
        self.consciousness_system = consciousness_system
        self.tool_registry = ToolRegistry()
        self.executor = FunctionCallExecutor(self.tool_registry)
        self.function_calling_module = FunctionCallingModule()
        
        # Function calling patterns
        self.function_patterns = [
            r"(?:please\s+)?(?:can you\s+)?(?:could you\s+)?(?:use|call|run|execute)\s+(\w+)",
            r"(?:i need to|let me|i want to)\s+(\w+)",
            r"calculate\s+(.+)",
            r"search\s+(?:for\s+)?(.+)",
            r"read\s+(?:the\s+)?file\s+(.+)",
            r"analyze\s+(.+)",
            r"get\s+(?:the\s+)?(.+?)\s+(?:info|information)"
        ]
        
    def parse_function_call_intent(self, text: str) -> List[FunctionCall]:
        """Parse text to identify function call intents"""
        import re
        
        function_calls = []
        text_lower = text.lower().strip()
        
        # Check for calculation requests
        if any(word in text_lower for word in ['calculate', 'compute', 'math', '=', '+', '-', '*', '/']):
            # Extract mathematical expression
            calc_patterns = [
                r"calculate\s+(.+)",
                r"compute\s+(.+)",
                r"what\s+is\s+(.+)",
                r"solve\s+(.+)"
            ]
            
            for pattern in calc_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    expression = match.group(1).strip()
                    function_calls.append(FunctionCall(
                        tool_name="calculate",
                        parameters={"expression": expression},
                        confidence=0.8,
                        reasoning=f"Detected calculation request: {expression}"
                    ))
                    break
        
        # Check for search requests
        if any(word in text_lower for word in ['search', 'look up', 'find information']):
            search_patterns = [
                r"search\s+(?:for\s+)?(.+)",
                r"look\s+up\s+(.+)",
                r"find\s+information\s+about\s+(.+)"
            ]
            
            for pattern in search_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    query = match.group(1).strip()
                    function_calls.append(FunctionCall(
                        tool_name="web_search",
                        parameters={"query": query, "max_results": 3},
                        confidence=0.7,
                        reasoning=f"Detected search request: {query}"
                    ))
                    break
        
        # Check for file reading requests
        if any(word in text_lower for word in ['read file', 'open file', 'show file']):
            file_patterns = [
                r"read\s+(?:the\s+)?file\s+(.+)",
                r"open\s+(?:the\s+)?file\s+(.+)",
                r"show\s+(?:me\s+)?(?:the\s+)?file\s+(.+)"
            ]
            
            for pattern in file_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    file_path = match.group(1).strip()
                    function_calls.append(FunctionCall(
                        tool_name="read_file",
                        parameters={"file_path": file_path, "max_lines": 50},
                        confidence=0.9,
                        reasoning=f"Detected file reading request: {file_path}"
                    ))
                    break
        
        # Check for system info requests
        if any(word in text_lower for word in ['time', 'date', 'system info', 'current time']):
            info_type = "time" if "time" in text_lower else "date" if "date" in text_lower else "all"
            function_calls.append(FunctionCall(
                tool_name="get_system_info",
                parameters={"info_type": info_type},
                confidence=0.9,
                reasoning=f"Detected system info request: {info_type}"
            ))
        
        # Check for text analysis requests
        if any(word in text_lower for word in ['analyze', 'analyze text', 'text analysis']):
            # Try to extract the text to analyze
            analysis_patterns = [
                r"analyze\s+(?:the\s+)?(?:text\s+)?[\"'](.*?)[\"']",
                r"analyze\s+(?:this\s+)?text:\s*(.+)"
            ]
            
            for pattern in analysis_patterns:
                match = re.search(pattern, text)
                if match:
                    text_to_analyze = match.group(1).strip()
                    function_calls.append(FunctionCall(
                        tool_name="analyze_text",
                        parameters={"text": text_to_analyze, "analysis_type": "detailed"},
                        confidence=0.8,
                        reasoning=f"Detected text analysis request"
                    ))
                    break
        
        return function_calls
    
    def execute_function_calls(self, function_calls: List[FunctionCall]) -> List[FunctionResult]:
        """Execute a list of function calls"""
        results = []
        
        for function_call in function_calls:
            result = self.executor.execute_function_call(function_call)
            results.append(result)
            
            # Add to consciousness working memory if available
            if (self.consciousness_system and 
                hasattr(self.consciousness_system, 'working_memory')):
                
                self.consciousness_system.working_memory.add_experience({
                    'type': 'function_call',
                    'tool_name': function_call.tool_name,
                    'parameters': function_call.parameters,
                    'result': result.result if result.success else result.error_message,
                    'success': result.success,
                    'execution_time': result.execution_time,
                    'timestamp': self.consciousness_system.get_current_time() if hasattr(self.consciousness_system, 'get_current_time') else 0,
                    'significance': function_call.confidence
                })
        
        return results
    
    def process_with_tools(self, input_text: str) -> Dict[str, Any]:
        """Process input text and execute any identified function calls"""
        
        # Parse for function call intents
        function_calls = self.parse_function_call_intent(input_text)
        
        if not function_calls:
            return {
                'function_calls_detected': False,
                'message': 'No function calls detected in input text',
                'available_tools': self.tool_registry.get_tool_descriptions()
            }
        
        # Execute function calls
        results = self.execute_function_calls(function_calls)
        
        # Format results
        formatted_results = []
        for call, result in zip(function_calls, results):
            formatted_result = {
                'tool_name': call.tool_name,
                'parameters': call.parameters,
                'confidence': call.confidence,
                'reasoning': call.reasoning,
                'success': result.success,
                'result': result.result,
                'execution_time': result.execution_time
            }
            
            if not result.success:
                formatted_result['error'] = result.error_message
                
            formatted_results.append(formatted_result)
        
        return {
            'function_calls_detected': True,
            'calls_executed': len(function_calls),
            'results': formatted_results,
            'execution_summary': self._create_execution_summary(function_calls, results)
        }
    
    def _create_execution_summary(self, function_calls: List[FunctionCall], results: List[FunctionResult]) -> str:
        """Create a human-readable summary of function execution"""
        
        if not function_calls:
            return "No functions were executed."
        
        summary = f"🛠️ **Function Execution Summary** ({len(function_calls)} tools used):\n\n"
        
        for call, result in zip(function_calls, results):
            status_emoji = "✅" if result.success else "❌"
            summary += f"{status_emoji} **{call.tool_name}** ({result.execution_time:.3f}s)\n"
            
            if result.success:
                if isinstance(result.result, dict):
                    if 'result' in result.result:
                        summary += f"   Result: {result.result['result']}\n"
                    elif 'analysis' in result.result:
                        analysis = result.result['analysis']
                        summary += f"   Analysis: {analysis.get('word_count', 'N/A')} words, {analysis.get('character_count', 'N/A')} chars\n"
                    else:
                        summary += f"   Success: {str(result.result)[:100]}...\n"
                else:
                    summary += f"   Result: {str(result.result)[:100]}...\n"
            else:
                summary += f"   Error: {result.error_message}\n"
            
            summary += f"   Confidence: {call.confidence:.1%}\n\n"
        
        return summary

# Integration function for consciousness system
def integrate_function_calling(consciousness_system, input_text: str) -> Dict[str, Any]:
    """Integrate function calling with consciousness system"""
    
    function_system = FunctionCallingSystem(consciousness_system)
    
    # Process input for function calls
    result = function_system.process_with_tools(input_text)
    
    # Log function calling activity in consciousness
    if hasattr(consciousness_system, 'thought_log'):
        consciousness_system.thought_log.append({
            'timestamp': consciousness_system.get_current_time() if hasattr(consciousness_system, 'get_current_time') else 0,
            'type': 'function_calling',
            'input': input_text,
            'calls_detected': result['function_calls_detected'],
            'calls_executed': result.get('calls_executed', 0),
            'summary': result.get('execution_summary', 'No functions executed')
        })
    
    return result