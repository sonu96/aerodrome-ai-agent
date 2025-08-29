"""
Mock implementations for LLM and AI service components.

Provides mock objects that simulate OpenAI, LangGraph, and other
AI service behavior for testing.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Generator
from unittest.mock import MagicMock, AsyncMock
import json


class MockOpenAIClient:
    """Mock OpenAI client for testing."""
    
    def __init__(self):
        self.embeddings = MockEmbeddings()
        self.chat = MockChat()
        self._should_fail = False
        self._failure_reason = "APIError"
        self._rate_limited = False
    
    def set_should_fail(self, should_fail: bool, reason: str = "APIError"):
        """Configure mock to simulate failures."""
        self._should_fail = should_fail
        self._failure_reason = reason
        self.embeddings.set_should_fail(should_fail, reason)
        self.chat.set_should_fail(should_fail, reason)
    
    def set_rate_limited(self, rate_limited: bool):
        """Configure mock to simulate rate limiting."""
        self._rate_limited = rate_limited
        self.embeddings.set_rate_limited(rate_limited)
        self.chat.set_rate_limited(rate_limited)


class MockEmbeddings:
    """Mock OpenAI embeddings API."""
    
    def __init__(self):
        self._should_fail = False
        self._failure_reason = "APIError"
        self._rate_limited = False
        self._custom_embeddings = {}
    
    def create(self, input: str | List[str], model: str = "text-embedding-ada-002"):
        """Mock embedding creation."""
        if self._should_fail:
            raise Exception(self._failure_reason)
        
        if self._rate_limited:
            raise Exception("Rate limit exceeded")
        
        # Convert single input to list
        if isinstance(input, str):
            inputs = [input]
        else:
            inputs = input
        
        # Generate mock embeddings
        data = []
        for i, text in enumerate(inputs):
            # Check for custom embeddings
            if text in self._custom_embeddings:
                embedding = self._custom_embeddings[text]
            else:
                # Generate deterministic mock embedding based on text
                embedding = self._generate_mock_embedding(text)
            
            data.append(MockEmbeddingData(embedding, i))
        
        return MockEmbeddingResponse(data, model, len(inputs))
    
    def _generate_mock_embedding(self, text: str, dimension: int = 1536) -> List[float]:
        """Generate deterministic mock embedding."""
        # Use hash of text to generate consistent embeddings
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert hash bytes to float values between -1 and 1
        embedding = []
        for i in range(dimension):
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] - 128) / 128.0
            embedding.append(value)
        
        # Normalize to unit vector
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    def set_custom_embedding(self, text: str, embedding: List[float]):
        """Set custom embedding for specific text."""
        self._custom_embeddings[text] = embedding
    
    def set_should_fail(self, should_fail: bool, reason: str = "APIError"):
        """Configure mock to simulate failures."""
        self._should_fail = should_fail
        self._failure_reason = reason
    
    def set_rate_limited(self, rate_limited: bool):
        """Configure mock to simulate rate limiting."""
        self._rate_limited = rate_limited


class MockEmbeddingData:
    """Mock embedding data response."""
    
    def __init__(self, embedding: List[float], index: int):
        self.embedding = embedding
        self.index = index


class MockEmbeddingResponse:
    """Mock embedding response."""
    
    def __init__(self, data: List[MockEmbeddingData], model: str, total_tokens: int):
        self.data = data
        self.model = model
        self.usage = MockUsage(total_tokens)


class MockUsage:
    """Mock usage information."""
    
    def __init__(self, total_tokens: int):
        self.total_tokens = total_tokens
        self.prompt_tokens = total_tokens


class MockChat:
    """Mock OpenAI chat API."""
    
    def __init__(self):
        self.completions = MockCompletions()
    
    def set_should_fail(self, should_fail: bool, reason: str = "APIError"):
        """Configure mock to simulate failures."""
        self.completions.set_should_fail(should_fail, reason)
    
    def set_rate_limited(self, rate_limited: bool):
        """Configure mock to simulate rate limiting."""
        self.completions.set_rate_limited(rate_limited)


class MockCompletions:
    """Mock OpenAI chat completions API."""
    
    def __init__(self):
        self._should_fail = False
        self._failure_reason = "APIError"
        self._rate_limited = False
        self._custom_responses = {}
        self._response_templates = {
            "analysis": "Based on the market data analysis, I identify a potential arbitrage opportunity with 85% confidence.",
            "decision": "I recommend executing a SWAP operation with the following parameters: amount=1000 USDC, expected_output=0.398 ETH, confidence=0.85",
            "summary": "Transaction executed successfully with profit of $25.50. Market conditions were favorable with low volatility.",
            "error": "Unable to execute transaction due to excessive slippage. Recommend waiting for better market conditions."
        }
    
    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ):
        """Mock chat completion creation."""
        if self._should_fail:
            raise Exception(self._failure_reason)
        
        if self._rate_limited:
            raise Exception("Rate limit exceeded")
        
        # Extract context from messages
        context = self._extract_context(messages)
        
        # Check for custom response
        if context in self._custom_responses:
            response_content = self._custom_responses[context]
        else:
            response_content = self._generate_response(messages, context)
        
        if stream:
            return self._create_streaming_response(response_content)
        else:
            return MockChatCompletion(
                response_content, model, len(messages), max_tokens or 1000
            )
    
    def _extract_context(self, messages: List[Dict[str, str]]) -> str:
        """Extract context from messages for response selection."""
        last_message = messages[-1] if messages else {}
        content = last_message.get("content", "").lower()
        
        if "analyze" in content or "analysis" in content:
            return "analysis"
        elif "decide" in content or "decision" in content:
            return "decision"
        elif "summary" in content or "summarize" in content:
            return "summary"
        elif "error" in content or "failed" in content:
            return "error"
        else:
            return "general"
    
    def _generate_response(self, messages: List[Dict[str, str]], context: str) -> str:
        """Generate appropriate response based on context."""
        if context in self._response_templates:
            return self._response_templates[context]
        
        # Default response
        return "I understand your request. Based on the provided information, I recommend proceeding with caution and monitoring market conditions."
    
    def _create_streaming_response(self, content: str) -> Generator:
        """Create streaming response generator."""
        words = content.split()
        for i, word in enumerate(words):
            chunk_content = word + (" " if i < len(words) - 1 else "")
            yield MockChatCompletionChunk(chunk_content, i == len(words) - 1)
    
    def set_custom_response(self, context: str, response: str):
        """Set custom response for specific context."""
        self._custom_responses[context] = response
    
    def set_response_template(self, template_name: str, template: str):
        """Set response template."""
        self._response_templates[template_name] = template
    
    def set_should_fail(self, should_fail: bool, reason: str = "APIError"):
        """Configure mock to simulate failures."""
        self._should_fail = should_fail
        self._failure_reason = reason
    
    def set_rate_limited(self, rate_limited: bool):
        """Configure mock to simulate rate limiting."""
        self._rate_limited = rate_limited


class MockChatCompletion:
    """Mock chat completion response."""
    
    def __init__(self, content: str, model: str, prompt_tokens: int, completion_tokens: int):
        self.id = f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.object = "chat.completion"
        self.created = int(datetime.now().timestamp())
        self.model = model
        self.choices = [MockChoice(content)]
        self.usage = MockChatUsage(prompt_tokens, completion_tokens)


class MockChoice:
    """Mock chat completion choice."""
    
    def __init__(self, content: str):
        self.index = 0
        self.message = MockMessage(content)
        self.finish_reason = "stop"


class MockMessage:
    """Mock chat message."""
    
    def __init__(self, content: str):
        self.role = "assistant"
        self.content = content


class MockChatUsage:
    """Mock chat completion usage."""
    
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class MockChatCompletionChunk:
    """Mock chat completion streaming chunk."""
    
    def __init__(self, content: str, is_final: bool = False):
        self.id = f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.object = "chat.completion.chunk"
        self.created = int(datetime.now().timestamp())
        self.model = "gpt-4"
        self.choices = [MockDelta(content, is_final)]


class MockDelta:
    """Mock chat completion delta."""
    
    def __init__(self, content: str, is_final: bool = False):
        self.index = 0
        self.delta = {"content": content} if not is_final else {}
        self.finish_reason = "stop" if is_final else None


class MockLangGraphState:
    """Mock LangGraph state for testing."""
    
    def __init__(self, initial_data: Dict[str, Any] = None):
        self._data = initial_data or {}
    
    def __getitem__(self, key: str):
        return self._data[key]
    
    def __setitem__(self, key: str, value: Any):
        self._data[key] = value
    
    def __contains__(self, key: str):
        return key in self._data
    
    def get(self, key: str, default: Any = None):
        return self._data.get(key, default)
    
    def update(self, data: Dict[str, Any]):
        self._data.update(data)
    
    def to_dict(self) -> Dict[str, Any]:
        return self._data.copy()


class MockLangGraphNode:
    """Mock LangGraph node for testing."""
    
    def __init__(self, name: str, execute_func=None):
        self.name = name
        self._execute_func = execute_func or self._default_execute
        self._should_fail = False
        self._failure_reason = "NodeExecutionError"
        self._execution_time = 0.1
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Mock node execution."""
        if self._should_fail:
            raise Exception(f"{self.name}: {self._failure_reason}")
        
        await asyncio.sleep(self._execution_time)  # Simulate processing time
        
        if self._execute_func:
            return await self._execute_func(state)
        else:
            return await self._default_execute(state)
    
    async def _default_execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Default execution logic."""
        # Add node execution timestamp
        state[f"{self.name}_timestamp"] = datetime.now().isoformat()
        state[f"{self.name}_executed"] = True
        return state
    
    def set_should_fail(self, should_fail: bool, reason: str = "NodeExecutionError"):
        """Configure mock to simulate failures."""
        self._should_fail = should_fail
        self._failure_reason = reason
    
    def set_execution_time(self, seconds: float):
        """Set execution time for testing."""
        self._execution_time = seconds
    
    def set_execute_func(self, func):
        """Set custom execute function."""
        self._execute_func = func


class MockLangGraph:
    """Mock LangGraph for testing."""
    
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.entry_point = None
        self._should_fail = False
        self._failure_reason = "GraphExecutionError"
    
    def add_node(self, name: str, func):
        """Add node to graph."""
        self.nodes[name] = MockLangGraphNode(name, func)
    
    def add_edge(self, from_node: str, to_node: str):
        """Add edge to graph."""
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append(to_node)
    
    def set_entry_point(self, node_name: str):
        """Set entry point for graph."""
        self.entry_point = node_name
    
    def compile(self, **kwargs):
        """Mock graph compilation."""
        return MockCompiledGraph(self)
    
    def set_should_fail(self, should_fail: bool, reason: str = "GraphExecutionError"):
        """Configure mock to simulate failures."""
        self._should_fail = should_fail
        self._failure_reason = reason


class MockCompiledGraph:
    """Mock compiled LangGraph for testing."""
    
    def __init__(self, graph: MockLangGraph):
        self.graph = graph
        self._execution_history = []
    
    async def ainvoke(self, initial_state: Dict[str, Any], config: Dict[str, Any] = None):
        """Mock graph invocation."""
        if self.graph._should_fail:
            raise Exception(self.graph._failure_reason)
        
        state = MockLangGraphState(initial_state)
        
        # Simple linear execution for testing
        if self.graph.entry_point:
            current_node = self.graph.entry_point
            executed_nodes = []
            
            while current_node and current_node in self.graph.nodes:
                node = self.graph.nodes[current_node]
                state_dict = state.to_dict()
                
                # Execute node
                result = await node.execute(state_dict)
                state.update(result)
                
                executed_nodes.append(current_node)
                self._execution_history.append({
                    "node": current_node,
                    "timestamp": datetime.now().isoformat(),
                    "state_after": state.to_dict().copy()
                })
                
                # Get next node (simple linear flow)
                next_nodes = self.graph.edges.get(current_node, [])
                current_node = next_nodes[0] if next_nodes else None
        
        return state.to_dict()
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history for testing."""
        return self._execution_history.copy()
    
    def clear_execution_history(self):
        """Clear execution history."""
        self._execution_history = []