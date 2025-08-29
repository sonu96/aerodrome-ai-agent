"""
Test suite for the Aerodrome AI Agent API

Basic tests to validate API endpoints and functionality.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import json

# Import the FastAPI app
from src.api.main import app, Config
from src.api.models import QueryRequest, SystemStatus
from src.brain.core import BrainConfig, AerodromeBrain


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_brain():
    """Create mock brain instance"""
    brain = MagicMock(spec=AerodromeBrain)
    brain.get_system_health = AsyncMock(return_value={
        "system_status": "healthy",
        "uptime": "0:01:30",
        "component_health": {},
        "metrics": {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "success_rate": 0.0,
            "avg_response_time": 0.0,
            "memory_items": 0,
            "confidence_avg": 0.0
        }
    })
    return brain


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_liveness_check(self, client):
        """Test liveness endpoint"""
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert "timestamp" in data
    
    @patch('src.api.main.get_brain')
    def test_readiness_check_healthy(self, mock_get_brain, client, mock_brain):
        """Test readiness check when system is healthy"""
        mock_get_brain.return_value = mock_brain
        
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
    
    @patch('src.api.main.get_brain')
    def test_readiness_check_unhealthy(self, mock_get_brain, client):
        """Test readiness check when system is unhealthy"""
        mock_get_brain.side_effect = Exception("Brain not initialized")
        
        response = client.get("/health/ready")
        assert response.status_code == 503
    
    @patch('src.api.main.get_brain')
    def test_health_check_detailed(self, mock_get_brain, client, mock_brain):
        """Test detailed health check"""
        mock_get_brain.return_value = mock_brain
        
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "uptime" in data


class TestMetricsEndpoints:
    """Test metrics endpoints"""
    
    def test_prometheus_metrics(self, client):
        """Test Prometheus metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
    
    @patch('src.api.main.get_brain')
    @patch('src.api.main.verify_api_key')
    def test_detailed_metrics(self, mock_verify, mock_get_brain, client, mock_brain):
        """Test detailed metrics endpoint"""
        mock_verify.return_value = "test-key"
        mock_get_brain.return_value = mock_brain
        
        headers = {"Authorization": "Bearer test-key"}
        response = client.get("/metrics/detailed", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "system_metrics" in data
        assert "component_health" in data
        assert "api_metrics" in data


class TestQueryEndpoints:
    """Test query processing endpoints"""
    
    @patch('src.api.main.get_brain')
    @patch('src.api.main.verify_api_key')
    def test_process_query_success(self, mock_verify, mock_get_brain, client, mock_brain):
        """Test successful query processing"""
        mock_verify.return_value = "test-key"
        mock_get_brain.return_value = mock_brain
        
        # Mock brain response
        mock_response = MagicMock()
        mock_response.query_id = "test-query-123"
        mock_response.response = "This is a test response"
        mock_response.confidence = 0.95
        mock_response.sources = []
        mock_response.metadata = {}
        mock_brain.process_query = AsyncMock(return_value=mock_response)
        
        headers = {"Authorization": "Bearer test-key"}
        query_data = {
            "query": "What is Aerodrome?",
            "user_id": "test-user",
            "context": {"test": "context"}
        }
        
        response = client.post("/query", json=query_data, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["query_id"] == "test-query-123"
        assert data["response"] == "This is a test response"
        assert data["confidence"] == 0.95
    
    @patch('src.api.main.get_brain')
    @patch('src.api.main.verify_api_key')
    def test_process_query_failure(self, mock_verify, mock_get_brain, client):
        """Test query processing failure"""
        mock_verify.return_value = "test-key"
        mock_get_brain.side_effect = Exception("Brain error")
        
        headers = {"Authorization": "Bearer test-key"}
        query_data = {"query": "What is Aerodrome?"}
        
        response = client.post("/query", json=query_data, headers=headers)
        assert response.status_code == 500
    
    def test_query_without_auth(self, client):
        """Test query without authentication"""
        query_data = {"query": "What is Aerodrome?"}
        
        response = client.post("/query", json=query_data)
        assert response.status_code == 401


class TestValidation:
    """Test request validation"""
    
    @patch('src.api.main.verify_api_key')
    def test_query_validation_empty(self, mock_verify, client):
        """Test query validation with empty query"""
        mock_verify.return_value = "test-key"
        headers = {"Authorization": "Bearer test-key"}
        query_data = {"query": ""}
        
        response = client.post("/query", json=query_data, headers=headers)
        assert response.status_code == 422
    
    @patch('src.api.main.verify_api_key')
    def test_query_validation_too_long(self, mock_verify, client):
        """Test query validation with too long query"""
        mock_verify.return_value = "test-key"
        headers = {"Authorization": "Bearer test-key"}
        query_data = {"query": "x" * 3000}  # Exceeds max length
        
        response = client.post("/query", json=query_data, headers=headers)
        assert response.status_code == 422


class TestAuthentication:
    """Test authentication and authorization"""
    
    def test_invalid_api_key(self, client):
        """Test with invalid API key"""
        headers = {"Authorization": "Bearer invalid-key"}
        
        # Mock Config.API_KEY to have a value
        with patch.object(Config, 'API_KEY', 'valid-key'):
            response = client.get("/metrics/detailed", headers=headers)
            assert response.status_code == 401
    
    def test_missing_api_key(self, client):
        """Test with missing API key"""
        response = client.get("/metrics/detailed")
        assert response.status_code == 401


class TestWebSocket:
    """Test WebSocket functionality"""
    
    def test_websocket_connection(self, client):
        """Test WebSocket connection"""
        with client.websocket_connect("/ws") as websocket:
            # Should receive welcome message
            data = websocket.receive_json()
            assert data["type"] == "welcome"
            assert "Connected to Aerodrome AI Agent" in data["message"]
    
    def test_websocket_ping_pong(self, client):
        """Test WebSocket ping/pong"""
        with client.websocket_connect("/ws") as websocket:
            # Skip welcome message
            websocket.receive_json()
            
            # Send ping
            websocket.send_json({"type": "ping"})
            
            # Should receive pong
            data = websocket.receive_json()
            assert data["type"] == "pong"


class TestAdminEndpoints:
    """Test admin endpoints"""
    
    @patch('src.api.main.verify_api_key')
    def test_brain_restart_unauthorized(self, mock_verify, client):
        """Test brain restart without proper authorization"""
        mock_verify.side_effect = Exception("Unauthorized")
        headers = {"Authorization": "Bearer invalid-key"}
        
        response = client.post("/admin/brain/restart", headers=headers)
        assert response.status_code == 401


class TestErrorHandling:
    """Test error handling"""
    
    def test_404_handler(self, client):
        """Test 404 error handling"""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
    
    @patch('src.api.main.get_brain')
    def test_500_handler(self, mock_get_brain, client):
        """Test 500 error handling"""
        mock_get_brain.side_effect = Exception("Internal error")
        
        response = client.get("/health")
        # The error should be caught and return 500 status
        assert response.status_code == 500 or response.status_code == 503


# Integration test markers
@pytest.mark.integration
class TestIntegration:
    """Integration tests (require actual brain setup)"""
    
    @pytest.mark.skip(reason="Requires actual brain initialization")
    def test_full_query_flow(self, client):
        """Test full query processing flow"""
        # This would test the complete flow with a real brain instance
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])