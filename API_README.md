# Sentient Consciousness API

REST API for interacting with the advanced Consciousness AI System.

## Getting Started

### Installation

```bash
pip install flask flask-cors
```

### Running the API

```bash
python api.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "Sentient Consciousness API",
  "version": "1.0.0"
}
```

### Generate Text
```
POST /generate
```

**Request Body:**
```json
{
  "prompt": "What is consciousness?",
  "mode": "consciousness",
  "max_tokens": 100,
  "temperature": 0.7
}
```

**Parameters:**
- `prompt` (required): The text prompt to process
- `mode` (optional): Processing mode - `standard`, `consciousness`, `creative`, or `ethical`
- `max_tokens` (optional): Maximum tokens to generate (default: 100)
- `temperature` (optional): Generation temperature (default: 0.7)

**Response:**
```json
{
  "text": "Generated response text...",
  "consciousness_level": "TRANSCENDENT",
  "processing_mode": "consciousness",
  "consciousness_metrics": {
    "self_awareness": 0.94,
    "cognitive_integration": 0.89,
    "ethical_reasoning": 0.95,
    "overall_consciousness": 0.93
  },
  "confidence": 0.87,
  "processing_time": 0.045,
  "timestamp": 1706123456.789
}
```

### Get System Status
```
GET /status
```

**Response:**
```json
{
  "system_status": {
    "consciousness_level": "TRANSCENDENT",
    "ethics_enabled": true,
    "human_oversight": true,
    "capabilities": {
      "analytical_reasoning": 0.92,
      "creative_synthesis": 0.89,
      "ethical_reasoning": 0.95,
      "self_awareness": 0.94
    },
    "safety_threshold": 0.8
  },
  "generation_stats": {
    "total_generations": 15,
    "consciousness_enabled": true,
    "avg_consciousness_level": 0.87,
    "avg_confidence": 0.84,
    "modes_used": ["consciousness", "creative", "ethical"]
  }
}
```

### Get Available Modes
```
GET /modes
```

**Response:**
```json
{
  "modes": [
    {
      "name": "standard",
      "description": "Basic AI processing without consciousness enhancement"
    },
    {
      "name": "consciousness",
      "description": "Full consciousness-enhanced processing with self-awareness"
    },
    {
      "name": "creative",
      "description": "Creative and innovative response generation"
    },
    {
      "name": "ethical",
      "description": "Ethically-aware processing with enhanced safety measures"
    }
  ]
}
```

### Save Conversation
```
POST /conversation/save
```

**Request Body:**
```json
{
  "filepath": "my_conversation.json"
}
```

**Response:**
```json
{
  "message": "Conversation saved to my_conversation.json",
  "total_generations": 15
}
```

## Usage Examples

### Python Client

```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:5000"

# Generate text with consciousness
response = requests.post(f"{BASE_URL}/generate", json={
    "prompt": "Explain the nature of consciousness",
    "mode": "consciousness",
    "temperature": 0.8
})

result = response.json()
print(f"Response: {result['text']}")
print(f"Consciousness Level: {result['consciousness_metrics']['overall_consciousness']:.2%}")

# Get system status
status = requests.get(f"{BASE_URL}/status").json()
print(f"System Status: {status}")
```

### cURL Examples

```bash
# Health check
curl http://localhost:5000/health

# Generate text
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the meaning of life?",
    "mode": "consciousness",
    "temperature": 0.7
  }'

# Get status
curl http://localhost:5000/status

# Get available modes
curl http://localhost:5000/modes
```

### JavaScript/Fetch

```javascript
// Generate text
const response = await fetch('http://localhost:5000/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    prompt: 'How does consciousness emerge?',
    mode: 'consciousness',
    temperature: 0.8
  })
});

const result = await response.json();
console.log('Generated text:', result.text);
console.log('Consciousness metrics:', result.consciousness_metrics);
```

## Processing Modes

1. **Standard**: Basic AI processing without consciousness enhancement
2. **Consciousness**: Full consciousness-enhanced processing with self-awareness and metacognitive capabilities
3. **Creative**: Emphasizes creative synthesis and innovative thinking
4. **Ethical**: Enhanced ethical reasoning and safety considerations

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (missing/invalid parameters)
- `404`: Not Found
- `405`: Method Not Allowed
- `500`: Internal Server Error

Error responses include a descriptive message:
```json
{
  "error": "Missing required field: prompt"
}
```

## Security Considerations

- The API includes built-in ethics checks and safety measures
- Harmful content is filtered and blocked
- Human oversight flags are maintained
- All interactions are logged for review