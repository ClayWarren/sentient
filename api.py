"""
Sentient API - REST API for the Consciousness AI System
Provides endpoints to interact with the advanced consciousness capabilities
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from typing import Dict, Any
from consciousness_core import ConsciousnessAI, ProcessingMode, ConsciousnessLevel

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the consciousness AI system
consciousness_ai = ConsciousnessAI(consciousness_enabled=True)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Sentient Consciousness API',
        'version': '1.0.0'
    })

@app.route('/generate', methods=['POST'])
def generate_text():
    """Generate text with consciousness enhancement"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Missing required field: prompt'}), 400
        
        prompt = data['prompt']
        mode = data.get('mode', 'consciousness')
        max_tokens = data.get('max_tokens', 100)
        temperature = data.get('temperature', 0.7)
        
        # Validate mode
        try:
            processing_mode = ProcessingMode(mode)
        except ValueError:
            return jsonify({'error': f'Invalid mode: {mode}. Valid modes: {[m.value for m in ProcessingMode]}'}), 400
        
        # Generate response
        result = consciousness_ai.generate(
            prompt=prompt,
            mode=processing_mode,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return jsonify({
            'text': result.text,
            'consciousness_level': result.consciousness_level.name,
            'processing_mode': result.processing_mode.value,
            'consciousness_metrics': {
                'self_awareness': result.consciousness_metrics.self_awareness,
                'cognitive_integration': result.consciousness_metrics.cognitive_integration,
                'ethical_reasoning': result.consciousness_metrics.ethical_reasoning,
                'overall_consciousness': result.consciousness_metrics.overall_consciousness
            },
            'confidence': result.confidence,
            'processing_time': result.processing_time,
            'timestamp': result.timestamp
        })
        
    except Exception as e:
        logger.error(f"Error in generate_text: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get system status and statistics"""
    try:
        stats = consciousness_ai.get_stats()
        
        if consciousness_ai.consciousness_enabled:
            consciousness_status = consciousness_ai.consciousness.get_system_status()
        else:
            consciousness_status = {'consciousness_enabled': False}
        
        return jsonify({
            'system_status': consciousness_status,
            'generation_stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error in get_status: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/modes', methods=['GET'])
def get_modes():
    """Get available processing modes"""
    return jsonify({
        'modes': [
            {
                'name': mode.value,
                'description': {
                    'standard': 'Basic AI processing without consciousness enhancement',
                    'consciousness': 'Full consciousness-enhanced processing with self-awareness',
                    'creative': 'Creative and innovative response generation',
                    'ethical': 'Ethically-aware processing with enhanced safety measures'
                }.get(mode.value, 'Advanced processing mode')
            }
            for mode in ProcessingMode
        ]
    })

@app.route('/conversation/save', methods=['POST'])
def save_conversation():
    """Save conversation history to file"""
    try:
        data = request.get_json()
        filepath = data.get('filepath', 'api_conversation.json')
        
        consciousness_ai.save_conversation(filepath)
        
        return jsonify({
            'message': f'Conversation saved to {filepath}',
            'total_generations': len(consciousness_ai.generation_history)
        })
        
    except Exception as e:
        logger.error(f"Error in save_conversation: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

if __name__ == '__main__':
    logger.info("🚀 Starting Sentient Consciousness API...")
    logger.info("📡 API endpoints available:")
    logger.info("   POST /generate - Generate text with consciousness")
    logger.info("   GET  /status - Get system status")
    logger.info("   GET  /modes - Get available processing modes")
    logger.info("   GET  /health - Health check")
    logger.info("   POST /conversation/save - Save conversation")
    
    app.run(host='0.0.0.0', port=5000, debug=True)