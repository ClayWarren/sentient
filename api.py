"""
Sentient API - REST API for the Consciousness AI System
Provides endpoints to interact with the advanced consciousness capabilities
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import logging
from typing import Dict, Any
import os
import tempfile
from werkzeug.utils import secure_filename
from consciousness_core import ConsciousnessAI, ConsciousnessLevel
from advanced_features import initialize_advanced_features

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the consciousness AI system with search
consciousness_ai = ConsciousnessAI(consciousness_enabled=True)

# Initialize advanced features
advanced = initialize_advanced_features(consciousness_ai)
file_processor = advanced['file_processor']
voice_processor = advanced['voice_processor']
video_processor = advanced['video_processor']
memory_system = advanced['memory']
reasoner = advanced['reasoner']

# Configure upload settings
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'xls', 'xlsx', 'png', 'jpg', 'jpeg', 'gif', 'json', 'md', 'py', 'js', 'html', 'css'}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        max_tokens = data.get('max_tokens', 100)
        temperature = data.get('temperature', 0.7)
        
        # Generate response with natural consciousness
        result = consciousness_ai.generate(
            prompt=prompt,
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

@app.route('/search', methods=['POST'])
def web_search():
    """Perform consciousness-enhanced web search"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing required field: query'}), 400
        
        query = data['query']
        
        # Get consciousness context
        consciousness_context = consciousness_ai.consciousness._get_consciousness_context(query)
        
        # Perform search
        search_result = consciousness_ai.consciousness.search_engine.search(query, consciousness_context)
        
        return jsonify(search_result)
        
    except Exception as e:
        logger.error(f"Error in web_search: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/search/stats', methods=['GET'])
def get_search_stats():
    """Get search engine statistics"""
    try:
        stats = consciousness_ai.consciousness.search_engine.get_search_statistics()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error in get_search_stats: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/mode', methods=['GET'])
def get_mode():
    """Get current processing mode"""
    return jsonify({
        'mode': 'consciousness',
        'description': 'Natural consciousness with adaptive responses and web search capabilities',
        'features': ['self_awareness', 'ethical_reasoning', 'creative_synthesis', 'web_search']
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

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload and process a file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            # Read file content
            file_content = file.read()
            
            # Process file
            result = file_processor.process_file(filename, file_content)
            
            # Generate AI response about the file
            if result.content:
                prompt = f"I've uploaded a {result.metadata.get('type', 'file')} file. Here's the content:\n\n{result.content[:1000]}..."
                ai_response = consciousness_ai.generate(prompt, max_tokens=200)
                
                response_data = {
                    'filename': result.filename,
                    'content_preview': result.content[:500],
                    'metadata': result.metadata,
                    'processing_time': result.processing_time,
                    'ai_analysis': ai_response.text,
                    'consciousness_level': ai_response.consciousness_level.name
                }
            else:
                response_data = {
                    'filename': result.filename,
                    'error': 'Could not extract content from file',
                    'metadata': result.metadata
                }
            
            return jsonify(response_data)
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/memory/store', methods=['POST'])
def store_memory():
    """Store information in long-term memory"""
    try:
        data = request.get_json()
        
        if not data or 'content' not in data:
            return jsonify({'error': 'Missing required field: content'}), 400
        
        content = data['content']
        memory_type = data.get('memory_type', 'short_term')
        metadata = data.get('metadata', {})
        
        memory_id = memory_system.store_memory(content, memory_type, metadata)
        
        return jsonify({
            'memory_id': memory_id,
            'memory_type': memory_type,
            'message': 'Memory stored successfully'
        })
        
    except Exception as e:
        logger.error(f"Error in store_memory: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/memory/retrieve', methods=['POST'])
def retrieve_memory():
    """Retrieve memories based on query"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing required field: query'}), 400
        
        query = data['query']
        memory_type = data.get('memory_type', 'all')
        top_k = data.get('top_k', 5)
        
        memories = memory_system.retrieve_memories(query, memory_type, top_k)
        
        return jsonify({
            'query': query,
            'memories': memories,
            'count': len(memories)
        })
        
    except Exception as e:
        logger.error(f"Error in retrieve_memory: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/reason', methods=['POST'])
def chain_of_thought():
    """Perform chain-of-thought reasoning"""
    try:
        data = request.get_json()
        
        if not data or 'problem' not in data:
            return jsonify({'error': 'Missing required field: problem'}), 400
        
        problem = data['problem']
        max_steps = data.get('max_steps', 5)
        
        result = reasoner.reason(problem, max_steps)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in chain_of_thought: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/voice/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio to text (simulated for now)"""
    try:
        # This would normally process audio data
        # For now, return a simulated response
        return jsonify({
            'text': 'Voice transcription endpoint ready',
            'confidence': 0.95,
            'duration': 2.5,
            'note': 'Actual voice processing requires microphone access from client'
        })
        
    except Exception as e:
        logger.error(f"Error in transcribe_audio: {str(e)}")
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
    logger.info("   POST /search - Web search with consciousness")
    logger.info("   POST /upload - Upload and process files")
    logger.info("   POST /memory/store - Store in long-term memory")
    logger.info("   POST /memory/retrieve - Retrieve from memory")
    logger.info("   POST /reason - Chain-of-thought reasoning")
    logger.info("   POST /voice/transcribe - Voice transcription")
    logger.info("   GET  /status - Get system status")
    logger.info("   GET  /health - Health check")
    logger.info("   POST /conversation/save - Save conversation")
    
    app.run(host='0.0.0.0', port=5000, debug=True)