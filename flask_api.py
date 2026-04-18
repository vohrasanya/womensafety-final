"""
Flask REST API for Multi-Camera Safety System
Allows external applications to integrate with the safety system
"""
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from functools import wraps
import threading
import json
import os
from datetime import datetime
from pathlib import Path

# Import system components
from multi_camera_main import SafetySystemOrchestrator
from config import FLASK_PORT, FLASK_HOST

app = Flask(__name__)
CORS(app)

# Global orchestrator instance
orchestrator = None
orchestrator_thread = None


# ==========================================
# AUTHENTICATION DECORATOR
# ==========================================

API_KEY_HEADER = "X-API-Key"
API_KEY = "your-secret-api-key-here"  # Change this in production!


def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get(API_KEY_HEADER)
        
        if api_key != API_KEY:
            return jsonify({'error': 'Invalid or missing API key'}), 401
        
        return f(*args, **kwargs)
    
    return decorated_function


# ==========================================
# SYSTEM MANAGEMENT ENDPOINTS
# ==========================================

@app.route('/api/v1/system/init', methods=['POST'])
def init_system():
    """Initialize the safety system"""
    global orchestrator, orchestrator_thread
    
    try:
        if orchestrator is not None:
            return jsonify({'error': 'System already running'}), 400
        
        orchestrator = SafetySystemOrchestrator()
        
        if not orchestrator.setup_cameras():
            return jsonify({'error': 'Failed to setup cameras'}), 500
        
        return jsonify({
            'status': 'success',
            'message': 'System initialized',
            'cameras': list(orchestrator.camera_manager.cameras.keys())
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/system/start', methods=['POST'])
@require_api_key
def start_system():
    """Start the safety system"""
    global orchestrator, orchestrator_thread
    
    try:
        if orchestrator is None:
            return jsonify({'error': 'System not initialized'}), 400
        
        if orchestrator.is_running:
            return jsonify({'error': 'System already running'}), 400
        
        # Start in background thread
        orchestrator_thread = threading.Thread(
            target=orchestrator.start_system,
            daemon=True
        )
        orchestrator_thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'System started'
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/system/stop', methods=['POST'])
@require_api_key
def stop_system():
    """Stop the safety system"""
    global orchestrator
    
    try:
        if orchestrator is None or not orchestrator.is_running:
            return jsonify({'error': 'System not running'}), 400
        
        orchestrator.stop_system()
        
        return jsonify({
            'status': 'success',
            'message': 'System stopped'
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/system/status', methods=['GET'])
def get_system_status():
    """Get current system status"""
    try:
        if orchestrator is None:
            return jsonify({
                'status': 'not_initialized',
                'is_running': False
            }), 200
        
        status = orchestrator.get_system_status()
        
        return jsonify({
            'status': 'success',
            'is_running': status['is_running'],
            'cameras': status['cameras'],
            'statistics': status['statistics'],
            'active_threads': status['threads_alive']
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==========================================
# CAMERA MANAGEMENT ENDPOINTS
# ==========================================

@app.route('/api/v1/cameras', methods=['GET'])
def get_cameras():
    """Get all camera information"""
    try:
        if orchestrator is None:
            return jsonify({'error': 'System not initialized'}), 400
        
        cameras = orchestrator.camera_manager.get_camera_info()
        
        return jsonify({
            'status': 'success',
            'total_cameras': len(cameras),
            'cameras': cameras
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/cameras/<camera_id>', methods=['GET'])
def get_camera(camera_id):
    """Get specific camera information"""
    try:
        if orchestrator is None:
            return jsonify({'error': 'System not initialized'}), 400
        
        camera_info = orchestrator.camera_manager.get_camera_info(camera_id)
        
        if not camera_info:
            return jsonify({'error': 'Camera not found'}), 404
        
        return jsonify({
            'status': 'success',
            'camera': camera_info
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/cameras/<camera_id>/location', methods=['GET'])
def get_camera_location(camera_id):
    """Get camera GPS location"""
    try:
        if orchestrator is None:
            return jsonify({'error': 'System not initialized'}), 400
        
        location = orchestrator.camera_manager.get_camera_location(camera_id)
        
        if not location:
            return jsonify({'error': 'Camera not found'}), 404
        
        return jsonify({
            'status': 'success',
            'location': location
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==========================================
# ALERT ENDPOINTS
# ==========================================

@app.route('/api/v1/alerts', methods=['GET'])
def get_alerts():
    """Get alert history"""
    try:
        if orchestrator is None:
            return jsonify({'error': 'System not initialized'}), 400
        
        alerts_file = 'alerts_data.json'
        
        if not os.path.exists(alerts_file):
            return jsonify({
                'status': 'success',
                'total_alerts': 0,
                'alerts': []
            }), 200
        
        with open(alerts_file, 'r') as f:
            alerts = json.load(f)
        
        return jsonify({
            'status': 'success',
            'total_alerts': len(alerts),
            'alerts': alerts
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/alerts/statistics', methods=['GET'])
def get_alert_statistics():
    """Get alert statistics"""
    try:
        if orchestrator is None:
            return jsonify({'error': 'System not initialized'}), 400
        
        stats = orchestrator.alert_manager.get_statistics()
        
        return jsonify({
            'status': 'success',
            'statistics': stats
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/snapshots', methods=['GET'])
def get_snapshots():
    """Get list of snapshot files"""
    try:
        snapshot_dir = 'alerts/snapshots'
        
        if not os.path.exists(snapshot_dir):
            return jsonify({
                'status': 'success',
                'snapshots': []
            }), 200
        
        snapshots = [f for f in os.listdir(snapshot_dir) if f.endswith('.jpg')]
        snapshots.sort(reverse=True)
        
        return jsonify({
            'status': 'success',
            'total_snapshots': len(snapshots),
            'snapshots': snapshots
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/snapshots/<filename>', methods=['GET'])
def get_snapshot(filename):
    """Download specific snapshot"""
    try:
        snapshot_path = Path('alerts/snapshots') / filename
        
        if not snapshot_path.exists():
            return jsonify({'error': 'Snapshot not found'}), 404
        
        return send_file(str(snapshot_path), mimetype='image/jpeg')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==========================================
# MAP ENDPOINTS
# ==========================================

@app.route('/api/v1/maps/heatmap', methods=['GET'])
def get_heatmap():
    """Download heatmap HTML"""
    try:
        if not os.path.exists('heatmap.html'):
            return jsonify({'error': 'Heatmap not generated'}), 404
        
        return send_file('heatmap.html', mimetype='text/html')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/maps/detailed', methods=['GET'])
def get_detailed_map():
    """Download detailed alert map HTML"""
    try:
        if not os.path.exists('detailed_map.html'):
            return jsonify({'error': 'Map not generated'}), 404
        
        return send_file('detailed_map.html', mimetype='text/html')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==========================================
# HEALTH & INFO ENDPOINTS
# ==========================================

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }), 200


@app.route('/api/v1/info', methods=['GET'])
def get_info():
    """Get system information"""
    return jsonify({
        'name': 'Multi-Camera Safety Detection System',
        'version': '1.0.0',
        'description': 'Educational AI safety system for women safety detection',
        'endpoints': {
            'system': '/api/v1/system/*',
            'cameras': '/api/v1/cameras/*',
            'alerts': '/api/v1/alerts/*',
            'maps': '/api/v1/maps/*'
        }
    }), 200


# ==========================================
# ERROR HANDLERS
# ==========================================

@app.errorhandler(404)
def not_found(error):
    """404 error handler"""
    return jsonify({
        'error': 'Endpoint not found',
        'status_code': 404
    }), 404


@app.errorhandler(500)
def server_error(error):
    """500 error handler"""
    return jsonify({
        'error': 'Internal server error',
        'status_code': 500
    }), 500


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("="*60)
    print("🚀 Multi-Camera Safety System - REST API")
    print("="*60)
    print(f"API Key: {API_KEY}")
    print(f"Host: {FLASK_HOST}:{FLASK_PORT}")
    print(f"Docs: http://localhost:{FLASK_PORT}/api/v1/info")
    print("="*60)
    
    # Initialize system on startup
    orchestrator = SafetySystemOrchestrator()
    orchestrator.setup_cameras()
    
    # Start Flask server
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)
