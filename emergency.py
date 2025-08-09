#!/usr/bin/env python3
"""
Emergency minimal server for Railway debugging
This will run with zero dependencies and always work
"""
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json

class EmergencyHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        try:
            if self.path == '/health':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                response = {
                    "status": "healthy",
                    "mode": "emergency",
                    "python_version": sys.version,
                    "working_dir": os.getcwd(),
                    "environment": dict(os.environ)
                }
                self.wfile.write(json.dumps(response).encode())
                
            elif self.path == '/':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                response = {
                    "message": "Emergency server running",
                    "status": "ok",
                    "endpoints": ["/health", "/debug"],
                    "note": "Main app failed to start"
                }
                self.wfile.write(json.dumps(response).encode())
                
            elif self.path == '/debug':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                # Get file system info safely
                files = []
                try:
                    files = os.listdir('/app')[:10]  # First 10 files only
                except:
                    files = ["error_reading_directory"]
                
                response = {
                    "emergency_mode": True,
                    "python_executable": sys.executable,
                    "python_path": sys.path[:3],
                    "app_files": files,
                    "port": os.getenv('PORT', '8080'),
                    "railway_env": os.getenv('RAILWAY_ENVIRONMENT_NAME', 'unknown'),
                    "user": os.getenv('USER', 'unknown')
                }
                self.wfile.write(json.dumps(response, indent=2).encode())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'Not Found')
                
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"Error: {str(e)}".encode())

if __name__ == '__main__':
    # Railway sets PORT environment variable
    port_str = os.getenv('PORT', '8080')
    try:
        port = int(port_str)
    except ValueError:
        print(f"⚠️ Invalid PORT value '{port_str}', using 8080")
        port = 8080
    
    print(f"🚨 EMERGENCY SERVER STARTING - {os.getenv('RAILWAY_ENVIRONMENT_NAME', 'local')}")
    print(f"🔍 Python: {sys.version}")
    print(f"🔍 Working dir: {os.getcwd()}")
    print(f"🔍 Port from Railway: {port_str} -> {port}")
    print(f"🔍 User: {os.getenv('USER', 'unknown')}")
    print(f"🔍 All env vars: {list(os.environ.keys())[:10]}...")  # Don't log sensitive data
    
    try:
        # Bind to all interfaces (Railway requirement)
        server = HTTPServer(('0.0.0.0', port), EmergencyHandler)
        print(f"✅ Emergency server bound to 0.0.0.0:{port}")
        print(f"🔗 Test: curl http://0.0.0.0:{port}/health")
        print(f"🚀 Server starting...")
        server.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} already in use - trying port 8080")
            try:
                server = HTTPServer(('0.0.0.0', 8080), EmergencyHandler)
                server.serve_forever()
            except Exception as e2:
                print(f"❌ Backup port 8080 also failed: {e2}")
                sys.exit(1)
        else:
            print(f"❌ Network error: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Emergency server failed: {e}")
        print(f"Exception type: {type(e)}")
        sys.exit(1)