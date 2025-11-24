"""
Flask web application for AirGuard dashboard.
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
import os
from datetime import datetime, timedelta

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Load environment variables from .env file first
from dotenv import load_dotenv
load_dotenv()

# FastAPI backend URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_TOKEN = os.getenv("API_TOKEN", "demo-token")


@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("index.html", api_base_url=API_BASE_URL, api_token=API_TOKEN)


@app.route("/mobile")
def mobile():
    """Mobile-optimized dashboard."""
    return render_template("mobile.html", api_base_url=API_BASE_URL, api_token=API_TOKEN)


@app.route("/mobile-app")
def mobile_app():
    """Enhanced mobile app."""
    return render_template("mobile_app.html", api_base_url=API_BASE_URL, api_token=API_TOKEN)


@app.route("/login")
def login():
    """Login page."""
    return render_template("login.html", api_base_url=API_BASE_URL)


@app.route("/register", methods=["GET"])
def register_page():
    """Registration page."""
    return render_template("register.html", api_base_url=API_BASE_URL)


@app.route("/api/proxy/register", methods=["POST"])
def proxy_register():
    """Proxy endpoint for user registration."""
    try:
        data = request.json
        
        # Debug logging
        print(f"Received registration data: {data}")
        
        # Required fields validation
        required_fields = ["full_name", "mobile", "age", "password", "confirm_password"]
        for field in required_fields:
            if not data.get(field):
                return jsonify({"detail": f"{field} is required"}), 400
        
        # Password length validation (bcrypt has a 72 byte limit)
        password = data.get("password", "")
        password_length = len(password.encode('utf-8'))
        print(f"Password length in bytes: {password_length}")
        if password_length > 72:
            return jsonify({"detail": "Password is too long. Please use a password with fewer than 72 characters."}), 400
        
        # Confirm password validation
        if password != data.get("confirm_password"):
            return jsonify({"detail": "Passwords do not match."}), 400
        
        # Debug: Log the exact data being sent
        print(f"Sending to FastAPI: {data}")
        
        # Call FastAPI register endpoint
        response = requests.post(
            f"{API_BASE_URL}/api/v1/auth/register",
            json=data,
            timeout=10
        )
        
        print(f"FastAPI request URL: {API_BASE_URL}/api/v1/auth/register")
        print(f"FastAPI request headers: {response.request.headers}")
        print(f"FastAPI request body: {response.request.body}")
        
        print(f"FastAPI response status: {response.status_code}")
        print(f"FastAPI response text: {response.text}")
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            error_data = response.json() if response.text else {}
            error_msg = error_data.get("detail", "Registration failed")
            # Handle the specific bcrypt password length error
            if "password cannot be longer than 72 bytes" in str(error_msg):
                return jsonify({"detail": "Password is too long. Please use a password with fewer than 72 characters."}), 400
            return jsonify({"detail": error_msg}), response.status_code
    
    except Exception as e:
        print(f"Exception in proxy_register: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"detail": str(e)}), 500


@app.route("/api/proxy/login", methods=["POST"])
def proxy_login():
    """Proxy endpoint for user login."""
    try:
        data = request.json
        identifier = data.get("identifier") or data.get("username")  # Accept both for compatibility
        password = data.get("password")
        
        if not identifier or not password:
            return jsonify({"error": "Identifier (mobile/email) and password are required"}), 400
        
        # Call FastAPI login endpoint with correct field names
        response = requests.post(
            f"{API_BASE_URL}/api/v1/auth/login",
            json={"identifier": identifier, "password": password},
            timeout=10
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            error_data = response.json() if response.text else {}
            error_msg = error_data.get("detail", "Login failed")
            return jsonify({"error": error_msg}), response.status_code
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/proxy/current-air-quality")
def proxy_current_air_quality():
    """Proxy endpoint for current air quality."""
    try:
        latitude = request.args.get("latitude", type=float)
        longitude = request.args.get("longitude", type=float)
        location_id = request.args.get("location_id", "Unknown")
        include_forecast = request.args.get("include_forecast", "false").lower() == "true"
        
        if not latitude or not longitude:
            return jsonify({"error": "latitude and longitude are required"}), 400
        
        # Check if FastAPI server is running
        try:
            health_check = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if health_check.status_code != 200:
                return jsonify({
                    "error": "FastAPI server is not responding properly. Please ensure the API server is running on port 8000."
                }), 503
        except requests.exceptions.ConnectionError:
            return jsonify({
                "error": "Cannot connect to FastAPI server. Please start it with: python main.py --mode api"
            }), 503
        except Exception as e:
            return jsonify({
                "error": f"Error connecting to API server: {str(e)}"
            }), 503
        
        # Call FastAPI backend
        params = {
            "latitude": latitude, 
            "longitude": longitude, 
            "location_id": location_id,
            "include_forecast": str(include_forecast).lower()
        }
        
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/v1/air-quality/current",
                params=params,
                headers={"Authorization": f"Bearer {API_TOKEN}"},
                timeout=30  # Increased timeout for API calls
            )
            
            if response.status_code == 200:
                return jsonify(response.json())
            elif response.status_code == 503:
                # Service unavailable - likely API keys missing
                error_data = response.json() if response.text else {}
                error_msg = error_data.get("detail", "Service unavailable")
                return jsonify({
                    "error": f"API service error: {error_msg}. Please check if API keys are configured in .env file."
                }), 503
            else:
                error_data = response.json() if response.text else {}
                error_msg = error_data.get("detail", response.text[:200])
                return jsonify({
                    "error": f"Failed to fetch data: {error_msg}"
                }), response.status_code
        
        except requests.exceptions.Timeout:
            return jsonify({
                "error": "Request timeout. The API call took too long. This might be due to missing API keys or network issues."
            }), 504
        except requests.exceptions.RequestException as e:
            return jsonify({
                "error": f"Network error: {str(e)}"
            }), 500
    
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Server error: {str(e)}",
            "details": traceback.format_exc() if app.debug else None
        }), 500


@app.route("/api/proxy/predictions", methods=["POST"])
def proxy_predictions():
    """Proxy endpoint for predictions."""
    try:
        data = request.json
        response = requests.post(
            f"{API_BASE_URL}/api/v1/predictions/generate",
            json=data,
            headers={"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            error_data = response.json() if response.text else {}
            error_msg = error_data.get("detail", "Failed to generate predictions")
            return jsonify({"error": error_msg}), response.status_code
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/proxy/advisory")
def proxy_advisory():
    """Proxy endpoint for health advisory."""
    try:
        location_id = request.args.get("location_id")
        aqi = request.args.get("aqi", type=float)
        latitude = request.args.get("latitude", type=float)
        longitude = request.args.get("longitude", type=float)
        vulnerable_groups = request.args.get("vulnerable_groups", "")
        language = request.args.get("language", "en")
        
        if not location_id:
            return jsonify({"error": "location_id is required"}), 400
        
        params = {"location_id": location_id, "language": language}
        if aqi is not None:
            params["aqi"] = aqi
        if latitude is not None and longitude is not None:
            params["latitude"] = latitude
            params["longitude"] = longitude
        if vulnerable_groups:
            params["vulnerable_groups"] = vulnerable_groups
        
        response = requests.get(
            f"{API_BASE_URL}/api/v1/advisory/health",
            params=params,
            headers={"Authorization": f"Bearer {API_TOKEN}"},
            timeout=10
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            error_data = response.json() if response.text else {}
            error_msg = error_data.get("detail", "Failed to fetch advisory")
            return jsonify({"error": error_msg}), response.status_code
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/proxy/historical")
def proxy_historical():
    """Proxy endpoint for historical data."""
    try:
        location_id = request.args.get("location_id")
        start_time = request.args.get("start_time")
        end_time = request.args.get("end_time")
        limit = request.args.get("limit", 100, type=int)
        
        if not location_id:
            return jsonify({"error": "location_id is required"}), 400
        
        params = {"location_id": location_id, "limit": limit}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        
        response = requests.get(
            f"{API_BASE_URL}/api/v1/air-quality/historical",
            params=params,
            headers={"Authorization": f"Bearer {API_TOKEN}"},
            timeout=10
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            error_data = response.json() if response.text else {}
            error_msg = error_data.get("detail", "Failed to fetch historical data")
            return jsonify({"error": error_msg}), response.status_code
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/proxy/insights")
def proxy_insights():
    """Proxy endpoint for AI insights."""
    try:
        latitude = request.args.get("latitude", type=float)
        longitude = request.args.get("longitude", type=float)
        location_id = request.args.get("location_id")
        days = request.args.get("days", 7, type=int)
        
        if not latitude or not longitude:
            return jsonify({"error": "latitude and longitude are required"}), 400
        
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "days": days
        }
        if location_id:
            params["location_id"] = location_id
        
        response = requests.get(
            f"{API_BASE_URL}/api/v1/insights/generate",
            params=params,
            headers={"Authorization": f"Bearer {API_TOKEN}"},
            timeout=15
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            error_data = response.json() if response.text else {}
            error_msg = error_data.get("detail", "Failed to fetch insights")
            return jsonify({"error": error_msg}), response.status_code
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/proxy/trends")
def proxy_trends():
    """Proxy endpoint for trend analysis."""
    try:
        location_id = request.args.get("location_id")
        days = request.args.get("days", 30, type=int)
        
        if not location_id:
            return jsonify({"error": "location_id is required"}), 400
        
        params = {"location_id": location_id, "days": days}
        
        response = requests.get(
            f"{API_BASE_URL}/api/v1/analysis/trends",
            params=params,
            headers={"Authorization": f"Bearer {API_TOKEN}"},
            timeout=15
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            error_data = response.json() if response.text else {}
            error_msg = error_data.get("detail", "Failed to fetch trend analysis")
            return jsonify({"error": error_msg}), response.status_code
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/proxy/system-status")
def proxy_system_status():
    """Proxy endpoint for system status."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/system/health",
            headers={"Authorization": f"Bearer {API_TOKEN}"},
            timeout=5
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            error_data = response.json() if response.text else {}
            error_msg = error_data.get("detail", "Failed to fetch system status")
            return jsonify({"error": error_msg}), response.status_code
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)