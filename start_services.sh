#!/bin/bash

# Drone Detection System Startup Script
# This script starts both the Flask API and Streamlit UI

echo "🚁 Starting Drone Detection System..."
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Function to start Flask API
start_flask() {
    echo "🔧 Starting Flask API on port 5000..."
    cd app
    python drone_detection.py &
    FLASK_PID=$!
    echo "Flask API started with PID: $FLASK_PID"
    cd ..
}

# Function to start Streamlit UI
start_streamlit() {
    echo "🌐 Starting Streamlit UI on port 8501..."
    sleep 3  # Wait for Flask to start
    streamlit run app/drone_detection_app.py --server.port 8501 &
    STREAMLIT_PID=$!
    echo "Streamlit UI started with PID: $STREAMLIT_PID"
}

# Start services
start_flask
start_streamlit

echo ""
echo "✅ Services started successfully!"
echo "=================================="
echo "🔧 Flask API: http://localhost:5000"
echo "🌐 Streamlit UI: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $FLASK_PID 2>/dev/null
    kill $STREAMLIT_PID 2>/dev/null
    echo "Services stopped."
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Wait for user interrupt
wait
