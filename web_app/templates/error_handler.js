// Enhanced error handling for dashboard

function handleApiError(error, context = "") {
    console.error(`Error in ${context}:`, error);
    
    let userMessage = "An error occurred";
    
    if (error.message) {
        if (error.message.includes("Cannot connect")) {
            userMessage = "Cannot connect to API server. Please ensure the FastAPI server is running on port 8000.\n\nStart it with: python main.py --mode api";
        } else if (error.message.includes("timeout")) {
            userMessage = "Request timed out. This might be due to:\n- Missing API keys in .env file\n- Network connectivity issues\n- API service temporarily unavailable";
        } else if (error.message.includes("API keys")) {
            userMessage = "API keys not configured. Please add at least one API key to .env file:\n\nOPENWEATHER_API_KEY=your_key_here";
        } else {
            userMessage = error.message;
        }
    }
    
    // Show user-friendly alert
    alert(`Error: ${userMessage}`);
    
    // Update UI to show error state
    document.getElementById('aqiValue').textContent = '--';
    document.getElementById('aqiStatus').textContent = 'Error - Check console';
    document.getElementById('aqiCard').className = 'card text-center bg-danger text-white';
}

