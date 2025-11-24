// Check if user is authenticated
document.addEventListener('DOMContentLoaded', function() {
    const token = localStorage.getItem('airguard_token');
    if (!token) {
        // Redirect to login if not authenticated
        window.location.href = '/login';
        return;
    }
    
    // Set username display
    const user = JSON.parse(localStorage.getItem('airguard_user') || '{}');
    if (user.full_name) {
        document.getElementById('usernameDisplay').textContent = user.full_name;
    }
    
    // Set up logout button
    document.getElementById('logoutBtn').addEventListener('click', function(e) {
        e.preventDefault();
        localStorage.removeItem('airguard_token');
        localStorage.removeItem('airguard_user');
        window.location.href = '/login';
    });
    
    // Initialize map
    initMap();
    
    // Try to detect user location on page load
    detectUserLocation();
});

// Global variables for map
let map;
let userMarker;
let hotspotMarkers = [];

function initMap() {
    // Initialize Leaflet map
    map = L.map('map').setView([28.6139, 77.2090], 13);
    
    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);
    
    // Add legend
    const legend = L.control({ position: 'bottomright' });
    legend.onAdd = function() {
        const div = L.DomUtil.create('div', 'info legend');
        div.innerHTML = `
            <div style="background: white; padding: 10px; border-radius: 5px; box-shadow: 0 0 15px rgba(0,0,0,0.2);">
                <h6>AQI Legend</h6>
                <div><i style="background: #00e400; width: 18px; height: 18px; display: inline-block; margin-right: 5px;"></i> 0-50 (Good)</div>
                <div><i style="background: #ffff00; width: 18px; height: 18px; display: inline-block; margin-right: 5px;"></i> 51-100 (Moderate)</div>
                <div><i style="background: #ff7e00; width: 18px; height: 18px; display: inline-block; margin-right: 5px;"></i> 101-150 (Unhealthy SG)</div>
                <div><i style="background: #ff0000; width: 18px; height: 18px; display: inline-block; margin-right: 5px;"></i> 151-200 (Unhealthy)</div>
                <div><i style="background: #8f3f97; width: 18px; height: 18px; display: inline-block; margin-right: 5px;"></i> 201-300 (Very Unhealthy)</div>
                <div><i style="background: #7e0023; width: 18px; height: 18px; display: inline-block; margin-right: 5px;"></i> 301+ (Hazardous)</div>
                <div><i style="background: #0000ff; width: 18px; height: 18px; display: inline-block; margin-right: 5px; border-radius: 50%;"></i> Your Location</div>
            </div>
        `;
        return div;
    };
    legend.addTo(map);
}

function getColorFromAQI(aqi) {
    if (aqi <= 50) return '#00e400';      // Green
    else if (aqi <= 100) return '#ffff00'; // Yellow
    else if (aqi <= 150) return '#ff7e00'; // Orange
    else if (aqi <= 200) return '#ff0000'; // Red
    else if (aqi <= 300) return '#8f3f97'; // Purple
    else return '#7e0023';                 // Maroon
}

function getAQICategory(aqi) {
    if (aqi <= 50) return 'Good';
    else if (aqi <= 100) return 'Moderate';
    else if (aqi <= 150) return 'Unhealthy for Sensitive Groups';
    else if (aqi <= 200) return 'Unhealthy';
    else if (aqi <= 300) return 'Very Unhealthy';
    else return 'Hazardous';
}

function detectUserLocation() {
    const statusDiv = document.getElementById('locationStatus');
    statusDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Detecting your location...';
    
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            function(position) {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                
                document.getElementById('latitudeInput').value = lat.toFixed(4);
                document.getElementById('longitudeInput').value = lon.toFixed(4);
                document.getElementById('locationInput').value = `Current Location (${lat.toFixed(4)}, ${lon.toFixed(4)})`;
                
                statusDiv.innerHTML = `<i class="fas fa-check-circle text-success"></i> Location detected: ${lat.toFixed(4)}, ${lon.toFixed(4)}`;
                
                // Update map view to user location
                map.setView([lat, lon], 13);
                
                // Add/update user marker
                if (userMarker) {
                    map.removeLayer(userMarker);
                }
                userMarker = L.marker([lat, lon], {
                    title: 'Your Location'
                }).addTo(map)
                  .bindPopup('<b>Your Location</b><br>Detected automatically')
                  .openPopup();
                
                // Fetch air quality data for detected location
                fetchCurrentData();
                
                // Fetch and display hotspots
                fetchHotspots(lat, lon);
            },
            function(error) {
                let errorMessage = '';
                switch(error.code) {
                    case error.PERMISSION_DENIED:
                        errorMessage = "Location access denied. Please enable location services in your browser settings.";
                        break;
                    case error.POSITION_UNAVAILABLE:
                        errorMessage = "Location information is unavailable.";
                        break;
                    case error.TIMEOUT:
                        errorMessage = "The request to get user location timed out.";
                        break;
                    case error.UNKNOWN_ERROR:
                        errorMessage = "An unknown error occurred.";
                        break;
                }
                statusDiv.innerHTML = `<i class="fas fa-exclamation-triangle text-warning"></i> ${errorMessage}`;
            },
            {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 300000 // 5 minutes
            }
        );
    } else {
        statusDiv.innerHTML = '<i class="fas fa-exclamation-triangle text-danger"></i> Geolocation is not supported by this browser.';
    }
}

async function fetchHotspots(lat, lon) {
    try {
        // Clear existing hotspot markers
        hotspotMarkers.forEach(marker => map.removeLayer(marker));
        hotspotMarkers = [];
        
        // Fetch hotspots with authentication
        const token = localStorage.getItem('airguard_token');
        const response = await fetch(`/api/proxy/air-quality/hotspots?lat=${lat}&lon=${lon}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const hotspots = await response.json();
        
        // Add hotspot markers to map
        hotspots.forEach(hotspot => {
            const color = getColorFromAQI(hotspot.aqi);
            const category = getAQICategory(hotspot.aqi);
            
            const circle = L.circle([hotspot.lat, hotspot.lon], {
                color: color,
                fillColor: color,
                fillOpacity: 0.5,
                radius: 500 // 500 meters radius
            }).addTo(map)
              .bindPopup(`
                <b>AQI: ${Math.round(hotspot.aqi)}</b><br>
                Category: ${category}<br>
                Location: ${hotspot.lat.toFixed(4)}, ${hotspot.lon.toFixed(4)}
              `);
            
            hotspotMarkers.push(circle);
        });
        
    } catch (error) {
        console.error('Error fetching hotspots:', error);
        document.getElementById('locationStatus').innerHTML = 
            '<i class="fas fa-exclamation-triangle text-danger"></i> Error loading air quality hotspots';
    }
}

async function fetchCurrentData() {
    const lat = document.getElementById('latitudeInput').value;
    const lon = document.getElementById('longitudeInput').value;
    const locationName = document.getElementById('locationInput').value || 'Unknown Location';
    
    if (!lat || !lon) {
        alert('Please enter both latitude and longitude');
        return;
    }
    
    try {
        // Show loading state
        document.getElementById('aqiValue').innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        document.getElementById('aqiStatus').textContent = 'Loading...';
        
        // Fetch air quality data with authentication
        const token = localStorage.getItem('airguard_token');
        const response = await fetch(`/api/proxy/air-quality/current?latitude=${lat}&longitude=${lon}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Update UI with air quality data
        updateAirQualityDisplay(data);
        
        // Fetch personalized health advisory
        fetchHealthAdvisory(data.aqi, data);
        
        // Update hotspots for this location
        fetchHotspots(parseFloat(lat), parseFloat(lon));
        
    } catch (error) {
        console.error('Error fetching air quality data:', error);
        document.getElementById('aqiValue').textContent = 'Error';
        document.getElementById('aqiStatus').textContent = 'Failed to load data';
    }
}

function updateAirQualityDisplay(data) {
    // Update AQI display
    document.getElementById('aqiValue').textContent = Math.round(data.aqi);
    
    // Set AQI status and color
    let statusText, statusClass, progressWidth, progressClass;
    
    if (data.aqi <= 50) {
        statusText = 'Good';
        statusClass = 'text-success';
        progressWidth = '20%';
        progressClass = 'bg-success';
    } else if (data.aqi <= 100) {
        statusText = 'Moderate';
        statusClass = 'text-warning';
        progressWidth = '40%';
        progressClass = 'bg-warning';
    } else if (data.aqi <= 150) {
        statusText = 'Unhealthy for Sensitive Groups';
        statusClass = 'text-orange';
        progressWidth = '60%';
        progressClass = 'bg-orange';
    } else if (data.aqi <= 200) {
        statusText = 'Unhealthy';
        statusClass = 'text-danger';
        progressWidth = '80%';
        progressClass = 'bg-danger';
    } else if (data.aqi <= 300) {
        statusText = 'Very Unhealthy';
        statusClass = 'text-purple';
        progressWidth = '90%';
        progressClass = 'bg-purple';
    } else {
        statusText = 'Hazardous';
        statusClass = 'text-danger fw-bold';
        progressWidth = '100%';
        progressClass = 'bg-danger';
    }
    
    document.getElementById('aqiStatus').textContent = statusText;
    document.getElementById('aqiStatus').className = `card-text ${statusClass}`;
    
    // Update progress bar
    const progressBar = document.getElementById('aqiProgress');
    progressBar.style.width = progressWidth;
    progressBar.className = `progress-bar ${progressClass}`;
    
    // Update confidence badge if available
    const confidenceBadge = document.getElementById('confidenceBadge');
    if (data.confidence_score) {
        const confidencePercent = Math.round(data.confidence_score * 100);
        confidenceBadge.innerHTML = `<span class="badge bg-info">Confidence: ${confidencePercent}%</span>`;
    }
    
    // Update pollutant chart (simplified for this example)
    updatePollutantChart(data);
}

function updatePollutantChart(data) {
    // This is a simplified implementation
    // In a real application, you would use Chart.js to create a proper chart
    console.log('Pollutant data:', data);
}

async function fetchHealthAdvisory(aqi, airQualityData) {
    try {
        const token = localStorage.getItem('airguard_token');
        const response = await fetch(`/api/proxy/advisory/health?aqi=${aqi}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (response.ok) {
            const advisory = await response.json();
            document.getElementById('advisoryContent').innerHTML = `
                <h6>Risk Level: ${advisory.risk_level}</h6>
                <p>${advisory.message}</p>
                <div class="mt-3">
                    <h6>Recommendations:</h6>
                    <ul>
                        ${advisory.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error fetching health advisory:', error);
    }
}

function startRealTimeUpdates() {
    // Implementation for real-time updates
    alert('Real-time updates feature would be implemented here');
}

// Add custom CSS for additional colors
const style = document.createElement('style');
style.textContent = `
    .text-orange { color: #ffa500 !important; }
    .bg-orange { background-color: #ffa500 !important; }
    .text-purple { color: #800080 !important; }
    .bg-purple { background-color: #800080 !important; }
    .legend {
        background: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
    }
    .legend h6 {
        margin: 0 0 10px;
        font-weight: bold;
    }
    .legend div {
        margin: 2px 0;
    }
`;
document.head.appendChild(style);