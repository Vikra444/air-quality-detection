// AirGuard Mobile App JavaScript

const API_BASE_URL = "{{ api_base_url }}" || "http://localhost:8001";
let mobilePredictionChart;
let mobileTrendChart;
let currentMobileData = null;
let darkMode = false;
let realTimeInterval = null;

document.addEventListener("DOMContentLoaded", function() {
    initializeMobileApp();
    fetchMobileData();
    
    // Set up event listeners
    document.getElementById('alertThreshold').addEventListener('input', function() {
        document.getElementById('thresholdValue').textContent = this.value;
    });
    
    // Check for dark mode preference
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        toggleDarkMode();
    }
});

function initializeMobileApp() {
    initializeMobileCharts();
    loadUserPreferences();
}

function initializeMobileCharts() {
    // Initialize prediction chart
    const predCtx = document.getElementById('mobilePredictionChart').getContext('2d');
    mobilePredictionChart = new Chart(predCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Predicted AQI',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
    
    // Initialize trend chart
    const trendCtx = document.getElementById('mobileTrendChart').getContext('2d');
    mobileTrendChart = new Chart(trendCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Historical AQI',
                data: [],
                backgroundColor: 'rgba(13, 110, 253, 0.7)',
                borderColor: 'rgba(13, 110, 253, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function showTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName + '-tab').classList.add('active');
    
    // Add active class to clicked button (using event delegation)
    event.target.closest('.tab-btn').classList.add('active');
}

function fetchMobileData() {
    const lat = parseFloat(document.getElementById('mobileLat').value);
    const lon = parseFloat(document.getElementById('mobileLon').value);
    const loc = document.getElementById('mobileLocation').value;

    // Show loading state
    document.getElementById('mobileAqi').classList.add('loading');
    
    fetch(`/api/proxy/current-air-quality?latitude=${lat}&longitude=${lon}&location_id=${loc}&include_forecast=true`)
        .then(response => response.json())
        .then(data => {
            if (data.error) throw new Error(data.error);
            
            currentMobileData = data;
            updateMobileDashboard(data);
            
            // Fetch additional data
            fetchMobilePredictions(lat, lon, loc);
            fetchMobileAdvisory(loc, lat, lon, data.aqi);
            fetchMobileTrends(loc);
            
            // Update last updated time
            document.getElementById('lastUpdated').textContent = 
                `Last updated: ${new Date().toLocaleTimeString()}`;
        })
        .catch(error => {
            console.error('Error fetching data:', error);
            alert("Error: " + error.message);
        })
        .finally(() => {
            document.getElementById('mobileAqi').classList.remove('loading');
        });
}

function updateMobileDashboard(data) {
    const aqi = Math.round(data.aqi || 0);
    document.querySelector('.aqi-value').textContent = aqi;

    let status, color;
    if (aqi <= 50) {
        status = "Good";
        color = "#28a745";
    } else if (aqi <= 100) {
        status = "Moderate";
        color = "#ffc107";
    } else if (aqi <= 150) {
        status = "Unhealthy for Sensitive";
        color = "#fd7e14";
    } else if (aqi <= 200) {
        status = "Unhealthy";
        color = "#dc3545";
    } else if (aqi <= 300) {
        status = "Very Unhealthy";
        color = "#6f42c1";
    } else {
        status = "Hazardous";
        color = "#343a40";
    }

    document.querySelector('.aqi-status').textContent = status;
    document.querySelector('.aqi-display').style.borderColor = color;
    
    // Update weather conditions
    document.getElementById('temperature').textContent = `${Math.round(data.temperature || 0)}°C`;
    document.getElementById('humidity').textContent = `${Math.round(data.humidity || 0)}%`;
    document.getElementById('pressure').textContent = `${Math.round(data.pressure || 0)} hPa`;
    
    // Update pollutants
    updatePollutantDisplay(data);
}

function updatePollutantDisplay(data) {
    const pollutantsContainer = document.getElementById('mobilePollutants');
    const pollutants = [
        { name: 'PM2.5', value: data.pm25, unit: 'μg/m³' },
        { name: 'PM10', value: data.pm10, unit: 'μg/m³' },
        { name: 'NO2', value: data.no2, unit: 'μg/m³' },
        { name: 'CO', value: data.co, unit: 'mg/m³' },
        { name: 'O3', value: data.o3, unit: 'μg/m³' },
        { name: 'SO2', value: data.so2, unit: 'μg/m³' }
    ];
    
    let html = '';
    pollutants.forEach(p => {
        if (p.value !== undefined && p.value !== null) {
            html += `
                <div class="pollutant-item">
                    <div>
                        <div class="pollutant-name">${p.name}</div>
                        <div class="pollutant-unit">${p.unit}</div>
                    </div>
                    <div class="pollutant-value">${p.value.toFixed(1)}</div>
                </div>
            `;
        }
    });
    
    pollutantsContainer.innerHTML = html;
}

function fetchMobilePredictions(lat, lon, loc) {
    fetch('/api/proxy/predictions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            location_id: loc,
            latitude: lat,
            longitude: lon,
            prediction_horizon: 24
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) throw new Error(data.error);
        
        // Update prediction chart
        const labels = [];
        const values = [];
        
        for (let i = 1; i <= 24; i++) {
            const key = `hour_${i}`;
            if (data.predictions && data.predictions[key]) {
                labels.push(`${i}h`);
                values.push(data.predictions[key].aqi);
            }
        }
        
        mobilePredictionChart.data.labels = labels;
        mobilePredictionChart.data.datasets[0].data = values;
        mobilePredictionChart.update();
        
        // Update forecast details
        updateForecastDetails(data);
    })
    .catch(error => {
        console.error('Error fetching predictions:', error);
    });
}

function updateForecastDetails(data) {
    const detailsContainer = document.getElementById('forecastDetails');
    let html = '<div class="forecast-summary">';
    
    if (data.confidence_score) {
        html += `<p>Confidence: ${(data.confidence_score * 100).toFixed(1)}%</p>`;
    }
    
    if (data.predictions) {
        const nextHour = data.predictions.hour_1;
        if (nextHour) {
            html += `<p>Next hour prediction: ${Math.round(nextHour.aqi)} AQI</p>`;
        }
    }
    
    html += '</div>';
    detailsContainer.innerHTML = html;
}

function fetchMobileAdvisory(loc, lat, lon, aqi) {
    const children = document.getElementById('checkChildren').checked;
    const elderly = document.getElementById('checkElderly').checked;
    const asthma = document.getElementById('checkAsthma').checked;
    
    let vulnerableGroups = [];
    if (children) vulnerableGroups.push('children');
    if (elderly) vulnerableGroups.push('elderly');
    if (asthma) vulnerableGroups.push('asthma_patients');
    
    const params = new URLSearchParams({
        location_id: loc,
        latitude: lat,
        longitude: lon,
        aqi: aqi,
        vulnerable_groups: vulnerableGroups.join(',')
    });
    
    fetch(`/api/proxy/advisory?${params}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) throw new Error(data.error);
            updateAdvisoryDisplay(data);
        })
        .catch(error => {
            console.error('Error fetching advisory:', error);
            document.getElementById('mobileAdvisory').innerHTML = 
                '<p class="text-danger">Failed to load health advisory</p>';
        });
}

function updateAdvisoryDisplay(data) {
    const advisoryContainer = document.getElementById('mobileAdvisory');
    let html = `
        <h6><i class="fas fa-heartbeat"></i> Health Advisory</h6>
        <div class="alert alert-${getAdvisoryAlertClass(data.risk_level)}">
            <strong>Risk Level:</strong> ${data.risk_level || 'Unknown'}
        </div>
        <div class="mt-3">
            <h6>Recommendations:</h6>
            <ul>
    `;
    
    if (data.recommendations && Array.isArray(data.recommendations)) {
        data.recommendations.forEach(rec => {
            html += `<li>${rec}</li>`;
        });
    }
    
    html += `
            </ul>
        </div>
    `;
    
    advisoryContainer.innerHTML = html;
}

function getAdvisoryAlertClass(riskLevel) {
    switch (riskLevel) {
        case 'GOOD': return 'success';
        case 'MODERATE': return 'info';
        case 'UNHEALTHY_FOR_SENSITIVE': return 'warning';
        case 'UNHEALTHY': return 'warning';
        case 'VERY_UNHEALTHY': return 'danger';
        case 'HAZARDOUS': return 'danger';
        default: return 'secondary';
    }
}

function fetchMobileTrends(loc) {
    fetch(`/api/proxy/trends?location_id=${loc}&days=7`)
        .then(response => response.json())
        .then(data => {
            if (data.error) throw new Error(data.error);
            updateTrendChart(data);
        })
        .catch(error => {
            console.error('Error fetching trends:', error);
        });
}

function updateTrendChart(data) {
    if (!data.statistics) return;
    
    // Generate last 7 days labels
    const labels = [];
    const values = [];
    
    for (let i = 6; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        labels.push(date.toLocaleDateString('en-US', { weekday: 'short' }));
        // For demo purposes, we'll generate some values
        values.push(Math.random() * 150 + 50);
    }
    
    mobileTrendChart.data.labels = labels;
    mobileTrendChart.data.datasets[0].data = values;
    mobileTrendChart.update();
}

function refreshData() {
    fetchMobileData();
}

function getCurrentLocation() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            position => {
                document.getElementById('mobileLat').value = position.coords.latitude.toFixed(4);
                document.getElementById('mobileLon').value = position.coords.longitude.toFixed(4);
                fetchMobileData();
            },
            error => {
                alert('Unable to get your location: ' + error.message);
            }
        );
    } else {
        alert('Geolocation is not supported by your browser');
    }
}

function startRealTimeMonitoring() {
    if (realTimeInterval) {
        clearInterval(realTimeInterval);
        realTimeInterval = null;
        event.target.innerHTML = '<i class="fas fa-play-circle"></i> Start Real-time Monitoring';
        return;
    }
    
    realTimeInterval = setInterval(() => {
        fetchMobileData();
    }, 30000); // Update every 30 seconds
    
    event.target.innerHTML = '<i class="fas fa-stop-circle"></i> Stop Real-time Monitoring';
}

function shareAirQuality() {
    if (!currentMobileData) {
        alert('No data to share');
        return;
    }
    
    const text = `Current Air Quality in ${document.getElementById('mobileLocation').value}:\n` +
                 `AQI: ${Math.round(currentMobileData.aqi)}\n` +
                 `PM2.5: ${currentMobileData.pm25?.toFixed(1)} μg/m³\n` +
                 `Temperature: ${currentMobileData.temperature?.toFixed(1)}°C\n` +
                 `Check more at AirGuard!`;
    
    if (navigator.share) {
        navigator.share({
            title: 'Air Quality Report',
            text: text
        }).catch(error => {
            console.log('Error sharing:', error);
            copyToClipboard(text);
        });
    } else {
        copyToClipboard(text);
    }
}

function copyToClipboard(text) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);
    alert('Air quality data copied to clipboard!');
}

function updateAdvisory() {
    if (currentMobileData) {
        fetchMobileAdvisory(
            document.getElementById('mobileLocation').value,
            parseFloat(document.getElementById('mobileLat').value),
            parseFloat(document.getElementById('mobileLon').value),
            currentMobileData.aqi
        );
    }
}

function toggleDarkMode() {
    darkMode = !darkMode;
    document.body.classList.toggle('dark-mode', darkMode);
    
    // Update icon
    const icon = document.querySelector('.mobile-header .fa-moon');
    if (icon) {
        icon.classList.toggle('fa-sun', darkMode);
        icon.classList.toggle('fa-moon', !darkMode);
    }
    
    // Save preference
    localStorage.setItem('darkMode', darkMode);
}

function loadUserPreferences() {
    // Load dark mode preference
    const savedDarkMode = localStorage.getItem('darkMode') === 'true';
    if (savedDarkMode) {
        darkMode = true;
        document.body.classList.add('dark-mode');
        const icon = document.querySelector('.mobile-header .fa-moon');
        if (icon) {
            icon.classList.remove('fa-moon');
            icon.classList.add('fa-sun');
        }
    }
    
    // Load user info
    const userName = localStorage.getItem('userName');
    const userEmail = localStorage.getItem('userEmail');
    if (userName) document.getElementById('userName').value = userName;
    if (userEmail) document.getElementById('userEmail').value = userEmail;
}

function openQuickActions() {
    alert('Quick actions menu would open here');
}

// Save user preferences when they change
document.getElementById('userName').addEventListener('change', function() {
    localStorage.setItem('userName', this.value);
});

document.getElementById('userEmail').addEventListener('change', function() {
    localStorage.setItem('userEmail', this.value);
});