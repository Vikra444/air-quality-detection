"""
Streamlit dashboard for AirGuard.
Advanced dashboard with real-time updates and enhanced visualizations.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import time

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_TOKEN = os.getenv("API_TOKEN", "demo-token")

st.set_page_config(
    page_title="AirGuard Dashboard",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .aqi-badge {
        display: inline-block;
        padding: 0.25em 0.4em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🌬️ AirGuard Dashboard v2.0</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📍 Location Settings")
    
    location_id = st.text_input("Location Name", value="Delhi")
    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input("Latitude", value=28.6139, step=0.0001, format="%.4f")
    with col2:
        longitude = st.number_input("Longitude", value=77.2090, step=0.0001, format="%.4f")
    
    st.header("📅 Time Settings")
    days = st.selectbox("Historical Data", [1, 7, 30, 90], index=1)
    
    st.header("👥 Vulnerable Groups")
    children = st.checkbox("Children", value=True)
    elderly = st.checkbox("Elderly", value=True)
    asthma = st.checkbox("Asthma Patients", value=False)
    
    st.header("🔄 Auto Refresh")
    auto_refresh = st.checkbox("Enable Auto Refresh", value=False)
    refresh_interval = st.slider("Refresh Interval (seconds)", 30, 300, 60)
    
    if st.button("🔄 Refresh Data", type="primary"):
        st.rerun()

# Auto-refresh functionality
# Note: Auto-refresh implementation depends on Streamlit version

# Main Content
@st.cache_data(ttl=60)
def fetch_air_quality(latitude, longitude, location_id, include_forecast=False):
    """Fetch current air quality data."""
    try:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "location_id": location_id
        }
        if include_forecast:
            params["include_forecast"] = "true"
            
        response = requests.get(
            f"{API_BASE_URL}/api/v1/air-quality/current",
            params=params,
            headers={"Authorization": f"Bearer {API_TOKEN}"},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

@st.cache_data(ttl=300)
def fetch_predictions(latitude, longitude, location_id):
    """Fetch predictions."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/predictions/generate",
            json={
                "location_id": location_id,
                "latitude": latitude,
                "longitude": longitude,
                "prediction_horizon": 24
            },
            headers={"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching predictions: {e}")
        return None

@st.cache_data(ttl=60)
def fetch_advisory(location_id, latitude, longitude, aqi, vulnerable_groups):
    """Fetch health advisory."""
    try:
        params = {"location_id": location_id}
        if aqi is not None:
            params["aqi"] = aqi
        if latitude is not None and longitude is not None:
            params["latitude"] = latitude
            params["longitude"] = longitude
        if vulnerable_groups:
            params["vulnerable_groups"] = ",".join(vulnerable_groups)
        
        response = requests.get(
            f"{API_BASE_URL}/api/v1/advisory/health",
            params=params,
            headers={"Authorization": f"Bearer {API_TOKEN}"},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching advisory: {e}")
        return None

@st.cache_data(ttl=300)
def fetch_historical_data(location_id, days=7):
    """Fetch historical data."""
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        params = {
            "location_id": location_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "limit": 1000
        }
        
        response = requests.get(
            f"{API_BASE_URL}/api/v1/air-quality/historical",
            params=params,
            headers={"Authorization": f"Bearer {API_TOKEN}"},
            timeout=15
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return None

@st.cache_data(ttl=300)
def fetch_trend_analysis(location_id, days=30):
    """Fetch trend analysis."""
    try:
        params = {"location_id": location_id, "days": days}
        
        response = requests.get(
            f"{API_BASE_URL}/api/v1/analysis/trends",
            params=params,
            headers={"Authorization": f"Bearer {API_TOKEN}"},
            timeout=15
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching trend analysis: {e}")
        return None

@st.cache_data(ttl=60)
def fetch_system_status():
    """Fetch system status."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/system/health",
            headers={"Authorization": f"Bearer {API_TOKEN}"},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching system status: {e}")
        return None

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dashboard", "Predictions", "Health Advisory", "Trends", "System Status"])

# Tab 1: Dashboard
with tab1:
    # Fetch data
    data = fetch_air_quality(latitude, longitude, location_id, include_forecast=True)
    
    if data:
        aqi = data.get("aqi", 0)
        
        # Determine AQI color and status
        if aqi <= 50:
            aqi_color = "🟢"
            aqi_status = "Good"
            status_color = "#00E400"
        elif aqi <= 100:
            aqi_color = "🟡"
            aqi_status = "Moderate"
            status_color = "#FFFF00"
        elif aqi <= 150:
            aqi_color = "🟠"
            aqi_status = "Unhealthy for Sensitive"
            status_color = "#FF7E00"
        elif aqi <= 200:
            aqi_color = "🔴"
            aqi_status = "Unhealthy"
            status_color = "#FF0000"
        elif aqi <= 300:
            aqi_color = "🟣"
            aqi_status = "Very Unhealthy"
            status_color = "#8F3F97"
        else:
            aqi_color = "⚫"
            aqi_status = "Hazardous"
            status_color = "#7E0023"
        
        # Main Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Air Quality Index", f"{aqi_color} {int(aqi)}", aqi_status)
        
        with col2:
            st.metric("PM2.5", f"{data.get('pm25', 0):.1f} μg/m³")
        
        with col3:
            st.metric("PM10", f"{data.get('pm10', 0):.1f} μg/m³")
        
        with col4:
            st.metric("Temperature", f"{data.get('temperature', 0):.1f}°C")
        
        # Confidence indicator
        if "confidence_score" in data:
            st.markdown(f"**Confidence Score:** {data['confidence_score']:.2f}")
        
        # Pollutant Chart
        st.subheader("📊 Pollutant Levels")
        pollutants_df = pd.DataFrame({
            "Pollutant": ["PM2.5", "PM10", "NO2", "CO", "O3", "SO2"],
            "Value": [
                data.get("pm25", 0),
                data.get("pm10", 0),
                data.get("no2", 0),
                data.get("co", 0),
                data.get("o3", 0),
                data.get("so2", 0)
            ]
        })
        
        fig = px.bar(pollutants_df, x="Pollutant", y="Value", 
                     title="Current Pollutant Concentrations",
                     color="Value",
                     color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional Info
        with st.expander("ℹ️ Additional Information"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Humidity:** {data.get('humidity', 0):.1f}%")
                st.write(f"**Wind Speed:** {data.get('wind_speed', 0):.1f} m/s")
                st.write(f"**Pressure:** {data.get('pressure', 0):.1f} hPa")
            with col2:
                st.write(f"**Source API:** {data.get('source_api', 'Unknown')}")
                if "confidence_score" in data:
                    st.write(f"**Confidence Score:** {data.get('confidence_score', 0):.2f}")
                if "quality_score" in data:
                    st.write(f"**Quality Score:** {data.get('quality_score', 0):.2f}")
        
        # Last update time
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.error("⚠️ Failed to fetch air quality data. Please check your API connection and location coordinates.")

# Tab 2: Predictions
with tab2:
    st.subheader("🔮 24-Hour Prediction")
    predictions = fetch_predictions(latitude, longitude, location_id)
    
    if predictions and predictions.get("predictions"):
        pred_data = []
        for i in range(1, 25):
            key = f"hour_{i}"
            if key in predictions["predictions"]:
                pred_data.append({
                    "Hour": i,
                    "AQI": predictions["predictions"][key]["aqi"],
                    "Time": (datetime.now() + timedelta(hours=i)).strftime("%H:00")
                })
        
        if pred_data:
            pred_df = pd.DataFrame(pred_data)
            fig_pred = px.line(pred_df, x="Time", y="AQI", 
                              title="Predicted Air Quality Index (24 hours)",
                              markers=True)
            fig_pred.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Display prediction details
            st.subheader("📈 Prediction Details")
            st.dataframe(pred_df.set_index("Time"))
        else:
            st.warning("No prediction data available")
    else:
        st.warning("Unable to fetch predictions. Please check your API connection.")

# Tab 3: Health Advisory
with tab3:
    st.subheader("❤️ Health Advisory")
    vulnerable_groups = []
    if children:
        vulnerable_groups.append("children")
    if elderly:
        vulnerable_groups.append("elderly")
    if asthma:
        vulnerable_groups.append("asthma_patients")
    
    data = fetch_air_quality(latitude, longitude, location_id)
    if data:
        aqi = data.get("aqi", 0)
        advisory = fetch_advisory(location_id, latitude, longitude, aqi, vulnerable_groups)
        
        if advisory:
            st.info(f"**Risk Level:** {advisory.get('risk_level', 'Unknown')}")
            
            st.write("**Primary Concerns:**")
            for concern in advisory.get("primary_concerns", []):
                st.write(f"- {concern}")
            
            st.write("**Recommendations:**")
            for rec in advisory.get("recommendations", []):
                st.write(f"- {rec}")
            
            if advisory.get("vulnerable_group_advisories"):
                st.write("**Vulnerable Group Advisories:**")
                for group, advisories in advisory["vulnerable_group_advisories"].items():
                    with st.expander(f"Advisory for {group.title()}"):
                        for adv in advisories:
                            st.write(f"- {adv}")
        else:
            st.warning("Unable to fetch health advisory")
    else:
        st.warning("Please fetch air quality data first")

# Tab 4: Trends
with tab4:
    st.subheader("📈 Historical Trends")
    
    # Fetch historical data
    historical = fetch_historical_data(location_id, days)
    
    if historical and historical.get("data"):
        # Convert to DataFrame
        hist_data = historical["data"]
        df = pd.DataFrame(hist_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601')
        df = df.sort_values("timestamp")
        
        # Plot historical AQI
        fig_hist = px.line(df, x="timestamp", y="aqi", 
                          title=f"Historical AQI (Last {days} days)")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Statistics
        st.subheader("📊 Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average AQI", f"{df['aqi'].mean():.1f}")
        with col2:
            st.metric("Max AQI", f"{df['aqi'].max():.1f}")
        with col3:
            st.metric("Min AQI", f"{df['aqi'].min():.1f}")
        with col4:
            st.metric("Std Dev", f"{df['aqi'].std():.1f}")
        
        # Show raw data
        with st.expander("Raw Data"):
            st.dataframe(df[["timestamp", "aqi", "pm25", "pm10"]].tail(20))
    else:
        st.warning("No historical data available")
    
    # Fetch trend analysis
    st.subheader("🔍 Trend Analysis")
    trend_data = fetch_trend_analysis(location_id, days)
    
    if trend_data and trend_data.get("statistics"):
        stats = trend_data["statistics"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Trend Direction", stats["trend_direction"].title())
        with col2:
            st.metric("Trend Magnitude", f"{stats['trend_magnitude']:.2f}")
        with col3:
            st.metric("Mean AQI", f"{stats['mean_aqi']:.1f}")

# Tab 5: System Status
with tab5:
    st.subheader("🖥️ System Status")
    status = fetch_system_status()
    
    if status:
        col1, col2, col3 = st.columns(3)
        with col1:
            if status["components"]["database"]:
                st.success("Database: Connected")
            else:
                st.error("Database: Disconnected")
        with col2:
            if status["components"]["cache"]:
                st.success("Cache: Active")
            else:
                st.warning("Cache: Inactive")
        with col3:
            st.info(f"Version: {status['version']}")
        
        # Cache stats if available
        if status.get("cache_stats"):
            st.subheader("キャッシング統計")
            cache_stats = status["cache_stats"]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Cache Keys", cache_stats.get("keys", "N/A"))
            with col2:
                st.metric("Cache Hits", cache_stats.get("hits", "N/A"))
    else:
        st.error("Unable to fetch system status")

# Footer
st.markdown("---")
st.markdown("**AirGuard** - Advanced Real-Time Air Quality & Health Risk Prediction System")