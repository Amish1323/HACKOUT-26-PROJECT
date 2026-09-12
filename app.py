# ==========================================
# CONFIGURATION & PROJECT NAME
# Change this variable to your team's project name!
PROJECT_NAME = "AeroSolar"  # e.g., "SolarWind Sentinel", "EcoPulse", etc.
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import urllib.request
import urllib.error

# ==========================================
# PAGE CONFIGURATION (Python 3.12 / Streamlit Modern)
# ==========================================
st.set_page_config(
    page_title=f"{PROJECT_NAME} | AI Predictive Maintenance",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS STYLING
# ==========================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .status-ok {
        color: #16A34A;
        font-weight: bold;
    }
    .status-warning {
        color: #D97706;
        font-weight: bold;
    }
    .status-critical {
        color: #DC2626;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# TELEMETRY GENERATOR (Python 3.12 Optimized)
# ==========================================
def generate_asset_data(soiling_val: float, vibration_val: float, temp_val: float):
    np.random.seed(42)
    
    # Solar Assets Data
    solar_assets = [
        {
            "Asset_ID": "SOLAR-P101", 
            "Type": "Solar Panel Array A", 
            "Capacity_kW": 250, 
            "Soiling_%": float(soiling_val), 
            "Temp_C": round(42.0 + (soiling_val * 0.25) + float(np.random.normal(0, 1)), 1), 
            "Current_A": round(max(5.0, 45.0 - (soiling_val * 0.35)), 1), 
            "Inverter_Eff_%": round(max(60.0, 98.0 - (soiling_val * 0.3)), 1)
        },
        {"Asset_ID": "SOLAR-P102", "Type": "Solar Panel Array B", "Capacity_kW": 250, "Soiling_%": 12.0, "Temp_C": 41.2, "Current_A": 43.8, "Inverter_Eff_%": 97.5},
        {"Asset_ID": "SOLAR-P103", "Type": "Solar Panel Array C", "Capacity_kW": 500, "Soiling_%": 18.5, "Temp_C": 44.0, "Current_A": 88.2, "Inverter_Eff_%": 96.8},
        {"Asset_ID": "SOLAR-P104", "Type": "Solar Panel Array D", "Capacity_kW": 500, "Soiling_%": 8.0, "Temp_C": 39.5, "Current_A": 92.1, "Inverter_Eff_%": 98.2},
    ]
    
    # Wind Assets Data
    wind_assets = [
        {
            "Asset_ID": "WIND-T201", 
            "Type": "2.5MW Wind Turbine #1", 
            "Capacity_kW": 2500, 
            "Vibration_Hz": float(vibration_val), 
            "Gearbox_Temp_C": float(temp_val), 
            "RPM": round(max(0.0, 18.0 - (vibration_val * 0.08)), 1), 
            "Oil_Pressure_bar": round(max(1.5, 4.2 - (temp_val * 0.02)), 2)
        },
        {"Asset_ID": "WIND-T202", "Type": "2.5MW Wind Turbine #2", "Capacity_kW": 2500, "Vibration_Hz": 28.5, "Gearbox_Temp_C": 58.0, "RPM": 17.5, "Oil_Pressure_bar": 4.1},
        {"Asset_ID": "WIND-T203", "Type": "2.5MW Wind Turbine #3", "Capacity_kW": 2500, "Vibration_Hz": 32.0, "Gearbox_Temp_C": 61.2, "RPM": 17.2, "Oil_Pressure_bar": 3.9},
    ]
    
    df_solar = pd.DataFrame(solar_assets)
    df_wind = pd.DataFrame(wind_assets)
    
    # Determine Health Status
    def check_solar_health(row):
        if row['Soiling_%'] > 40 or row['Temp_C'] > 60:
            return "CRITICAL"
        elif row['Soiling_%'] > 25 or row['Temp_C'] > 50:
            return "WARNING"
        return "NORMAL"

    def check_wind_health(row):
        if row['Vibration_Hz'] > 75 or row['Gearbox_Temp_C'] > 80:
            return "CRITICAL"
        elif row['Vibration_Hz'] > 50 or row['Gearbox_Temp_C'] > 70:
            return "WARNING"
        return "NORMAL"

    df_solar['Status'] = df_solar.apply(check_solar_health, axis=1)
    df_wind['Status'] = df_wind.apply(check_wind_health, axis=1)
    
    return df_solar, df_wind

# ==========================================
# LIVE GEMINI API CALLER (No external dependency needed)
# ==========================================
def call_gemini_api(api_key: str, prompt_text: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{
            "parts": [{"text": prompt_text}]
        }]
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            res_body = response.read().decode('utf-8')
            res_json = json.loads(res_body)
            return res_json['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"API Call Error: {str(e)}"

# ==========================================
# SIDEBAR CONTROLS & FAULT INJECTION
# ==========================================
st.sidebar.image("https://img.icons8.com/color/96/lightning-bolt.png", width=60)
st.sidebar.title(PROJECT_NAME)
st.sidebar.markdown("**AI Predictive Maintenance Platform**")
st.sidebar.caption("Compatible with Python 3.12 / 3.13+")
st.sidebar.markdown("---")

st.sidebar.subheader("📍 Site Selection")
site = st.sidebar.selectbox("Select Renewable Energy Park", ["Rajasthan Solar & Wind Complex (Site Alpha)", "Gujarat Hybrid Energy Park (Site Beta)"])

st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Fault Simulation (Demo Controls)")
st.sidebar.info("Adjust sliders to simulate live IoT sensor anomalies:")

sim_soiling = st.sidebar.slider("Solar Panel Soiling (%)", min_value=5.0, max_value=80.0, value=52.0, step=1.0)
sim_vibration = st.sidebar.slider("Turbine Vibration (Hz)", min_value=15.0, max_value=120.0, value=82.0, step=1.0)
sim_temp = st.sidebar.slider("Gearbox Temp (°C)", min_value=40.0, max_value=105.0, value=85.0, step=1.0)

st.sidebar.markdown("---")
use_ai_api = st.sidebar.checkbox("Use Live Gemini API for Diagnostics", value=False)
api_key = ""
if use_ai_api:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

# ==========================================
# MAIN DASHBOARD CONTENT
# ==========================================
st.markdown(f'<div class="main-header">⚡ {PROJECT_NAME}: Renewable Fleet Monitor</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Real-time IoT Telemetry & AI Diagnostic Engine | Currently Monitoring: <b>{site}</b></div>', unsafe_allow_html=True)

df_solar, df_wind = generate_asset_data(sim_soiling, sim_vibration, sim_temp)

# Count Statuses
total_assets = len(df_solar) + len(df_wind)
critical_count = int((df_solar['Status'] == 'CRITICAL').sum() + (df_wind['Status'] == 'CRITICAL').sum())
warning_count = int((df_solar['Status'] == 'WARNING').sum() + (df_wind['Status'] == 'WARNING').sum())
normal_count = total_assets - critical_count - warning_count

# Calculate Est Daily Revenue Loss
revenue_loss_solar = float(df_solar[df_solar['Status'] != 'NORMAL']['Capacity_kW'].sum() * 5 * 4.5)  # 5 hrs @ Rs 4.5/kWh
revenue_loss_wind = float(df_wind[df_wind['Status'] != 'NORMAL']['Capacity_kW'].sum() * 12 * 4.5)
total_rev_loss = int(revenue_loss_solar + revenue_loss_wind)

# KPI ROW
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric(label="Total Assets Monitored", value=total_assets)
with col2:
    st.metric(label="Optimal Operation", value=f"{normal_count}", delta=f"{int(normal_count/total_assets*100)}%")
with col3:
    st.metric(label="Warnings Flagged", value=f"{warning_count}", delta_color="off")
with col4:
    st.metric(label="Critical Alerts", value=f"{critical_count}", delta=f"-{critical_count}", delta_color="inverse")
with col5:
    st.metric(label="Est. Daily Revenue Loss", value=f"₹{total_rev_loss:,}", delta_color="inverse")

st.markdown("---")

# TABS NAVIGATION
tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Telemetry & Fleet Status", "🤖 AI Diagnostic Assistant", "📉 Financial & ROI Impact", "📜 Dispatch & Maintenance Log"])

# Helper function for pandas styling compatible with Pandas 2.1+ / 2.2+ (Python 3.12+)
def apply_status_style(styler):
    def highlight_status(val):
        if val == 'CRITICAL':
            return 'background-color: #FEE2E2; color: #991B1B; font-weight: bold;'
        elif val == 'WARNING':
            return 'background-color: #FEF3C7; color: #92400E; font-weight: bold;'
        return 'background-color: #DCFCE7; color: #166534; font-weight: bold;'
    
    # Use Styler.map for Pandas 2.1+, fallback to Styler.applymap for older Pandas
    if hasattr(styler, 'map'):
        return styler.map(highlight_status, subset=['Status'])
    else:
        return styler.applymap(highlight_status, subset=['Status'])

# ------------------------------------------
# TAB 1: FLEET MONITOR
# ------------------------------------------
with tab1:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("☀️ Solar Assets Telemetry")
        st.dataframe(apply_status_style(df_solar.style), use_container_width=True)
        
        fig_solar = px.bar(
            df_solar, x="Asset_ID", y="Current_A", color="Status",
            color_discrete_map={"NORMAL": "#22C55E", "WARNING": "#F59E0B", "CRITICAL": "#EF4444"},
            title="Solar Output Current (Amperes)", height=280
        )
        st.plotly_chart(fig_solar, use_container_width=True)

    with col_b:
        st.subheader("🌪️ Wind Turbine Telemetry")
        st.dataframe(apply_status_style(df_wind.style), use_container_width=True)
        
        fig_wind = px.scatter(
            df_wind, x="Vibration_Hz", y="Gearbox_Temp_C", color="Status", size="Capacity_kW",
            color_discrete_map={"NORMAL": "#22C55E", "WARNING": "#F59E0B", "CRITICAL": "#EF4444"},
            title="Wind Component Stress (Vibration vs Gearbox Temp)", height=280
        )
        st.plotly_chart(fig_wind, use_container_width=True)

# ------------------------------------------
# TAB 2: AI DIAGNOSTIC ASSISTANT
# ------------------------------------------
with tab2:
    st.subheader("🤖 Generative AI Maintenance Diagnostic Engine")
    st.write("Select an asset flagged with an anomaly to generate an automated root cause analysis and field technician dispatch plan.")
    
    all_critical_assets = df_solar[df_solar['Status'] != 'NORMAL']['Asset_ID'].tolist() + df_wind[df_wind['Status'] != 'NORMAL']['Asset_ID'].tolist()
    
    if not all_critical_assets:
        st.success("✅ All assets are currently operating within nominal parameters. No diagnostics needed.")
    else:
        selected_asset = st.selectbox("Select Anomalous Asset for AI Analysis:", all_critical_assets)
        
        if st.button("🔍 Generate AI Diagnostic Report"):
            with st.spinner("Analyzing telemetry patterns, cross-referencing maintenance manuals, and computing financial risk..."):
                
                # Check if asset is solar or wind
                if "SOLAR" in selected_asset:
                    row = df_solar[df_solar['Asset_ID'] == selected_asset].iloc[0]
                    asset_type = "Solar Panel Array"
                    telemetry_str = f"Soiling Rate: {row['Soiling_%']}%, Surface Temp: {row['Temp_C']}°C, Output Current: {row['Current_A']} A"
                    
                    if use_ai_api and api_key:
                        prompt = f"Act as an expert solar engineer. Anomaly on {selected_asset} ({asset_type}): {telemetry_str}. Provide: 1) Root Cause, 2) 3 Step-by-step repair actions, 3) Estimated daily revenue loss."
                        ai_response_text = call_gemini_api(api_key, prompt)
                        st.markdown("### 🌟 Live Gemini API Diagnostic Result")
                        st.write(ai_response_text)
                    else:
                        ai_root_cause = f"Heavy dust & particulate accumulation on PV glass (Soiling Level: {row['Soiling_%']}%). Soiling exceeds the 25% threshold, leading to hotspot creation and a surface temperature elevation to {row['Temp_C']}°C."
                        ai_actions = [
                            "Dispatch Field Team Bravo for dry microfiber robotic cleaning.",
                            "Inspect inverter string junction box for localized thermal stress.",
                            "Verify bypass diode integrity on Panel Array Section B."
                        ]
                        daily_loss = int(row['Capacity_kW'] * 5 * 4.5 * (row['Soiling_%'] / 100))
                else:
                    row = df_wind[df_wind['Asset_ID'] == selected_asset].iloc[0]
                    asset_type = "2.5MW Wind Turbine"
                    telemetry_str = f"Vibration: {row['Vibration_Hz']} Hz, Gearbox Temp: {row['Gearbox_Temp_C']}°C, Oil Pressure: {row['Oil_Pressure_bar']} bar"
                    
                    if use_ai_api and api_key:
                        prompt = f"Act as an expert wind energy engineer. Anomaly on {selected_asset} ({asset_type}): {telemetry_str}. Provide: 1) Root Cause, 2) 3 Step-by-step repair actions, 3) Estimated daily revenue loss."
                        ai_response_text = call_gemini_api(api_key, prompt)
                        st.markdown("### 🌟 Live Gemini API Diagnostic Result")
                        st.write(ai_response_text)
                    else:
                        ai_root_cause = f"Severe mechanical resonance detected in main high-speed shaft bearing (Vibration: {row['Vibration_Hz']} Hz, Gearbox Temp: {row['Gearbox_Temp_C']}°C). Indication of lubrication breakdown and early bearing pitting."
                        ai_actions = [
                            "Immediately curtail turbine speed to idling mode (8 RPM) to prevent catastrophic mechanical fracture.",
                            "Schedule urgent high-grade gear oil flush and particle contamination inspection.",
                            "Order replacement SKF Spherical Roller Bearing (Part #SKF-32948) for primary drive assembly."
                        ]
                        daily_loss = int(row['Capacity_kW'] * 12 * 4.5 * 0.7)

                st.markdown("---")
                
                if not (use_ai_api and api_key):
                    # Default Structured AI Layout
                    res_col1, res_col2 = st.columns([2, 1])
                    
                    with res_col1:
                        st.markdown(f"### 📋 Diagnostic Summary for `{selected_asset}` ({asset_type})")
                        st.info(f"**Live Telemetry Ingested:** {telemetry_str}")
                        
                        st.markdown("#### **1. Root Cause Identification**")
                        st.write(ai_root_cause)
                        
                        st.markdown("#### **2. Step-by-Step Field Technician Action Plan**")
                        for i, act in enumerate(ai_actions, 1):
                            st.markdown(f"**Step {i}:** {act}")
                    
                    with res_col2:
                        st.markdown("### ⚠️ Risk & Priority")
                        st.error("Priority: **HIGH (DISPATCH REQUIRED)**")
                        st.metric(label="Est. Daily Financial Risk", value=f"₹{daily_loss:,}")
                        st.metric(label="Energy Loss per Day", value=f"{int(daily_loss / 4.5):,} kWh")
                        
                        st.markdown("---")
                        if st.button("📩 Auto-Generate Technician Work Order"):
                            st.success("Work Order #WO-8942 dispatched to Lead Technician via WhatsApp/SMS!")

# ------------------------------------------
# TAB 3: FINANCIAL IMPACT
# ------------------------------------------
with tab3:
    st.subheader("📉 Predictive vs. Reactive Maintenance ROI Calculator")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("#### Key Metrics Comparison")
        metrics_df = pd.DataFrame({
            "Metric": ["Unplanned Downtime (Hours/Year)", "Annual Maintenance Cost (₹)", "Equipment Lifespan (Years)", "Catastrophic Failure Rate (%)"],
            "Reactive Approach": ["450 Hours", "₹18,50,000", "15 Years", "8.5%"],
            f"{PROJECT_NAME} Predictive AI": ["85 Hours (-81%)", "₹6,20,000 (-66%)", "22 Years (+46%)", "0.8% (-90%)"]
        })
        st.table(metrics_df)
        
    with col_f2:
        st.markdown("#### 5-Year Cumulative Savings Projection")
        years = [f"Year {i}" for i in range(1, 6)]
        savings = [12.5, 26.0, 41.2, 58.0, 76.5]  # in Lakhs
        
        fig_roi = px.line(x=years, y=savings, labels={'x': 'Timeline', 'y': 'Cumulative Savings (Lakhs ₹)'}, markers=True)
        fig_roi.update_traces(line_color='#16A34A', line_width=3)
        st.plotly_chart(fig_roi, use_container_width=True)

# ------------------------------------------
# TAB 4: DISPATCH & MAINTENANCE LOG
# ------------------------------------------
with tab4:
    st.subheader("📜 Recent Work Orders & Maintenance History")
    
    history_data = [
        {"Ticket_ID": "WO-8941", "Asset_ID": "WIND-T201", "Severity": "CRITICAL", "Assigned_Tech": "Ramesh Kumar", "Status": "IN-PROGRESS", "Timestamp": "Today, 08:30 AM"},
        {"Ticket_ID": "WO-8939", "Asset_ID": "SOLAR-P101", "Severity": "WARNING", "Assigned_Tech": "Priya Sharma", "Status": "SCHEDULED", "Timestamp": "Today, 07:15 AM"},
        {"Ticket_ID": "WO-8932", "Asset_ID": "SOLAR-P104", "Severity": "NORMAL", "Assigned_Tech": "Vikram Singh", "Status": "COMPLETED", "Timestamp": "Yesterday, 04:45 PM"},
        {"Ticket_ID": "WO-8928", "Asset_ID": "WIND-T203", "Severity": "WARNING", "Assigned_Tech": "Ramesh Kumar", "Status": "COMPLETED", "Timestamp": "10 Sep 2026"},
    ]
    st.dataframe(pd.DataFrame(history_data), use_container_width=True)

st.markdown("---")
st.caption(f"⚡ {PROJECT_NAME} | Built with Python for HackOut'26 Ideation & Prototype Demonstration")
