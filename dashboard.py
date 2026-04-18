"""
Streamlit Dashboard for Multi-Camera Safety System
Real-time visualization of alerts, maps, and camera feeds
"""
import streamlit as st
import cv2
import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import folium
from streamlit_folium import st_folium
import glob

st.set_page_config(page_title="Safety System Dashboard", layout="wide")


def load_alerts_log() -> list:
    """Load alerts from log file"""
    log_file = "alerts/alerts.log"
    alerts = []
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            content = f.read()
            # Parse alerts (simple parsing)
            alert_blocks = content.split('-' * 60)
            for block in alert_blocks:
                if 'ALERT' in block:
                    alerts.append(block.strip())
    
    return alerts


def load_alerts_json() -> list:
    """Load structured alert data from JSON"""
    json_file = "alerts_data.json"
    
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            return json.load(f)
    
    return []


def get_latest_snapshots(limit: int = 5) -> list:
    """Get latest snapshot files"""
    snapshot_dir = "alerts/snapshots"
    if not os.path.exists(snapshot_dir):
        return []
    
    snapshots = glob.glob(os.path.join(snapshot_dir, "*.jpg"))
    snapshots.sort(key=os.path.getmtime, reverse=True)
    
    return snapshots[:limit]


def get_statistics() -> dict:
    """Calculate statistics"""
    alerts = load_alerts_json()
    
    if not alerts:
        return {
            'total_alerts': 0,
            'high_severity': 0,
            'medium_severity': 0,
            'low_severity': 0,
            'avg_males': 0,
            'avg_females': 0
        }
    
    stats = {
        'total_alerts': len(alerts),
        'high_severity': sum(1 for a in alerts if a.get('severity') == 'high'),
        'medium_severity': sum(1 for a in alerts if a.get('severity') == 'medium'),
        'low_severity': sum(1 for a in alerts if a.get('severity') == 'low'),
        'avg_males': sum(a.get('details', {}).get('male_count', 0) for a in alerts) / len(alerts),
        'avg_females': sum(a.get('details', {}).get('female_count', 0) for a in alerts) / len(alerts)
    }
    
    return stats


def display_heatmap():
    """Display heatmap"""
    heatmap_file = "heatmap.html"
    map_file = "detailed_map.html"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Incident Heatmap")
        if os.path.exists(heatmap_file):
            with open(heatmap_file, 'r') as f:
                map_html = f.read()
                st.components.v1.html(map_html, height=500)
        else:
            st.info("Heatmap not generated yet")
    
    with col2:
        st.subheader("📍 Detailed Alert Map")
        if os.path.exists(map_file):
            with open(map_file, 'r') as f:
                map_html = f.read()
                st.components.v1.html(map_html, height=500)
        else:
            st.info("Detailed map not generated yet")


def display_snapshots():
    """Display latest snapshots"""
    st.subheader("📸 Latest Snapshots")
    
    snapshots = get_latest_snapshots()
    
    if snapshots:
        cols = st.columns(3)
        for idx, snapshot_path in enumerate(snapshots):
            with cols[idx % 3]:
                image = cv2.imread(snapshot_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image, caption=Path(snapshot_path).name, use_column_width=True)
    else:
        st.info("No snapshots available yet")


def display_alert_statistics():
    """Display alert statistics"""
    st.subheader("📊 Alert Statistics")
    
    stats = get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Alerts", stats['total_alerts'])
    
    with col2:
        st.metric("High Severity", stats['high_severity'], delta=f"🔴")
    
    with col3:
        st.metric("Medium Severity", stats['medium_severity'], delta=f"🟠")
    
    with col4:
        st.metric("Low Severity", stats['low_severity'], delta=f"🟡")
    
    # Additional stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Avg Males per Alert", f"{stats['avg_males']:.1f}")
    
    with col2:
        st.metric("Avg Females per Alert", f"{stats['avg_females']:.1f}")


def display_alerts_table():
    """Display alerts in table format"""
    st.subheader("📋 Alert Log")
    
    alerts = load_alerts_json()
    
    if alerts:
        # Create dataframe
        rows = []
        for alert in alerts:
            details = alert.get('details', {})
            rows.append({
                'Camera': details.get('camera_id', 'N/A'),
                'Timestamp': alert.get('timestamp', 'N/A')[:19],
                'Females': details.get('female_count', 0),
                'Males': details.get('male_count', 0),
                'Severity': alert.get('severity', 'N/A').upper(),
                'Latitude': f"{alert.get('latitude', 0):.4f}",
                'Longitude': f"{alert.get('longitude', 0):.4f}"
            })
        
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        
        # Export option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="alerts_report.csv",
            mime="text/csv"
        )
    else:
        st.info("No alerts recorded yet")


def main():
    """Main dashboard"""
    st.title("🛡️ Multi-Camera Safety Detection System Dashboard")
    
    st.markdown("""
    Real-time monitoring and incident tracking for women safety detection system.
    """)
    
    # Sidebar
    st.sidebar.title("🔧 Controls")
    
    page = st.sidebar.radio("Select View", [
        "📊 Dashboard",
        "🔥 Maps & Heatmaps",
        "📸 Snapshots",
        "📋 Alert Logs"
    ])
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Main content
    if page == "📊 Dashboard":
        st.divider()
        display_alert_statistics()
        
        st.divider()
        st.subheader("📈 Recent Activity")
        
        alerts = load_alerts_json()
        if alerts:
            recent_alert = alerts[-1]
            details = recent_alert.get('details', {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"Latest Alert\n{recent_alert.get('timestamp', 'N/A')[:19]}")
            
            with col2:
                st.warning(f"From: {details.get('camera_id', 'N/A')}")
            
            with col3:
                severity = recent_alert.get('severity', 'medium').upper()
                if severity == 'HIGH':
                    st.error(f"Severity: {severity}")
                elif severity == 'MEDIUM':
                    st.warning(f"Severity: {severity}")
                else:
                    st.info(f"Severity: {severity}")
        else:
            st.info("No alerts recorded yet")
    
    elif page == "🔥 Maps & Heatmaps":
        st.divider()
        display_heatmap()
    
    elif page == "📸 Snapshots":
        st.divider()
        display_snapshots()
    
    elif page == "📋 Alert Logs":
        st.divider()
        display_alerts_table()
    
    # Footer
    st.divider()
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <p>🛡️ Women Safety Detection System | Educational Purpose Only</p>
        <p>Last Updated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
