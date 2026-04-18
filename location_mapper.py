"""
Location Mapping and Heatmap Generation
"""
import folium
from folium.plugins import HeatMap, MarkerCluster
import json
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict
from config import HEATMAP_OUTPUT, HEATMAP_CENTER_LAT, HEATMAP_CENTER_LON, HEATMAP_ZOOM


class LocationMapper:
    """
    Handles GPS location mapping and heatmap generation
    """
    
    def __init__(self, center_lat: float = HEATMAP_CENTER_LAT, 
                 center_lon: float = HEATMAP_CENTER_LON,
                 zoom: int = HEATMAP_ZOOM):
        """
        Initialize mapper
        
        Args:
            center_lat: Center latitude for map
            center_lon: Center longitude for map
            zoom: Initial zoom level
        """
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoom = zoom
        self.alerts = []  # Store all alerts: {lat, lon, timestamp, severity, details}
    
    def add_alert(self, latitude: float, longitude: float, 
                  severity: str = "medium", details: dict = None):
        """
        Add an alert location
        
        Args:
            latitude: Alert latitude
            longitude: Alert longitude
            severity: 'low', 'medium', 'high' (affects heatmap color)
            details: Additional details (camera_id, male_count, female_count, etc.)
        """
        alert = {
            'latitude': latitude,
            'longitude': longitude,
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'details': details or {}
        }
        self.alerts.append(alert)
    
    def create_heatmap(self, output_path: str = HEATMAP_OUTPUT):
        """
        Generate heatmap from alert locations
        
        Args:
            output_path: Path to save heatmap HTML file
        """
        # Create base map
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=self.zoom,
            tiles='OpenStreetMap'
        )
        
        # Prepare heatmap data (list of [lat, lon, intensity])
        heat_data = []
        for alert in self.alerts:
            # Intensity based on severity
            intensity = {'low': 0.5, 'medium': 0.75, 'high': 1.0}
            severity = alert.get('severity', 'medium')
            
            heat_data.append([
                alert['latitude'],
                alert['longitude'],
                intensity.get(severity, 0.5)
            ])
        
        # Add heatmap layer if we have data
        if heat_data:
            HeatMap(heat_data, radius=20, blur=15, max_zoom=1).add_to(m)
        
        # Save map
        m.save(output_path)
        print(f"✅ Heatmap saved to {output_path}")
        return output_path
    
    def create_detailed_map(self, output_path: str = None):
        """
        Create map with detailed markers for each alert
        
        Args:
            output_path: Path to save map HTML file
        
        Returns:
            Path to saved map
        """
        if output_path is None:
            output_path = 'detailed_map.html'
        
        # Create base map
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=self.zoom,
            tiles='OpenStreetMap'
        )
        
        # Add marker cluster
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add each alert as a marker
        for idx, alert in enumerate(self.alerts):
            severity_color = {
                'low': 'blue',
                'medium': 'orange',
                'high': 'red'
            }
            color = severity_color.get(alert.get('severity', 'medium'), 'blue')
            
            # Create popup text
            details = alert.get('details', {})
            popup_text = f"""
            <b>Alert #{idx + 1}</b><br>
            Camera ID: {details.get('camera_id', 'N/A')}<br>
            Females: {details.get('female_count', 0)}<br>
            Males: {details.get('male_count', 0)}<br>
            Time: {alert.get('timestamp', 'N/A')[:19]}<br>
            Severity: {alert.get('severity', 'N/A').upper()}
            """
            
            folium.Marker(
                location=[alert['latitude'], alert['longitude']],
                popup=folium.Popup(popup_text, max_width=250),
                icon=folium.Icon(color=color, icon='info-sign'),
                tooltip=f"Alert {idx + 1}"
            ).add_to(marker_cluster)
        
        m.save(output_path)
        print(f"✅ Detailed map saved to {output_path}")
        return output_path
    
    def export_alerts_json(self, output_path: str = 'alerts.json'):
        """
        Export all alerts to JSON file
        
        Args:
            output_path: Path to save JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(self.alerts, f, indent=4)
        print(f"✅ Alerts exported to {output_path}")
    
    def get_alert_statistics(self) -> dict:
        """Get statistics about alerts"""
        if not self.alerts:
            return {
                'total_alerts': 0,
                'high_severity': 0,
                'medium_severity': 0,
                'low_severity': 0
            }
        
        return {
            'total_alerts': len(self.alerts),
            'high_severity': sum(1 for a in self.alerts if a.get('severity') == 'high'),
            'medium_severity': sum(1 for a in self.alerts if a.get('severity') == 'medium'),
            'low_severity': sum(1 for a in self.alerts if a.get('severity') == 'low'),
            'average_latitude': sum(a['latitude'] for a in self.alerts) / len(self.alerts),
            'average_longitude': sum(a['longitude'] for a in self.alerts) / len(self.alerts),
        }
    
    def clear_alerts(self):
        """Clear all stored alerts"""
        self.alerts.clear()
    
    def get_hotspots(self, radius_km: float = 0.5) -> List[dict]:
        """
        Find clusters/hotspots of alerts
        
        Args:
            radius_km: Radius in kilometers to consider as same location
        
        Returns:
            List of hotspots with coordinates and alert count
        """
        if not self.alerts:
            return []
        
        hotspots = []
        used_indices = set()
        
        for i, alert1 in enumerate(self.alerts):
            if i in used_indices:
                continue
            
            # Simple clustering: find alerts within radius_km
            cluster = [alert1]
            used_indices.add(i)
            
            for j, alert2 in enumerate(self.alerts):
                if j <= i or j in used_indices:
                    continue
                
                # Simple distance calculation (rough approximation)
                lat_diff = (alert2['latitude'] - alert1['latitude']) * 111  # km per degree
                lon_diff = (alert2['longitude'] - alert1['longitude']) * 111 * \
                           np.cos(np.radians(alert1['latitude']))  # km per degree
                
                distance = np.sqrt(lat_diff**2 + lon_diff**2)
                
                if distance <= radius_km:
                    cluster.append(alert2)
                    used_indices.add(j)
            
            # Calculate cluster center
            avg_lat = sum(a['latitude'] for a in cluster) / len(cluster)
            avg_lon = sum(a['longitude'] for a in cluster) / len(cluster)
            
            hotspots.append({
                'latitude': avg_lat,
                'longitude': avg_lon,
                'alert_count': len(cluster),
                'severity': max([a.get('severity', 'medium') for a in cluster],
                               key=lambda x: {'low': 0, 'medium': 1, 'high': 2}.get(x, 1))
            })
        
        return hotspots
