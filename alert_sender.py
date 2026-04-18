"""
Alert Sender Module - Send alerts via Telegram and Email
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    EMAIL_ENABLED, EMAIL_SENDER, EMAIL_PASSWORD,
    EMAIL_RECIPIENT, SMTP_SERVER, SMTP_PORT
)


class AlertSender:
    """
    Send alerts via multiple channels (Telegram, Email)
    """
    
    def __init__(self):
        self.telegram_token = TELEGRAM_BOT_TOKEN
        self.telegram_chat_id = TELEGRAM_CHAT_ID
        self.telegram_api_url = f"https://api.telegram.org/bot{self.telegram_token}"
    
    def send_telegram_alert(self, camera_id: str, latitude: float, longitude: float,
                           male_count: int, female_count: int,
                           image_path: Optional[str] = None) -> bool:
        """
        Send alert via Telegram
        
        Args:
            camera_id: Camera ID
            latitude: Latitude
            longitude: Longitude
            male_count: Number of males detected
            female_count: Number of females detected
            image_path: Optional path to snapshot
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create alert message
            message = (
                f"🚨 **SAFETY ALERT** 🚨\n\n"
                f"📷 Camera: {camera_id}\n"
                f"👩: {female_count} Female(s)\n"
                f"👨: {male_count} Male(s)\n"
                f"📍 Coordinates: {latitude:.4f}, {longitude:.4f}\n"
                f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"🔗 Map: https://maps.google.com/?q={latitude},{longitude}"
            )
            
            # Send photo with caption if image exists
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as photo:
                    files = {'photo': photo}
                    data = {
                        'chat_id': self.telegram_chat_id,
                        'caption': message,
                        'parse_mode': 'Markdown'
                    }
                    response = requests.post(
                        f"{self.telegram_api_url}/sendPhoto",
                        data=data,
                        files=files,
                        timeout=10
                    )
            else:
                # Send text message only
                data = {
                    'chat_id': self.telegram_chat_id,
                    'text': message,
                    'parse_mode': 'Markdown'
                }
                response = requests.post(
                    f"{self.telegram_api_url}/sendMessage",
                    json=data,
                    timeout=10
                )
            
            if response.status_code == 200:
                print(f"✅ Telegram alert sent from {camera_id}")
                return True
            else:
                print(f"❌ Failed to send Telegram alert: {response.text}")
                return False
        
        except Exception as e:
            print(f"❌ Error sending Telegram alert: {e}")
            return False
    
    def send_telegram_video(self, camera_id: str, video_path: str,
                           message: str = None) -> bool:
        """
        Send video alert via Telegram
        
        Args:
            camera_id: Camera ID
            video_path: Path to video file
            message: Optional caption
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(video_path):
                print(f"❌ Video file not found: {video_path}")
                return False
            
            if message is None:
                message = f"📹 Incident video from {camera_id}"
            
            with open(video_path, 'rb') as video:
                files = {'video': video}
                data = {
                    'chat_id': self.telegram_chat_id,
                    'caption': message,
                    'parse_mode': 'Markdown'
                }
                response = requests.post(
                    f"{self.telegram_api_url}/sendVideo",
                    data=data,
                    files=files,
                    timeout=30  # Videos take longer
                )
            
            if response.status_code == 200:
                print(f"✅ Telegram video sent from {camera_id}")
                return True
            else:
                print(f"❌ Failed to send Telegram video: {response.text}")
                return False
        
        except Exception as e:
            print(f"❌ Error sending Telegram video: {e}")
            return False
    
    def send_email_alert(self, camera_id: str, latitude: float, longitude: float,
                        male_count: int, female_count: int,
                        image_path: Optional[str] = None,
                        video_path: Optional[str] = None) -> bool:
        """
        Send alert via Email
        
        Args:
            camera_id: Camera ID
            latitude: Latitude
            longitude: Longitude
            male_count: Number of males detected
            female_count: Number of females detected
            image_path: Optional path to snapshot
            video_path: Optional path to video clip
        
        Returns:
            True if successful, False otherwise
        """
        if not EMAIL_ENABLED:
            return False
        
        try:
            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"🚨 Safety Alert from {camera_id}"
            msg['From'] = EMAIL_SENDER
            msg['To'] = EMAIL_RECIPIENT
            
            # Create email body
            html_body = f"""
            <html>
                <body style="font-family: Arial, sans-serif;">
                    <h2 style="color: red;">🚨 SAFETY ALERT 🚨</h2>
                    <p><b>Camera:</b> {camera_id}</p>
                    <p><b>Females Detected:</b> {female_count}</p>
                    <p><b>Males Detected:</b> {male_count}</p>
                    <p><b>Location:</b> {latitude:.4f}, {longitude:.4f}</p>
                    <p><b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><a href="https://maps.google.com/?q={latitude},{longitude}">View on Map</a></p>
                </body>
            </html>
            """
            
            # Attach HTML body
            msg.attach(MIMEText(html_body, 'html'))
            
            # Attach image if provided
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as attachment:
                    image = MIMEImage(attachment.read())
                    image.add_header('Content-ID', '<image>')
                    msg.attach(image)
            
            # Attach video if provided
            if video_path and os.path.exists(video_path):
                with open(video_path, 'rb') as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header('Content-Disposition', f'attachment; filename= {Path(video_path).name}')
                    msg.attach(part)
            
            # Send email
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.send_message(msg)
            
            print(f"✅ Email alert sent from {camera_id}")
            return True
        
        except Exception as e:
            print(f"❌ Error sending email: {e}")
            return False
    
    def send_multi_channel_alert(self, camera_id: str, latitude: float, longitude: float,
                                male_count: int, female_count: int,
                                image_path: Optional[str] = None,
                                video_path: Optional[str] = None) -> Dict:
        """
        Send alert via all enabled channels
        
        Args:
            camera_id: Camera ID
            latitude: Latitude
            longitude: Longitude
            male_count: Number of males detected
            female_count: Number of females detected
            image_path: Optional path to snapshot
            video_path: Optional path to video clip
        
        Returns:
            Dictionary with results from each channel
        """
        results = {
            'telegram_alert': False,
            'telegram_video': False,
            'email': False
        }
        
        # Send Telegram alert
        results['telegram_alert'] = self.send_telegram_alert(
            camera_id, latitude, longitude, male_count, female_count, image_path
        )
        
        # Send Telegram video if available
        if video_path:
            message = (
                f"📹 Incident Video\n"
                f"Camera: {camera_id}\n"
                f"Females: {female_count}\n"
                f"Males: {male_count}"
            )
            results['telegram_video'] = self.send_telegram_video(camera_id, video_path, message)
        
        # Send Email if enabled
        if EMAIL_ENABLED:
            results['email'] = self.send_email_alert(
                camera_id, latitude, longitude, male_count, female_count, image_path, video_path
            )
        
        return results


class AlertLogger:
    """
    Log all alerts to file and database
    """
    
    def __init__(self, log_file: str = 'alerts/alerts.log'):
        self.log_file = log_file
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def log_alert(self, alert_data: Dict):
        """
        Log alert to file
        
        Args:
            alert_data: Dictionary with alert information
        """
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            log_entry = (
                f"[{timestamp}] ALERT\n"
                f"  Camera: {alert_data.get('camera_id', 'Unknown')}\n"
                f"  Location: {alert_data.get('latitude', 'N/A')}, "
                f"{alert_data.get('longitude', 'N/A')}\n"
                f"  Females: {alert_data.get('female_count', 0)}\n"
                f"  Males: {alert_data.get('male_count', 0)}\n"
                f"  Severity: {alert_data.get('severity', 'Unknown')}\n"
                f"  Image: {alert_data.get('image_path', 'None')}\n"
                f"  Video: {alert_data.get('video_path', 'None')}\n"
                f"-" * 60 + "\n"
            )
            
            with open(self.log_file, 'a') as f:
                f.write(log_entry)
            
            print(f"✅ Alert logged to {self.log_file}")
        
        except Exception as e:
            print(f"❌ Error logging alert: {e}")
    
    def get_alert_history(self) -> List[str]:
        """Get all logged alerts"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    return f.readlines()
            return []
        except Exception as e:
            print(f"❌ Error reading logs: {e}")
            return []
