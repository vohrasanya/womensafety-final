import cv2

class VideoCamera:
    def __init__(self, source=0):
        # Default source is webcam (0), change to IP address if using mobile camera
        self.video = cv2.VideoCapture(source)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        if success:
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
        return None
