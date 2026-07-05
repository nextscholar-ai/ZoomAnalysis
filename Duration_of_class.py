import cv2

video = cv2.VideoCapture("Class_1.mp4")

fps = video.get(cv2.CAP_PROP_FPS)
frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

duration = frame_count / fps  


minutes = int(duration // 60)
seconds = int(duration % 60)

print(f"Duration: {minutes}m {seconds}s")