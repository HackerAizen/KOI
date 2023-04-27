from imageai.Detection import VideoObjectDetection
import os
import cv2

execution_path = os.getcwd()


camera = cv2.VideoCapture(0)
cv2.startWindowThread()

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "retinanet_resnet50_fpn_coco-eeacb38b.pth"))
detector.loadModel()

def forFrame(frame_number, output_array, output_count, detected_frame):
    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")
    cv2.imshow('Tool detection', detected_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        return

video_path = detector.detectObjectsFromVideo(camera_input=camera,
                        output_file_path=os.path.join(execution_path, "camera_detected_video")
                        , frames_per_second=20, log_progress=True, minimum_percentage_probability=40, detection_timeout=120,
                                             save_detected_video=True, return_detected_frame=True, per_frame_function=forFrame)