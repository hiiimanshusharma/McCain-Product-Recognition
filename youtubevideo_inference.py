# load config
import json
import cv2
import base64
import numpy as np
import requests
import time
from vidgear.gears import CamGear

# Load configuration from JSON
with open(r"C:\Users\91830\Downloads\roboflow_config.json") as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]

    FRAMERATE = config["FRAMERATE"]
    BUFFER = config["BUFFER"]

# Construct the Roboflow Infer URL
upload_url = "".join([
    "https://detect.roboflow.com/",
    ROBOFLOW_MODEL,
    "?api_key=",
    ROBOFLOW_API_KEY,
    "&format=image",
    "&stroke=5"
])

source = input("enter the link of video")

stream = CamGear(source, stream_mode = True, logging=True).start() # YouTube Video URL as input


# Get video file path
# video_path = r"C:\Users\91830\Downloads\McCain Emotibites #LetsSnackchat.mp4" # Replace with your input video file path
#
# # Open the video file
# video = cv2.VideoCapture(video_path)
#
# # Get video properties
# frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
# frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
# fps = video.get(cv2.CAP_PROP_FPS)

# Define the output video writer
# output_path = r"C:\Users\91830\Downloads\video-inference\output1.mp4"  # Replace with your desired output video file path
# output_codec = cv2.VideoWriter_fourcc(*"mp4v")  # Output video codec
# output = cv2.VideoWriter(output_path, output_codec, fps, (frame_width, frame_height))

# Infer via the Roboflow Infer API and return the result
def infer(image):
    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = image.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    resized_image = cv2.resize(image, (round(scale * width), round(scale * height)))

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', resized_image)
    img_str = base64.b64encode(buffer)

    # Get prediction from Roboflow Infer API
    resp = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, stream=True).raw

    # Parse result image
    result_image = np.asarray(bytearray(resp.read()), dtype="uint8")
    result_image = cv2.imdecode(result_image, cv2.IMREAD_COLOR)

    return result_image

# frame counter
frame_counter = 0
# number of frames to skip
skip_frames = 5



# Process each frame of the video
while True:
    # Read the next frame
    frame = stream.read()

    # check if frame is None
    if frame is None:
        #if True break the infinite loop
        break

    if frame_counter % (skip_frames + 1) != 0:
        frame_counter += 1
        continue


    # If the frame was not successfully read, then we have reached the end of the video
    #if not ret:
    #   break

    # Perform inference on the frame
    result_frame = infer(frame)

    # Display the inference results
    cv2.imshow('Inferred Frame', result_frame)

    # Write the inferred frame to the output video
    # output.write(result_frame)

    # Check for 'q' keypress to stop the program
    if cv2.waitKey(1) == ord('q'):
        break


    # increment frame counter
    frame_counter += 1



# Release resources
# video.release()
# output.release()
cv2.destroyAllWindows()
