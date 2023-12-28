def extractCode():
    import re
    pattern = r"```(\w*)\n(.*?)\n```"
    text = '''```python
    def is_prime(n):
        if n <= 1:
            return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0:
                return False
        return True
    ```'''

    match = re.findall(pattern, text, flags=re.DOTALL)

    print(match, False)

def mergeTwoMJpeg():
    import cv2
    import time

    # Open the first MJPEG file for reading
    first_mjpeg_file = '/Users/sdayma/Downloads/sample_640x360.mjpeg'
    cap1 = cv2.VideoCapture(first_mjpeg_file)

    # Open the second MJPEG file for reading
    second_mjpeg_file = '/Users/sdayma/Downloads/test2.mjpeg'
    cap2 = cv2.VideoCapture(second_mjpeg_file)

    # Check if the video files were opened successfully
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both of the MJPEG files.")
        exit()

    # Get the video's width, height, and frame rate (assuming both videos have the same properties)
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap1.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object to write the merged video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_file = '/Users/sdayma/Downloads/sample_640x360.mjpeg'
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))

    # Check if the VideoWriter was opened successfully
    if not out.isOpened():
        print("Error: Could not open the output MJPEG file for writing.")
        cap1.release()
        cap2.release()
        exit()

    # Process frames from the first video and add them to the merged video
    while True:
        ret, frame = cap1.read()
        if not ret:
            break  # No more frames to read from the first video
        out.write(frame)  # Write the frame to the merged video

    time.sleep(2)
    # Process frames from the second video and add them to the merged video
    while True:
        ret, frame = cap2.read()
        if not ret:
            break  # No more frames to read from the second video
        out.write(frame)  # Write the frame to the merged video

    # Release the VideoCapture and VideoWriter objects
    cap1.release()
    cap2.release()
    out.release()

    print("Merged MJPEG video saved as:", output_file)


mergeTwoMJpeg()