
def generate_video():
    import cv2
    import numpy as np

    # Define the video output file and codec
    output_file = 'output_video.mjpeg'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    avatar = 'avatars/shubham/'
    consentrating = cv2.VideoCapture(avatar+"consentrating.mov")
    laughing = cv2.VideoCapture(avatar+"laughing.mov")
    lol = cv2.VideoCapture(avatar+"lol.mov")
    thinking = cv2.VideoCapture(avatar+"thinking.mov")
    image = cv2.imread(avatar+"avatar.jpg")
    
    width = int(consentrating.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(consentrating.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(consentrating.get(cv2.CAP_PROP_FPS))

    # Create a VideoWriter object to write the video
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    sequence = [consentrating, thinking, laughing, lol]

    while True:
        # Create a frame (you can generate this frame from images or any other source)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        resized_image = cv2.resize(image, (width, height))
        frame = cv2.addWeighted(frame, 1, resized_image, 0.5, 0)

        # Write the frame to the video
        # if sequence:
        #     ret, frame = consentrating.read()
        #     if not ret:
        #         resized_image = cv2.resize(image, (width, height))
        #         frame = cv2.addWeighted(frame, 1, resized_image, 0.5, 0)
            
        out.write(frame)

        # Display the frame in a window (optional)
        cv2.imshow("Generated Video Stream", frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoWriter object and close the display window
    out.release()
    consentrating.release()
    laughing.release()
    lol.release()
    thinking.release()
    cv2.destroyAllWindows()

def generate_reply():
    import autogen

    def print_stream_response(content):
        print('===={}=====\n'.format(content))
        # generate_video(content)
        
    config_list = [
        {
            "model": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            "base_url": "http://localhost:1234/v1",  #litellm compatible endpoint
            "api_key": "NULL", # just a placeholder
        }
    ]

    llm_config = {
            "cache_seed": None,
            "config_list": config_list,  # a list of OpenAI API configurations
            "temperature": 0,  # temperature for sampling
            "timeout": 600,
            "stream": True,
            "stream_output_func": print_stream_response
        }

    assistant = autogen.AssistantAgent(
        name="assistant",
        system_message="You are a normal human that has different facial expression based on the replies and context of the conversion.",
        llm_config=llm_config,
    )

    # create a UserProxyAgent instance named "user_proxy"
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("EXIT") or "execution succeeded" in x.get("content", ""),
        llm_config=llm_config,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False,  # set to True or image name like "python:3" to use docker
        },
        system_message="""Reply "EXIT" if the given task is completed with all the specification."""
    )

    
    # the assistant receives a message from the user_proxy, which contains the task description
    user_proxy.initiate_chat(
        assistant,
        message="""Always define a facial expression under square branckets. what is full form of LOL?.""",
    )

def generate_audio():
    import numpy as np
    import sounddevice as sd
    from transformers import AutoProcessor, BarkModel
    import os
    
    os.environ["SUNO_OFFLOAD_CPU"] = "True"
    os.environ["SUNO_ENABLE_MPS"] = "True"
    os.environ["USE_SMALL_MODELS"] = "True"

    model = BarkModel.from_pretrained("suno/bark")
    model =  model.to_bettertransformer()

    processor = AutoProcessor.from_pretrained("suno/bark")
    voice_preset = "v2/en_speaker_6"
    
    inputs = processor("Hello, my dog is cute", voice_preset=voice_preset)
    audio_array = model.generate(**inputs)
    
    audio_array = audio_array.cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate
    
    # audio_array = np.loadtxt('my_array.txt')
    # sample_rate = 24000

    if len(audio_array.shape) == 2:
        # Transpose the array if necessary
        audio_array = audio_array.T
    
    # Convert the array to float32 if it's not already
    audio_array = audio_array.astype(np.float32)

    def callback(stream, audio_array):
        try:
            while True:
                stream.write(audio_array)
        except KeyboardInterrupt:
            # Stop the stream when the user interrupts the program
            stream.stop()
    
    # Open a stream with the specified parameters
    channels = 1
    if len(audio_array.shape) > 1:
        channels = audio_array.shape[1]

    stream = sd.OutputStream(channels=channels, samplerate=sample_rate)

    # Start the stream
    stream.start()

    callback(stream, audio_array)

    # Stop and close the stream
    stream.stop()

def speechToText():
    import whisper
    model = whisper.load_model("tiny")
    result = model.transcribe("WisperAudio.webm")
    print(result["text"])


speechToText()