import asyncio
import autogen
import cv2
import numpy as np
import os
import sounddevice as sd
from transformers import AutoProcessor, BarkModel


# Intitialize the webcam
# Intitialize the AutoGen
# On output stream, add video frame to webcam feed
class LiveStream:
    def __init__(self):
        self.C1 = cv2
        self.VIDEO_WRITER = None
        self.VIDEO_FRAMES = []
        avatar = 'avatars/shubham/'
        self.DEFAULT_FRAME = self.C1.imread(avatar+"avatar.jpg")
        self.VIDEO_WIDTH = int(1620)
        self.VIDEO_HEIGHT = int(1080)
        self.FPS = int(30)
        self.SPEAK = ""
    
    def init_video(self):
        # Define the video output file and codec
        output_file = 'output_video.mjpeg'
        fourcc = self.C1.VideoWriter_fourcc(*'mp4v')

        # Create a VideoWriter object to write the video
        self.VIDEO_WRITER = self.C1.VideoWriter(output_file, fourcc, self.FPS, (self.VIDEO_WIDTH, self.VIDEO_HEIGHT))
    
    def destory_video(self):
        # Release the VideoWriter object and close the display window
        self.VIDEO_WRITER.release()
        self.C1.destroyAllWindows()
    
    def add_frame(self, expression=None):
        # Create a frame (you can generate this frame from images or any other source)
        frame = np.zeros((self.VIDEO_HEIGHT, self.VIDEO_WIDTH, 3), dtype=np.uint8)
        if expression is None:
            resized_image = self.C1.resize(self.DEFAULT_FRAME, (self.VIDEO_WIDTH, self.VIDEO_HEIGHT))
            self.VIDEO_FRAMES.append(self.C1.addWeighted(frame, 1, resized_image, 0.5, 0))
        else:
            # Write the frame to the video based on given expression
            pass
    
    async def stream_video(self):
        while True:
            if self.VIDEO_FRAMES:
                frame = self.VIDEO_FRAMES.pop()
                self.VIDEO_WRITER.write(frame)

            # Display the frame in a window (optional)
            self.C1.imshow("Generated Video Stream", frame)
            
            # Exit the loop if the 'q' key is pressed
            if self.C1.waitKey(1) & 0xFF == ord('q'):
                break

            await asyncio.sleep(0)
            
        self.destory_video()
        self.destory_audio()
    
    def init_audio(self):
        os.environ["SUNO_OFFLOAD_CPU"] = "True"
        os.environ["SUNO_ENABLE_MPS"] = "True"
        os.environ["USE_SMALL_MODELS"] = "True"

        self.AUDIO_MODEL = BarkModel.from_pretrained("suno/bark")
        self.AUDIO_MODEL =  self.AUDIO_MODEL.to_bettertransformer()

        self.AUDIO_PROCESSOR = AutoProcessor.from_pretrained("suno/bark")
        self.VOICE_PRESENT = "v2/en_speaker_6"

        self.AUDIO_STREAM = sd.OutputStream(channels=1, samplerate=44100)

        # Start the stream
        self.AUDIO_STREAM.start()
    
    def stream_audio(self, text):
        inputs = self.AUDIO_PROCESSOR(text, voice_preset=self.VOICE_PRESENT)
        audio_array = self.AUDIO_MODEL.generate(**inputs)
        
        audio_array = audio_array.cpu().numpy().squeeze()

        # sample_rate and channels are statically set in init_audio
        # sample_rate = self.AUDIO_MODEL.generation_config.sample_rate

        # Convert the array to float32 if it's not already
        audio_array = audio_array.astype(np.float32)

        self.AUDIO_STREAM.write(audio_array)

    def destory_audio(self):
        # Stop and close the stream
        self.AUDIO_STREAM.stop()
    
    def init_ai(self):
        llm_config = {
                "cache_seed": None,
                "config_list": [
                    {
                        "model": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                        "base_url": "http://localhost:1234/v1",  #litellm compatible endpoint
                        "api_key": "NULL", # just a placeholder
                    }
                ],
                "temperature": 0,  # temperature for sampling
                "timeout": 600,
                "stream": True,
                "stream_output_func": self.construct_ai_response
            }

        self.AUTOGEN_ASST = autogen.AssistantAgent(
            name="assistant",
            system_message="You are a normal human that has different facial expression based on the replies and context of the conversion.",
            llm_config=llm_config,
        )

        # create a UserProxyAgent instance named "user_proxy"
        self.AUTOGEN_USER = autogen.UserProxyAgent(
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
    
    def construct_ai_response(self, content):
        self.SPEAK += content
        if len(self.SPEAK.split(' ')) > 9:
            self.stream_audio(self.SPEAK)
            self.SPEAK = ""

    async def start_conversion(self):
        # the assistant receives a message from the user_proxy, which contains the task description
        self.AUTOGEN_USER.initiate_chat(
            self.AUTOGEN_ASST,
            message="""Always define a facial expression under square branckets. What is full form of LOL?.""",
        )
    
    def stream(self):
        asyncio.run(self._stream())

    async def _stream(self):
        await asyncio.gather(
            self.stream_video(),
            self.start_conversion(),
        )

ls = LiveStream()
ls.init_audio()
ls.init_video()
ls.init_ai()
ls.add_frame()
ls.stream()


