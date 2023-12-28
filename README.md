## Using MS AutoGen
https://microsoft.github.io/autogen/

## Using litellm python package for open source model API handling. [Not Working]
This enables functionality of API (that would behave same as openAI APIs) for AI models stored locally.
Below command will load the model stored locally, the endpoint will look like http://0.0.0.0:8000
`litellm --model TheBloke/Mistral-7B-Instruct-v0.1-GGUF`
Perform below command to test th loaded model
`litellm --test`
https://docs.litellm.ai/docs/proxy_server#tutorial-use-with-aiderautogencontinue-devlangroid


## Project Flow
Real Time AI Video Assistant
- AI Models
-- Speech to text -> 
-- Text to speech -> https://github.com/suno-ai/bark, https://docs.google.com/document/d/13_l1bd1Osgz7qlAZn-zhklCbHpVRk6bYOuAuB78qmsE/edit
-- Text to human video -> https://huggingface.co/spaces/fffiloni/ControlVideo/tree/main, https://github.com/s0md3v/roop
-- Text to human expression

## Chromium Tasks
Flow Discussion with AI: https://chat.openai.com/share/c11e1396-f59f-4615-8c7a-c7ae0e4c95de
Chromium Midea stream: https://www.w3.org/TR/mediacapture-streams/

## Usefull AI Models
Youtuber AI Jason: BakLLaVA+StyleTTS2+dolphin-2.5-mixtral-8x7b+SpeechBrain