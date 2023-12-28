import autogen

"""
config_list2 = [
    {
        'model': 'gpt-3.5-turbo',
        'api_key': "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",
    },
]
"""
config_list = [
    {
        "model": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        "base_url": "http://localhost:1234/v1",  #litellm compatible endpoint
        "api_key": "NULL", # just a placeholder
    }
]

llm_config = {
        "cache_seed": 1,
        "config_list": config_list,  # a list of OpenAI API configurations
        "temperature": 0,  # temperature for sampling
        "timeout": 600
    }


assistant = autogen.AssistantAgent(
    name="assistant",
    system_message="You are a computer programmer.",
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
    message="""What date is today? Compare the year-to-date gain for META and TESLA. Always provide asked code block within triple backticks. Start the code block with "# filename: suggested_file_name" where the suggested_file_name is suggest the placeholder""",
)

