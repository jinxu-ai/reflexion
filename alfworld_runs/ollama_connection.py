from ollama import generate, chat


import os
import sys
# import openai
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)

from typing import Optional, List
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


Model = Literal["deepseek-r1:8b", "llama3.2"]


def ollama_generate(model_name="deepseek-r1:8b", prompt=None):
    """
    Generates text using the specified model based on a single prompt.
    
    :param model_name: The name of the model to use for text generation.
    :param prompt: The prompt string to generate text form.
    :return: Generated text response.
    """
    try:
        response = generate(model=model_name, prompt=prompt)
        
        # Debug: Print the response structure
        # print("Response Debug:", response)
        
        # Directly return the generated text
        # many components in the response variable, use get function to extract the response part. \
        # If there is no response part, return 'No text generated in the response'
        return response.get('response', "No text generated in the response")
    except Exception as e:
        return f"Error during text generation: {e}"



def ollama_chat(model_name="deepseek-r1:8b", messages=None):
    """
    Conducts a chat conversation using the specified model.
    
    :param model_name: The name of the model to use for chatting.
    :param messages: A list of message dictionaries containing roles and contents.
    :return: Generated response from the model.
    """

    if isinstance(messages, str):  # 检查 messages 是否是字符串
        messages = [{"role": "user", "content": messages}]  # 转换成正确的格式
    
    try:
        response = chat(model=model_name, messages=messages)
        return response['message']['content']
    except Exception as e:
        return f"Error during chat interaction: {e}"



# messages in ollama_chat: list[dict]
# original code: str (for pppGPT)
        
# # Example usage for chat
# model_name_chat = "qwen2.5:1.5b"
# messages = [
#     {
#         'role': 'user',
#         'content': 'Why is the sky blue?',
#     },
#     {
#         'role': 'assistant',
#         'content': 'The sky appears blue because of Rayleigh scattering.',
#     },
#     {
#         'role': 'user',
#         'content': 'Can you explain it in more detail?',
#     },
# ]
# chat_response = ollama_chat(model_name_chat, messages)
# print("Chat Response:", chat_response)