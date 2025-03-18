from ollama import generate, chat
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

# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
# Function to generate text based on a given prompt
def ollama_generate(model_name, prompt):
    """
    Generates text using the specified model based on a single prompt.
    
    :param model_name: The name of the model to use for text generation.
    :param prompt: The prompt string to generate text from.
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


# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
# Function to conduct a chat conversation using the specified model
def ollama_chat(model_name, messages):
    """
    Conducts a chat conversation using the specified model.
    
    :param model_name: The name of the model to use for chatting.
    :param messages: A list of message dictionaries containing roles and contents.
    :return: Generated response from the model.
    """
    try:
        response = chat(model=model_name, messages=messages)
        return response['message']['content']
    except Exception as e:
        return f"Error during chat interaction: {e}"
    

# # Example usage for chat
# model_name_chat = "llama3.2"
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

# Example usage for text generation
prompt = "Explain the process of photosynthesis."
model_name_generate = "llama3.2"
generated_text = ollama_generate(model_name_generate, prompt)
print("Generated Text:", generated_text)