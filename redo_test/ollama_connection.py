from ollama import chat
from ollama import ChatResponse
import ollama

# response: ChatResponse = chat(
#     model='llama3.2', 
#     messages=[{
#         'role': 'user',
#         'content': 'Why is the sky blue?'}])
# print(response['message']['content'])
# # or access fields directly from the response object
# print(response.message.content)

chat_test = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])
print(type(chat_test))
