from ollama import chat
from ollama import ChatResponse
import ollama
# import aisuite as ai 

# # Initialize the client
# client = ai.Client()

# # Access the completions interface
# chat_completions = client.chat.completions

# # Use the correct 'provider:model' format
# response = chat_completions.create(
#     model='ollama:llama3.2',
#     messages=[{
#         'role': 'user',
#         'content': 'Why is the sky blue?'
#     }]
# )

# # Extract and print the generated content
# if isinstance(response.choices, list) and len(response.choices) > 0:
#     first_choice = response.choices[0]
#     if isinstance(first_choice, dict) and 'text' in first_choice:
#         print("Generated Text:", first_choice['text'])
#     else:
#         print("Unexpected structure in first choice:", first_choice)
# else:
#     print("No choices returned in the response.")



# print(dir(response.choices))
# help(response.choices)




# response = client.chat(
#     model='llama3.2',
#     messages=[{
#         'role': 'user',
#         'content': 'Why is the sky blue?'
#     }]
# )

# print(response)


response: ChatResponse = chat(
    model='llama3.2', 
    messages=[{
        'role': 'user',
        'content': 'Why is the sky blue?'}])
print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)

# chat_test = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])
# print(type(chat_test))
