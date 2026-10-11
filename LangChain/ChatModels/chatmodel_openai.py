from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4") #, temperature=0, max_tokens=100)

"""
Temperature guide:
Factual Answers - 0.0 - 0.3
Balanced response - 0.4 - 0.7
Creative writing, storytelling - 0.8 - 1.2
Maximum randomness - 1.5+

max_tokens -> number of tokens to generate in the response
"""

result = model.invoke("What is the Capital of France?")

# print(result)
#Output: content='The capital of France is Paris.' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 7, 'prompt_tokens': 14, 'total_tokens': 21, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cache_write_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'id': 'chatcmpl-EXsZPtMVfI9z18IDxt6esw55rUWhn', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='run--01a12c39-0c95-7802-98ad-ef9c37442059-0' usage_metadata={'input_tokens': 14, 'output_tokens': 7, 'total_tokens': 21, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}} 

print(result.content)
# Output: The capital of France is Paris.