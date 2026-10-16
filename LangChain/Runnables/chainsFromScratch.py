import random

class DummyLLM:
    def __init__(self):
        print('LLM Created')

    def predict(self, prompt):
        
        response_list = [
            "The weather is sunny and perfect for a walk.",
            "A quick review shows the plan is on track and efficient.",
            "The new idea is creative, practical, and worth testing."
        ]

        return {'response': random.choice(response_list)}


class DummyPromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, input_dict):
        return self.template.format(**input_dict)



class DummyLLMChain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict):
        final_prompt = self.prompt.format(input_dict)
        result = self.llm.predict(final_prompt)
        return result["response"]


template = DummyPromptTemplate(
    template="Hello, {name}! Today is {day}.",
    input_variables=["name", "day"]
)

llm = DummyLLM()

chain = DummyLLMChain(llm, template)

result = chain.run({"name": "Alice", "day": "Monday"})
print(result)