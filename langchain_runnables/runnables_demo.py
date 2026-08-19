import random

class NakliLLM:
    def __init__(self):
        print("LLM Created")

    def predict(self, prompt):
        response_list = [
            'Islamabad is the Capital of Pakistan',
            'PSL is a cricket league',
            'AI stands for Artificial Intelligence'
        ]

        return random.choice(response_list)

# llm = NakliLLM()
# response = llm.predict("What is the Capital of Pakistan?")
# print(response)


class NakliPromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, input_dict):
        return self.template.format(**input_dict)

template = NakliPromptTemplate(
    template="Write a {length} poem about {topic}",
    input_variables=['topic']
)

prompt = template.format({'length':'short', 'topic':'Pakistan'})
print(prompt)

llm = NakliLLM()
result = llm.predict(prompt=prompt)
print(result)


class NakliLLMChain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict):
        final_prompt = self.prompt.format(input_dict)
        result = self.llm.predict(final_prompt)
        return result


chain = NakliLLMChain(llm, template)
res = chain.run({'length':'short', 'topic':'Pakistan'})
print(res)



