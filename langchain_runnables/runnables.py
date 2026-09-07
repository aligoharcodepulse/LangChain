import random
from abc import ABC, abstractmethod

class Runnable(ABC):
    @abstractmethod
    def invoke(input_data):
        pass

class NakliLLM(Runnable):
    def __init__(self):
        print("LLM Created")

    def invoke(self, prompt):
        response_list = [
            'Islamabad is the Capital of Pakistan',
            'PSL is a cricket league',
            'AI stands for Artificial Intelligence'
            ]
        return random.choice(response_list)

    def predict(self, prompt):
        response_list = [
            'Islamabad is the Capital of Pakistan',
            'PSL is a cricket league',
            'AI stands for Artificial Intelligence'
        ]

        return random.choice(response_list)



class NakliPromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict)

    def format(self, input_dict):
        return self.template.format(**input_dict)


# class NakliStrOutputParser(Runnable):
#     def __init__(self):
#         pass

#     def invoke(self, input_data):
#         return input_data['response']


class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)

        return input_data

template = NakliPromptTemplate(
    template="Write a {length} poem about {topic}",
    input_variables=['length', 'topic']
)

llm = NakliLLM()
# parser = NakliStrOutputParser()

chain = RunnableConnector([template,llm])
response = chain.invoke({'length':'long', 'topic':'Pakistan'})
print(response)


