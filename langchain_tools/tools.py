from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, StructuredTool, BaseTool
from pydantic import BaseModel, Field
from typing import Type

#Built-in tools
search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke("Pakistan Super League News")
#print(results)


from langchain_community.tools import ShellTool

shell_tool = ShellTool()
results = shell_tool.invoke("whoami")
#print(results)


# Custom Tools
# Step - 1 Create a Function
def multiply(a,b):
    """Multiply Two Numbers"""
    return a*b

# Step - 2 Add type hints
def multiply(a:int, b:int) -> int:
    """Multiply Two Numbers"""
    return a*b

# Step - 3 Add tool decorator
@tool   #LLM can communicate with this function
def multiply(a:int, b:int) -> int:
    """Multiply Two Numbers"""
    return a*b

result = multiply.invoke({"a":3,"b":5})
# print(result)
# print(multiply.name)
# print(multiply.description)
# print(multiply.args)



# Using Structured Tool
class MultiplyInput(BaseModel):
    a: int = Field(
        required=True,
        description="The first number to add"
    )
    b: int = Field(
        required=True,
        description="The first number to add"
    )


def multiply_func(a: int, b: int) -> int:
    return a * b


multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)

result = multiply_tool.invoke({"a":3,"b":3})
# print(result)
# print(multiply_tool.name)
# print(multiply_tool.description)
# print(multiply_tool.args)


# Using Base Tool
# arg schema using pydantic
class MultiplyInput(BaseModel):
    a: int = Field(
        required=True,
        description="The first number to add"
    )
    b: int = Field(
        required=True,
        description="The second number to add"
    )


class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"

    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b


multiply_tool = MultiplyTool()
result = result = multiply_tool.invoke({"a":6,"b":3})
print(result)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)