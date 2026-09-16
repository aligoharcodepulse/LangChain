from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

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
print(result)
print(multiply.name)
print(multiply.description)
print(multiply.args)