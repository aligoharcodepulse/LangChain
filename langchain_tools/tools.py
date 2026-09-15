from langchain_community.tools import DuckDuckGoSearchRun

#Built-in tools
search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke("Pakistan Super League News")
#print(results)


from langchain_community.tools import ShellTool

shell_tool = ShellTool()
results = shell_tool.invoke("whoami")
print(results)
