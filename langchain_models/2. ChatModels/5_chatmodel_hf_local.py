from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
    },
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the name of President of the USA?")

print(result.content)