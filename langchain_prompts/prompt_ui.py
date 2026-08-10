from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512
)

model = ChatHuggingFace(llm=llm)

st.header('Research Tool')


paper_input = st.selectbox("Select Research Paper Name", ['Attention Is All You Need',
            'BERT: Pre-Training of Deep Bidirectional Transformers',
            'GPT-3: Language Models are Few Shot Learners', 
            'Diffusion Models Beat GANs on Image Synthesis'])

style_input = st.selectbox("Select Explanation Style", ['Beginner-Friendly', 'Technical',
            'Code Oriented', 'Mathematical'])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)",
            'Medium (3-5 paragraphs)', 'Long (detailes explanation)'])

template = load_prompt('template.json')

# Fill the placeholder
prompt = template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)
