import streamlit as st 
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain_community.llms import GPT4All
from langchain_core.prompts import PromptTemplate
import os

# Define the base path
base_path = "C:/Users/Ce PC/AppData/Local/nomic.ai/GPT4All"
# Define the model name
model_name = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
# Combine the base path and model name to create the full path
PATH = os.path.join(base_path, model_name)
callbacks = [StreamingStdOutCallbackHandler()]

llm = GPT4All(model=PATH,callbacks=callbacks, verbose=True)

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)
llm_chain = LLMChain(prompt=prompt, llm=llm)

st.title('🦜🔗 GPT4ALL Y\'All')
st.info('This is using the MPT model!')
prompt = st.text_input('Enter your prompt here!')

if prompt: 
    response = llm_chain.run(prompt)
    print(response)
    st.write(response)

