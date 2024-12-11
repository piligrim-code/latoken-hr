import json
import openai
import os
from openai import OpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from vector import get_embedding_function

with open('prompts.json', 'r', encoding='utf-8') as file:
    prompts = json.load(file)

CHROMA_PATH = "chroma"

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


def query_chatgpt(prompt):
    client = OpenAI(
    api_key=openai_api_key, 
)
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system",
            "content": prompt}
        ],
        model="gpt-4o"
    )
    return response.choices[0].message.content

def search_and_respond(query_text):
    embedding_function = get_embedding_function()
    
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    results = db.similarity_search_with_score(query_text, k=5)
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt1_text = "\n".join(prompts['prompt1'])
    
    prompt_template = ChatPromptTemplate.from_template(prompt1_text)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    response_text = query_chatgpt(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    
    return response_text

