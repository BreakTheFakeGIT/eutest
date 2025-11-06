from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to French. Translate the user sentence.",
#     ),
#     ("human", "I love programming."),
# ]
# ai_msg = llm.invoke(messages)
# ai_msg

def test_plumm1():

    template = """Question: {question}

    Answer: Pomyslę i odpowiem po polsku."""

    prompt = ChatPromptTemplate.from_template(template)

    model = OllamaLLM(model="PRIHLOP/PLLuM:12b", temperature=0.9, max_tokens=1024)

    chain = prompt | model

    response = chain.invoke({"question": "Jakie są rodzaje podatków w Polsce?"})
    print(response)


if __name__ == "__main__":
    test_plumm1()