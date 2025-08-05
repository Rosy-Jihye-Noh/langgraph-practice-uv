##02_uv_exam -> export 한 것
# %%
from dotenv import load_dotenv
load_dotenv()

# %%
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# %%
from langchain_chroma import Chroma

#데이터를 처음 저장할 때
# vector_store = Chroma.from_documents(
#     documents=document_lists,
#     embedding=embedding,
#     collection_name = "chroma_tax",
#     persist_directory="./chroma_tax"
# )


vector_store = Chroma(
    collection_name = "chroma_tax",
    embedding_function=embedding,
    persist_directory="./chroma_tax"
)

retriever=vector_store.as_retriever(search_kwargs={"k":3})
retriever

# %%
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    query: str #질문
    context: List[Document]
    answer: str

graph_builder = StateGraph(AgentState)


# %%
#retrieve 노드 #state받고 state return
def retrieve(state: AgentState) -> AgentState: 
    """
    사용자의 질문에 기반하여 벡터 스토어에서 관련 문서를 검색합니다.
    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트의 현재 state
    
    Return:
        AgentState: 검색된 문서가 추가된 state를 반환합니다.
    """

    query = state['query']
    docs = retriever.invoke(query)
    return {'context': docs}


# %%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model = 'gpt-4o')


# %%
from langchain import hub

generate_prompt = hub.pull('rlm/rag-prompt')

#generate 노드
def generate(state: AgentState) -> AgentState:
    """
    주어진 state를 기반으로 RAG 체인을 사용하여 응답을 생성합니다.
    Args:
        state (AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state

    Returns:
        AgentState : 생성된 응답을 포함하는 state를 반환합니다.
    """
    context = state['context']
    query = state['query']
    
    rag_chain = generate_prompt | llm

    response = rag_chain.invoke({'question': query, 'context': context})
    #generate_prompt에 전달->llm에 전달

    return {'answer': response}

# %%
from langchain import hub
from typing import Literal

doc_relevance_prompt = hub.pull("langchain-ai/rag-document-relevance")
#문서의 관련성이 적으면 다시 retriver할 수 있도록 #generate와 rewirte 중 분기 결정
#check_doc_relevacne 노드
def check_doc_relevance(state : AgentState) -> Literal['generate', 'rewrite']:
    """
    주어진 state를 기반으로 문서의 관련성을 판단합니다.

    Args:
        state (AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state
    
    Returns:
        Literal['generate', 'rewrite']: 문서가 관련성이 높으면 'generate', 그렇지 않으면 'rewrite'를 반환합니다.
    """

    query = state['query']
    context = state['context']

    doc_relevance_chain = doc_relevance_prompt | llm

    response = doc_relevance_chain.invoke({'question': query, 'documents':context})
    
    if response['Score'] == 1:
        print('관련성 : relevant')
        return 'relevant'
    print('관련성 : irrelevant')
    return 'irrelevant'

# %%
from urllib import response
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

dictionary = ['사람과 관련된 표현 -> 거주자']

rewrite_prompt = PromptTemplate.from_template(f"""
사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
사전: {dictionary}
질문: {{query}} 
""") #f string이어서 중괄호 2개

#rewrite 노드
def rewrite(state: AgentState) -> AgentState:
    """
    사용자의 질문을 사전을 고려하여 변경합니다.

    Args:
        state (AgentState) : 사용자의 질문을 포함한 에이전트 현재 state

    Returns:
        AgentState : 변경된 질문을 포함한 state를 반환합니다.
    """

    query = state['query']

    rewrite_chain = rewrite_prompt | llm | StrOutputParser() #또 다른 질문을 생성해야하기에 string

    response = rewrite_chain.invoke({'query': query})

    return{'query': response} #질문 변경

# %%
#hallucination parser
#hallucination할 때 자유도는 0

from langchain_core.output_parsers import StrOutputParser

hallucination_prompt = PromptTemplate.from_template("""
You are a teacher tasked with evaluating whether a student's answer is based on documents or not,
Given documents, which are excerpts from income tax law, and a student's answer;
If the student's answer is based on documents, respond with "not hallucinated",
If the student's answer is not based on documents, respond with "hallucinated".

documents: {documents}
student_answer: {student_answer}
""")

hallucination_llm = ChatOpenAI(model='gpt-4o', temperature=0)
def check_hallucination(state: AgentState) -> Literal['hallucinated', 'not hallucinatied']:
    #state에는 answer와 context가 필요
    answer = state['answer'] #generate에서 만들어진 값
    context = state['context']
    context = [doc.page_content for doc in context]
    hallucination_chain = hallucination_prompt | hallucination_llm | StrOutputParser()
    response = hallucination_chain.invoke({'student_answer' : answer, 'documents': context})
    print('거짓말: ', response)

    return response #hallucination 통과한 것

# %%
# query = '연봉 5천만원 거주자의 소득세는 얼마인가요?'

# context = retriever.invoke(query)
# generate_state = {'query': query, 'context': context} 
# answer = generate(generate_state)
# #generate를 통해 만들어진 answer
# #검색된 context
# print(f'answer: {answer}')

# hallucination_state = {'answer': answer, 'context': context}
# #hallucination_state에 answer과 context를 넣음
# check_hallucination(hallucination_state) #그리고 check_hallucination 함수에 넣음


# %%
#answer이 query와 관련있는 것인지 체크
from langchain import hub

helpfulness_prompt = hub.pull('langchain-ai/rag-answer-helpfulness')

def check_helpfulness_grader(state: AgentState) -> str:
    """
    사용자의 질문에 기반하여 생성된 답변의 유용성을 평가합니다.
    
    Args:
        state (AgentState): 사용자의 질문과 생성된 답변을 포함한 에이전트의 현재 state
        
    Returns:
        str: 답변이 유용하다고 판단되면 'helpful', 그렇지 않으면 'unhelpful'을 반환합니다.
    """

    query = state['query']
    answer = state['answer']
    
    helpfulness_chain = helpfulness_prompt | llm
    response = helpfulness_chain.invoke({'question': query, 'student_answer': answer})

    if response['Score'] == 1:
        print('유용성: helpful')
        return 'helpful'

    print('유용성: unhelpful')
    return 'unhelpful'

def check_helpfulness(state: AgentState) -> AgentState:
    """
    유용성을 확인하는 자리 표시자 함수 입니다.
    graph에서 conditional_edge를 연속으로 사용하지 않고 node를 추가해
    가독성을 높이기 위해 사용합니다.

    Args:
        state (AgentState): 에이전트의 현재 state

    Returns:
        AgentState: 변경되지 않은 state를 반환합니다.
    """

    return state

# %%
#테스트
# query = '연봉 5천만원 거주자의 소득세는 얼마인가요?'

# context = retriever.invoke(query)
# generate_state = {'query': query, 'context': context} 
# answer = generate(generate_state)

# helpfullness_state = {'query': query, 'answer': answer}
# check_helpfulness_grader(helpfullness_state)

# %%
graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('generate', generate)
graph_builder.add_node('rewrite', rewrite)
graph_builder.add_node('check_helpfulness', check_helpfulness)

# %%
from langgraph.graph import START, END

graph_builder.add_edge(START, 'retrieve')
graph_builder.add_conditional_edges(
    'retrieve',
    check_doc_relevance, #검색이 관련 있는지 없는지
    {
        'relevant': 'generate',
        'irrelevant': END
    }
)
graph_builder.add_conditional_edges(
    'generate',
    check_hallucination,
    {
        'not hallucinated': 'check_helpfulness',
        'hallucinated': 'generate'
    }
)
graph_builder.add_conditional_edges(
    'check_helpfulness',
    check_helpfulness_grader,
    {
        'helpful': END,
        'unhelpful': 'rewrite'
    }
)

graph_builder.add_edge('rewrite', 'retrieve')



# # %%
graph = graph_builder.compile()

# # %%
# from IPython.display import Image, display

# display(Image(graph.get_graph().draw_mermaid_png()))

# # %%
# initial_state = {'query': '연봉 5천만원 직장인의 소득세는?'}
# graph.invoke(initial_state)

# %%



