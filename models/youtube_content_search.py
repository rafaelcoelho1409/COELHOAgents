import streamlit as st
import stqdm
import os
import uuid
import pandas as pd
from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from youtube_search import YoutubeSearch
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.text_splitter import TokenTextSplitter
from langgraph.graph import END, StateGraph, START
from neo4j import GraphDatabase
#from pytubefix import Search
from pytubefix.contrib.search import Search, Filter

load_dotenv()
#------------------------------------------------
###STRUCTURES
class SearchQuery(BaseModel):
    search_query: str = Field(
        #"An accurate search query for YouTube videos.",
        title = "Search Query",
        description = """
        Search query for YouTube videos. 
        Must be only 1 search query at the maximum.
        You must provide a search query with the best keywords that is accurate and concise.
        It can be an extense search query if necessary, but it must be only one.
        Give the best and most accurate search query you can.
        """,
        example = "How to make a cake"
    )


# Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""
    names: List[str] = Field(
        ...,
        description = "All the person, organization, or business entities that "
        "appear in the text",
    )
    

class State(TypedDict):
    error: str
    messages: List
    streamlit_actions: List
    user_input: str
    search_results: List
    unique_videos: List
#------------------------------------------------

class YouTubeContentSearch:
    def __init__(self, framework, temperature_filter, model_name, shared_memory):
        self.shared_memory = shared_memory
        self.config = {
            "configurable": {"thread_id": "1"},
            "callbacks": [StreamlitCallbackHandler(st.container())]}
        self.llm_framework = {
            "Groq": ChatGroq,
            "Ollama": ChatOllama,
            "Google Generative AI": ChatGoogleGenerativeAI,
            "SambaNova": ChatSambaNovaCloud,
            "Scaleway": ChatOpenAI
        }
        self.llm_model = self.llm_framework[framework]
        if framework == "Scaleway":
            self.llm = ChatOpenAI(
                base_url = os.getenv("SCW_GENERATIVE_APIs_ENDPOINT"),
                api_key = os.getenv("SCW_SECRET_KEY"),
                model = model_name,
                temperature =  temperature_filter
            )
        else:
            self.llm = self.llm_model(
                model = model_name,
                temperature = temperature_filter,
            )

    def load_model(self, max_results, search_type, upload_date, video_type, duration, features, sort_by):
        self.max_results = max_results
        self.search_type = search_type
        self.upload_date = upload_date
        self.video_type = video_type
        self.duration = duration
        self.features = features
        self.sort_by = sort_by
        #clearing all previous Neo4J relationships to avoid context confusion
        self.clear_neo4j_graph()
        #------------------------------------------------
        self.neo4j_graph = Neo4jGraph()
        self.vector_index = Neo4jVector.from_existing_graph(
            HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2"),
            search_type = "hybrid",
            node_label = "Document",
            text_node_properties = ["text"],
            embedding_node_property = "embedding"
        )
        self.youtube_search_agent = self.build_youtube_search_agent()
        self.entity_chain = self.build_entity_chain()
        self.rag_chain = self.build_rag_chain()
        self.llm_transformer = LLMGraphTransformer(llm = self.llm)
        self.workflow = StateGraph(State)
        ###NODES
        self.workflow.add_node("search_youtube_videos", self.search_youtube_videos)
        self.workflow.add_node("set_knowledge_graph", self.set_knowledge_graph)
        self.workflow.add_node("final_step", self.final_step)
        ###EDGES
        self.workflow.add_edge(START, "search_youtube_videos")
        self.workflow.add_edge("search_youtube_videos", "set_knowledge_graph")
        self.workflow.add_edge("set_knowledge_graph", "final_step")
        self.workflow.add_edge("final_step", END)
        self.graph = self.workflow.compile(
            checkpointer = st.session_state["shared_memory"],#self.shared_memory
            #interrupt_before = ["chatbot_loop"],
        )
    
    ###AGENTS
    def build_youtube_search_agent(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a YouTube search agent.\n
                    Based on the following prompt:\n\n
                    {user_input}\n\n
                    You must take this user input and transform it into 
                    the most efficient youtube search query you can.\n
                    It must be in a string format.\n
                    """,
                ),
                ("placeholder", "{messages}"),
            ]
        )
        chain = prompt | self.llm.with_structured_output(SearchQuery)
        return chain
    
    def build_entity_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are extracting organization and person entities from the text.",
                ),
                (
                    "human",
                    "Use the given format to extract information from the following "
                    "input: {question}",
                ),
            ]
        )
        entity_chain = prompt | self.llm.with_structured_output(Entities)
        return entity_chain
    
    def build_rag_chain(self):
        # Condense a chat history and follow-up question into a standalone question
        _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
            in its original language.
            Chat History:
            {chat_history}
            Follow Up Input: {question}
            Standalone question:"""  # noqa: E501
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
        _search_query = RunnableBranch(
            # If input includes chat_history, we condense it with the follow-up question
            (
                RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                    run_name = "HasChatHistoryCheck"
                ),  # Condense follow-up question and chat into a standalone_question
                RunnablePassthrough.assign(
                    chat_history = lambda x: self._format_chat_history(x["chat_history"])
                )
                | CONDENSE_QUESTION_PROMPT
                | self.llm#ChatOpenAI(temperature=0)
                | StrOutputParser(),
            ),
            # Else, we have no chat history, so just pass through the question
            RunnableLambda(lambda x : x["question"]),
        )
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        Use natural language and be concise.
        Answer:"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            RunnableParallel(
                {
                    "context": _search_query | self.retriever,
                    "question": RunnablePassthrough(),
                }
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    #------------------------------------------------
    ###TOOLS
    def generate_full_text_query(self, input):
        """
        Generate a full-text search query for a given input string.

        This function constructs a query string suitable for a full-text search.
        It processes the input string by splitting it into words and appending a
        similarity threshold (~2 changed characters) to each word, then combines
        them using the AND operator. Useful for mapping entities from user questions
        to database values, and allows for some misspelings.
        """
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()
    
    # Fulltext index query
    def structured_retriever(self, question):
        """
        Collects the neighborhood of entities mentioned
        in the question
        """
        result = ""
        entities = self.entity_chain.invoke({"question": question})
        for entity in entities.names:
            response = self.neo4j_graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL {
                  WITH node
                  MATCH (node)-[r:!MENTIONS]->(neighbor)
                  RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                  UNION ALL
                  WITH node
                  MATCH (node)<-[r:!MENTIONS]-(neighbor)
                  RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": self.generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response])
        return result
    
    def retriever(self, question):
        print(f"Search query: {question}")
        structured_data = self.structured_retriever(question)
        unstructured_data = [el.page_content for el in self.vector_index.similarity_search(question)]
        final_data = f"""Structured data:
            {structured_data}
            Unstructured data:
            {"#Document ". join(unstructured_data)}
        """
        return final_data
    
    def _format_chat_history(self, chat_history):
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content = human))
            buffer.append(AIMessage(content = ai))
        return buffer
    
    def clear_neo4j_graph(self):
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"), 
            auth = (
                os.getenv("NEO4J_USERNAME"), 
                os.getenv("NEO4J_PASSWORD")
                )
            )
        with driver.session(database = "neo4j") as session:
            session.run("MATCH (n) DETACH DELETE n")

    #------------------------------------------------
    ###NODES
    def search_youtube_videos(self, state: State):
        messages = state["messages"]
        streamlit_actions = state["streamlit_actions"]
        user_input = state["user_input"]
        streamlit_action = []
        youtube_search_query = self.youtube_search_agent.invoke({
            "user_input": user_input
        })
        search_query = [youtube_search_query.search_query]
        if self.search_type == "Search":
            search_results_dict, search_filters_dict = {}, {}
            search_filters_dict_ = {
                "upload_date": self.upload_date,
                "type": self.video_type,
                "duration": self.duration,
                "features": self.features,
                "sort_by": self.sort_by
            }
            for key, value in search_filters_dict_.copy().items():
                if value in [[], None]:
                    search_filters_dict_.pop(key)
            for key, value in search_filters_dict_.items():
                if key == "features":
                    search_filters_dict[key] = [Filter.get_features(x) for x in value]
                else:
                    search_filters_dict__ = {
                        "upload_date": (Filter.get_upload_date, self.upload_date),
                        "type":        (Filter.get_type,        self.video_type),
                        "duration":    (Filter.get_duration,    self.duration),
                        "sort_by":     (Filter.get_sort_by,     self.sort_by)      
                    }
                    search_filters_dict[key] = search_filters_dict__[key][0](search_filters_dict__[key][1])
            for query in stqdm.stqdm(search_query, desc = "Searching YouTube videos"):
                search_results = Search(
                    query,
                    filters = search_filters_dict).videos[:self.max_results]
                search_results_dict[query] = {
                    "title": [video.title for video in search_results],
                    "author": [video.author for video in search_results],
                    "publish_date": [video.publish_date for video in search_results],
                    "views": [video.views for video in search_results],
                    "length": [video.length for video in search_results],
                    "captions": [str(list(video.captions.lang_code_index.keys())) for video in search_results],
                    "keywords": [video.keywords for video in search_results],
                    "description": [video.description for video in search_results],
                    "video_id": [video.video_id for video in search_results],
                }
        messages += [
            (
                "assistant",
                list(search_results_dict.keys())
            )
        ]
        streamlit_action += [(
            "markdown", 
            {
                "body": "- " + "\n- ".join(str(x) for x in messages[-1][1]), 
                #"expanded": True
                },
            ("Youtube search query", False),
            messages[-1][0],
            )]
        unique_videos_df = pd.concat(
            [pd.DataFrame(search_results_dict[key]) for key in search_results_dict.keys()],
            axis = 0)
        unique_videos = unique_videos_df[~unique_videos_df["video_id"].duplicated()].to_dict()
        messages += [
            (
                "assistant",
                unique_videos
            )
        ]
        streamlit_action += [(
            "dataframe", 
            {
                "data": messages[-1][1],
                "use_container_width": True,
                #"expanded": False
                },
            ("Youtube videos searched", False),
            messages[-1][0],
            )]
        streamlit_actions += [streamlit_action]
        return {
            "messages": messages,
            "streamlit_actions": streamlit_actions,
            #"search_results": search_results,
            "unique_videos": unique_videos
        }
    
    def set_knowledge_graph(self, state: State):
        messages = state["messages"]
        streamlit_actions = state["streamlit_actions"]
        unique_videos = state["unique_videos"]
        streamlit_action = []
        transcripts_ids = unique_videos["video_id"].values()#[video["id"] for video in unique_videos]
        transcriptions = {}
        for video_id in stqdm.stqdm(transcripts_ids, desc = "Getting YouTube videos transcripts"):
            try:
                transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
                for transcript in transcripts:
                    transcription = YouTubeTranscriptApi.get_transcript(
                        video_id,
                        languages = [transcript.language_code])
                    transcriptions[video_id] = Document(
                        page_content = " ".join([line["text"] for line in transcription]))
            except:
                pass
        #Building Knowledge Graphs
        text_splitter = TokenTextSplitter(chunk_size = 512, chunk_overlap = 24)
        documents = text_splitter.split_documents(transcriptions.values())
        #Transforming documents to graphs take a little more time, we need better ways to make it faster
        graph_documents = []
        for document in stqdm.stqdm(documents, desc = "Transforming documents to graphs"):
            graph_documents += self.llm_transformer.convert_to_graph_documents([document])
        #graph_documents = self.llm_transformer.convert_to_graph_documents(documents)
        #Graph nodes
        nodes, nodes_dict = [], {}
        for x in graph_documents:
            nodes += x.nodes
        nodes_dict["id"] = [x.id for x in nodes]
        nodes_dict["type"] = [x.type for x in nodes]
        nodes_dict["properties"] = [x.properties for x in nodes]
        messages += [
            (
                "assistant",
                #[x.nodes for x in graph_documents]
                nodes_dict
            )
        ]
        streamlit_action += [(
            "dataframe", 
            {
                "data": messages[-1][1], 
                "use_container_width": True,
                #"expanded": False
                },
            ("Graph nodes", False),
            messages[-1][0],
            )]
        #Graph relationships
        relationships, relationships_dict = [], {}
        for x in graph_documents:
            relationships += x.relationships
        relationships_dict["source_id"] = [x.source.id for x in relationships]
        relationships_dict["source_type"] = [x.source.type for x in relationships]
        relationships_dict["source_properties"] = [x.source.properties for x in relationships]
        relationships_dict["target_id"] = [x.target.id for x in relationships]
        relationships_dict["target_type"] = [x.target.type for x in relationships]
        relationships_dict["target_properties"] = [x.target.properties for x in relationships]
        relationships_dict["type"] = [x.type for x in relationships]
        relationships_dict["properties"] = [x.properties for x in relationships] 
        messages += [
            (
                "assistant",
                #[x.relationships for x in graph_documents]
                relationships_dict
            )
        ]
        streamlit_action += [(
            "dataframe", 
            {
                "data": messages[-1][1], 
                "use_container_width": True,
                #"expanded": False
                },
            ("Graph relationships", False),
            messages[-1][0],
            )]
        #Graph source
        messages += [
            (
                "assistant",
                [x.source.page_content for x in graph_documents]
            )
        ]
        streamlit_action += [(
            "markdown", 
            {
                "body": "\n\n---\n\n".join(x for x in messages[-1][1]), 
                #"expanded": False
                },
            ("Graph source", False),
            messages[-1][0],
            )]
        self.neo4j_graph.add_graph_documents(
            graph_documents,
            baseEntityLabel = True,
            include_source = True
        )
        # Retriever
        self.neo4j_graph.query(
            "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
        streamlit_actions += [streamlit_action]
        return {
            "messages": messages,
            "streamlit_actions": streamlit_actions,
        }
    
    def final_step(self, state: State):
        messages = state["messages"]
        streamlit_actions = state["streamlit_actions"]
        user_input = state["user_input"]
        streamlit_action = []
        question_answer = self.rag_chain.invoke({"question": user_input})
        messages += [
            (
                "assistant",
                question_answer
            )
        ]
        streamlit_action += [(
            "markdown", 
            {"body": question_answer},
            ("Assistant response", True),
            "assistant"
            )]
        streamlit_actions += [streamlit_action]
        return {
                "messages": messages,
                "streamlit_actions": streamlit_actions,
                "user_input": user_input
            }



    #------------------------------------------------
    def stream_graph_updates(self, user_input):
        # The config is the **second positional argument** to stream() or invoke()!
        events = self.graph.stream(
            {
                "messages": [("user", user_input)], 
                "streamlit_actions": [[(
                    "markdown", 
                    {"body": user_input},
                    ("User request", True),
                    "user"
                    )]],
                "error": "",
                "user_input": user_input,
                "search_results": {},
                },
            self.config, 
            stream_mode = "values"
        )
        for event in events:
            actions = event["streamlit_actions"][-1]
            if actions != []:
                for action in actions:
                    st.chat_message(
                        action[3]
                    ).expander(
                        action[2][0], 
                        expanded = action[2][1]
                    ).__getattribute__(
                        action[0]
                    )(
                        **action[1]
                    )



class YouTubeChatbot:
    def __init__(self, framework, temperature_filter, model_name, shared_memory):
        self.shared_memory = shared_memory
        self.config = {
            "configurable": {"thread_id": "1"},
            "callbacks": [StreamlitCallbackHandler(st.container())]}
        self.llm_framework = {
            "Groq": ChatGroq,
            "Ollama": ChatOllama,
            "Google Generative AI": ChatGoogleGenerativeAI,
            "SambaNova": ChatSambaNovaCloud,
            "Scaleway": ChatOpenAI
        }
        self.llm_model = self.llm_framework[framework]
        if framework == "Scaleway":
            self.llm = ChatOpenAI(
                base_url = os.getenv("SCW_GENERATIVE_APIs_ENDPOINT"),
                api_key = os.getenv("SCW_SECRET_KEY"),
                model = model_name,
                temperature =  temperature_filter
            )
        else:
            self.llm = self.llm_model(
                model = model_name,
                temperature = temperature_filter,
            )

    def load_model(self, rag_chain):
        ###GRAPH
        self.rag_chain = rag_chain
        self.graph_builder = StateGraph(State)
        self.graph_builder.add_node("chatbot", self.chatbot)
        self.graph_builder.add_edge(START, "chatbot")
        self.graph_builder.add_edge("chatbot", END)
        self.graph = self.graph_builder.compile(
            checkpointer = self.shared_memory
        )

    def chatbot(self, state: State):
        messages = state["messages"]
        streamlit_actions = state["streamlit_actions"]
        user_input = state["user_input"]
        streamlit_action = []
        messages += [
            (
                "user",
                user_input
            )
        ]
        streamlit_action += [(
            "markdown", 
            {"body": user_input},
            ("User request", True),
            "user"
            )]
        answer = self.rag_chain.invoke({"question": user_input})
        messages += [
            (
                "assistant",
                answer
            )
        ]
        streamlit_action += [(
            "markdown", 
            {"body": answer},
            ("Assistant response", True),
            "assistant"
            )]
        streamlit_actions += [streamlit_action]
        return {
            "messages": messages,
            "streamlit_actions": streamlit_actions,
            "user_input": user_input
        }
    
    def stream_graph_updates(self, user_input):
        # The config is the **second positional argument** to stream() or invoke()!
        events = self.graph.stream(
            {
                "error": "",
                "user_input": user_input,
                "search_results": {}
                },
            self.config, 
            stream_mode = "values"
        )
        for i, event in enumerate(events):
            if i != 0:
                actions = event["streamlit_actions"][-1]
                if actions != []:
                    for action in actions:
                        st.chat_message(
                            action[3]
                        ).expander(
                            action[2][0], 
                            expanded = action[2][1]
                        ).__getattribute__(
                            action[0]
                        )(
                            **action[1]
                        )

