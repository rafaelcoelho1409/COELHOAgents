import streamlit as st
import re
from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph, START


###STRUCTURES
#Data model
class Code(BaseModel):
    """
    Schema for code solutions to questions about the programming language.
    """
    prefix: str = Field(description = """
        Description of the problem and approach""")
    imports: str = Field(description = """
        Code block import statements."""
        )
    code: str = Field(description = """
        Code block not including import statements.""" 
        )


class State(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries
    """
    error: str
    messages: List
    generation: str
    iterations: int
#---------------------------------------------

class SoftwareDeveloper:
    def __init__(self, framework, temperature_filter, model_name, shared_memory):
        self.shared_memory = shared_memory
        self.config = {
            "configurable": {"thread_id": "1"},
            "callbacks": [StreamlitCallbackHandler(st.container())]}
        self.llm_framework = {
            "Groq": ChatGroq,
            "Ollama": ChatOllama
        }
        self.llm_model = self.llm_framework[framework]
        self.llm = self.llm_model(
            model = model_name,
            temperature = temperature_filter
        )

    def load_model(self, language):
        self.language = language
        # Grader prompt
        # Grader prompt
        self.code_gen_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a coding assistant with expertise in the following language:  
                    \n ------- \n  {language} \n ------- \n 
                    Answer the user question based on the programming language. \n
                    Ensure any code you provide can be executed 
                    with all required imports and variables defined. \n 
                    Structure your answer with a description of the code solution. \n
                    Then list the imports. And finally list the functioning code block. \n
                    Format the code to be shown very organized in a markdown. \n
                    Here is the user question:
                    """,
                ),
                ("placeholder", "{messages}"),
            ]
        )
        self.code_gen_chain = self.code_gen_prompt | self.llm.with_structured_output(Code)
        # Max tries
        self.max_iterations = 3
        # Reflect
        # flag = 'reflect'
        self.flag = "do not reflect"
        self.workflow = StateGraph(State)
        ###NODES
        self.workflow.add_node("llm_invoke", self.llm_invoke)
        ###EDGES
        self.workflow.add_edge(START, "llm_invoke")
        self.workflow.add_edge("llm_invoke", END)
        self.graph = self.workflow.compile(checkpointer = self.shared_memory)

    ###Nodes
    def llm_invoke(self, state: State):
        st.chat_message("Tool").info("GENERATING CODE SOLUTION")
        messages = state["messages"]
        iterations = state["iterations"]
        error = state["error"]
        # We have been routed back to generation with an error
        if error == "yes":
            messages += [
                (
                    "user",
                    "Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:",
                )
            ]
        code_solution = self.code_gen_chain.invoke({
            "language": self.language,
            "messages": messages
        })
        messages += [
            (
                "assistant",
                f"""
                **Description:**\n{code_solution.prefix}\n
                **Imports:**\n```{self.language.lower()}\n{code_solution.imports}\n```\n
                **Code:**\n```{self.language.lower()}\n{code_solution.code}\n```\n
                """,
            )
        ]
        # Increment
        iterations = iterations + 1
        return {"generation": code_solution, "messages": messages, "iterations": iterations}

    def stream_graph_updates(self, language, user_input):
        # The config is the **second positional argument** to stream() or invoke()!
        events = self.graph.stream(
            {
                "language": language, 
                "messages": [("user", user_input)], 
                "iterations": 0, 
                "error": ""},
            self.config, 
            stream_mode = "values"
        )
        for event in events:
            #st.write(event["messages"])
            st.chat_message(
                event["messages"][-1][0]#.type
            ).markdown(
                event["messages"][-1][1]#.content
            )
        #response = self.graph.invoke(
        #        {
        #            "language": language, 
        #            "messages": [("user", user_input)], 
        #            "iterations": 0, 
        #            "error": ""},
        #        self.config
        #    )
        #st.chat_message("human").write(user_input)
        #st.chat_message("assistant").write(response)