import streamlit as st
from bs4 import BeautifulSoup as Soup
from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START


###STRUCTURES
# Data model
class code(BaseModel):
    """Schema for code solutions to questions about LCEL."""
    prefix: str = Field(description = "Description of the problem and approach")
    imports: str = Field(description = "Code block import statements")
    code: str = Field(description = "Code block not including import statements")


class GraphState(TypedDict):
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

#--------------------------------------------------------------


class CodeGeneration:
    def __init__(self, framework, temperature_filter, model_name, shared_memory):
        self.concatenated_content = self.load_docs()
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
        # Max tries
        self.max_iterations = 3
        # Reflect
        # flag = 'reflect'
        self.flag = "do not reflect"

    def load_model(self):
        # Grader prompt
        self.code_gen_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a coding assistant with expertise in LCEL, LangChain expression language. \n 
            Here is a full set of LCEL documentation:  \n ------- \n  {context} \n ------- \n Answer the user 
            question based on the above provided documentation. Ensure any code you provide can be executed \n 
            with all required imports and variables defined. Structure your answer with a description of the code solution. \n
            Then list the imports. And finally list the functioning code block. Here is the user question:""",
                ),
                ("placeholder", "{messages}"),
            ]
        )
        self.code_gen_chain = self.code_gen_prompt | self.llm.with_structured_output(code)
        self.workflow = StateGraph(GraphState)
        # Define the nodes
        self.workflow.add_node("generate", self.generate)  # generation solution
        self.workflow.add_node("check_code", self.code_check)  # check code
        self.workflow.add_node("reflect", self.reflect)  # reflect
        # Build graph
        self.workflow.add_edge(START, "generate")
        self.workflow.add_edge("generate", "check_code")
        self.workflow.add_conditional_edges(
            "check_code",
            self.decide_to_finish,
            {
                "end": END,
                "reflect": "reflect",
                "generate": "generate",
            },
        )
        self.workflow.add_edge("reflect", "generate")
        self.graph = self.workflow.compile(checkpointer = self.shared_memory)

    def load_docs(self):
        # LCEL docs
        url = "https://python.langchain.com/docs/concepts/lcel/"
        loader = RecursiveUrlLoader(
            url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
        )
        docs = loader.load()

        # Sort the list based on the URLs and get the text
        d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
        d_reversed = list(reversed(d_sorted))
        concatenated_content = "\n\n\n --- \n\n\n".join(
            [doc.page_content for doc in d_reversed]
        )
        return concatenated_content
    
    ###---NODES---###
    def generate(self, state: GraphState):
        """
        Generate a code solution

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation
        """
        print("---GENERATING CODE SOLUTION---")
        # State
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
        # Solution
        code_solution = self.code_gen_chain.invoke(
            {"context": self.concatenated_content, "messages": messages}
        )
        messages += [
            (
                "assistant",
                f"{code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}",
            )
        ]
        # Increment
        iterations = iterations + 1
        return {"generation": code_solution, "messages": messages, "iterations": iterations}
    
    def code_check(self, state: GraphState):
        """
        Check code

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, error
        """
        print("---CHECKING CODE---")
        # State
        messages = state["messages"]
        code_solution = state["generation"]
        iterations = state["iterations"]
        # Get solution components
        imports = code_solution.imports
        code = code_solution.code
        # Check imports
        try:
            exec(imports)
        except Exception as e:
            print("---CODE IMPORT CHECK: FAILED---")
            error_message = [("user", f"Your solution failed the import test: {e}")]
            messages += error_message
            return {
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": "yes",
            }
        # Check execution
        try:
            exec(imports + "\n" + code)
        except Exception as e:
            print("---CODE BLOCK CHECK: FAILED---")
            error_message = [("user", f"Your solution failed the code execution test: {e}")]
            messages += error_message
            return {
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": "yes",
            }
        # No errors
        print("---NO CODE TEST FAILURES---")
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "no",
        }
    
    def reflect(self, state: GraphState):
        """
        Reflect on errors

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation
        """
        print("---GENERATING CODE SOLUTION---")
        # State
        messages = state["messages"]
        iterations = state["iterations"]
        code_solution = state["generation"]
        # Prompt reflection
        # Add reflection
        reflections = self.code_gen_chain.invoke(
            {"context": self.concatenated_content, "messages": messages}
        )
        messages += [("assistant", f"Here are reflections on the error: {reflections}")]
        return {"generation": code_solution, "messages": messages, "iterations": iterations}
    
    ###---EDGES---###
    def decide_to_finish(self, state: GraphState):
        """
        Determines whether to finish.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """
        error = state["error"]
        iterations = state["iterations"]
        if error == "no" or iterations == self.max_iterations:
            print("---DECISION: FINISH---")
            return "end"
        else:
            print("---DECISION: RE-TRY SOLUTION---")
            if self.flag == "reflect":
                return "reflect"
            else:
                return "generate"
    
    ###Others
    def stream_graph_updates(self, user_input: str):
        # The config is the **second positional argument** to stream() or invoke()!
        events = self.graph.stream(
            {"messages": [("user", user_input)], "iterations": 0, "error": ""}, 
            self.config, 
            stream_mode = "values"
        )
        for event in events:
            st.write(event)
            #st.chat_message(event["messages"][-1].type).markdown(event["messages"][-1].content)