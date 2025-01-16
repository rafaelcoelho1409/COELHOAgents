import streamlit as st
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
        self.code_gen_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a coding assistant with expertise in the following language:  
                    \n ------- \n  {language} \n ------- \n 
                    Answer the user question based on the programming language. 
                    Ensure any code you provide can be executed \n 
                    with all required imports and variables defined. 
                    Structure your answer with a description of the code solution. \n
                    Then list the imports. And finally list the functioning code block.
                    Here is the user question:""",
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

    ### Nodes
    def generate(self, state: GraphState):
        """
        Generate a code solution

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation
        """
        st.chat_message("Tool").info("GENERATING CODE SOLUTION")
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
            {"language": self.language, "messages": messages}
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
        st.chat_message("Tool").info("CHECKING CODE")
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
            st.chat_message("Tool").error("CODE IMPORT CHECK: FAILED")
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
            st.chat_message("Tool").error("CODE BLOCK CHECK: FAILED")
            error_message = [("user", f"Your solution failed the code execution test: {e}")]
            messages += error_message
            return {
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": "yes",
            }
        # No errors
        st.chat_message("Tool").success("NO CODE TEST FAILURES")
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
        st.chat_message("Tool").info("GENERATING CODE SOLUTION")
        # State
        messages = state["messages"]
        iterations = state["iterations"]
        code_solution = state["generation"]
        # Prompt reflection
        # Add reflection
        reflections = self.code_gen_chain.invoke(
            {"language": self.language, "messages": messages}
        )
        messages += [("assistant", f"Here are reflections on the error: {reflections}")]
        return {"generation": code_solution, "messages": messages, "iterations": iterations}
    
    ### Edges
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
            st.chat_message("Tool").info("DECISION: FINISH")
            return "end"
        else:
            st.chat_message("Tool").info("DECISION: RE-TRY SOLUTION")
            if self.flag == "reflect":
                return "reflect"
            else:
                return "generate"

    def stream_graph_updates_test(self, language, user_input):
        st.chat_message("human").write(user_input)
        st.chat_message("assistant").write(
            self.graph.invoke(
                {
                    "language": language, 
                    "messages": [("user", user_input)], 
                    "iterations": 0, 
                    "error": ""},
                self.config
            )
        )

#How can I build a simple computer vision software using MediaPipe?