import streamlit as st
import subprocess
import os
import re
from dotenv import load_dotenv
from typing import List, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START

load_dotenv()


###STRUCTURES
#Data model
class Code(BaseModel):
    """
    Schema for code solutions to questions about the programming language.
    """
    prefix: str = Field(description = """
        Description of the problem and approach""")
    filenames: List = Field(description = """
        File names for this code solution. 
        Can be one or more files.
        Names must be into a list of strings.""")
    requirements: List = Field(description = """
        Required imports for the code solution.
        It's mandatory to be in requirements file format to be ingested by the language runner.""")
    codes: List = Field(description = """
        Code block statements for each file to be created in the project. 
        Can be one or more code files, according to the file names.
        Codes must be into a list of strings.""" 
        )

class CodeBlock(BaseModel):
    """
    Represents the code block to be executed.

    Attributes:
        code : Code block
    """
    code: str

class State(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        error_message : Error message
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries
        technology : Programming language or technology stack
    """
    error: str
    error_message: str
    messages: List
    streamlit_actions: List
    generation: str
    iterations: int
    technology: str
    project_name: str
#---------------------------------------------

class SoftwareDeveloper:
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

    def load_model(self, technology, project_folder):
        self.project_folder = project_folder
        self.technology = technology
        self.code_gen_chain = self.build_code_generator()
        self.code_runner_chain = self.build_code_runner()
        # Max tries
        self.max_iterations = 3
        # Reflect
        # flag = 'reflect'
        self.flag = "do not reflect"
        self.workflow = StateGraph(State)
        ###NODES
        self.workflow.add_node("check_install", self.check_install)
        self.workflow.add_node("generate_code", self.generate_code)
        self.workflow.add_node("run_code", self.run_code)
        ###EDGES
        self.workflow.add_edge(START, "check_install")
        #self.workflow.add_edge(START, "generate_code")
        self.workflow.add_conditional_edges("check_install", self.check_install_error)
        self.workflow.add_edge("generate_code", "run_code")
        self.workflow.add_edge("run_code", END)
        #self.workflow.add_edge("generate_code", END)
        #self.workflow.add_edge("check_install", END)
        self.graph = self.workflow.compile(checkpointer = self.shared_memory)

    def build_code_generator(self):
        # Grader prompt
        code_gen_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a coding assistant with expertise in the following language:  
                    \n ------- \n  {technology} \n ------- \n 
                    Answer the user question based on the programming language. \n
                    Ensure any code you provide can be executed 
                    with all required imports and variables defined. \n 
                    Structure your answer with a description of the code solution. \n
                    Then list the imports in a requirements format file. \n 
                    And finally list the functioning code block. \n
                    Format the code to be shown very organized in a markdown. \n
                    Here is the user question:
                    """,
                ),
                ("placeholder", "{messages}"),
            ]
        )
        code_gen_chain = code_gen_prompt | self.llm.with_structured_output(
            Code,
            #method = "json_mode",
            #include_raw = True
            )
        return code_gen_chain
    
    def build_code_runner(self):
        code_runner_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a coding assistant with expertise in the following language:  
                    \n ------- \n  {technology} \n ------- \n
                    These are all files created: \n
                    {filenames} \n\n
                    Based on this technology, return all necessary commands to run the main file. \n
                    Return only the commands to be executed in the terminal,
                    only the commands inside a code block and nothing more. \n
                    Make sure the commands can be run in only one row. \n
                    In the commands below, you need to put "[...]/" before EVERY file names (a special attention here, please),\n
                    because this "[...]" will be replace by the real folder name using Python .replace method.
                    The commands need to be in the following standard format: \n
                    ```{technology}\nCOMMANDS HERE```\n\n
                    """,
                ),
                ("placeholder", "{messages}"),
            ]
        )
        code_runner_chain = code_runner_prompt | self.llm.with_structured_output(
            CodeBlock,
            #method = "json_mode",
            #include_raw = True
        )
        return code_runner_chain
    
    ###Nodes
    def check_install(self, state: State):
        messages = state["messages"]
        streamlit_actions = state["streamlit_actions"]
        error = state["error"]
        technology = state["technology"]
        check_install_commands = {
            "Python": "python3 --version",
            "Java": "java -version",
            "Go": "go version",
            "C++": "g++ --version",
            "C#": "dotnet --list-sdks",
            ".NET": "dotnet --version",
            "MySQL": "mysql --version",
            "PostgreSQL": "psql --version"
        }
        try:
            check_install = subprocess.run(
                check_install_commands[technology].split(),
                capture_output = True,
                text = True
            )
            error = "no"
        except FileNotFoundError as e:
            messages += [
                (
                    "assistant",
                    f"""
                    You need to install {technology} on your system.\n
                    Please follow the instructions on the official website.
                    """
                )
            ]
            streamlit_actions += ["error"]
            error = "yes"
        return {"messages": messages, "error": error}

    def check_install_error(self, state: State):
        error = state["error"]
        if error == "yes":
            return END
        else:
            return "generate_code"


    def generate_code(self, state: State):
        #st.chat_message("Tool").info("GENERATING CODE SOLUTION")
        messages = state["messages"]
        streamlit_actions = state["streamlit_actions"]
        iterations = state["iterations"]
        error = state["error"]
        # We have been routed back to generation with an error
        if error == "yes":
            messages += [
                (
                    "user",
                    """
                    Now, try again. 
                    Invoke the code tool to structure the output with a prefix, 
                    imports, and code block:""",
                )
            ]
            streamlit_actions += ["write"]
        code_solution = self.code_gen_chain.invoke({
            "technology": self.technology,
            "messages": messages
        })
        #SAVING FILES
        #ALERT: INSERT ALL THESE FILES INTO A NEW FOLDER TO MAKE THINGS ORGANIZED
        for filename, code in zip(code_solution.filenames, code_solution.codes):
            with open(self.project_folder / filename, "w") as file:
                file.write(code)
        #------------
        messages += [
            (
                "assistant",
                f"""
                **Description:**\n{code_solution.prefix}\n
                **Requirements:**\n```{code_solution.requirements}\n```\n
                """ + "---\n".join(
                [
                f"""
                **File Names:**\n{filename}\n
                **Codes:**\n```{self.technology.lower()}\n{code}\n```\n
                """
                for filename, code in zip(code_solution.filenames, code_solution.codes)],
                )
            )
        ]
        streamlit_actions += ["write"]
        # Increment
        iterations = iterations + 1
        return {"generation": code_solution, "messages": messages, "iterations": iterations}
    
    def run_code(self, state: State):
        messages = state["messages"]
        streamlit_actions = state["streamlit_actions"]
        code_solution = state["generation"]
        technology = state["technology"]
        code_runner = self.code_runner_chain.invoke({
            "technology": self.technology,
            "messages": messages,
            "filenames": code_solution.filenames
        })
        #st.write(code_runner)
        #st.stop()
        code_runner_content = code_runner.code#["code"]
        #for filename in code_solution.filenames:
        code_runner_content = code_runner_content.replace(
            "[...]", 
            str(self.project_folder)
            )
        messages += [
            (
                "assistant",
                f"""
                ```{technology}\n
                {code_runner_content}\n
                ```
                """
            )
        ]
        streamlit_actions += ["write"]
        command_status = subprocess.run(
            code_runner_content,
            shell = True,
            capture_output = True,
            text = True
        )
        if command_status.returncode == 0:
            messages += [
                (
                    "assistant",
                    f"""
                    The code was executed successfully.\n
                    Output: {command_status.stdout}
                    """
                )
            ]
            streamlit_actions += ["success"]
        else:
            messages += [
                (
                    "assistant",
                    f"""
                    The code was not executed successfully.\n
                    Output: {command_status.stderr}
                    """
                )
            ]
            print(code_runner_content)
            print(command_status.stderr)
            streamlit_actions += ["error"]
        return {"messages": messages}

    def stream_graph_updates(self, technology, project_name, user_input):
        # The config is the **second positional argument** to stream() or invoke()!
        events = self.graph.stream(
            {
                "messages": [("user", user_input)], 
                "streamlit_actions": ["write"],
                "iterations": 0, 
                "error": "",
                "error_message": "",
                "project_name": project_name,
                "technology": technology},
            self.config, 
            stream_mode = "values"
        )
        for event in events:
            st.chat_message(
                event["messages"][-1][0]#.type
            ).__getattribute__(event["streamlit_actions"][-1])(
                event["messages"][-1][1]#.content
            )