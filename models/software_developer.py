import streamlit as st
import subprocess
import os
import json
import stqdm
from dotenv import load_dotenv
from typing import List, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
from langchain_community.utilities import StackExchangeAPIWrapper
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START

load_dotenv()

#---------------------------------------------


###STRUCTURES
class CodeRequirements(BaseModel):
    dependencies_commands: str = Field(description = """
        Represents the code block to be executed.
        Commands to install the dependencies.""")
        

#Data model
class CodeGeneration(BaseModel):
    """
    Schema for code solutions to questions about the programming language.
    """
    project_name: str = Field(description = """
        Name of the project""")
    prefix: str = Field(description = """
        Description of the problem and approach""")
    filenames: List = Field(description = """
        File names for this code solution. 
        Can be one or more files.
        Names must be into a list of strings.""")
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


class CodeError(BaseModel):
    error_message: str = Field(description = """
        Summarized error message in only one row, to be searched on StackOverflow API.""" 
        )


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
    messages: List
    streamlit_actions: List
    generation: str
    #dependencies_command: str
    iterations: int
    technology: str
    project_name: str
    command: str
    command_error: str
    #error_search_term: str
    #error_search_results: str
    #iterations_search: int
#---------------------------------------------

class SoftwareDeveloper:
    def __init__(self, framework, temperature_filter, model_name, shared_memory):
        self.shared_memory = shared_memory
        self.config = {
            "configurable": {"thread_id": "1"},
            "callbacks": [StreamlitCallbackHandler(st.container())]}
        with open("technologies.json") as file:
            self.technologies_json = json.load(file)
        self.technologies_json = dict(sorted(self.technologies_json.items()))
        self.llm_framework = {
            "Groq": ChatGroq,
            "Ollama": ChatOllama,
            "Google Generative AI": ChatGoogleGenerativeAI,
            "SambaNova": ChatSambaNovaCloud,
            "Scaleway": ChatOpenAI,
            "OpenAI": ChatOpenAI,
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
            try:
                self.llm = self.llm_model(
                    model = model_name,
                    temperature = temperature_filter,
                )
            except:
                self.llm = self.llm_model(
                    model = model_name,
                    #temperature = temperature_filter,
                )

    def load_model(self, technology, project_folder):
        self.project_folder = project_folder
        self.technology = technology
        self.code_gen_chain = self.build_code_generator()
        self.code_runner_chain = self.build_code_runner()
        self.dep_checker_chain = self.build_dependencies_checker()
        self.code_runner_correct_chain = self.build_code_runner_correct_search()
        # Max tries
        self.max_iterations = 3
        # Reflect
        # flag = 'reflect'
        self.flag = "do not reflect"
        self.workflow = StateGraph(State)
        ###NODES
        self.workflow.add_node("check_install", self.check_install)
        self.workflow.add_node("generate_code", self.generate_code)
        self.workflow.add_node("check_dependencies", self.check_dependencies)
        self.workflow.add_node("run_code", self.run_code)
        #self.workflow.add_node("correct_run_code_search_term", self.correct_run_code_search_term)
        #self.workflow.add_node("correct_run_code_search_results", self.correct_run_code_search_results)
        ###EDGES
        self.workflow.add_edge(START, "check_install")
        self.workflow.add_conditional_edges("check_install", self.check_install_error)
        self.workflow.add_edge("generate_code", "check_dependencies")
        self.workflow.add_edge("check_dependencies", "run_code")
        #self.workflow.add_conditional_edges("run_code", self.search_error_online)
        #self.workflow.add_edge("correct_run_code_search_term", "correct_run_code_search_results")
        #self.workflow.add_edge("correct_run_code_search_results", END)
        self.workflow.add_edge("run_code", END)
        #self.workflow.add_edge("check_install", END)
        #self.workflow.add_edge("check_dependencies", END)
        self.graph = self.workflow.compile(
            checkpointer = st.session_state["shared_memory"]#self.shared_memory
        )
    
    def build_code_generator(self):
        # Grader prompt
        code_gen_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a coding assistant with expertise in the following language:  
                    \n ------- \n  {technology} \n ------- \n 
                    ### Instructions about code solution generation ### \n
                    Answer the user question 
                    based on the programming language. \n
                    Ensure any code you provide can be executed 
                    with all required imports and variables defined. \n 
                    Structure your answer with a description of the code solution. \n
                    The project name must be in a folder name format, Pascal case specifically. \n
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
            CodeGeneration,
            )
        return code_gen_chain
    
    def build_dependencies_checker(self):
        dep_checker_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    ### Instructions about dependencies install ###\n
                    You are a coding assistant with expertise in the following language:\n
                    **{technology}**\n
                    \n
                    Based on the generated code below:\n\n
                    {code}\n\n 
                    Your task is to:\n
                    1. Analyze and return all necessary dependencies to run the main file.\n
                    2. Provide these dependencies in the format of a `requirements` file, which lists each dependency on a new line.\n
                    3. List the file names for these dependencies in a list of strings.\n
                    4. Generate the terminal commands to install the dependencies. The commands should:\n
                       - Be in a single row.\n
                       - Use the standard format for the specified technology.\n
                    \n
                    5. Always put "[...]" before all file names, because this "[...]" block 
                    will be replaced by the project folder path when the code is executed.\n
                    6. If technology is Python, >>>it's mandatory to use uv commands 
                    along with pip (virtual enviroment creation and requirements install)<<< 
                    instead of pure pip to avoid errors like 
                    "This environment is externally managed".\n
                    Example: ```plaintext\n
                    uv venv $FOLDER_NAME/.venv && 
                    source $FOLDER_NAME/.venv/bin/activate &&
                    uv pip install numpy pandas\n
                    ```\n
                    ```\n
                    """
                ),
                ("placeholder", "{messages}"),
            ]
        )
        dep_checker_chain = dep_checker_prompt | self.llm.with_structured_output(
            CodeRequirements,
        )
        return dep_checker_chain

    def build_code_runner(self):
        code_runner_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a coding assistant with expertise in the following language:  
                    \n ------- \n  {technology} \n ------- \n 
                    Answer the user question based on the programming language. \n
                    Ensure any code you provide can be executed 
                    with all required imports and variables defined. \n 
                    Based on the generated code below:\n\n
                    {code}\n\n 
                    Your task is to:\n
                    1. Generate the terminal commands to install the dependencies. The commands should:\n
                       - Be in a single row.\n
                       - Use the standard format for the specified technology.\n
                    \n
                    2. Always put "[...]" before all file names, because this "[...]" block 
                    will be replaced by the project folder path when the code is executed.\n
                    3. If technology is Python, >>>it's mandatory to use uv commands 
                    along with pip (virtual enviroment creation and requirements install)<<< 
                    instead of pure pip to avoid errors like 
                    "This environment is externally managed".\n
                    Example: ```plaintext\n
                    uv venv $FOLDER_NAME/.venv && 
                    source $FOLDER_NAME/.venv/bin/activate &&
                    uv pip install numpy pandas &&
                    python3 $PYTHON_FILE_TO_BE_EXECUTED\n
                    ```\n
                    """,
                ),
                ("placeholder", "{messages}"),
            ]
        )
        code_runner_chain = code_runner_prompt | self.llm.with_structured_output(
            CodeBlock,
        )
        return code_runner_chain
    
    def build_code_runner_correct_search(self):
        code_runner_correct_search_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a coding assistant with expertise in the following language:  
                    \n ------- \n  {technology} \n ------- \n 
                    Answer the user question based on the programming language. \n
                    Ensure any code you provide can be executed 
                    with all required imports and variables defined. \n 
                    Based on the code executed below:\n\n
                    {command}\n\n 
                    And based on the error after the code execution:\n\n
                    {command_error}\n\n 
                    Your task is to:\n
                    1. Summarize the error message in only one row, 
                    to be searched on StackOverflow in its most optimized way.\n
                    """,
                ),
                ("placeholder", "{messages}"),
            ]
        )
        #stack_api = StackExchangeAPIWrapper()
        code_runner_correct_chain = code_runner_correct_search_prompt | self.llm.with_structured_output(
            CodeBlock,
        )
        return code_runner_correct_chain
    
    ###Nodes
    def check_install(self, state: State):
        messages = state["messages"]
        streamlit_actions = state["streamlit_actions"]
        error = state["error"]
        technology = state["technology"]
        streamlit_action = []
        check_install_commands = self.technologies_json
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
            streamlit_action += [(
                "error", 
                {"body": messages[-1][1]},
                ("Error", True),
                messages[-1][0],
                )]
            error = "yes"
        streamlit_actions += [streamlit_action]
        return {
            "messages": messages, 
            "streamlit_actions": streamlit_actions,
            "error": error}

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
        streamlit_action = []
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
            streamlit_action += [(
                "markdown", 
                {"body": messages[-1][1]},
                ("", True),
                messages[-1][0],
                )]
        code_solution = self.code_gen_chain.invoke({
            "technology": self.technology,
            "messages": messages
        })
        #SAVING FILES
        #ALERT: INSERT ALL THESE FILES INTO A NEW FOLDER TO MAKE THINGS ORGANIZED
        os.makedirs(
            os.path.join(
                str(self.project_folder), 
                code_solution.project_name), 
                exist_ok = True
                )
        for filename, code in zip(code_solution.filenames, code_solution.codes):
            with open(self.project_folder / os.path.join(
                code_solution.project_name, 
                filename), "w") as file:
                file.write(code)
        #------------
        messages += [
            (
                "assistant",
                f"""
                **Project Name:**\n{code_solution.project_name}\n
                """
            )
        ]
        streamlit_action += [(
            "markdown", 
            {"body": messages[-1][1]},
            ("Project Name", True),
            messages[-1][0],
            )]
        messages += [
            (
                "assistant",
                f"""
                **Description:**\n{code_solution.prefix}\n
                """
            )
        ]
        streamlit_action += [(
            "markdown", 
            {"body": messages[-1][1]},
            ("Code description", True),
            messages[-1][0],
            )]
        for filename, code in zip(code_solution.filenames, code_solution.codes):
            messages += [
                (
                    "assistant",
                    f"""
                    **Codes:**\n```{self.technology.lower()}\n{code}\n```\n
                    """
                )
            ]
            streamlit_action += [(
                "markdown", 
                {"body": messages[-1][1]},
                (filename, True),
                messages[-1][0],
                )]
        # Increment
        iterations = iterations + 1
        streamlit_actions += [streamlit_action]
        return {
            "generation": code_solution, 
            "messages": messages, 
            "streamlit_actions": streamlit_actions,
            "iterations": iterations}
    
    def check_dependencies(self, state: State):
        messages = state["messages"]
        streamlit_actions = state["streamlit_actions"]
        error = state["error"]
        code_solution = state["generation"]
        streamlit_action = []
        dependencies = self.dep_checker_chain.invoke({
            "technology": self.technology,
            "code": code_solution.codes,
            "messages": messages,
        })
        dependencies_commands = dependencies.dependencies_commands.replace(
            "[...]", 
            os.path.join(
                str(self.project_folder), 
                code_solution.project_name
                )
            )
        # We have been routed back to dependencies check with an error
        if error == "yes":
            messages += [
                (
                    "user",
                    """
                    Now, try again. 
                    Invoke the code tool to structure the output with requirements
                    and dependencies install commands:""",
                )
            ]
            streamlit_action += [(
                "markdown", 
                {"body": messages[-1][1]},
                ("", True),
                messages[-1][0],
                )]
        messages += [
            (
                "assistant",
                f"""
                **Dependencies install commands:**\n
                ```{dependencies_commands}\n```\n
                """
            )
        ]
        streamlit_action += [(
            "markdown", 
            {"body": messages[-1][1]},
            ("Dependencies install commands", True),
            messages[-1][0],
        )]
        streamlit_actions += [streamlit_action]
        return {
            "messages": messages, 
            "streamlit_actions": streamlit_actions,
            "error": error}
    
    def run_code(self, state: State):
        messages = state["messages"]
        streamlit_actions = state["streamlit_actions"]
        code_solution = state["generation"]
        technology = state["technology"]
        error = state["error"]
        streamlit_action = []
        code_runner = self.code_runner_chain.invoke({
            "technology": self.technology,
            "messages": messages,
            #"filenames": code_solution.filenames,
            "code": code_solution.codes,
        })
        code_runner_content = code_runner.code#["code"]
        code_runner_content = code_runner_content.replace(
            "[...]", 
            str(self.project_folder)
            )
        messages += [
            (
                "assistant",
                f"""
                Suggested code to be executed on the generated folder (terminal):\n\n
                ```{technology}\n
                {code_runner_content}\n
                ```
                """
            )
        ]
        streamlit_action += [(
            "info", 
            {"body": messages[-1][1]},
            ("Run code suggestion", True),
            messages[-1][0],
            )]
        #command_status = subprocess.run(
        #    code_runner_content,
        #    shell = True,
        #    capture_output = True,
        #    text = True
        #)
        #if command_status.returncode == 0:
        #    messages += [
        #        (
        #            "assistant",
        #            f"""
        #            The code was executed successfully.\n
        #            Output: {command_status.stdout}
        #            """
        #        )
        #    ]
        #    streamlit_action += [(
        #        "success", 
        #        {"body": messages[-1][1]},
        #        ("Success", True),
        #        messages[-1][0],
        #        )]
        #    error = "no"
        #else:
        #    messages += [
        #        (
        #            "assistant",
        #            f"""
        #            The code was not executed successfully.\n
        #            Output: {command_status.stderr}
        #            """
        #        )
        #    ]
        #    command_error = command_status.stderr
        #    print(code_runner_content)
        #    print(command_status.stderr)
        #    streamlit_action += [(
        #        "error", 
        #        {"body": messages[-1][1]},
        #        ("Error", True),
        #        messages[-1][0],
        #        )]
        #    error = "yes"
        streamlit_actions += [streamlit_action]
        return {
            "messages": messages,
            "streamlit_actions": streamlit_actions,
            "error": error,
            "command": code_runner_content,
            #"command_error": command_error
            }
    
    #def search_error_online(self, state: State):
    #    error = state["error"]
    #    if error == "yes":
    #        return "correct_run_code_search_term"
    #    else:
    #        return END
    
    #def correct_run_code_search_term(self, state: State):
    #    messages = state["messages"]
    #    streamlit_actions = state["streamlit_actions"]
    #    technology = state["technology"]
    #    command = state["command"]
    #    command_error = state["command_error"]
    #    streamlit_action = []
    #    error_search_term = self.code_runner_correct_chain.invoke({
    #        "technology": technology,
    #        "messages": messages,
    #        "command": command,
    #        "command_error": command_error
    #    })
    #    messages += [
    #        (
    #            "assistant",
    #            f"""
    #            Term to be searched on StackExchange API:\n
    #            {error_search_term}
    #            """
    #        )
    #    ]
    #    streamlit_action += [(
    #        "info", 
    #        {"body": messages[-1][1]},
    #        ("Search error online", True),
    #        messages[-1][0],
    #        )]
    #    streamlit_actions += [streamlit_action]
    #    return {
    #        "messages": messages,
    #        "streamlit_actions": streamlit_actions,
    #        "error_search_term": error_search_term,
    #        }
    
    #def correct_run_code_search_results(self, state: State):
    #    messages = state["messages"]
    #    streamlit_actions = state["streamlit_actions"]
    #    error_search_term = state["error_search_term"]
    #    iterations_search = state["iterations_search"]
    #    streamlit_action = []
    #    stackexchange = StackExchangeAPIWrapper()
    #    search_results = stackexchange.run(error_search_term)
    #    messages += [
    #        (
    #            "assistant",
    #            f"""
    #            Results about the error on StackExchange API:\n
    #            {search_results}
    #            """
    #        )
    #    ]
    #    streamlit_action += [(
    #        "success", 
    #        {"body": messages[-1][1]},
    #        ("Search error online - results", True),
    #        messages[-1][0],
    #        )]
    #    streamlit_actions += [streamlit_action]
    #    iterations_search += 1
    #    return {
    #        "messages": messages,
    #        "streamlit_actions": streamlit_actions,
    #        "error_search_term": error_search_term,
    #        "iterations_search": iterations_search
    #        }


    def stream_graph_updates(self, technology, project_name, user_input):
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
                "iterations": 0, 
                "error": "",
                "error_message": "",
                "project_name": project_name,
                "technology": technology,
                "command": "",
                "command_error": "",
                #"error_search_term": "",
                #"iterations_search": 0,
                #"error_search_results": ""
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
