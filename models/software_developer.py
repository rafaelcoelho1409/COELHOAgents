import streamlit as st
import subprocess
import os
import json
import stqdm
from dotenv import load_dotenv
from typing import List, Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
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
    filenames: List[str] = Field(description = """
        File names for this code solution. 
        Can be one or more files.
        Names must be into a list of strings.""")
    imports: List[str] = Field(description = """
        Code block import statements for each file to be created in the project,
        excluding the rest of the code that are not the import statements. 
        Can be one or more code files, according to the file names.
        Code imports must be into a list of strings.""")
    codes: List[str] = Field(description = """
        Code block statements for each file to be created in the project,
        without the code block import statements. 
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
    

class FixType(BaseModel):
    fix_type: Literal["dependencies", "code"] = Field(description = """
        Type of error to fix. Can be dependencies or code.""")

class FixCode(BaseModel):
    filename: List[str] = Field(description = """
        Name of the file(s) to be fixed.""")
    codes: List[str] = Field(description = """
        Code block(s) to be fixed. Must match with each file name""")


class State(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        error_message : Error message
        messages : With user question, error messages, reasoning
        generation : Code solution
        technology : Programming language or technology stack
    """
    error: str
    error_message: str
    messages: List[str]
    streamlit_actions: List[str]
    filenames: List[str]
    generation: str
    technology: str
    project_name: str
    command: str
    command_error: str
    dependencies: str
    fix_dependencies_iterations: int
    fix_code_iterations: int
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
        self.fix_error_dependencies_chain = self.build_fix_error_dependencies_chain()
        self.fix_code_type_chain = self.build_fix_code_type_chain()
        self.fix_code_chain = self.build_fix_code_chain()
        # Max tries
        self.max_iterations = 3
        self.workflow = StateGraph(State)
        ###NODES
        self.workflow.add_node("check_install", self.check_install)
        self.workflow.add_node("generate_code", self.generate_code)
        self.workflow.add_node("check_dependencies", self.check_dependencies)
        self.workflow.add_node("run_dependencies", self.run_dependencies)
        self.workflow.add_node("fix_error_dependencies", self.fix_error_dependencies)
        self.workflow.add_node("run_code", self.run_code)
        #self.workflow.add_node("fix_code_type", self.fix_code_type)
        self.workflow.add_node("fix_code", self.fix_code)
        #self.workflow.add_node("correct_run_code_search_results", self.correct_run_code_search_results)
        ###EDGES
        self.workflow.add_edge(START, "check_install")
        self.workflow.add_conditional_edges("check_install", self.check_install_error)
        self.workflow.add_edge("generate_code", "check_dependencies")
        self.workflow.add_edge("check_dependencies", "run_dependencies")
        self.workflow.add_conditional_edges("run_dependencies", self.fix_error_dependencies_conditional)
        self.workflow.add_conditional_edges("fix_error_dependencies", self.from_fix_dependencies_to_run_code)
        self.workflow.add_conditional_edges("run_code", self.fix_code_conditional)
        self.workflow.add_edge("fix_code", "run_code")
        #self.workflow.add_edge("run_dependencies", END)
        #self.workflow.add_edge("check_dependencies", "run_code")
        #self.workflow.add_conditional_edges("run_code", self.search_error_online)
        #self.workflow.add_edge("run_code", END)
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
                    You are a coding assistant specialist with expertise in the following language:  
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
                    Based on the generated code imports below:\n\n
                    {imports}\n\n 
                    And based on this code executed to check if the technology is installed locally:\n\n
                    """ + self.technologies_json[self.technology] + """
                    Your task is to:\n
                    1. Analyze and return all necessary dependencies to run the main file.\n
                    2. List the file names for these dependencies in a list of strings.\n
                    3. Generate the terminal commands to install the dependencies. The commands should:\n
                       - Be in a single row.\n
                       - Use the standard format for the specified technology.\n
                    4. It must be in a raw text row format.\n
                    5. Any command piece involving something like sudo must be dropped to not stop the execution.\n
                    """
                    #2. Provide these dependencies in the format of a `requirements` file, which lists each dependency on a new line.\n
                ),
                ("placeholder", "{messages}"),
            ]
        )
        dep_checker_chain = dep_checker_prompt | self.llm.with_structured_output(
            CodeRequirements,
        )
        return dep_checker_chain
    
    def build_fix_error_dependencies_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a coding assistant with expertise in the following language:
                    \n ------- \n  {technology} \n ------- \n
                    Based on the following error message:\n\n
                    {error_message}
                    \n\n
                    And based on the following code executed to check if the technology is installed locally:\n\n
                    """ + self.technologies_json[self.technology] + """\n\n
                    You must supply a command to fix this error and install the necessary dependencies,
                    based on your own knowledge. \n
                    It must be in only one row and the most optimized and accurate possible.\n
                    Any command piece involving something like sudo must be dropped to not stop the execution.\n
                    """
                ),
                ("placeholder", "{messages}"),
            ]
        )
        chain = prompt | self.llm
        return chain

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
                    Based on the following file names:\n\n
                    {filenames}\n\n
                    And based on the code executed below:\n\n
                    {code}\n\n 
                    Your task is to:\n
                    1. Generate the terminal commands to run the main file(s). The commands should:\n
                       - Be in a single row.\n
                       - Use the standard format for the specified technology.\n
                    """,
                ),
                ("placeholder", "{messages}"),
            ]
        )
        code_runner_chain = code_runner_prompt | self.llm.with_structured_output(
            CodeBlock,
        )
        return code_runner_chain
    
    def build_fix_code_type_chain(self):
        fix_code_prompt = ChatPromptTemplate.from_messages(
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
                    1. Define the error type: dependencies or code.\n
                    """,
                ),
                ("placeholder", "{messages}"),
            ]
        )
        fix_code_chain = fix_code_prompt | self.llm.with_structured_output(
            FixType
        )
        return fix_code_chain
    

    def build_fix_code_chain(self):
        fix_code_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a coding assistant with expertise in the following language:  
                    \n ------- \n  {technology} \n ------- \n 
                    Ensure any code you provide can be executed 
                    with all required imports and variables defined. \n
                    Based on the following file names:\n\n
                    {filenames}\n\n
                    And based on the codes generated below:\n\n
                    {codes}\n\n 
                    And based on the error after the code execution:\n\n
                    {error_message}\n\n
                    Your task is to:\n
                    1. Return the list of file name(s) to be fixed.\n
                    2. Return the list of code(s) to be fixed.\n
                    """,
                ),
                ("placeholder", "{messages}"),
            ]
        )
        fix_code_chain = fix_code_prompt | self.llm.with_structured_output(
            FixCode,
        )
        return fix_code_chain
    
    
    ###NODES###
    def check_install(self, state: State):
        print("Node: check_install")
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
        print("Node: check_install_error")
        error = state["error"]
        if error == "yes":
            return END
        else:
            return "generate_code"

    def generate_code(self, state: State):
        print("Node: generate_code")
        messages = state["messages"]
        streamlit_actions = state["streamlit_actions"]
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
                    **Codes:**\n```{self.technology.lower()}\n\n{code}\n\n```\n
                    """
                )
            ]
            streamlit_action += [(
                "markdown", 
                {"body": messages[-1][1]},
                (filename, True),
                messages[-1][0],
                )]
        streamlit_actions += [streamlit_action]
        return {
            "generation": code_solution, 
            "messages": messages, 
            "streamlit_actions": streamlit_actions,
            "filenames": code_solution.filenames,
            "project_name": code_solution.project_name
        }
    
    def check_dependencies(self, state: State):
        print("Node: check_dependencies")
        messages = state["messages"]
        streamlit_actions = state["streamlit_actions"]
        code_solution = state["generation"]
        streamlit_action = []
        dependencies = self.dep_checker_chain.invoke({
            "technology": self.technology,
            "imports": code_solution.imports,
            #"code": code_solution.codes,
            "messages": messages,
        })
        dependencies_commands = dependencies.dependencies_commands
        messages += [
            (
                "assistant",
                dependencies_commands
            )
        ]
        streamlit_action += [(
            "code", 
            {"body": messages[-1][1]},
            ("Dependencies install commands", True),
            messages[-1][0],
        )]
        streamlit_actions += [streamlit_action]
        return {
            "messages": messages, 
            "streamlit_actions": streamlit_actions,
            "dependencies": dependencies_commands}
    
    def run_dependencies(self, state: State):
        print("Node: run_dependencies")
        messages = state["messages"]
        streamlit_actions = state["streamlit_actions"]
        dependencies = state["dependencies"]
        streamlit_action = []
        command_status = subprocess.run(
            dependencies,
            shell = True,
            capture_output = True,
            text = True
        )
        if command_status.returncode == 0:
            messages += [
                (
                    "assistant",
                    f"""
                    The dependencies install was executed successfully.\n
                    Output: {command_status.stdout}
                    """
                )
            ]
            streamlit_action += [(
                "success", 
                {"body": messages[-1][1]},
                ("Success", True),
                messages[-1][0],
                )]
            error = "no"
            error_message = ""
        else:
            messages += [
                (
                    "assistant",
                    f"""
                    The dependencies install was not executed successfully.\n
                    Output: \n\n{command_status.stderr}
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
            error_message = command_status.stderr
        streamlit_actions += [streamlit_action]
        return {
            "messages": messages, 
            "streamlit_actions": streamlit_actions,
            "error": error,
            "error_message": error_message,
        }
    
    def fix_error_dependencies_conditional(self, state: State):
        print("Node: fix_error_dependencies_conditional")
        error = state["error"]
        if error == "yes":
            return "fix_error_dependencies"
        else:
            return "run_code"
        

    def fix_error_dependencies(self, state: State):
        print("Node: fix_error_dependencies")
        messages = state["messages"]
        streamlit_actions = state["streamlit_actions"]
        error_message = state["error_message"]
        fix_dependencies_iterations = state["fix_dependencies_iterations"]
        streamlit_action = []
        #------------------------------------------------------------------------------
        print(f"fix_dependencies_iterations: {fix_dependencies_iterations}")
        command_result = self.fix_error_dependencies_chain.invoke({
            "technology": self.technology,
            "error_message": error_message
        })
        messages += [
            (
                "assistant",
                command_result.content
            )
        ]
        streamlit_action += [(
            "code", 
            {"body": messages[-1][1]},
            (f"Dependencies command fix to be run - attempt {fix_dependencies_iterations + 1}", True),
            messages[-1][0],
        )]
        command_status = subprocess.run(
            command_result.content,
            shell = True,
            capture_output = True,
            text = True
        )
        if command_status.returncode == 0:
            messages += [
                (
                    "assistant",
                    f"""
                    The dependencies install was executed successfully.\n
                    Output: {command_status.stdout}
                    """
                )
            ]
            streamlit_action += [(
                "success", 
                {"body": messages[-1][1]},
                ("Success", True),
                messages[-1][0],
                )]
            error = "no"
            error_message = ""
        else:
            messages += [
                (
                    "assistant",
                    f"""
                    The dependencies install was not executed successfully.\n
                    Output: \n\n{command_status.stderr}
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
            error_message = command_status.stderr
            fix_dependencies_iterations += 1
        if fix_dependencies_iterations >= 3:
            messages += [
                (
                    "assistant",
                    "The maximum number of attempts to fix dependencies was reached. The execution is going to be stopped."
                )
            ]
            streamlit_action += [(
                "info",
                {"body": messages[-1][1]},
                ("Error trying to fix dependencies install", True),
                messages[-1][0],
            )]
        streamlit_actions += [streamlit_action]
        return {
            "messages": messages, 
            "streamlit_actions": streamlit_actions,
            "error": error,
            "error_message": error_message,
            "fix_dependencies_iterations": fix_dependencies_iterations
        }
    
    def from_fix_dependencies_to_run_code(self, state: State):
        print("Node: from_fix_dependencies_to_run_code")
        error = state["error"]
        fix_dependencies_iterations = state["fix_dependencies_iterations"]
        if error == "yes" and fix_dependencies_iterations < 3:
            return "fix_error_dependencies"
        elif error == "yes" and fix_dependencies_iterations >= 3:
            return END
        else:
            return "run_code"
    
    def run_code(self, state: State):
        print("Node: run_code")
        messages = state["messages"]
        streamlit_actions = state["streamlit_actions"]
        fix_code_iterations = state["fix_code_iterations"]
        error = state["error"]
        project_name = state["project_name"]
        streamlit_action = []
        #------------------------------------------------------------------------------
        print(f"fix_code_iterations: {fix_code_iterations}")
        code_solution = {}
        code_solution["filenames"] = os.listdir(str(self.project_folder / project_name))
        code_solution["codes"] = [
            open(str(self.project_folder / os.path.join(project_name, x)), "r").read() 
            for x 
            in code_solution["filenames"]]
        code_runner = self.code_runner_chain.invoke({
            "technology": self.technology,
            "messages": messages,
            "filenames": code_solution["filenames"],
            "code": code_solution["codes"],
        })
        original_command = code_runner.code
        def replace_path_on_command(_term):
            if _term in code_solution["filenames"]:
                return str(self.project_folder / os.path.join(project_name, _term))
            return _term
        updated_command = " ".join([replace_path_on_command(x) for x in original_command.split()])
        messages += [
            (
                "assistant",
                updated_command
            )
        ]
        streamlit_action += [(
            "code", 
            {"body": messages[-1][1]},
            ("Code to be executed on the generated folder (terminal)", True),
            messages[-1][0],
            )]
        command_status = subprocess.run(
            updated_command,
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
            streamlit_action += [(
                "success", 
                {"body": messages[-1][1]},
                ("Success", True),
                messages[-1][0],
                )]
            error = "no"
            error_message = ""
        else:
            messages += [
                (
                    "assistant",
                    f"""
                    The code was not executed successfully.\n
                    Output: \n\n{command_status.stderr}
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
            error_message = command_status.stderr.replace(str(self.project_folder / project_name), "[REDACTED]")
            fix_code_iterations += 1
        if fix_code_iterations >= 3:
            messages += [
                (
                    "assistant",
                    "The maximum number of attempts to fix codes was reached. The execution is going to be stopped."
                )
            ]
            streamlit_action += [(
                "info",
                {"body": messages[-1][1]},
                ("Error trying to fix dependencies install", True),
                messages[-1][0],
            )]
        streamlit_actions += [streamlit_action]
        return {
            "messages": messages,
            "streamlit_actions": streamlit_actions,
            "error": error,
            "command": updated_command,
            "error_message": error_message
            }
    
    def fix_code_conditional(self, state: State):
        print("Node: fix_code_conditional")
        messages = state["messages"]
        command = state["command"]
        error_message = state["error_message"]
        error = state["error"]
        fix_code_iterations = state["fix_code_iterations"]
        fix_type = self.fix_code_type_chain.invoke({
            "technology": self.technology,
            "messages": messages,
            "command": command,
            "command_error": error_message
        })
        if error == "yes" and fix_code_iterations < 3:
            if fix_type.fix_type == "dependencies":
                return "fix_error_dependencies"
            elif fix_type.fix_type == "code":
                return "fix_code"
        else:
            return END
    
    #def fix_code_type(self, state: State): 
    #    print("Node: fix_code_type") 
    #    messages = state["messages"]
    #    command = state["command"]
    #    error_message = state["error_message"]
    #    fix_type = self.fix_code_type_chain.invoke({
    #        "technology": self.technology,
    #        "messages": messages,
    #        "command": command,
    #        "command_error": error_message
    #    })
    #    if fix_type.fix_type == "dependencies":
    #        return "fix_error_dependencies"
    #    elif fix_type.fix_type == "code":
    #        return "fix_code"
        

    def fix_code(self, state: State):
        print("Node: fix_code")
        messages = state["messages"]
        streamlit_actions = state["streamlit_actions"]
        #filenames = state["generation"].filenames
        #codes = state["generation"].codes
        error_message = state["error_message"]
        project_name = state["project_name"]
        streamlit_action = []
        code_solution = {}
        code_solution["filenames"] = os.listdir(str(self.project_folder / project_name))
        code_solution["codes"] = [
            open(str(self.project_folder / os.path.join(project_name, x)), "r").read() 
            for x 
            in code_solution["filenames"]]
        fix_code_results = self.fix_code_chain.invoke({
            "technology": self.technology,
            "messages": messages,
            "filenames": code_solution["filenames"],
            "codes": code_solution["codes"],
            "error_message": error_message
        })
        for filename, code in zip(code_solution["filenames"], code_solution["codes"]):
            messages += [
                (
                    "assistant",
                    f"""
                    **Codes:**\n```{self.technology.lower()}\n\n{code}\n\n```\n
                    """
                )
            ]
            streamlit_action += [(
                "markdown", 
                {"body": messages[-1][1]},
                (filename, True),
                messages[-1][0],
                )]
            with open(self.project_folder / os.path.join(
                project_name, 
                filename), "w") as file:
                file.write(code)
        streamlit_actions += [streamlit_action]
        return {
            "messages": messages,
            "streamlit_actions": streamlit_actions,
        }


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
                "error": "",
                "error_message": "",
                "project_name": project_name,
                "technology": technology,
                "command": "",
                "command_error": "",
                "dependencies": "",
                "fix_dependencies_iterations": 0,
                "fix_code_iterations": 0,
                "filenames": [],
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
