![COELHO Agents](assets/coelho_agents_logo.png)
# COELHO Agents
Complete demonstration on YouTube: [COELHO Agents Presentation](https://www.youtube.com/watch?v=lxXcUA0jmcM)

COELHO Agents is my project involving AI Autonomous Agents with three functionalities:  

**1) Simple Assistant:** A simple chatbot with memory and real-time response  

**2) Software Developer:** Tool formed by AI Agents to generate software code and to run this generated code, with cycles of code fixing autonomously.  

**3) YouTube Content Search:** A distinguished tool that extracts YouTube videos transcriptions and use Knowledge Graph to get important informations and to store it into a graph database (Neo4J). The main advantage of this tool is that you can extract specific informations coming from personal points of views, which are not available on Google search, that doesn't index YouTube videos transcriptions.  


This project was built using Streamlit to get the interface, LangChain to build each AI Agent, and LangGraph to build each Multi-Agent approaches. In addition, it's used Neo4J to store entities and relationships among them in order to build a GraphRAG with Knowledge Graph.

---

## Details about the project  

COELHO Agents allows you to use 5 different APIs services to function with AI Agents, which 4 of them you need to get an API key in order to use the project:  
- [Groq](https://console.groq.com)
- [Ollama](https://ollama.com/download)
- [SambaNova](https://cloud.sambanova.ai/)
- [ScaleWay](https://account.scaleway.com/)
- [OpenAI](https://platform.openai.com/)

About Ollama, you can install LLM local models through [Ollama Models](https://ollama.com/library) or [HuggingFace Models](https://huggingface.co/models).

---

## How to install this project

1) Clone this repository:  
> git clone https://github.com/rafaelcoelho1409/COELHOAgents  
2) Enter this repository folder:  
> cd COELHOAgents  
3) Install UV for Python libraries management - [UV install](https://docs.astral.sh/uv/getting-started/installation/)  
4) Install Neo4J to use graph database on YouTube Content Search - [Neo4J](https://neo4j.com/docs/operations-manual/current/installation/)

4.1) If there's an error in Neo4J with APOC, do the following:  

4.1.1) Copy APOC into Neo4J plugins folder:  
> sudo cp /opt/neo4j-community-2025.01.0/labs/apoc-2025.01.0-core.jar /opt/neo4j-community-2025.01.0/plugins/  

4.1.2) Open the following file:  
> sudo nano /opt/neo4j-community-2025.01.0/conf/neo4j.conf  

4.1.3) And replace the following lines:  
> dbms.security.procedures.unrestricted=apoc.*  
> dbms.security.procedures.allowlist=apoc.*  
5) Set a virtual environment and install requeriments  
> uv venv  
> source .venv/bin/activate  
> uv pip install -r requirements.txt
6) Finally, run COELHO Agents on Streamlit:  
> streamlit run app.py