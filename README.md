# 🤖 LangChain Agent Framework  
**Building Intelligent Queryable AI with Pandas, SQL, and Hugging Face**

---

## 📖 Overview

This project demonstrates how to build intelligent, queryable LLM-based systems using the [LangChain](https://www.langchain.com/) framework. By leveraging LangChain agents, tools, memory, and evaluation modules, this suite enables natural language interfaces for querying **Pandas DataFrames**, interacting with **SQL databases**, and integrating **Hugging Face** language models.

The codebase is structured around a sequence of well-annotated Jupyter notebooks, each progressively introducing and demonstrating LangChain capabilities—from model prompt parsing to full-fledged autonomous agents.

---

## 🚀 Features

- ✅ Natural language queries over Pandas DataFrames
- ✅ SQL querying via LangChain agents
- ✅ Hugging Face LLM integration (open-source models)
- ✅ In-depth LangChain prompt engineering & parsing
- ✅ Memory-enabled conversational agents
- ✅ Chain construction for multi-step reasoning
- ✅ Evaluation tools for LLM output scoring
- ✅ Modular design with reusable components

---

## 🧠 Model Architectures Used

| Model Source         | Use Case                     | Description                          |
|----------------------|------------------------------|--------------------------------------|
| Hugging Face         | LLM Integration              | Uses `HuggingFaceHub` via LangChain  |
| OpenAI (Optional)    | Chat-based agent (optional)  | Replaceable backend                  |

> ⚠️ Hugging Face API keys may be required for some models. Replace with OpenAI-compatible APIs if desired.

---

## 🛠 Tech Stack Used

| Layer               | Technology                    |
|---------------------|-------------------------------|
| **Core Framework**  | LangChain                     |
| **Notebook Kernel** | Jupyter + Python 3.10         |
| **LLMs**            | Hugging Face Transformers     |
| **Data**            | Pandas, SQLite                |
| **Prompt Logic**    | LangChain PromptTemplate      |
| **Memory**          | ConversationBufferMemory      |
| **Evaluation**      | LangChain's EvaluationModule  |

---

## 🍎 Mac M1 Optimization

This project runs perfectly on **Mac M1/M2 chips** with no GPU dependencies. All Hugging Face and LangChain integrations are CPU-friendly. For performance boosts, you may optionally run server-hosted models or leverage OpenAI endpoints.

---

## 📦 Outputs

- ✅ Natural language responses from agents
- ✅ SQL queries auto-generated from prompts
- ✅ Intermediate thought steps from agents
- ✅ Chat-style memory tracebacks
- ✅ Evaluation metrics on model outputs
- ✅ Hugging Face model completions

---

## 📊 Results

The notebooks walk through complete use cases, such as:

- **Pandas Agent** responding to “What is the average sales in Q2?”
- **SQL Agent** parsing natural queries into valid SQL
- **LLM Chains** combining prompts and memory for multi-turn conversations
- **LangChain Evaluation** auto-scoring LLM responses against references

These results demonstrate the feasibility of building **intelligent internal data tools, AI copilots, and interactive analytics agents** with zero manual scripting.

---

## 📁 Code Breakdown

### 📒 `L1_langchain_model_prompt_parser.ipynb`
- Parses and formats prompts using `PromptTemplate`  
- Demonstrates prompt injection and custom variables

### 📒 `L2_langchain_memory.ipynb`
- Introduces conversational memory using `ConversationBufferMemory`  
- Builds a persistent memory trace for multi-turn queries

### 📒 `L3_langchain_chains.ipynb`
- Constructs chained LLM workflows  
- Executes multi-step logic using `LLMChain`

### 📒 `L4_langchain_questions.ipynb`
- Evaluates LLM question-answering pipelines  
- Compares baseline vs chain-enhanced performance

### 📒 `L5_langchain_evaluation.ipynb`
- Uses LangChain’s built-in evaluation framework  
- Automatically grades LLM outputs using metrics

### 📒 `L6_langchain_agents.ipynb`
- Builds custom LangChain agents  
- Introduces tool usage, self-reflection, and planning steps

### 📒 `pandas-dataframe-agent-notebook.ipynb`
- Creates a natural language Pandas DataFrame query agent  
- Converts questions into `.loc` and `.groupby` operations

### 📒 `sql-agent-notebook.ipynb`
- Uses LangChain SQLDatabaseChain to query SQLite  
- Translates prompts into parameterized SQL queries

### 🐍 `langchain_hugging_face.py`
- Integrates `HuggingFaceHub` as a language model backend  
- Configures repo IDs, temperature, and tokens

### 🐍 `test2.py`
- Experimental script for prototyping Hugging Face completions  
- Tests transformers independently of LangChain

---
