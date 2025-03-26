import os
import gc
import re
import tempfile
import uuid
import logging
from typing import Optional, Tuple, Any, List, Union, Dict
from io import StringIO
import sys
from langchain_core.messages import BaseMessage

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai.llm import OpenAI as PandasAI_OpenAI

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create a simple in-memory chat message history storage
class InMemoryChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._messages: List[BaseMessage] = []
    
    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)
    
    def clear(self) -> None:
        self._messages = []
    
    @property
    def messages(self) -> List[BaseMessage]:
        return self._messages

# Capturing stdout for code execution
class CaptureOutput:
    def __init__(self):
        self.output = StringIO()
        self._stdout = sys.stdout
        
    def __enter__(self):
        sys.stdout = self.output
        return self
        
    def __exit__(self, *args):
        sys.stdout = self._stdout
        
    def get_output(self):
        return self.output.getvalue()

# Initialize session state
if "id" not in st.session_state:
    st.session_state.id = str(uuid.uuid4())
    st.session_state.file_cache = {}
    st.session_state.dataframes = {}
    st.session_state.messages = []
    st.session_state.context = None
    st.session_state.current_file = None
    st.session_state.chat_histories = {}


def reset_chat() -> None:
    """Reset the chat history and clear conversation memory."""
    st.session_state.messages = []
    if st.session_state.current_file:
        file_key = f"{st.session_state.id}-{st.session_state.current_file}"
        if file_key in st.session_state.chat_histories:
            st.session_state.chat_histories[file_key].clear()
    st.session_state.context = None
    gc.collect()

def display_csv(file_obj: Any) -> pd.DataFrame:
    """
    Display a CSV file preview and return the DataFrame.
    
    Args:
        file_obj: The uploaded CSV file object.
        
    Returns:
        pd.DataFrame: The pandas DataFrame containing the CSV data.
    """
    try:
        st.markdown("### Data Preview")
        
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                # Reset file pointer to beginning
                file_obj.seek(0)
                # Read the CSV file with current encoding
                df = pd.read_csv(file_obj, encoding=encoding)
                logger.info(f"Successfully read CSV with {encoding} encoding")
                break
            except UnicodeDecodeError:
                logger.warning(f"Failed to read with {encoding} encoding")
                continue
            except Exception as e:
                logger.error(f"Error reading CSV with {encoding} encoding: {str(e)}")
                raise
        
        if df is None:
            raise ValueError("Could not read CSV file with any supported encoding")
        
        # Display basic information about the DataFrame
        st.write("**DataFrame Info:**")
        st.write(f"Shape: {df.shape}")
        st.write(f"Columns: {df.columns.tolist()}")
        st.write(f"Data Types:\n{df.dtypes}")
        
        # Display the dataframe
        st.dataframe(df)
        
        return df
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error details: {str(e)}")
        st.error(f"Error reading CSV file: {str(e)}")
        st.error("Please check if your CSV file is properly formatted and not corrupted.")
        raise

def display_excel(file_obj: Any) -> pd.DataFrame:
    """
    Display an Excel file preview and return the DataFrame.
    
    Args:
        file_obj: The uploaded Excel file object.
        
    Returns:
        pd.DataFrame: The pandas DataFrame containing the Excel data.
    """
    try:
        st.markdown("### Data Preview")
        
        # Read the Excel file
        df = pd.read_excel(file_obj)
        
        # Display the dataframe
        st.dataframe(df)
        
        return df
    except Exception as e:
        logger.error(f"Error reading Excel file: {str(e)}")
        st.error(f"Error reading Excel file: {str(e)}")
        raise

def extract_code_from_response(response: str) -> List[str]:
    """
    Extract all Python code snippets from a text response.
    
    Args:
        response: The text response potentially containing code blocks.
        
    Returns:
        List of extracted code snippets.
    """
    # Look for Python code blocks with explicit language tag
    python_pattern = r"```python\s*(.*?)\s*```"
    python_blocks = re.findall(python_pattern, response, re.DOTALL)
    
    # If none found, try looking for generic code blocks
    if not python_blocks:
        generic_pattern = r"```\s*(.*?)\s*```"
        generic_blocks = re.findall(generic_pattern, response, re.DOTALL)
        
        # Only use generic blocks if they look like Python
        python_keywords = ['import', 'def', 'print', 'for', 'while', 'if', 'df', 'pd', 'np']
        for block in generic_blocks:
            if any(keyword in block for keyword in python_keywords):
                python_blocks.append(block)
    
    # Clean up the code blocks
    cleaned_blocks = [block.strip() for block in python_blocks]
    return cleaned_blocks

def execute_code_safely(code: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Execute Python code safely and capture different types of output.
    
    Args:
        code: Python code string to execute.
        df: DataFrame to make available to the code.
        
    Returns:
        Dictionary containing:
            - result: The explicit return value, if any
            - df_result: DataFrame result if code outputs a DataFrame
            - plot: Matplotlib or Plotly figure if code produces one
            - plot_type: Type of plot (plotly or matplotlib)
            - stdout: Captured standard output
            - error: Error message if execution fails
    """
    # Create a safe locals dictionary with only necessary libraries
    local_vars = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "px": px,
        "go": go,
        "make_subplots": make_subplots,
        "df": df.copy()  # Use a copy to prevent modification of original
    }
    
    result = {
        "result": None,
        "df_result": None,
        "plot": None,
        "plot_type": None,
        "stdout": None,
        "error": None
    }
    
    # Add plt.switch_backend if not in code (helps with rendering issues)
    if 'plt.' in code and 'plt.switch_backend' not in code:
        code = "plt.switch_backend('Agg')\n" + code
    
    # Check if code looks dangerous
    dangerous_modules = ['os', 'sys', 'subprocess', 'shutil', 'requests']
    dangerous_funcs = ['eval', 'exec', 'compile', 'open', 'write', 'system']
    
    if any(f"import {module}" in code for module in dangerous_modules) or \
       any(f"from {module}" in code for module in dangerous_modules) or \
       any(func in code for func in dangerous_funcs):
        result["error"] = "Code contains potentially unsafe operations and was not executed."
        return result
    
    try:
        # Capture stdout
        with CaptureOutput() as captured:
            # Execute the code
            exec(code, {}, local_vars)
            result["stdout"] = captured.get_output()
        
        # Look for specific result types
        
        # Check for DataFrames
        for var_name, var_value in local_vars.items():
            if isinstance(var_value, pd.DataFrame) and var_name != 'df':
                result["df_result"] = var_value
                break
        
        # If no explicit DataFrame was found but 'df' was modified
        if result["df_result"] is None and not local_vars["df"].equals(df):
            result["df_result"] = local_vars["df"]
        
        # Check for Plotly figures first (preferred)
        if 'fig' in local_vars and isinstance(local_vars['fig'], go.Figure):
            result["plot"] = local_vars['fig']
            result["plot_type"] = "plotly"
        # Then check for other variables that might be Plotly figures
        elif any(isinstance(local_vars[var_name], go.Figure) for var_name in local_vars if isinstance(var_name, str)):
            for var_name, var_value in local_vars.items():
                if isinstance(var_name, str) and isinstance(var_value, go.Figure):
                    result["plot"] = var_value
                    result["plot_type"] = "plotly"
                    break
        # Only then check for Matplotlib figures
        elif plt.get_fignums():
            fig = plt.gcf()
            result["plot"] = fig
            result["plot_type"] = "matplotlib"
            # Don't close the figure here, we'll render it with st.pyplot()
        
        # Check for other result variables
        result_var_names = ['result', 'output', 'summary', 'analysis']
        for var_name in result_var_names:
            if var_name in local_vars:
                result["result"] = local_vars[var_name]
                break
        
        return result
    
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}")
        result["error"] = str(e)
        return result

def process_response_with_code_execution(response: str, df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    """
    Process a response, extract and execute any code, and format the results.
    
    Args:
        response: The text response potentially containing code.
        df: DataFrame to execute the code against.
        
    Returns:
        Tuple containing:
            - Modified response with code results instead of code
            - Dictionary of execution results
    """
    # Extract code blocks
    code_blocks = extract_code_from_response(response)
    
    if not code_blocks:
        return response, {}
    
    results = {}
    modified_response = response
    
    # Check for newly created images in the exports/charts directory
    charts_dir = "exports/charts"
    existing_charts = set()
    if os.path.exists(charts_dir):
        existing_charts = set([f for f in os.listdir(charts_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    for i, code in enumerate(code_blocks):
        # Execute the code
        execution_result = execute_code_safely(code, df)
        results[f"block_{i}"] = execution_result
        
        # Replace code block with a summary of its execution
        result_summary = ""
        
        if execution_result["error"]:
            result_summary = f"❌ Code execution error: {execution_result['error']}"
        else:
            # Add DataFrame output for commands like df.head() even if there's no explicit result
            if "df.head()" in code or "df.tail()" in code or "df.sample(" in code or "df.iloc" in code:
                if execution_result["df_result"] is not None:
                    result_summary = "**Data Preview:**"  # Will be followed by the dataframe visualization
                else:
                    # If we couldn't capture a dataframe result but the code should show one
                    head_df = df.head() if "df.head()" in code else df.tail() if "df.tail()" in code else df.sample(5)
                    execution_result["df_result"] = head_df
                    result_summary = "**Data Preview:**"
            elif execution_result["result"] is not None:
                result_summary = f"Result: {execution_result['result']}"
            
            if execution_result["stdout"] and execution_result["stdout"].strip():
                if result_summary:
                    result_summary += "\n\n"
                result_summary += f"**Output:** {execution_result['stdout'].strip()}"
        
        # Replace the code block with the result summary if needed
        if result_summary:
            code_pattern = r"```python\s*" + re.escape(code) + r"\s*```"
            if re.search(code_pattern, modified_response, re.DOTALL):
                modified_response = re.sub(code_pattern, result_summary, modified_response, flags=re.DOTALL)
            else:
                # Try generic code block pattern
                generic_pattern = r"```\s*" + re.escape(code) + r"\s*```"
                if re.search(generic_pattern, modified_response, re.DOTALL):
                    modified_response = re.sub(generic_pattern, result_summary, modified_response, flags=re.DOTALL)
                    
    # Check for newly saved charts
    new_charts = []
    if os.path.exists(charts_dir):
        current_charts = set([f for f in os.listdir(charts_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        new_charts = list(current_charts - existing_charts)
    
    # Add newly saved charts to the results
    if new_charts:
        results["saved_charts"] = [os.path.join(charts_dir, chart) for chart in new_charts]
    
    return modified_response, results

def get_openai_model(model_name: str = "gpt-4o") -> ChatOpenAI:
    """
    Initialize and return the OpenAI model.
    
    Args:
        model_name: The name of the OpenAI model to use.
        
    Returns:
        ChatOpenAI: The initialized OpenAI chat model.
    """
    return ChatOpenAI(
        model_name=model_name,
        temperature=0,
        streaming=True
    )

def get_message_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create a new chat message history for the session."""
    if session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[session_id] = InMemoryChatMessageHistory(session_id)
    return st.session_state.chat_histories[session_id]

def create_retrieval_chain(df: pd.DataFrame, file_key: str) -> RunnableWithMessageHistory:
    """
    Create a conversational retrieval chain for the given DataFrame.
    
    Args:
        df: The pandas DataFrame to create a chain for.
        file_key: Unique identifier for the file.
        
    Returns:
        RunnableWithMessageHistory: The initialized chain for question answering.
    """
    try:
        # Ensure DataFrame is not empty and has data
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Convert DataFrame to string representation for document creation
        df_str = df.to_string()
        
        # Create documents from DataFrame
        documents = [Document(page_content=df_str, metadata={"source": file_key})]
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Create a retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Create LLM chain
        llm = get_openai_model()
        
        # Create custom prompt that emphasizes code execution
        qa_prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template("""
            You are an expert data analyst helping to analyze a dataset. You will be given a question about the data.

            Previous conversation:
            {chat_history}
            
            Context information from the file is below:
            ---
            {context}
            ---
            
            IMPORTANT: The user has already uploaded a DataFrame which is available as the variable 'df'. This DataFrame ALREADY CONTAINS ALL THE NECESSARY DATA for analysis. 
            
            Think step by step to provide a precise answer. When asked to analyze data or create visualizations:
            1. ALWAYS use the existing 'df' DataFrame that is ALREADY LOADED - DO NOT create ANY sample/synthetic data or initialize new dataframes
            2. ALWAYS start your code by examining the structure of the existing data with df.head() or df.columns.tolist() to understand what's available
            3. ALWAYS provide runnable Python code that performs the requested analysis or creates the visualization using ONLY the existing 'df'
            4. STRONGLY PREFER using plotly over matplotlib/seaborn for visualizations as it creates interactive charts and has fewer restrictions
            5. When using plotly, always end your code with something like 'fig = px.line(...)' or 'fig = go.Figure(...)' and don't call fig.show()
            6. If using matplotlib/seaborn, NEVER use plt.gca(), plt.axis(), plt.subplot(), or other advanced matplotlib functions that might be restricted
            7. For matplotlib, use simple commands like plt.figure(), plt.plot(), plt.bar(), plt.title(), plt.xlabel(), plt.ylabel() only
            8. NEVER create fake/sample data - all data should come directly from operations on the existing 'df' DataFrame
            9. DO NOT use any code that requires reading files from disk
            10. When returning results, make sure your code explicitly creates a result object - for charts this is 'fig', for tables this could be a DataFrame
            
            When providing code, ALWAYS wrap it in ```python ``` tags. The code will be executed and the results will be shown to the user.
            
            User Question: {question}
            """)
        ])
        
        # Create chain with updated parameters
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
            return_generated_question=True
        )
        
        # Wrap the chain with message history capabilities
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_message_history,
            input_messages_key="question",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        return chain_with_history
    except Exception as e:
        logger.error(f"Error creating retrieval chain: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error details: {str(e)}")
        raise

def create_pandasai_df(df: pd.DataFrame) -> SmartDataframe:
    """
    Create a PandasAI SmartDataframe for advanced data analysis.
    
    Args:
        df: The pandas DataFrame to enhance.
        
    Returns:
        SmartDataframe: The enhanced DataFrame with AI capabilities.
    """
    try:
        # Ensure OpenAI API key is set
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        llm = PandasAI_OpenAI(api_token=api_key)
        
        smart_df = SmartDataframe(
            df,
            config={
                "llm": llm,
                "enable_cache": True,
                "verbose": True,
                "open_charts": True,
                "save_charts": True,
                "custom_whitelisted_dependencies": ["plotly", "pandas", "numpy", "plotly.express", "plotly.graph_objects"],
                "custom_instructions": """
                    ALWAYS use plotly for visualizations. DO NOT use matplotlib or seaborn.
                    Always use px.bar(), px.line(), px.scatter(), etc. for creating charts.
                    Always save the figure object as 'fig' and return it.
                    Do not use any matplotlib functions at all.
                    For bar charts of categorical data, use px.bar(df_grouped, x='category_column', y='value_column').
                    For horizontal bar charts, add orientation='h' to px.bar() and swap x/y parameters.
                    For line charts, use px.line(df, x='x_column', y='y_column').
                    For scatter plots, use px.scatter(df, x='x_column', y='y_column').
                    For histograms, use px.histogram(df, x='column').
                    For pie charts, use px.pie(df, values='value_column', names='category_column').
                """
            }
        )
        
        return smart_df
    except Exception as e:
        logger.error(f"Error creating PandasAI DataFrame: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error details: {str(e)}")
        raise

def display_saved_charts() -> None:
    """
    Display all saved charts from the exports/charts directory.
    """
    try:
        charts_dir = "exports/charts"
        if os.path.exists(charts_dir):
            # Get all PNG files from the charts directory
            chart_files = [f for f in os.listdir(charts_dir) if f.endswith('.png')]
            
            if chart_files:
                st.markdown("### 📊 Saved Charts")
                
                # Create columns for displaying charts
                cols = st.columns(3)  # Display 3 charts per row
                
                for idx, chart_file in enumerate(chart_files):
                    with cols[idx % 3]:
                        # Read and display the image
                        image_path = os.path.join(charts_dir, chart_file)
                        st.image(image_path, caption=chart_file, use_container_width=True)
            else:
                st.info("No saved charts found in the exports/charts directory.")
        else:
            st.info("Charts directory not found. Charts will be saved here when generated.")
    except Exception as e:
        st.error(f"Error displaying saved charts: {str(e)}")
        
def initialize_loaded_data():
    """
    Initialize RAG components for all loaded dataframes in the session state.
    This replaces the file upload functionality in the original ExcelChat.
    """
    # Get data from session state
    data_dict = st.session_state.get("dataframes_dict", {})
    if not data_dict:
        st.error("No data available. Please go back and select files.")
        return False
    
    # Process each dataframe
    for file_name, df in data_dict.items():
        file_key = f"{st.session_state.id}-{file_name}"
        
        # Skip if already in cache
        if file_key in st.session_state.get('file_cache', {}):
            continue
            
        try:
            # Store original DataFrame if not already stored
            if file_key not in st.session_state.dataframes:
                st.session_state.dataframes[file_key] = df.copy()
            
            # Create retrieval chain
            chain = create_retrieval_chain(df, file_key)
            
            # Create PandasAI smart dataframe
            smart_df = create_pandasai_df(df)
            
            # Store in cache
            st.session_state.file_cache[file_key] = {
                "chain": chain,
                "smart_df": smart_df
            }
            
            # Set as current file if none is set
            if not st.session_state.current_file:
                st.session_state.current_file = file_name
                
        except Exception as e:
            logger.error(f"Error initializing RAG for {file_name}: {str(e)}")
            st.error(f"Error setting up AI for {file_name}: {str(e)}")
    
    return True

def ExcelChat_main():
    """Main application function."""
    if "id" not in st.session_state:
        st.session_state.id = str(uuid.uuid4())
        st.session_state.file_cache = {}
        st.session_state.dataframes = {}
        st.session_state.messages = []
        st.session_state.context = None
        st.session_state.current_file = None
        st.session_state.chat_histories = {}

    col1, col2 = st.columns([6, 1])
    
    with col2:
        st.button("Reset Chat ↺", on_click=reset_chat)
    
    # Initialize data that's already loaded
    success = initialize_loaded_data()
    if not success:
        return
    
    # File selector if multiple files
    data_dict = st.session_state.get("dataframes_dict", {})
    if len(data_dict) > 1:
        selected_file = st.selectbox("Select data to chat with:", list(data_dict.keys()))
        st.session_state.current_file = selected_file
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display additional components based on message type
            if "df_result" in message and message["df_result"] is not None:
                st.dataframe(message["df_result"])
            
            if "plot" in message and message["plot"] is not None:
                if hasattr(message["plot"], "savefig"):  # Matplotlib figure
                    st.pyplot(message["plot"])
                else:  # Assume Plotly figure
                    st.plotly_chart(message["plot"], use_container_width=True)
            
            # Display saved chart images
            if "saved_charts" in message and message["saved_charts"]:
                for chart_path in message["saved_charts"]:
                    if os.path.exists(chart_path):
                        st.image(chart_path, use_column_width=True)
    
    # Accept user input
    if prompt := st.chat_input("Ask me about your data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response if file has been uploaded
        if st.session_state.current_file and st.session_state.file_cache:
            # file_key = f"{st.session_state.current_file}"
            file_key = f"{st.session_state.id}-{st.session_state.current_file}"
            
            if file_key in st.session_state.file_cache:
                # Get cached resources
                chain = st.session_state.file_cache[file_key]["chain"]
                smart_df = st.session_state.file_cache[file_key]["smart_df"]
                df = st.session_state.dataframes[file_key]
                
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    
                    # Determine if this is a visualization request
                    viz_keywords = ["chart", "plot", "graph", "visualize", "visualization", 
                                   "show me", "display", "bar chart", "line graph", 
                                   "histogram"]
                    is_viz_request = any(keyword in prompt.lower() 
                                        for keyword in viz_keywords)
                    
                    if is_viz_request:
                        # Use PandasAI for visualization
                        try:
                            with st.spinner("Generating visualization..."):
                                # Generate visualization with PandasAI
                                result = smart_df.chat(prompt)
                                
                                # Check for newly created images in the exports/charts directory
                                charts_dir = "exports/charts"
                                existing_charts = set()
                                if os.path.exists(charts_dir):
                                    existing_charts = set([f for f in os.listdir(charts_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
                                
                                # Handle result based on type
                                if isinstance(result, dict) and 'type' in result and result['type'] == 'plot' and 'value' in result:
                                    # It's a saved chart path in a dict
                                    chart_path = result['value']
                                    # Ensure it's absolute path
                                    if not os.path.isabs(chart_path):
                                        chart_path = os.path.abspath(chart_path)
                                    
                                    # Verify file exists
                                    if os.path.exists(chart_path):
                                        message_placeholder.write("Here's the visualization based on your request:")
                                        st.image(chart_path, use_column_width=True)
                                        st.session_state.messages.append({
                                            "role": "assistant", 
                                            "content": "Here's the visualization based on your request:", 
                                            "saved_charts": [chart_path]
                                        })
                                    else:
                                        # Check if file exists in the exports/charts directory instead
                                        alt_path = os.path.join("exports/charts", os.path.basename(chart_path))
                                        if os.path.exists(alt_path):
                                            message_placeholder.write("Here's the visualization based on your request:")
                                            st.image(alt_path, use_column_width=True)
                                            st.session_state.messages.append({
                                                "role": "assistant", 
                                                "content": "Here's the visualization based on your request:", 
                                                "saved_charts": [alt_path]
                                            })
                                        else:
                                            # Fall back to checking for any new charts
                                            new_charts = []
                                            if os.path.exists(charts_dir):
                                                current_charts = set([f for f in os.listdir(charts_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
                                                new_charts = list(current_charts - existing_charts)
                                            
                                            if new_charts:
                                                chart_paths = [os.path.join(charts_dir, chart) for chart in new_charts]
                                                message_placeholder.write("Here's the visualization based on your request:")
                                                for chart_path in chart_paths:
                                                    st.image(chart_path, use_column_width=True)
                                                st.session_state.messages.append({
                                                    "role": "assistant", 
                                                    "content": "Here's the visualization based on your request:", 
                                                    "saved_charts": chart_paths
                                                })
                                            else:
                                                message_placeholder.write(f"Visualization was created but could not be displayed. Path: {chart_path}")

                                # Handle any type of result
                                elif hasattr(result, 'figure') or hasattr(result, 'plot'):
                                    # If it's a figure/plot object
                                    message_placeholder.write("Here's the visualization based on your request:")
                                    st.plotly_chart(result, use_container_width=True)
                                    st.session_state.messages.append({
                                        "role": "assistant", 
                                        "content": "Here's the visualization based on your request:", 
                                        "plot": result
                                    })
                                else:
                                    # Check for newly saved charts
                                    new_charts = []
                                    if os.path.exists(charts_dir):
                                        current_charts = set([f for f in os.listdir(charts_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
                                        new_charts = list(current_charts - existing_charts)
                                    
                                    if new_charts:
                                        # If charts were saved during execution
                                        chart_paths = [os.path.join(charts_dir, chart) for chart in new_charts]
                                        message_placeholder.write("Here's the visualization based on your request:")
                                        for chart_path in chart_paths:
                                            st.image(chart_path, use_column_width=True)
                                        st.session_state.messages.append({
                                            "role": "assistant", 
                                            "content": "Here's the visualization based on your request:", 
                                            "saved_charts": chart_paths
                                        })
                                    else:
                                        # If it's any other type (string, dict, list, etc.)
                                        message_placeholder.write(str(result))
                                        st.session_state.messages.append({
                                            "role": "assistant", 
                                            "content": str(result)
                                        })
                                
                                # Add to LangChain message history as string
                                if file_key in st.session_state.chat_histories:
                                    history = st.session_state.chat_histories[file_key]
                                    history.add_message(HumanMessage(content=prompt))
                                    history.add_message(AIMessage(content=str(result)))
                        except Exception as e:
                            # Fall back to LLM if PandasAI fails
                            st.warning(f"PandasAI visualization failed, falling back to LLM: {str(e)}")
                            config = RunnableConfig(configurable={"session_id": file_key})
                            
                            # Use spinner while generating response
                            with st.spinner("Generating response..."):
                                # Get response from chain
                                result = chain.invoke({"question": prompt, "context": "", "chat_history": ""}, config=config)
                                response = result.get("answer", "")
                                
                                # Process response by executing code and capturing output
                                modified_response, execution_results = process_response_with_code_execution(response, df)
                                
                                # Update the message placeholder with the modified response
                                message_placeholder.markdown(modified_response)
                                
                                # Display execution results
                                for block_key, block_result in execution_results.items():
                                    # Display DataFrame results
                                    if block_result["df_result"] is not None:
                                        st.dataframe(block_result["df_result"])
                                    
                                    # Display plot results
                                    if block_result["plot"] is not None:
                                        if hasattr(block_result["plot"], "savefig"):  # Matplotlib
                                            st.pyplot(block_result["plot"])
                                        else:  # Plotly
                                            st.plotly_chart(block_result["plot"], use_container_width=True)
                                
                                # Display any saved charts
                                if "saved_charts" in execution_results:
                                    st.markdown("### Generated Charts:")
                                    for chart_path in execution_results["saved_charts"]:
                                        st.image(chart_path, use_column_width=True)
                                
                                # Add response to chat history
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": modified_response,
                                    "df_result": next((r["df_result"] for r in execution_results.values() 
                                                     if r["df_result"] is not None), None),
                                    "plot": next((r["plot"] for r in execution_results.values() 
                                                if r["plot"] is not None), None),
                                    "saved_charts": execution_results.get("saved_charts", [])
                                })
                                
                                # Add to LangChain message history
                                if file_key in st.session_state.chat_histories:
                                    history = st.session_state.chat_histories[file_key]
                                    history.add_message(HumanMessage(content=prompt))
                                    history.add_message(AIMessage(content=modified_response))
                    else:
                        # Standard QA with the chain
                        config = RunnableConfig(configurable={"session_id": file_key})
                        
                        # Use spinner while generating response
                        with st.spinner("Analyzing your data..."):
                            # Get response from chain
                            result = chain.invoke({"question": prompt, "context": "", "chat_history": ""}, config=config)
                            response = result.get("answer", "")
                            
                            # Process response by executing code and capturing output
                            modified_response, execution_results = process_response_with_code_execution(response, df)
                            
                            # Update the message placeholder with the modified response
                            message_placeholder.markdown(modified_response)
                            
                            # Display execution results
                            for block_key, block_result in execution_results.items():
                                # Display DataFrame results
                                if block_result["df_result"] is not None:
                                    st.dataframe(block_result["df_result"])
                                
                                # Display plot results
                                if block_result["plot"] is not None:
                                    if hasattr(block_result["plot"], "savefig"):  # Matplotlib
                                        st.pyplot(block_result["plot"])
                                    else:  # Plotly
                                        st.plotly_chart(block_result["plot"], use_container_width=True)
                            
                            # Add response to chat history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": modified_response,
                                "df_result": next((r["df_result"] for r in execution_results.values() 
                                                 if r["df_result"] is not None), None),
                                "plot": next((r["plot"] for r in execution_results.values() 
                                            if r["plot"] is not None), None)
                            })
                            
                            # Add to LangChain message history
                            if file_key in st.session_state.chat_histories:
                                history = st.session_state.chat_histories[file_key]
                                history.add_message(HumanMessage(content=prompt))
                                history.add_message(AIMessage(content=modified_response))
            else:
                st.error("Something went wrong with the file processing. Please re-upload the file.")
        else:
            # No file uploaded yet
            with st.chat_message("assistant"):
                missing_file_message = "Please upload a data file first using the sidebar. I need data to answer your questions."
                st.warning(missing_file_message)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": missing_file_message
                })