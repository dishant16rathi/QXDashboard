from datetime import datetime
import os
import gc
import re
import tempfile
import time
import uuid
import logging
from typing import Optional, Tuple, Any, List, Union, Dict
from io import StringIO
import sys
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create a simple in-memory chat message history storage
class InMemoryChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._messages: List[BaseMessage] = []
    
    def add_user_message(self, message: str) -> None:
        """Add a user message to the history."""
        self._messages.append(HumanMessage(content=message))
    
    def add_ai_message(self, message: str) -> None:
        """Add an AI message to the history."""
        self._messages.append(AIMessage(content=message))
        
    def add_system_message(self, message: str) -> None:
        """Add a system message to the history."""
        self._messages.append(SystemMessage(content=message))
    
    def clear(self) -> None:
        """Clear the message history."""
        self._messages = []
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Return the message history."""
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
def init_session_state():
    if "id" not in st.session_state:
        st.session_state.id = str(uuid.uuid4())
    if "file_cache" not in st.session_state:
        st.session_state.file_cache = {}
    if "dataframes" not in st.session_state:
        st.session_state.dataframes = {}
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "context" not in st.session_state:
        st.session_state.context = None
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}
    if "vector_stores" not in st.session_state:
        st.session_state.vector_stores = {}

def reset_chat() -> None:
    """Reset the chat history and clear conversation memory."""
    st.session_state.messages = []
    if st.session_state.current_file:
        file_key = f"{st.session_state.id}-{st.session_state.current_file}"
        if file_key in st.session_state.chat_histories:
            st.session_state.chat_histories[file_key].clear()
    st.session_state.context = None
    gc.collect()
    
def extract_code_from_response(response: str) -> Tuple[List[str], str]:
    """
    Extract all Python code snippets from a text response.
    
    Args:
        response: The text response potentially containing code blocks.
        
    Returns:
        List of extracted code snippets.
    """
    python_pattern = r"```python\s*(.*?)\s*```"
    python_blocks = re.findall(python_pattern, response, re.DOTALL)
    
    # Clean text by removing code blocks
    clean_text = re.sub(r"```python\s*(.*?)\s*```", "", response, flags=re.DOTALL)
    
    # If none found, try looking for generic code blocks
    if not python_blocks:
        generic_pattern = r"```\s*(.*?)\s*```"
        generic_blocks = re.findall(generic_pattern, response, re.DOTALL)
        
        # Only use generic blocks if they look like Python
        python_keywords = ['import', 'def', 'print', 'for', 'while', 'if', 'df', 'pd', 'np', 'px', 'go']
        for block in generic_blocks:
            if any(keyword in block for keyword in python_keywords):
                python_blocks.append(block)
        
        # Remove generic code blocks that look like Python
        for block in generic_blocks:
            if any(keyword in block for keyword in python_keywords):
                clean_text = clean_text.replace(f"```{block}```", "")
    
    # Clean up the code blocks and clean text
    cleaned_blocks = [block.strip() for block in python_blocks]
    clean_text = clean_text.strip()
    
    # Remove any empty lines or excessive whitespace
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
    
    return cleaned_blocks, clean_text

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
        "error": None,
        "code": code  # Store the executed code for reference
    }
    
    # Check if code looks dangerous
    dangerous_modules = ['os', 'sys', 'subprocess', 'shutil', 'requests']
    dangerous_funcs = ['eval', 'exec', 'compile', 'open', 'write', 'system']
    
    if any(f"import {module}" in code for module in dangerous_modules) or \
       any(f"from {module}" in code for module in dangerous_modules) or \
       any(func in code for func in dangerous_funcs):
        result["error"] = "Code contains potentially unsafe operations and was not executed."
        return result
    
    try:
        # Log code execution for debugging
        logger.info(f"Executing code: {code}")

        # Capture stdout
        with CaptureOutput() as captured:
            # Execute the code
            with st.spinner("Executing code..."):
                # Execute the code
                exec(code, {}, local_vars)
                result["stdout"] = captured.get_output()
        
        # Check for DataFrames with specific names or patterns
        df_priorities = [
            'result_df', 'df_result', 'filtered_df', 'processed_df', 
            'output_df', 'final_df', 'summary_df', 'stats_df', 'analysis_df'
        ]
        
        # First look for DataFrames with priority names
        for var_name in df_priorities:
            if var_name in local_vars and isinstance(local_vars[var_name], pd.DataFrame):
                result["df_result"] = local_vars[var_name]
                break
        
        # If not found, check for any DataFrame
        if result["df_result"] is None:
            df_patterns = ['df_', '_df', 'dataframe', 'data_']
            for var_name, var_value in local_vars.items():
                if isinstance(var_value, pd.DataFrame) and var_name != 'df':
                    # Check if variable name matches a pattern
                    if any(pattern in var_name.lower() for pattern in df_patterns):
                        result["df_result"] = var_value
                        break
        
        # If still not found, check for any DataFrame
        if result["df_result"] is None:
            for var_name, var_value in local_vars.items():
                if isinstance(var_value, pd.DataFrame) and var_name != 'df':
                    result["df_result"] = var_value
                    break
        
        # If no explicit DataFrame was found but 'df' was modified
        if result["df_result"] is None and 'df' in local_vars and not local_vars["df"].equals(df):
            result["df_result"] = local_vars["df"]
        
        # Check for Plotly figures
        fig_var_names = ['fig', 'figure', 'plot', 'chart', 'viz', 'visualization', 'graph']
        for var_name in fig_var_names:
            if var_name in local_vars and isinstance(local_vars[var_name], go.Figure):
                result["plot"] = local_vars[var_name]
                result["plot_type"] = "plotly"
                break
                
        # If not found, check any variable for a Plotly figure
        if result["plot"] is None:
            all_vars = list(local_vars.keys())
            for var_name in all_vars:
                var_value = local_vars[var_name]
                if isinstance(var_value, go.Figure):
                    result["plot"] = var_value
                    result["plot_type"] = "plotly"
                    logger.info(f"Found Plotly figure in variable: {var_name}")
                    break
                
        # Look for other result variables
        result_vars = ['result', 'output', 'summary', 'analysis', 'stats', 'count', 'total', 'average', 'mean']
        for var_name in result_vars:
            if var_name in local_vars and local_vars[var_name] is not None:
                if result["result"] is None:  # Only if not already found
                    # result["result"] = local_vars[var_name]
                    for var_name, var_value in local_vars.items():
                        if isinstance(var_value, (list, str, int, float)) and var_name != 'df':
                            result["result"] = var_value
                            break
                break
            
        # If no result was found but stdout exists, use that
        if result["result"] is None and result["df_result"] is None and result["plot"] is None:
            if result["stdout"] and len(result["stdout"].strip()) > 0:
                result["result"] = result["stdout"].strip()
                    
        logger.info(f"Execution results: df_result={result['df_result'] is not None}, " 
                   f"plot={result['plot_type'] if result['plot'] is not None else None}, "
                   f"result={result['result'] is not None}")
                    
        return result
    
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}")
        # Include traceback for better debugging
        import traceback
        result["error"] = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return result

def get_message_history(file_key: str) -> InMemoryChatMessageHistory:
    """
    Return the chat message history for this session.
    Creates a new history if one doesn't exist.
    
    Args:
        file_key: Unique identifier for the chat session
        
    Returns:
        InMemoryChatMessageHistory: The chat history for this session
    """
    if file_key not in st.session_state.chat_histories:
        logger.info(f"Creating new chat history for session {file_key}")
        st.session_state.chat_histories[file_key] = InMemoryChatMessageHistory(file_key)
    
    return st.session_state.chat_histories[file_key]

def initialize_loaded_data():
    """
    Initialize RAG components for all loaded dataframes in the session state.
    This sets up the necessary chains and dataframes for AI interaction.
    """
    # Get data from session state
    data_dict = st.session_state.get("dataframes_dict", {})
    if not data_dict:
        st.error("No data available. Please upload files first.")
        return False

    # Show initialization progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Process each dataframe
    for idx, (file_name, df) in enumerate(data_dict.items()):
        progress = (idx / len(data_dict))
        progress_bar.progress(progress)
        status_text.text(f"Setting up AI for {file_name}... ({idx+1}/{len(data_dict)})")

        file_key = f"{st.session_state.id}-{file_name}"

        try:
            # Store original DataFrame if not already stored
            if file_key not in st.session_state.dataframes:
                st.session_state.dataframes[file_key] = df.copy()
            
            # Initialize chat history if not exists
            if file_key not in st.session_state.chat_histories:
                st.session_state.chat_histories[file_key] = InMemoryChatMessageHistory(file_key)
            
            # Set as current file if none is set
            if not st.session_state.current_file:
                st.session_state.current_file = file_name
                
        except Exception as e:
            logger.error(f"Error initializing RAG for {file_name}: {str(e)}")
            status_text.error(f"Error setting up AI for {file_name}: {str(e)}")
            st.warning(f"Failed to set up AI for {file_name}. You might need to check your API key or network connection.")
            return False
    
    # Complete the progress bar
    progress_bar.progress(1.0)
    status_text.text("AI setup complete!")
    time.sleep(0.5)  # Brief pause to show completion
    status_text.empty()
    progress_bar.empty()
    
    return True

def direct_llm_response(query, df, file_key):
    """
    Directly get a response from the LLM without using complex chain or message history.
    
    Args:
        query: User query
        df: DataFrame being analyzed
        file_key: Unique file identifier
        
    Returns:
        Response text and debug info
    """
    try:
        classifier_llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
            api_key=OPENAI_API_KEY,
        )
        
        # Classify if the query is data-related
        classification_prompt = f"""
        Determine if the following query is related to data analysis, Excel, dataframes, or data visualization.
        Only respond with 'YES' if the query is data-related or 'NO' if it's not data-related.
        
        Query: {query}
        """
        
        classification_response = classifier_llm.invoke([
            {"role": "user", "content": classification_prompt}
        ])
        
        # Check if query is not data-related
        if "NO" in classification_response.content.upper():
            return {
                "answer": "I can only answer questions related to your uploaded data. Please ask a question about data analysis, visualization, or specific aspects of your dataset."
            }, {"is_data_related": False}
        # # Step 1: Validate the query against the DataFrame
        # if df is None or df.empty:
        #     return {"answer": "Data not found or irrelevant questions."}, {}

        # # Extract keywords from the query
        # query_keywords = set(re.findall(r'\b\w+\b', query.lower()))

        # # Get column names and sample data from the DataFrame
        # df_columns = set(df.columns.str.lower())
        # df_sample_data = set(df.head(10).astype(str).stack().str.lower())

        # # Check if query keywords match the DataFrame columns or sample data
        # if not query_keywords.intersection(df_columns) and not query_keywords.intersection(df_sample_data):
        #     return {"answer": "Data not found or irrelevant questions."}, {}
        
        # Get and format the chat history for context
        chat_history = get_message_history(file_key)
        history_text = ""
        
        # Only include the last 10 messages to avoid token limit issues
        recent_messages = chat_history.messages[-10:] if len(chat_history.messages) > 10 else chat_history.messages
        
        if recent_messages:
            history_text = "Previous conversation:\n"
            for msg in recent_messages:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                # Limit message content for history to avoid token limits
                content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
                history_text += f"{role}: {content}\n\n"
        
        # Create LLM instance
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.8,
            streaming=True,
            api_key=OPENAI_API_KEY,
        )
        
        # Create context about the dataframe
        df_info = {
            "shape": str(df.shape),
            "columns": str(df.columns.tolist()),
            "dtypes": str(df.dtypes.to_dict()),
            "head": df.head(5).to_string(),
            "null_counts": str(df.isnull().sum().to_dict()),
            "description": df.describe().to_string() if not df.empty else "Empty DataFrame"
        }
        
        # Check if this is a visualization request with more comprehensive detection
        viz_terms = ["chart", "plot", "graph", "visualize", "visualization", "show me", "give me", "display", "draw"]
        viz_types = ["bar", "line", "scatter", "pie", "histogram", "box", "heatmap", "area", "bubble", 
                     "treemap", "funnel", "radar", "violin", "correlation", "distribution"]
        
        is_viz_request = (any(term in query.lower() for term in viz_terms) and 
                          any(viz_type in query.lower() for viz_type in viz_types))
        
        # Enhanced system message with more explicit instructions
        system_message = """You are an Excel data analysis expert with deep knowledge of pandas and data visualization.
        
        CONVERSATION HANDLING:
        1. Pay close attention to the conversation history
        2. Reference previous questions and answers when appropriate
        3. If the user asks about previous results or charts, refer to them specifically
        4. Make connections between current and past questions
        
        RESPONSE GUIDELINES:
        1. Analyze the data carefully before responding
        2. Always provide concise insights relevant to the user's question
        3. For calculations, show both the code and the result
        4. Format numbers and percentages appropriately for readability
        5. For complex analysis, break down your approach step by step
        
        DATA HANDLING GUIDELINES:
        1. ALWAYS check data types before performing operations
        2. Use pd.to_numeric(column, errors='coerce') when converting strings to numbers
        3. Handle missing values with appropriate techniques (fillna, dropna)
        4. For string columns that should be numeric, ALWAYS clean them first:
        - df['column'] = df['column'].str.replace(r'[^0-9.-]', '', regex=True)
        - df['column'] = pd.to_numeric(df['column'], errors='coerce')
        5. NEVER directly use .astype(float) on string columns
        
        CODE GUIDELINES:
        1. The DataFrame is available as the variable 'df'
        2. Make all code completely executable - it will be run automatically
        3. For visualizations, ALWAYS use Plotly (import plotly.express as px, import plotly.graph_objects as go)
        4. NEVER use matplotlib, seaborn, or other visualization libraries
        5. For Plotly visualizations:
        - Use appropriate chart types for the data
        - Add clear titles, labels, and legends
        - Set appropriate color schemes
        - DO NOT include fig.show() in your code
        - ALWAYS STORE THE FINAL FIGURE IN A VARIABLE NAMED 'fig'
        6. Store analysis results in descriptive variable names (e.g., 'filtered_df', 'result_df', etc.)
        7. PUT ALL CODE IN A SINGLE EXECUTABLE BLOCK - do not split into separate blocks

        OUTPUT GUIDELINES:
        1. Structure your response clearly with sections and bullet points where appropriate
        2. Always place executable Python code in ```python code blocks```
        3. When providing both text analysis and a visualization, put the code AFTER your textual insights
        4. Provide meaningful interpretation of any visualizations you create
        """
        
        # Enhanced user message with more context and clearer instructions
        user_message = f"""
        I'm analyzing a DataFrame with the following information:
        
        Shape: {df_info['shape']}
        Columns: {df_info['columns']}
        Data Types: {df_info['dtypes']}
        Null Values: {df_info['null_counts']}
        
        Sample data:
        {df_info['head']}
        
        Statistical summary:
        {df_info['description']}
        
        USER QUERY: {query}
        
        {
            "Please generate Python code using Plotly to create a clear, informative visualization. Include both the code and your interpretation of what the visualization shows." 
            if is_viz_request else 
            "Please analyze this data based on my query. Include Python code for any calculations, and if relevant, include a visualization using Plotly."
        }
        """

        # Get response from LLM
        start_time = time.time()
        response = llm.invoke([
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ])
        elapsed_time = time.time() - start_time
        
        # Extract content from response
        answer = response.content
        
        # Add user message to chat history
        chat_history.add_user_message(query)
        chat_history.add_ai_message(answer)
        
        # Debug info
        debug_info = {
            "execution_time": f"{elapsed_time:.2f} seconds",
            "response_length": len(answer),
            "is_viz_request": is_viz_request,
            "is_data_related": True,
        }
        
        # Extract and execute code from the response
        code_blocks, clean_text = extract_code_from_response(answer)
        result = {
            "answer": answer,
            "clean_text": clean_text  # Text without code blocks
        }
        
        if code_blocks:
            # Execute the first code block found
            combined_code = "\n\n".join(code_blocks)
            execution_result = execute_code_safely(combined_code, df)
            
            # Handle execution results
            if execution_result.get("error"):
                # If there's an error, add it to the response but don't replace the entire answer
                result["error"] = execution_result["error"]
                logger.error(f"Code execution error: {execution_result['error']}")
            else:
                # Add execution results to the response
                result.update(execution_result)
                
                # Log successful execution
                logger.info(f"Code executed successfully. Results: plot={execution_result['plot'] is not None}, df_result={execution_result['df_result'] is not None}")
        
        return result, debug_info
        
    except Exception as e:
        logger.error(f"Error in direct_llm_response: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Create error response
        error_response = {
            "answer": f"I encountered an error while processing your request: {str(e)}. Please try rephrasing your question or try a different analysis."
        }
        
        error_debug = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "query": query
        }
        
        return error_response, error_debug

def ExcelChat_main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Load environment variables
    load_dotenv()
    
    # Add custom CSS for better UI
    st.markdown("""
    <style>
    .chat-container {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #f9f9f9;
    }
    .code-block {
        background-color: #272822;
        color: #f8f8f2;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .plot-container {
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
        background-color: #f5f7ff;
        border: 1px solid #e0e6ff;
    }
    .result-container {
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
        background-color: #eef2f5;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if not OPENAI_API_KEY:
        st.error("⚠️ OpenAI API key not found. Please set your OPENAI_API_KEY in the .env file.")
        st.stop()

    # Initialize data if files are already in session
    if "dataframes_dict" in st.session_state and st.session_state.dataframes_dict:
        success = initialize_loaded_data()
        
        if not success:
            st.info("Please upload your Excel files to begin analysis.")
            st.stop()
                
        # File selector if multiple files
        data_dict = st.session_state.get("dataframes_dict", {})
        if len(data_dict) > 1:
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_file = st.selectbox(
                    "Select data to analyze:", 
                    list(data_dict.keys()),
                    index=list(data_dict.keys()).index(st.session_state.current_file) if st.session_state.current_file else 0
                )
            with col2:
                st.write("")  # Spacing
                st.write("")  # Spacing
                if st.button("Switch Dataset"):
                    if selected_file != st.session_state.current_file:
                        st.session_state.current_file = selected_file
                        reset_chat()
                        st.rerun()
        
            
        # Show current dataset info
        if st.session_state.current_file:
            file_key = f"{st.session_state.id}-{st.session_state.current_file}"
            if file_key in st.session_state.dataframes:
                current_df = st.session_state.dataframes[file_key]
                st.write(f"**Current dataset:** {st.session_state.current_file}")
                
                # Show data preview in expander
                with st.expander("Data Preview", expanded=False):
                    st.dataframe(current_df.head(10), use_container_width=True)
            
        # Add reset button
        if st.button("Reset Chat", key="reset_main"):
            reset_chat()
            st.success("Chat reset!")
            time.sleep(0.5)
            st.rerun()
            
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display additional components based on message type
                if "df_result" in message and message["df_result"] is not None:
                    with st.container(border=True):
                        st.dataframe(message["df_result"], use_container_width=True)
                        
                        # Add download button for results
                        if isinstance(message["df_result"], pd.DataFrame) and not message["df_result"].empty:
                            csv = message["df_result"].to_csv(index=False)
                            st.download_button(
                                label="Download as CSV",
                                data=csv,
                                file_name="analysis_result.csv",
                                mime="text/csv",
                                key=f"download_{uuid.uuid4()}"
                            )

                if "plot" in message and message["plot"] is not None:
                    st.plotly_chart(message["plot"])
            
        # Chat input
        if prompt := st.chat_input("Ask about your Excel data (e.g., 'Show me a bar chart of sales by region')"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # st.session_state.chat_histories[file_key].add_message(HumanMessage(content=prompt))
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Process the query
            if st.session_state.current_file and st.session_state.dataframes:
                file_key = f"{st.session_state.id}-{st.session_state.current_file}"
                
                if file_key in st.session_state.dataframes:
                    # Get the dataframe
                    df = st.session_state.dataframes[file_key]
                    
                    # Process with direct LLM approach
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        
                        try:
                            with st.spinner("Thinking..."):
                                # Get response directly from LLM
                                response, debug_info = direct_llm_response(prompt, df, file_key)
                                
                                # Extract the response text
                                answer = response.get("clean_text", response.get("answer", ""))
                                message_placeholder.markdown(answer)
                                # st.session_state.chat_histories[file_key].add_message(AIMessage(content=answer))
                                
                                # Initialize results for storing in chat history
                                response_data = {
                                    "role": "assistant",
                                    "content": answer
                                }
                                
                                # Handle visualization if present
                                if "plot" in response and response["plot"] is not None:
                                    st.plotly_chart(response["plot"], use_container_width=True)
                                    response_data["plot"] = response["plot"]
                                    response_data["plot_type"] = "plotly"
                                
                                # Show DataFrame result if available
                                if "df_result" in response and response["df_result"] is not None:
                                    with st.container(border=True):
                                        st.dataframe(response["df_result"], use_container_width=True)
                                    response_data["df_result"] = response["df_result"]
                                    
                                # Add this block to display the "result" if it exists
                                if "result" in response and response["result"] is not None:
                                    with st.container():
                                        st.write("Result:")
                                        st.write(response["result"])
                                
                                # Show error message if there was an error during code execution
                                if "error" in response and response["error"]:
                                    with st.expander("Execution Error", expanded=False):
                                        st.error(response["error"])
                            
                            # Add to chat history
                            st.session_state.messages.append(response_data)
                            
                        except Exception as e:
                            error_msg = f"Error generating response: {str(e)}"
                            message_placeholder.error(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": error_msg
                            })