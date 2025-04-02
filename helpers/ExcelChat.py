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
    
    class SafeMatplotlib:
        def __init__(self):
            self.figure_created = False
            
        def figure(self, *args, **kwargs):
            self.figure_created = True
            # Return a basic figure that won't cause issues
            return plt.figure(*args, **kwargs)
            
        def __getattr__(self, name):
            # Handle restricted functions safely
            if name in ['gca', 'gcf', 'subplot', 'subplots', 'axes']:
                return lambda *args, **kwargs: None
            # For other functions, use the actual matplotlib functions
            return getattr(plt, name)
    
    # Add safe matplotlib to locals
    local_vars["plt"] = SafeMatplotlib()
    local_vars["sns"] = sns  # Keep seaborn for compatibility
    
    # Check if code is trying to save a plot to a path
    save_fig_pattern = r'(plt\.savefig|fig\.savefig)\([\'"]([^\'"]+)[\'"]\)'
    save_match = re.search(save_fig_pattern, code)
    if save_match:
        result["plot_path"] = save_match.group(2)
    
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
            exec(code, {}, local_vars)
            result["stdout"] = captured.get_output()
        
        # Look for specific result types
        
        # Check for Analysis Result Variables - expanded list of common output variables
        analysis_var_names = [
            'result', 'output', 'summary', 'analysis', 'stats', 'metrics', 
            'top_industries', 'grouped_data', 'sorted_df', 'results', 'top_10',
            'top_10_industries', 'response', 'answer', 'final_result', 'data',
            'aggregated', 'filtered', 'processed', 'calculated', 'ranked'
        ]
        
        # First try to find any analysis result
        for var_name in analysis_var_names:
            if var_name in local_vars and local_vars[var_name] is not None:
                # If it's a DataFrame, prioritize as df_result
                if isinstance(local_vars[var_name], pd.DataFrame):
                    result["df_result"] = local_vars[var_name]
                else:
                    # Otherwise store as general result
                    result["result"] = local_vars[var_name]
                break
        
        # Check for DataFrames with standard naming patterns
        df_var_patterns = ['df_', '_df', 'dataframe', 'data_', 'filtered', 'sorted', 'grouped']
        for var_name, var_value in local_vars.items():
            if isinstance(var_value, pd.DataFrame) and var_name != 'df':
                # Check if this might be a result DataFrame based on naming
                is_result_df = any(pattern in var_name.lower() for pattern in df_var_patterns)
                if is_result_df or result["df_result"] is None:
                    result["df_result"] = var_value
                    # If this seems like a primary result, stop looking
                    if is_result_df:
                        break
        
        # If no explicit DataFrame was found but 'df' was modified
        if result["df_result"] is None and 'df' in local_vars and not local_vars["df"].equals(df):
            result["df_result"] = local_vars["df"]
        
        # Check for Plotly figures with various names
        fig_var_names = ['fig', 'figure', 'plot', 'chart', 'vis', 'visualization', 'graph']
        for var_name in fig_var_names:
            if var_name in local_vars and isinstance(local_vars[var_name], go.Figure):
                result["plot"] = local_vars[var_name]
                result["plot_type"] = "plotly"
                break
                
        # Generic check for any Plotly figure if none found by name
        if result["plot"] is None:
            for var_name, var_value in local_vars.items():
                if isinstance(var_name, str) and isinstance(var_value, go.Figure):
                    result["plot"] = var_value
                    result["plot_type"] = "plotly"
                    break
                    
        # Only check for Matplotlib figures if no Plotly figure found
        if result["plot"] is None and plt.get_fignums():
            fig = plt.gcf()
            result["plot"] = fig
            result["plot_type"] = "matplotlib"
            
        # If no result was found but we have stdout, try to extract information from it
        if result["result"] is None and result["df_result"] is None and result["plot"] is None:
            stdout = result["stdout"]
            if stdout and len(stdout.strip()) > 0:
                # For tabular data in stdout, try to convert to DataFrame
                if '\n' in stdout and '\t' in stdout or ',' in stdout:
                    try:
                        # Try to parse as CSV
                        result["df_result"] = pd.read_csv(StringIO(stdout), sep=None, engine='python')
                    except:
                        # If parsing failed, just use as text result
                        result["result"] = stdout.strip()
                else:
                    result["result"] = stdout.strip()
                    
        # Log the execution results for debugging
        logger.info(f"Execution results: df_result={result['df_result'] is not None}, " 
                   f"plot={result['plot_type'] if result['plot'] is not None else None}, "
                   f"result={result['result'] is not None}")
                    
        return result
    
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}")
        result["error"] = str(e)
        return result

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
        # Display a full-page loader
        with st.spinner("🔄 Loading data and creating embeddings. Please wait..."):
            # Ensure DataFrame is not empty and has data
            if df.empty:
                raise ValueError("DataFrame is empty")
        
            # Convert DataFrame to string representation for document creation
            df_str = df.to_string()
            
            # Create documents from DataFrame
            documents = [Document(page_content=df_str, metadata={"source": file_key})]
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=250
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
                You can use this DataFrame to answer the user's questions. If you need to perform any data processing or analysis, you can do so directly on this DataFrame.
                1. The DataFrame is already defined as variable 'data' - DO NOT redefine it or try to access it as dfs[0]
                2. If visualization is needed, use Streamlit native charts (st.bar_chart, st.line_chart) whenever possible.
                3. For more complex visualizations, use Plotly (px.line, px.bar, px.scatter, etc.) or Altair.
                4. Keep your answer concise and data-focused.
                5. For Plotly charts, always use st.plotly_chart(fig, use_container_width=True)
                6. For Altair charts, always use st.altair_chart(chart, use_container_width=True)
                7. Ensure visualizations have clear titles and labels.
                8. DO NOT use matplotlib or seaborn for visualizations.
                9. DO NOT create new DataFrames using `dfs[0]` - the DataFrame is already available as 'data'
                10. Always use the variable name 'data' to refer to the DataFrame - not 'df'
                11. DO NOT use plt.show() - use Streamlit's display functions instead

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

def create_pandasai_df(df):
    """
    Create a PandasAI SmartDataframe for advanced data analysis.
    
    Args:
        df: The pandas DataFrame to enhance.
        
    Returns:
        SmartDataframe: The enhanced DataFrame with AI capabilities.
    """
        
    return df

def generate_visualization_code(prompt: str, df: pd.DataFrame) -> str:
    """
    Use OpenAI directly to generate Python code for data visualization based on user prompt.
    
    Args:
        prompt: The user's visualization request
        df: The pandas DataFrame to visualize
        
    Returns:
        Generated Python code for visualization
    """
    try:
        # Ensure OpenAI API key is set
        api_key = OPENAI_API_KEY
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        # Create ChatOpenAI instance
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
            api_key=api_key
        )
        
        # Create context about the dataframe
        df_info = {
            "shape": str(df.shape),
            "columns": str(df.columns.tolist()),
            "dtypes": str(df.dtypes.to_dict()),
            "head": df.head(5).to_string(),
            "description": df.describe().to_string()
        }
        
        # Create system message with visualization guidelines
        system_message = """You are an expert data visualization assistant. 
        Your task is to generate Python code that creates visualizations based on user requests.
        The code must use the DataFrame 'df' which is already loaded.
        
        GUIDELINES:
        1. Always use Plotly (import plotly.express as px, import plotly.graph_objects as go) for visualization
        2. Don't use matplotlib or seaborn
        3. Create proper titles, labels, and legends
        4. Make the visualization colorful and professional
        5. Return ONLY the Python code, nothing else
        6. The code should be complete and ready to execute
        7. Don't include code to display the plot (like plt.show() or fig.show())
        8. Make sure every plot has a descriptive title
        9. Include insights or findings in the plot title or caption
        """
        
        # User message with dataframe info and visualization request
        user_message = f"""
        I have a pandas DataFrame with the following information:
        
        Shape: {df_info['shape']}
        Columns: {df_info['columns']}
        Data Types: {df_info['dtypes']}
        
        Sample data:
        {df_info['head']}
        
        Statistical summary:
        {df_info['description']}
        
        VISUALIZATION REQUEST: {prompt}
        
        Generate Python code to create this visualization using the DataFrame 'df'.
        """
        
        # Get response from OpenAI
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        response = llm.invoke(messages)
        
        # Extract code from response
        code = response.content
        
        # If the response contains code blocks, extract them
        if "```python" in code:
            code_blocks = re.findall(r"```python\n(.*?)\n```", code, re.DOTALL)
            if code_blocks:
                return code_blocks[0]
        
        # If no code blocks with ```python tag, look for general code blocks
        if "```" in code:
            code_blocks = re.findall(r"```\n(.*?)\n```", code, re.DOTALL)
            if code_blocks:
                return code_blocks[0]
                
        # If no code blocks found, return the whole response
        return code
        
    except Exception as e:
        logger.error(f"Error generating visualization code: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        return f"# Error generating visualization code: {str(e)}"


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

st.markdown(
    """
    <style>
    .st-emotion-cache-t74pzu {
        background: #fff;
        padding: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


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

    # with col2:
    #     st.button("Reset Chat ↺", on_click=reset_chat)
    
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
    

    # Input container for chat input and reset button
    # st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    
    # # Chat input box
    # prompt = st.text_input("Ask me about your data...", key="chat_input", label_visibility="collapsed")
    
    # # Reset button
    # if st.button("Reset ↺", on_click=reset_chat, key="reset_chat_bottom"):
    #     pass
    
    # st.markdown('</div>', unsafe_allow_html=True)
    
    # Accept user input
    if prompt := st.chat_input("Ask me about your data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response if file has been uploaded
        if st.session_state.current_file and st.session_state.file_cache:
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
                                   "show me", "display", "bar chart", "line graph", "show",
                                   "histogram"]
                    is_viz_request = any(keyword in prompt.lower() 
                                        for keyword in viz_keywords)
                    
                    if is_viz_request:
                        try:
                            with st.spinner("Generating visualization..."):
                                # Generate visualization code using OpenAI
                                generated_code = generate_visualization_code(prompt, df)
                                
                                # Execute the generated code
                                execution_result = execute_code_safely(generated_code, df)
                                
                                # Handle execution results
                                if execution_result["error"]:
                                    message_placeholder.write(f"❌ Error: {execution_result['error']}")
                                else:
                                    # Display plot if generated
                                    if execution_result["plot"] is not None:
                                        message_placeholder.write("Here's the visualization based on your request:")
                                        if execution_result["plot_type"] == "matplotlib":
                                            st.pyplot(execution_result["plot"])
                                        else:  # Plotly
                                            st.plotly_chart(execution_result["plot"], use_container_width=True)
                                        
                                        # Add to chat history
                                        st.session_state.messages.append({
                                            "role": "assistant",
                                            "content": "Here's the visualization based on your request:",
                                            "plot": execution_result["plot"]
                                        })
                                    else:
                                        # Display any other result
                                        message_placeholder.write("Visualization generated, but no plot was found.")
                        except Exception as e:
                            message_placeholder.write(f"❌ Error generating visualization: {str(e)}")
                    else:
                        # Handle non-visualization requests (fallback to standard QA)
                        try:
                            with st.spinner("Processing your request..."):
                                # Generate response code for non-visualization requests
                                generated_code = generate_visualization_code(prompt, df)
                                
                                # Execute the generated code
                                execution_result = execute_code_safely(generated_code, df)
                                
                                # Handle execution results
                                if execution_result["error"]:
                                    message_placeholder.write(f"❌ Error: {execution_result['error']}")
                                else:
                                    # Display DataFrame result if generated
                                    if execution_result["df_result"] is not None:
                                        message_placeholder.write("Here's the result of your query:")
                                        st.dataframe(execution_result["df_result"])
                                        
                                        # Add to chat history
                                        st.session_state.messages.append({
                                            "role": "assistant",
                                            "content": "Here's the result of your query:",
                                            "df_result": execution_result["df_result"]
                                        })
                                    else:
                                        # Display any other result
                                        message_placeholder.write(str(execution_result["result"]))
                                        st.session_state.messages.append({
                                            "role": "assistant",
                                            "content": str(execution_result["result"])
                                        })
                        except Exception as e:
                            message_placeholder.write(f"❌ Error processing your request: {str(e)}")
            else:
                st.error("Something went wrong with the file processing. Please re-upload the file.")
        else:
            # No file uploaded yet
            with st.chat_message("assistant"):
                missing_file_message = "Please upload a data file first. I need data to answer your questions."
                st.warning(missing_file_message)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": missing_file_message
                })