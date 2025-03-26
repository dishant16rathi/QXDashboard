import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
# from pandasai.responses.response_parser import ResponseParser
from streamlit_autorefresh import st_autorefresh
import altair as alt
import plotly.express as px
import tkinter as tk
from tkinter import filedialog
import plotly.graph_objects as go
import multiprocessing
import dotenv

# Load environment variables
dotenv.load_dotenv()

# ----------------------
# Page Initialization
# ----------------------
def page_initialization():
    st.set_page_config(page_title="Dashboard for Qx", page_icon="📊", layout="wide")
    
    # Create a layout for the logo in the top left corner
    header_container = st.container()
    with header_container:
        cols = st.columns([0.15, 0.85])
        with cols[0]:
            # You can replace "logo.png" with your actual image path
            # You can use a URL or a local image in your project directory
            st.image("logo.png", width=100)  # Adjust width as needed
    
    # Custom CSS injection
    st.markdown("""
    <style>
        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }
        .stApp {
            background: #f8f9fa;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        .stButton>button {
            background: blue;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
        .stSelectbox>div>div>div {
            border-radius: 8px;
        }
        
        /* Position the logo container to appear at the top left */
        [data-testid="stAppViewContainer"] > div:first-child {
            position: relative;
        }
        [data-testid="stAppViewContainer"] > div:first-child > div:first-child > div:first-child {
            position: absolute;
            top: 0;
            left: 0;
            z-index: 1000; /* Ensure it appears above other elements */
            background: transparent;
        }
        /* Make the column container take less space */
        [data-testid="stAppViewContainer"] > div:first-child > div:first-child > div:first-child > div {
            padding-top: 10px;
            padding-left: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state.page = "main"
    if "data" not in st.session_state:
        st.session_state.data = None
    if "file_selected" not in st.session_state:
        st.session_state.file_selected = None
    if "button_clicked" not in st.session_state:
        st.session_state.button_clicked = False
    if "dataframes_dict" not in st.session_state:
        st.session_state["dataframes_dict"] = {}
    if "Get_Ans_bt_clicked" not in st.session_state:
        st.session_state.Get_Ans_bt_clicked = False
    # Initialize session state for folder and selected files
    if "selected_folder" not in st.session_state:
        st.session_state.selected_folder = None
    
def switch_page(page):
    st.session_state.page = page

def load_data():
    """Loads data from the selected file and updates session state."""
    if "file_selected" in st.session_state and st.session_state.file_selected is not None:
        file_paths = st.session_state.file_selected
        dataframes_dict = {}  # Dictionary to store DataFrames
        
        # Initialize file_timestamps if it doesn't exist
        if "file_timestamps" not in st.session_state:
            st.session_state.file_timestamps = {}

        for path in file_paths:
            file_name = os.path.basename(path)  # Extract filename (without full path)
            
            try:
                # Check file modification time
                current_mtime = os.path.getmtime(path)
                if path not in st.session_state.file_timestamps:
                    st.session_state.file_timestamps[path] = current_mtime
                    load_needed = True
                else:
                    load_needed = current_mtime > st.session_state.file_timestamps[path]
                
                if load_needed:
                    # Read file based on extension
                    if path.endswith(".csv"):
                        df = pd.read_csv(path)
                    elif path.endswith(".xlsx"):
                        df = pd.read_excel(path, engine='openpyxl')  # Ensure 'openpyxl' is installed
                    else:
                        st.warning(f"⚠️ Unsupported file format: {file_name}")
                        continue
                    
                    # Store DataFrame in dictionary
                    dataframes_dict[file_name] = df
                    
                    # Update timestamp
                    st.session_state.file_timestamps[path] = current_mtime
                    st.toast(f"Refreshed data from {file_name}", icon="🔄")
                else:
                    # Use the existing dataframe from session state if available
                    if file_name in st.session_state.get("dataframes_dict", {}):
                        dataframes_dict[file_name] = st.session_state["dataframes_dict"][file_name]

            except Exception as e:
                st.error(f"⚠️ Unexpected error while reading {file_name}: {str(e)}")

        # Store dictionary in session state
        if dataframes_dict:
            st.session_state["dataframes_dict"] = dataframes_dict

            # Set the first dataframe as default for chat_with_data if not already set
            if st.session_state.data is None:
                first_key = next(iter(dataframes_dict))
                st.session_state.data = dataframes_dict[first_key]
        else:
            st.error("🚨 No valid files loaded. Please check your file selections.")

def Dashboard_page():
    col1, col2, col3 = st.columns([0.8, 0.2, 0.2])
    with col1:
        st.title("🤖 AI-Generated Insights")
    with col3:
        custom_dash = st.button("QX Intelligence Assitant ➡️")
        if custom_dash:
            switch_page("page_dashboard")
            st.session_state.button_clicked = False
    with col2:
        back = st.button("⬅️ Back to File Selection")
        if back:
            switch_page("main")
            st.session_state.button_clicked = False
    
    # Get data from session state
    data_dict = st.session_state.get("dataframes_dict", {})
    if not data_dict:
        st.error("No data available. Please go back and select files.")
        return
    
    # Add a selector for which dataframe to analyze
    if len(data_dict) > 1:
        selected_file = st.selectbox("Select file to analyze:", list(data_dict.keys()))
        st.session_state.data = data_dict[selected_file]
    
    # Then call the AI dashboard function
    from helpers.raw_utils import ai_dashboard
    ai_dashboard()

# ----------------------
# Function to List Files in a Folder
# ----------------------
def list_files_in_folder(folder_path):
    files_dict = {}
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            file_path_list = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path_list):
                files_dict[file_name] = file_path_list
    return files_dict

def preview_data():
    if st.session_state.dataframes_dict is not None:
        dataframes_dict = st.session_state.dataframes_dict
        for file, df in dataframes_dict.items():
            st.subheader(f"📂 Preview: {file}")
            st.dataframe(df.head())
    elif st.session_state.dataframes_dict is None:
        st.write("Please select the files the first.")

def pick_folder(queue):
    """Runs the Tkinter folder selection dialog in a separate process."""
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    queue.put(folder_path)
    root.destroy()
    
def select_file():
    st.title("📂 Please Enter Folder Path")

    # Text input for folder path
    folder_path = st.text_input("Enter the full path to your data folder:", 
                               placeholder="e.g., C:\\Users\\YourName\\Documents\\Data",
                               help="Enter the complete folder path where your CSV/Excel files are stored")
    
    # Button to confirm folder path
    if st.button("📂 Load Files from Folder"):
        if not folder_path:
            st.warning("⚠️ Please enter a folder path first.")
            return
            
        if not os.path.exists(folder_path):
            st.error(f"❌ The folder path '{folder_path}' does not exist. Please check and try again.")
            return
            
        if not os.path.isdir(folder_path):
            st.error(f"❌ '{folder_path}' is not a valid directory. Please enter a folder path.")
            return
            
        # Store valid folder in session state
        st.session_state.selected_folder = folder_path
        st.session_state.selected_files = []  # Reset selected files
        st.success(f"✅ Successfully loaded folder: **{folder_path}**")

    # Show file selection only if a folder is selected
    if st.session_state.get("selected_folder"):
        folder = st.session_state.selected_folder
        try:
            # Get only CSV and Excel files
            files = [f for f in os.listdir(folder) 
                    if os.path.isfile(os.path.join(folder, f)) 
                    and (f.lower().endswith('.csv') or f.lower().endswith('.xlsx') or f.lower().endswith('.xls'))]

            if files:
                # Multi-file selection with a max limit of 5
                selected_files = st.multiselect(
                    "📑 Select up to 5 files:", 
                    files, 
                    key="file_selector", 
                    default=[os.path.basename(f) for f in st.session_state.selected_files]  # Maintain selection
                )

                # Enforce the 5-file limit
                if len(selected_files) > 5:
                    st.warning("⚠️ You can select a maximum of **5 files** only!")
                else:
                    # Update session state with selected file paths
                    st.session_state.file_selected = [os.path.join(folder, f) for f in selected_files]

                # Display selected file paths
                if st.session_state.selected_files is not None:
                    for path in st.session_state.selected_files:
                        st.write(f"📄 `{path}`")
                    load_data()
                    preview_data()
                    if st.button("Create Dashboard"):
                        switch_page("Dashboard")
                
                else:
                    st.info("ℹ️ Please select at least one file.")
                    preview_data()
            else:
                st.warning(f"⚠️ No CSV or Excel files found in '{folder}'. Please check the folder content.")
        except Exception as e:
            st.error(f"❌ Error reading folder: {e}")


def select_file_old():
    st.title("📂 Please Select Folder Location")

   # Button to select folder
    if st.button("🗂️ Select Folder"):
        ctx = multiprocessing.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=pick_folder, args=(q,))
        p.start()
        p.join()  # Wait for the process to finish
        folder = q.get()

        if folder:
            st.session_state.selected_folder = folder  # Store folder in session state
            st.session_state.selected_files = []  # Reset selected files
            st.success(f"✅ Selected Folder: **{folder}**")

    # Show file selection only if a folder is selected
    if st.session_state.selected_folder:
        folder = st.session_state.selected_folder
        try:
            files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

            if files:
                # Multi-file selection with a max limit of 5
                selected_files = st.multiselect(
                    "📑 Select up to 5 files:", 
                    files, 
                    key="file_selector", 
                    default=[os.path.basename(f) for f in st.session_state.selected_files]  # Maintain selection
                )

                # Enforce the 5-file limit
                if len(selected_files) > 5:
                    st.warning("⚠️ You can select a maximum of **5 files** only!")
                else:
                    # Update session state with selected file paths
                    st.session_state.file_selected = [os.path.join(folder, f) for f in selected_files]

                # st.write(st.session_state.selected_files)
                # st.write(st.session_state.dataframes_dict)

                # Display selected file paths
                if st.session_state.selected_files is not None:
                    # st.success("✅ **Selected File Paths:**")
                    for path in st.session_state.selected_files:
                        st.write(f"📄 `{path}`")
                    load_data()
                    preview_data()
                    if st.button("Create Dashboard"):
                        switch_page("Dashboard")
                
                else:
                    st.info("ℹ️ Please select at least one file.")
                    preview_data()

                
            else:
                st.warning("⚠️ No files found in the selected folder.")
        except Exception as e:
            st.error(f"❌ Error reading folder: {e}")

# ----------------------
# Dashboard Page
# ----------------------
def dashboard_page():
    data_dict = st.session_state.get("dataframes_dict", {})
    if not data_dict:
        st.error("No data available. Please go back and select files.")
        if st.button("⬅️ Back to File Selection"):
            switch_page("main")
        if st.button("⬅️ Back to AI Dashboard"):
            switch_page("Dashboard")
        return

    col1, col2, col3 = st.columns([0.8, 0.4, 0.4])
    with col1:
        st.title("💬 QX Intelligence Assistant")
    with col2:
        aipage = st.button("⬅️ Back to Dashboard")
        if aipage:
            switch_page("Dashboard")
            st.session_state.button_clicked = False
    with col3:
        back = st.button("⬅️ Back to File Selection")
        if back:
            switch_page("main")
            st.session_state.button_clicked = False

    with st.expander("📂 Data Preview", expanded=False):
        for file_name, df in data_dict.items():
            st.subheader(f"📂 Data of {file_name}")
            st.dataframe(df)

    # Remove the tabs and directly show only the Chat with Data section
    # from extra_utils import chat_with_data
    # chat_with_data()
    
    with st.spinner("Loading AI Assistant... This may take a moment as we analyze your data."):
        from helpers.ExcelChat import ExcelChat_main
        ExcelChat_main()

# ----------------------
# Main Function
# ----------------------
def main():
    page_initialization()
    # Auto-refresh every 5 seconds (5000 milliseconds)
    # st_autorefresh(interval=5000, key="data_refresh")
    if st.session_state.page == "main":
        select_file()
    elif st.session_state.page == "page_dashboard":
        dashboard_page()
    elif st.session_state.page == "Dashboard":
        Dashboard_page()

if __name__ == "__main__":
    main()