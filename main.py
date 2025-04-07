import streamlit as st
import pandas as pd
import os
import dotenv


dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def page_initialization():
    st.set_page_config(page_title="QX Dashboard", page_icon="📊", layout="wide")

    # Initialize session state for the page if not already set
    if "page" not in st.session_state:
        st.session_state.page = "main"  # Default to the File Selection page
    if "dataframes_dict" not in st.session_state:
        st.session_state["dataframes_dict"] = {}  # Initialize as an empty dictionary
    if "selected_folder" not in st.session_state:
        st.session_state.selected_folder = None
    if "file_selected" not in st.session_state:
        st.session_state.file_selected = None
    if "data" not in st.session_state:
        st.session_state.data = None
    if "button_clicked" not in st.session_state:
        st.session_state.button_clicked = False  # Default to the File Selection page
        
    # Custom CSS injection
    st.markdown("""
    <style>
        [data-testid="stHeader"] {
            background: white !important;
            border-bottom: 1px solid #e0e0e0;
            padding: 0.5rem 2rem;
        }
        
        .stApp {
            background: #f8fafc;
        }
        
        .card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border: 1px solid #f1f1f1;
            margin-bottom: 1.5rem;
            transition: transform 0.2s ease;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        }
        
        .stButton>button {
            background: white !important;
            border-radius: 8px !important;
            border: 1px solid #473DEB !important;
            color: #473DEB !important;
            font-weight: 700 !important;
            padding: 0.5rem 1.5rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton>button:hover {
            background: white !important;
            transform: translateY(-1px);
            box-shadow: 0 3px 6px rgba(0, 0, 0, 0.15);
        }
        
        .stSelectbox>div>div>div {
            border-radius: 8px !important;
            border: 1px solid #e0e0e0 !important;
        }
        
        h1, h2, h3 {
            color: #292759 !important;
        }

        [data-testid="stAppViewContainer"] {
            background: #fff;
            padding: 15px;
        }

        [data-testid="stSidebar"] {
            background: #f0f0f0;
            padding: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Create a layout for the logo in the top left corner
    header_container = st.container()
    with header_container:
        cols = st.columns([0.25, 0.37, 0.38])  # Adjusted column ratios for logo, title, and buttons
        with cols[0]:
            # Sub-columns for logo and text
            logo_col, text_col = st.columns([0.25, 0.75])
            with logo_col:
                st.image("assets/qxgglogo.png", width=70)  # Slightly smaller logo
            with text_col:
                st.markdown(
                    '<h1 style="display: inline; color: #292759; font-size: 30px; '
                    'font-weight: 700;">AI DASHBOARD</h1>',
                    unsafe_allow_html=True
                )
        # Conditionally render buttons only on Dashboard and Chatbot pages
        # if st.session_state.page in ["Dashboard", "chatbot_page"]:
        if st.session_state.page == "Dashboard":
            with cols[2]:
                # Add buttons at the top-right corner
                col1, col2 = st.columns([0.5, 0.5])  # Two buttons side by side
                with col1:
                    if st.button("⬅️ Back to File Selection", key="back_button"):
                        switch_page("main")
                        st.session_state.button_clicked = False
                with col2:
                    if st.button("QX Intelligence Assistant ➡️", key="chatbot_button"):
                        switch_page("chatbot_page")
                        st.session_state.button_clicked = False
        elif st.session_state.page == "chatbot_page":
            with cols[2]:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("⬅️ Back to Dashboard", key="dashboard_button_chat"):
                        switch_page("Dashboard")
                        st.session_state.button_clicked = False
                with col2:
                    if st.button("File Selection ➡️", key="file_button_chat"):
                        switch_page("main")
                        st.session_state.button_clicked = False

    # Page-specific logic
    if st.session_state.page == "main":
        # File Selection Page
        select_file()

    elif st.session_state.page == "Dashboard":
        # Dashboard Page
        data_dict = st.session_state.get("dataframes_dict", {})
        if not data_dict:
            st.error("No data available. Please go back and select files.")
            return

        # Add a selector for which dataframe to analyze
        if len(data_dict) > 1:
            selected_file = st.selectbox("Select file to analyze:", list(data_dict.keys()))
            st.session_state.data = data_dict[selected_file]

        # Call the AI dashboard function
        from helpers.raw_utils import ai_dashboard
        ai_dashboard()

    elif st.session_state.page == "chatbot_page":
        # Chatbot Page
        st.markdown("<h3 style='font-size: 20px'>💬 QX Intelligence Assistant</h3>", unsafe_allow_html=True)
        data_dict = st.session_state.get("dataframes_dict", {})
        if not data_dict:
            st.error("No data available. Please go back and select files.")
            return

        # Display a full-page loader while loading the assistant
        with st.spinner("🔄 Loading QX Intelligence Assistant... This may take a moment as we analyze your data."):
            # Display data preview in an expander
            # with st.expander("📂 Data Preview", expanded=False):
            #     for file_name, df in data_dict.items():
            #         # st.subheader(f"📂 Data of {file_name}")
            #         st.markdown(f'<h2 style="font-size:18px;">📂 Data of {file_name}</h2>', unsafe_allow_html=True)
            #         st.dataframe(df)

            # Call the main function for the assistant
            # from helpers.ExcelChat import ExcelChat_main
            # ExcelChat_main()
            from helpers.rag_chat import ExcelChat_main
            ExcelChat_main()

def switch_page(page):
    st.session_state.page = page
    st.rerun()

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
        # else:
        #     st.error("🚨 No valid files loaded. Please check your file selections.")


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
            # st.subheader(f"📂 Preview: {file}")
            st.dataframe(df.head())
    elif st.session_state.dataframes_dict is None:
        st.write("Please select the files the first.")

def select_file():
    st.markdown(
        """
        <style>
        /* This targets the container div of the text input widget */
        div[data-testid="stTextInput"] {
            margin-top: -25px;  /* Adjust the negative margin as needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1 style='font-size:19px;'>📂 Upload Your File</h1>", unsafe_allow_html=True)
    
    # Display already loaded file if it exists
    existing_files = st.session_state.get("dataframes_dict", {})
    if existing_files:
        
        # Create a container with custom styling for the loaded file
        with st.container():
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
            """, unsafe_allow_html=True)
            
            for file_name, df in existing_files.items():
                file_info = f"📄 **{file_name}** ({df.shape[0]} rows × {df.shape[1]} columns)"
                col1, col2 = st.columns([0.85, 0.15])
                with col1:
                    st.markdown(file_info)
                with col2:
                    # Create a unique key for the remove button
                    if st.button("🗑️", key=f"remove_{file_name}", help=f"Remove {file_name}"):
                        # Remove the file from the dictionary
                        del st.session_state["dataframes_dict"][file_name]
                        st.session_state.data = None
                        st.rerun()  # Rerun to update the UI
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Show preview of loaded data
            preview_data()
            
            # Create dashboard button if a file is loaded
            if st.button("Create Dashboard", key="create_dashboard_button_existing"):
                switch_page("Dashboard")
    
    # Show file uploader only if no file is uploaded
    if not existing_files:
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file", 
            type=["csv", "xlsx", "xls"],
            help="Upload one file at a time"
        )
        
        if uploaded_file:
            # Process the uploaded file
            dataframes_dict = {}
            
            try:
                # Read file based on extension
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
                else:
                    st.warning(f"⚠️ Unsupported file format: {uploaded_file.name}")
                    df = None
                
                # Store DataFrame in dictionary if valid
                if df is not None:
                    dataframes_dict[uploaded_file.name] = df
                    
                    # Update the session state
                    st.session_state["dataframes_dict"] = dataframes_dict
                    st.session_state.data = df
                    
                    st.success(f"✅ Successfully loaded: **{uploaded_file.name}**")
                    st.rerun()  # Refresh to update UI and clear uploader
                
            except Exception as e:
                st.error(f"⚠️ Error reading {uploaded_file.name}: {str(e)}")
    
    # If no files are loaded at all, show an info message
    if not st.session_state.get("dataframes_dict"):
        st.info("ℹ️ Please upload a CSV or Excel file to get started.")

def select_file_old():
    st.markdown(
        """
        <style>
        /* This targets the container div of the text input widget */
        div[data-testid="stTextInput"] {
            margin-top: -25px;  /* Adjust the negative margin as needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1 style='font-size:22px;'>📂 Please Insert Folder Path", unsafe_allow_html=True)
    folder_path = st.text_input("", 
                               placeholder="e.g., C:\\Users\\YourName\\Documents\\Data",
                               help="Enter the complete folder path where your CSV/Excel files are stored",
                               key="folder_path_input")  # Unique key
    
    # Button to confirm folder path
    if st.button("📂 Load Files from Folder", key="load_files_button"):
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
                    if st.button("Create Dashboard", key="create_dashboard_button"):
                        switch_page("Dashboard")
                
                else:
                    st.info("ℹ️ Please select at least one file.")
                    preview_data()
            else:
                st.warning(f"⚠️ No CSV or Excel files found in '{folder}'. Please check the folder content.")
        except Exception as e:
            st.error(f"❌ Error reading folder: {e}")


# ----------------------
# Main Function
# ----------------------
def main():
    page_initialization()
    # Auto-refresh every 5 seconds (5000 milliseconds)
    # st_autorefresh(interval=5000, key="data_refresh")

if __name__ == "__main__":
    main()