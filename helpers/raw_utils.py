import streamlit as st
import pandas as pd
import numpy as np
from streamlit_autorefresh import st_autorefresh
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

def ai_dashboard():
    data_dict = st.session_state.get("dataframes_dict", {})

    if not data_dict:
        st.error("No data available. Please go back and select files.")
        return

    for file_name, df in data_dict.items():
        dataset_analysis, df_clean = analyze_dataset(df)
        create_basic_visualizations(df_clean)
        

def analyze_dataset(df):
    """Clean the DataFrame and analyze its structure to provide context for visualizations."""
    # Work on a copy to preserve the original DataFrame
    df_clean = df.copy()

    # --- Data Cleaning / Preprocessing ---
    # Trim strings and replace empty strings with NaN for object columns
    obj_cols = df_clean.select_dtypes(include=['object']).columns
    for col in obj_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip().replace('', np.nan)
    
    # Drop rows that are completely null
    df_clean.dropna(how="all", inplace=True)
    
    # Fill missing numerical values with the median
    num_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
    for col in num_cols:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # Fill missing categorical values with the mode
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        if not df_clean[col].mode().empty:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    # Detect date/time columns (simple check)
    date_cols = [col for col in df_clean.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    # --- Build Analysis ---
    analysis = []
    analysis.append(f"After cleaning, dataset contains {df_clean.shape[0]} rows and {df_clean.shape[1]} columns.")
    
    analysis.append(f"Numeric columns ({len(num_cols)}): {', '.join(num_cols[:5])}" + 
                    (f" and {len(num_cols) - 5} more" if len(num_cols) > 5 else ""))
    
    analysis.append(f"Categorical columns ({len(cat_cols)}): {', '.join(cat_cols[:5])}" + 
                    (f" and {len(cat_cols) - 5} more" if len(cat_cols) > 5 else ""))
    
    if date_cols:
        analysis.append(f"Detected date/time columns: {', '.join(date_cols)}")
    
    # Note if any missing values remain after cleaning
    missing_cols = df_clean.columns[df_clean.isna().any()].tolist()
    if missing_cols:
        analysis.append(f"Note: Missing values still present in: {', '.join(missing_cols)}")
    
    # Key statistical summaries for the main numeric columns (up to 3)
    if num_cols:
        for col in num_cols[:3]:
            stats = df_clean[col].describe()
            analysis.append(f"'{col}' ranges from {stats['min']:.2f} to {stats['max']:.2f} with mean {stats['mean']:.2f}")
    
    # Top distributions for a couple of categorical columns
    if cat_cols:
        for col in cat_cols[:2]:
            top_values = df_clean[col].value_counts().nlargest(3)
            top_vals_str = ", ".join([f"'{k}' ({v} rows)" for k, v in top_values.items()])
            analysis.append(f"'{col}' top values: {top_vals_str}")
    
    # Return both the analysis text and the cleaned DataFrame
    return "\n".join(analysis), df_clean


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

def create_basic_visualizations(df):
    """Create interactive creative visualizations using Streamlit charts.
    Charts included:
      1. Pie Chart
      2. Donut Chart
      3. Bar Graph
      4. Frequency Polygon Chart
      5. Waterfall Chart
      6. Bubble Chart (for variable relationships)
      7. Lollipop Chart
      8. Connected Scatterplot
      9. Stacked Bar Chart
    Displays charts in a 2-column layout for better screen utilization.
    """
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns available for visualization.")
        return

    # st.markdown("##### Interactive Creative Visualizations")
    st.markdown("<h1 style='font-size:20px;'>Interactive Creative Visualizations</h1>", unsafe_allow_html=True)
    
    # Add visualization selector for user to choose which charts to display
    viz_options = st.multiselect(
        "Select visualizations to display:",
        ["Bar Chart", "Pie Chart", "Stacked Bar Chart", "Donut Chart", "Frequency Chart", "Waterfall Chart", 
         "Bubble Chart", "Lollipop Chart", "Connected Scatterplot", "Heatmap"],
        default=["Bar Chart", "Pie Chart", "Stacked Bar Chart", "Donut Chart"], key="viz_selection_1"
    )
    
    # Add column selectors for more interactivity
    col1, col2 = st.columns(2)
    with col1:
        if categorical_cols:
            selected_cat_col = st.selectbox("Select categorical column:", categorical_cols)
        else:
            selected_cat_col = None
            st.info("No categorical columns available")
    
    with col2:
        if len(numeric_cols) > 0:
            selected_num_col = st.selectbox("Select primary numeric column:", numeric_cols)
            if len(numeric_cols) > 1:
                selected_num_col2 = st.selectbox("Select secondary numeric column:", 
                                               [col for col in numeric_cols if col != selected_num_col])
            else:
                selected_num_col2 = selected_num_col
        else:
            st.error("No numeric columns available for visualization")
            return
    
    # Display charts in a 2-column layout (2 charts per row)
    # Calculate number of rows needed
    num_charts = len(viz_options)
    num_rows = (num_charts + 1) // 2  # Ceiling division to ensure all charts are shown
    
    # Display selected charts in a 2-column grid
    for row in range(num_rows):
        # Create a new row with two columns
        left_col, right_col = st.columns(2)
        
        # Left column chart (even index)
        left_idx = row * 2
        if left_idx < num_charts:
            create_chart(viz_options[left_idx], left_col, df, selected_cat_col, selected_num_col, selected_num_col2, numeric_cols, categorical_cols, left_idx)
        
        # Right column chart (odd index)
        right_idx = row * 2 + 1
        if right_idx < num_charts:
            create_chart(viz_options[right_idx], right_col, df, selected_cat_col, selected_num_col, selected_num_col2, numeric_cols, categorical_cols, right_idx)

def create_chart(chart_type, column, df, selected_cat_col, selected_num_col, selected_num_col2, numeric_cols, categorical_cols, chart_idx):
    """Create a single chart in the specified column."""
    
    # Bar Chart
    if chart_type == "Bar Chart":
        with column:
            st.markdown("##### Interactive Bar Chart")
            # Let user customize the chart
            bar_type = st.radio(f"Bar chart type {chart_idx}:", ["Regular", "Stacked", "Grouped"], horizontal=True, key=f"bar_type_{chart_idx}")
            top_n = st.slider(f"Number of categories {chart_idx}:", 3, 15, 10, key=f"bar_top_n_{chart_idx}")
            
            if selected_cat_col:
                bar_data = df[selected_cat_col].value_counts().nlargest(top_n).reset_index()
                bar_data.columns = [selected_cat_col, "Count"]
                
                if bar_type == "Regular":
                    fig = px.bar(bar_data, x=selected_cat_col, y="Count", 
                                title=f"Bar Chart of Top {top_n} {selected_cat_col} Categories",
                                color_discrete_sequence=px.colors.qualitative.Bold)
                elif bar_type == "Stacked" and len(numeric_cols) > 1:
                    # For stacked, use the top categories and show multiple numeric values
                    top_cats = bar_data[selected_cat_col].tolist()
                    stacked_df = df[df[selected_cat_col].isin(top_cats)]
                    fig = px.bar(stacked_df, x=selected_cat_col, y=[selected_num_col, selected_num_col2], 
                                title=f"Stacked Bar Chart by {selected_cat_col}",
                                barmode="stack")
                elif bar_type == "Grouped" and len(numeric_cols) > 1:
                    # For grouped, similar approach
                    top_cats = bar_data[selected_cat_col].tolist()
                    grouped_df = df[df[selected_cat_col].isin(top_cats)]
                    fig = px.bar(grouped_df, x=selected_cat_col, y=[selected_num_col, selected_num_col2], 
                                title=f"Grouped Bar Chart by {selected_cat_col}",
                                barmode="group")
                else:
                    # Fallback to regular
                    fig = px.bar(bar_data, x=selected_cat_col, y="Count", 
                                title=f"Bar Chart of Top {top_n} {selected_cat_col} Categories")
            else:
                # Fall-back: use value counts of selected numeric grouped into bins
                bins = pd.cut(df[selected_num_col], bins=top_n)
                bar_data = bins.value_counts().reset_index()
                bar_data.columns = ["Bins", "Count"]
                fig = px.bar(bar_data, x="Bins", y="Count", 
                            title=f"Bar Chart of {selected_num_col} Bins")
            
            # Add more interactivity options
            fig.update_layout(
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=16,
                    font_family="Rockwell"
                )
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Pie Chart
    elif chart_type == "Pie Chart":
        with column:
            st.markdown("##### Interactive Pie Chart")
            pie_top_n = st.slider(f"Number of categories for pie chart {chart_idx}:", 3, 10, 5, key=f"pie_slider_{chart_idx}")
            
            if selected_cat_col:
                pie_data = df[selected_cat_col].value_counts().nlargest(pie_top_n).reset_index()
                pie_data.columns = [selected_cat_col, "Count"]
                
                # Add "Other" category for remaining values
                other_count = df[selected_cat_col].value_counts().sum() - pie_data["Count"].sum()
                if other_count > 0:
                    other_row = pd.DataFrame({selected_cat_col: ["Other"], "Count": [other_count]})
                    pie_data = pd.concat([pie_data, other_row], ignore_index=True)
                    
                fig = px.pie(pie_data, values="Count", names=selected_cat_col, 
                            title=f"Pie Chart of {selected_cat_col}",
                            color_discrete_sequence=px.colors.sequential.Plasma_r)
                fig.update_traces(textposition='inside', textinfo='percent+label')
            else:
                # fallback to categorizing selected numeric as discrete bins
                bins = pd.cut(df[selected_num_col], bins=6)
                pie_data = bins.value_counts().reset_index()
                pie_data.columns = ["Bins", "Count"]
                fig = px.pie(pie_data, values="Count", names="Bins", 
                            title=f"Pie Chart of {selected_num_col} Bins")
            st.plotly_chart(fig, use_container_width=True)
    
    # Donut Chart
    elif chart_type == "Donut Chart":
        with column:
            st.markdown("##### Interactive Donut Chart")
            donut_top_n = st.slider(f"Number of categories for donut chart {chart_idx}:", 3, 10, 5, key=f"donut_slider_{chart_idx}")
            
            if selected_cat_col:
                donut_data = df[selected_cat_col].value_counts().nlargest(donut_top_n).reset_index()
                donut_data.columns = [selected_cat_col, "Count"]
                
                # Add "Other" category for remaining values
                other_count = df[selected_cat_col].value_counts().sum() - donut_data["Count"].sum()
                if other_count > 0:
                    other_row = pd.DataFrame({selected_cat_col: ["Other"], "Count": [other_count]})
                    donut_data = pd.concat([donut_data, other_row], ignore_index=True)
                
                # Let user control the hole size
                hole_size = st.slider(f"Hole size {chart_idx}:", 0.0, 0.8, 0.4, step=0.1, key=f"hole_slider_{chart_idx}")
                
                fig = px.pie(donut_data, values="Count", names=selected_cat_col, 
                            title=f"Donut Chart of {selected_cat_col}", 
                            hole=hole_size,
                            color_discrete_sequence=px.colors.sequential.Viridis)
                fig.update_traces(textposition='inside', textinfo='percent+label')
            else:
                bins = pd.cut(df[selected_num_col], bins=6)
                donut_data = bins.value_counts().reset_index()
                donut_data.columns = ["Bins", "Count"]
                hole_size = st.slider(f"Hole size {chart_idx}:", 0.0, 0.8, 0.4, step=0.1, key=f"hole_slider_{chart_idx}")
                fig = px.pie(donut_data, values="Count", names="Bins", 
                            title=f"Donut Chart of {selected_num_col} Bins", 
                            hole=hole_size)
            st.plotly_chart(fig, use_container_width=True)
    
    # Frequency Chart
    elif chart_type == "Frequency Chart":
        with column:
            st.markdown("##### Interactive Frequency Chart")
            
            freq_type = st.radio(f"Frequency chart type {chart_idx}:", ["Polygon", "Histogram", "KDE"], horizontal=True, key=f"freq_type_{chart_idx}")
            num_bins = st.slider(f"Number of bins {chart_idx}:", 5, 30, 15, key=f"freq_bins_{chart_idx}")
            
            if freq_type == "Polygon":
                # Calculate histogram using numpy
                hist_vals, bin_edges = np.histogram(df[selected_num_col].dropna(), bins=num_bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                freq_data = pd.DataFrame({ selected_num_col: bin_centers, "Frequency": hist_vals })
                freq_poly = alt.Chart(freq_data).mark_line(point=True).encode(
                    x=alt.X(f"{selected_num_col}:Q", title=selected_num_col),
                    y=alt.Y("Frequency:Q", title="Frequency"),
                    tooltip=[selected_num_col, "Frequency"]
                ).properties(title=f"Frequency Polygon of {selected_num_col}")
                st.altair_chart(freq_poly, use_container_width=True)
            elif freq_type == "Histogram":
                fig = px.histogram(df, x=selected_num_col, nbins=num_bins,
                                  title=f"Histogram of {selected_num_col}")
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
            elif freq_type == "KDE":
                # KDE plot using plotly
                kde_data = df[selected_num_col].dropna()
                fig = px.density_contour(df, x=selected_num_col, title=f"KDE of {selected_num_col}")
                fig.update_traces(contours_coloring="fill", contours_showlabels=True)
                st.plotly_chart(fig, use_container_width=True)
    
    # Waterfall Chart
    elif chart_type == "Waterfall Chart":
        with column:
            st.markdown("##### Interactive Waterfall Chart")
            # Let user choose quantile divisions
            quantile_divisions = st.slider(f"Number of quantile divisions {chart_idx}:", 3, 10, 5, key=f"quantile_slider_{chart_idx}")
            q_values = np.linspace(0, 1, quantile_divisions)
            
            q = df[selected_num_col].quantile(q_values)
            q_df = q.reset_index()
            q_df.columns = ["Quantile", "Value"]
            
            # First measure is absolute, rest are relative
            measures = ["absolute"] + ["relative"] * (len(q_df) - 1)
            
            # Allow user to customize colors
            color_theme = st.selectbox(f"Color theme {chart_idx}:", 
                                      ["Blues", "Greens", "Reds", "Purples", "Oranges"],
                                      key=f"waterfall_colors_{chart_idx}")
            
            increasing_color = f"rgb(0, 100, 150)" if color_theme == "Blues" else f"rgb(50, 150, 50)" if color_theme == "Greens" else "rgb(150, 50, 50)" if color_theme == "Reds" else "rgb(100, 50, 150)" if color_theme == "Purples" else "rgb(150, 100, 50)"
            
            fig = go.Figure(go.Waterfall(
                measure = measures,
                x = [f"{v:.2f}" for v in q_df["Quantile"]],
                text = [f"{v:.2f}" for v in q_df["Value"]],
                y = q_df["Value"],
                connector = {"line": {"color": "rgb(63, 63, 63)"}},
                increasing = {"marker": {"color": increasing_color}},
                decreasing = {"marker": {"color": "rgba(100, 100, 100, 0.7)"}},
            ))
            fig.update_layout(title=f"Waterfall Chart of {selected_num_col} Quantiles")
            st.plotly_chart(fig, use_container_width=True)

    # Bubble Chart
    elif chart_type == "Bubble Chart" and len(numeric_cols) >= 3:
        with column:
            st.markdown("##### Interactive Bubble Chart")
            
            # Let user choose columns for bubble chart
            x_col = st.selectbox(f"X-axis column {chart_idx}:", numeric_cols, key=f"bubble_x_{chart_idx}")
            y_col = st.selectbox(f"Y-axis column {chart_idx}:", [col for col in numeric_cols if col != x_col], key=f"bubble_y_{chart_idx}")
            size_col = st.selectbox(f"Size column {chart_idx}:", [col for col in numeric_cols if col not in [x_col, y_col]], key=f"bubble_size_{chart_idx}")
            
            # Optional color column
            if categorical_cols:
                color_col = st.selectbox(f"Color by category {chart_idx}:", ["None"] + categorical_cols, key=f"bubble_color_{chart_idx}")
                if color_col != "None":
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        size=df[size_col].abs(),
                        color=color_col,
                        hover_name=color_col if len(df[color_col].unique()) <= 50 else None,
                        title=f"Bubble Chart: {x_col} vs {y_col} (Size: {size_col}, Color: {color_col})",
                        opacity=0.7,
                    )
                else:
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        size=df[size_col].abs(),
                        color=x_col,
                        title=f"Bubble Chart: {x_col} vs {y_col} (Size: {size_col})",
                        opacity=0.7,
                    )
            else:
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    size=df[size_col].abs(),
                    color=x_col,
                    title=f"Bubble Chart: {x_col} vs {y_col} (Size: {size_col})",
                    opacity=0.7,
                )
            
            # Add trendline if requested
            if st.checkbox(f"Show trendline {chart_idx}", key=f"bubble_trendline_{chart_idx}"):
                fig.update_layout(
                    shapes=[
                        dict(
                            type='line',
                            xref='x', yref='y',
                            x0=df[x_col].min(), 
                            y0=np.polyval(np.polyfit(df[x_col].fillna(0), df[y_col].fillna(0), 1), df[x_col].min()),
                            x1=df[x_col].max(), 
                            y1=np.polyval(np.polyfit(df[x_col].fillna(0), df[y_col].fillna(0), 1), df[x_col].max()),
                            line=dict(color='red', width=2, dash='dot')
                        )
                    ]
                )
                
            fig.update_layout(
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            )
            st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Bubble Chart":
        with column:
            st.info("At least 3 numeric columns needed for the bubble chart.")

    # Lollipop Chart
    elif chart_type == "Lollipop Chart":
        with column:
            st.markdown("##### Interactive Lollipop Chart")
            lollipop_n = st.slider(f"Number of values to display {chart_idx}:", 3, 15, 10, key=f"lollipop_slider_{chart_idx}")
            
            if selected_cat_col:
                # Use categorical column
                top_n = df[selected_cat_col].value_counts().nlargest(lollipop_n).reset_index()
                top_n.columns = [selected_cat_col, "Count"]
                
                # Create lollipop chart with Plotly
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=top_n[selected_cat_col],
                    y=top_n["Count"],
                    mode='markers',
                    marker=dict(size=12, color='#FFA500'),
                    name='Count'
                ))
                fig.add_trace(go.Scatter(
                    x=top_n[selected_cat_col],
                    y=top_n["Count"],
                    mode='lines',
                    line=dict(color='#008080', width=2),
                    name='Connection'
                ))
                fig.update_layout(
                    title=f"Lollipop Chart: Top {lollipop_n} {selected_cat_col} Counts",
                    showlegend=False
                )
            else:
                # Use numeric column bins
                bins = pd.cut(df[selected_num_col], bins=lollipop_n)
                top_n = bins.value_counts().reset_index()
                top_n.columns = ["Bins", "Count"]
                
                # Convert Bins to string for better display
                top_n["Bins"] = top_n["Bins"].astype(str)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=top_n["Bins"],
                    y=top_n["Count"],
                    mode='markers',
                    marker=dict(size=12, color='#FFA500'),
                    name='Count'
                ))
                fig.add_trace(go.Scatter(
                    x=top_n["Bins"],
                    y=top_n["Count"],
                    mode='lines',
                    line=dict(color='#008080', width=2),
                    name='Connection'
                ))
                fig.update_layout(
                    title=f"Lollipop Chart: {selected_num_col} Bins",
                    showlegend=False
                )
            
            # Add rotation for x-axis labels if they're long
            fig.update_layout(
                xaxis=dict(tickangle=45)
            )
            st.plotly_chart(fig, use_container_width=True)

    # Connected Scatterplot
    elif chart_type == "Connected Scatterplot" and len(numeric_cols) >= 2:
        with column:
            st.markdown("##### Interactive Connected Scatterplot")
            
            # Let user select columns
            x_col = st.selectbox(f"X-axis {chart_idx}:", numeric_cols, key=f"scatter_x_{chart_idx}")
            y_col = st.selectbox(f"Y-axis {chart_idx}:", [col for col in numeric_cols if col != x_col], key=f"scatter_y_{chart_idx}")
            
            # Sort order and sample size
            sort_by = st.radio(f"Sort by {chart_idx}:", [x_col, y_col, "No sorting"], horizontal=True, key=f"sort_by_{chart_idx}")
            if df.shape[0] > 100:
                use_sample = st.checkbox(f"Use sample {chart_idx}", value=True, key=f"use_sample_{chart_idx}")
                sample_size = st.slider(f"Sample size {chart_idx}:", 10, min(1000, df.shape[0]), 100, key=f"sample_size_{chart_idx}") if use_sample else df.shape[0]
                scatter_df = df.sample(sample_size) if use_sample else df
            else:
                scatter_df = df
            
            # Sort the data
            if sort_by != "No sorting":
                scatter_df = scatter_df.sort_values(sort_by)
            
            # Create connected scatterplot with Plotly
            fig = go.Figure()
            
            # Add the lines
            fig.add_trace(go.Scatter(
                x=scatter_df[x_col],
                y=scatter_df[y_col],
                mode='lines',
                line=dict(color='#7070DB', width=1.5),
                name='Connection'
            ))
            
            # Add the points
            fig.add_trace(go.Scatter(
                x=scatter_df[x_col],
                y=scatter_df[y_col],
                mode='markers',
                marker=dict(size=8, color='#DB7070'),
                name='Data Points'
            ))
            
            fig.update_layout(
                title=f"Connected Scatterplot: {x_col} vs {y_col}",
                xaxis_title=x_col,
                yaxis_title=y_col,
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Connected Scatterplot":
        with column:
            st.info("At least 2 numeric columns needed for the connected scatterplot.")

    # Heatmap
    elif chart_type == "Heatmap" and len(numeric_cols) > 1:
        with column:
            st.markdown("##### Interactive Correlation Heatmap")
            
            # Let user choose correlation method
            corr_method = st.radio(f"Correlation method {chart_idx}:", ["pearson", "spearman", "kendall"], horizontal=True, key=f"corr_method_{chart_idx}")
            
            # Select columns to include
            heatmap_cols = st.multiselect(
                f"Columns to include {chart_idx}:",
                numeric_cols,
                default=numeric_cols[:min(8, len(numeric_cols))],
                key=f"heatmap_cols_{chart_idx}"
            )
            
            if len(heatmap_cols) > 1:
                correlation = df[heatmap_cols].corr(method=corr_method).round(2)
                
                # Create heatmap with plotly
                fig = px.imshow(
                    correlation,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    title=f"{corr_method.capitalize()} Correlation Heatmap"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least 2 columns for the heatmap.")
    elif chart_type == "Heatmap":
        with column:
            st.info("At least 2 numeric columns needed for the heatmap.")
            
    # Stacked Bar Chart
    elif chart_type == "Stacked Bar Chart":
        with column:
            st.markdown("##### Interactive Stacked Bar Chart")
            
            # Auto-detect categorical columns for x-axis
            if categorical_cols:
                # Choose x-axis column with smart defaults
                x_col = st.selectbox(f"X-axis (categorical) {chart_idx}:", categorical_cols, key=f"bar_x_{chart_idx}")
                
                # Choose numeric columns for the stacked bars
                if len(numeric_cols) > 0:
                    stack_cols = st.multiselect(
                        f"Numeric columns to stack {chart_idx}:",
                        numeric_cols,
                        default=numeric_cols[:min(4, len(numeric_cols))],
                        key=f"stack_cols_{chart_idx}"
                    )
                else:
                    st.warning("No numeric columns available for the stacked bar chart.")
                    stack_cols = []
                
                # Let user customize the chart
                top_n = st.slider(f"Number of categories {chart_idx}:", 3, 20, 10, key=f"stacked_bar_top_n_{chart_idx}")
                normalize = st.checkbox(f"Normalize to percentage {chart_idx}", key=f"normalize_bars_{chart_idx}")
                orientation = st.radio(f"Orientation {chart_idx}:", ["Vertical", "Horizontal"], horizontal=True, key=f"bar_orientation_{chart_idx}")
                color_scheme = st.selectbox(
                    f"Color scheme {chart_idx}:",
                    ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo"],
                    key=f"stack_bar_colors_{chart_idx}"
                )
                
                # Check if we have required data
                if stack_cols and x_col:
                    # Prepare data
                    # Get the top N categories by count
                    top_categories = df[x_col].value_counts().nlargest(top_n).index.tolist()
                    
                    # Filter the dataframe to include only these categories
                    chart_data = df[df[x_col].isin(top_categories)].copy()
                    
                    # Select only the columns we need
                    chart_data = chart_data[[x_col] + stack_cols].copy()
                    
                    # Handle missing data
                    for col in stack_cols:
                        chart_data[col] = chart_data[col].fillna(0)
                    
                    # Group by the categorical column and calculate the mean for each numeric column
                    agg_data = chart_data.groupby(x_col)[stack_cols].mean().reset_index()
                    
                    # Sort by total value for better visualization
                    agg_data['_total'] = agg_data[stack_cols].sum(axis=1)
                    agg_data = agg_data.sort_values('_total', ascending=False).drop(columns=['_total'])
                    
                    # Normalize if requested
                    if normalize:
                        # Calculate row sums for normalization
                        row_sums = agg_data[stack_cols].sum(axis=1)
                        # Normalize each column by dividing by row sum
                        for col in stack_cols:
                            agg_data[col] = agg_data[col].div(row_sums) * 100
                    
                    # Create the stacked bar chart
                    if orientation == "Vertical":
                        fig = px.bar(
                            agg_data, 
                            x=x_col, 
                            y=stack_cols,
                            title=f"Stacked Bar Chart: {x_col} vs {', '.join(stack_cols[:3])}{' and more' if len(stack_cols) > 3 else ''}",
                            color_discrete_sequence=getattr(px.colors.sequential, color_scheme),
                            barmode='stack'
                        )
                    else:  # Horizontal
                        fig = px.bar(
                            agg_data, 
                            y=x_col, 
                            x=stack_cols,
                            title=f"Horizontal Stacked Bar Chart: {x_col} vs {', '.join(stack_cols[:3])}{' and more' if len(stack_cols) > 3 else ''}",
                            color_discrete_sequence=getattr(px.colors.sequential, color_scheme),
                            barmode='stack',
                            orientation='h'
                        )
                    
                    # Customize layout
                    fig.update_layout(
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        margin=dict(l=40, r=40, t=80, b=40)
                    )
                    
                    # Add labels
                    if orientation == "Vertical":
                        fig.update_layout(
                            xaxis_title=x_col,
                            yaxis_title="Percentage" if normalize else "Value"
                        )
                    else:
                        fig.update_layout(
                            yaxis_title=x_col,
                            xaxis_title="Percentage" if normalize else "Value"
                        )
                    
                    # Rotate x-axis labels if they might be long
                    if orientation == "Vertical":
                        fig.update_layout(xaxis=dict(tickangle=45))
                    
                    st.plotly_chart(fig, use_container_width=True)

                elif not stack_cols:
                    st.warning("Please select at least one numeric column for the stacked bars.")
                elif not x_col:
                    st.warning("Please select a categorical column for the x-axis.")
            else:
                st.warning("No categorical columns available for the Stacked Bar Chart. This chart requires at least one categorical column for the x-axis.")
