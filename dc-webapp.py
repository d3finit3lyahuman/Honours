import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
import io
import glob
import os
from functools import reduce  # For merging dataframes more efficiently
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

st.set_page_config(page_title="Depression Detection Comparison", layout="wide")


# Helper function to create a download button for a given matplotlib figure
def download_figure(fig, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="Download Figure",
        data=buf.getvalue(),
        file_name=filename,
        mime="image/png",
    )


# Load the data files
@st.cache_data
def load_all_data(pattern="data/*_results-525.csv"):
    """
    Loads all CSV files matching the pattern into a dictionary of DataFrames.
    The keys of the dictionary are derived from the filenames.
    """
    data_files = glob.glob(pattern)
    all_data = {}
    if not data_files:
        st.warning(f"No data files found matching pattern: {pattern}")
        return all_data  # Return empty dict

    for f in data_files:
        try:
            # Extract model name from filename (e.g., "gemini-flash-2.0" from "data/gemini-flash-2.0_results-525.csv")
            model_name = os.path.basename(f).replace("_results-525.csv", "")
            model_name = model_name.replace("-", " ").replace("_", " ").title()
            df = pd.read_csv(f)

            # Basic validation
            if "severity_rating" not in df.columns or "text" not in df.columns:
                st.error(
                    f"File '{f}' is missing required columns ('severity_rating', 'text'). Skipping."
                )
                continue
            if "label" not in df.columns:
                st.warning(
                    f"File '{f}' is missing 'label' column. Some features might be limited."
                )
                # Add a dummy label column if missing, maybe default to 0 or NaN?
                # df['label'] = 0 # Or handle appropriately later

            # Convert severity ratings to numeric
            df["severity_rating"] = pd.to_numeric(
                df["severity_rating"], errors="coerce"
            )

            # Store in dictionary
            all_data[model_name] = df
            print(f"Loaded {model_name} data with {len(df)} samples.")

        except Exception as e:
            st.error(f"Error loading or processing file {f}: {str(e)}")

    return all_data


# Create custom theme function
def set_custom_theme():
    st.markdown(
        """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


# Main dashboard
def main():
    # set_custom_theme()  # Apply custom theme

    # Set title and description
    st.title("Depression Severity Detection Comparison")
    st.markdown("### Comparing LLM Performance for Depression Assessment")

    try:
        # Load data for all models found
        all_data = load_all_data()  # Use the new loading function

        # Check data loading
        if not all_data:  # Check if the dictionary is empty
            st.error(
                "Error: No valid model data could be loaded. Please check the 'data' folder and file formats."
            )
            return  # Stop if no data

        model_names = list(all_data.keys())
        # st.info(f"Loaded models: {', '.join(model_names)}")
        print(f"Loaded models: {', '.join(model_names)}")

        # --- Define Tabs ---
        # Adjust tabs based on the number of models loaded
        tab_titles = ["Overall Overview", "Model Performance"]
        if len(model_names) >= 2:
            tab_titles.append("Pairwise Comparison")
        tab_titles.append("Text Sample Analysis")

        tabs = st.tabs(tab_titles)
        tab_map = {title: tab for title, tab in zip(tab_titles, tabs)}

        # --- Populate Tabs ---
        with tab_map["Overall Overview"]:
            overall_overview_tab(all_data)

        with tab_map["Model Performance"]:
            model_performance_tab(all_data)

        # Only show pairwise comparison tab if at least 2 models exist
        if "Pairwise Comparison" in tab_map:
            with tab_map["Pairwise Comparison"]:
                pairwise_comparison_tab(all_data)

        with tab_map["Text Sample Analysis"]:
            text_analysis_tab_modular(all_data)  # Use the new modular text analysis tab

    except Exception as e:
        st.error(f"An unexpected error occurred in the main application: {str(e)}")
        st.warning("Please ensure data files are correctly formatted and accessible.")


def plot_correlation_heatmap(merged_df):
    """
    Generate a heatmap showing the correlation between Gemini and DeepSeek severity ratings.
    """
    # Create a pivot table for counts
    heatmap_data = (
        merged_df.groupby(["severity_rating_gemini", "severity_rating_deepseek"])
        .size()
        .unstack(fill_value=0)
    )

    # Create a soft red gradient colormap using seaborn's light_palette
    cmap = sns.light_palette("red", as_cmap=True)

    # Plot the heatmap using the custom colormap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap=cmap, cbar=True, ax=ax)

    ax.set_xlabel("Gemini Severity Rating")
    ax.set_ylabel("DeepSeek Severity Rating")
    ax.set_title("Correlation Heatmap Between Model Ratings")

    return fig


def plot_joint_density(merged_df):
    """
    Generate a joint plot showing the correlation and distribution of severity ratings.
    """
    # Create the joint plot
    fig = sns.jointplot(
        data=merged_df,
        x="severity_rating_gemini",
        y="severity_rating_deepseek",
        kind="scatter",
        marginal_kws=dict(bins=10, fill=True),
        joint_kws=dict(alpha=0.7),
    )
    fig.fig.suptitle("Joint Plot of Severity Ratings (Gemini vs DeepSeek)", y=1.03)
    fig.set_axis_labels("Gemini Severity Rating", "DeepSeek Severity Rating")
    return fig


def plot_violin_distribution(merged_df):
    """
    Generate a violin plot showing the distribution of DeepSeek ratings for each Gemini rating.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(
        data=merged_df,
        x="severity_rating_gemini",
        y="severity_rating_deepseek",
        palette="muted",
        ax=ax,
    )
    ax.set_xlabel("Gemini Severity Rating")
    ax.set_ylabel("DeepSeek Severity Rating")
    ax.set_title("Violin Plot of Severity Ratings")
    return fig


def plot_scatter_correlation(merged_df):
    """
    Generate a scatter plot showing the correlation between Gemini and DeepSeek severity ratings.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Calculate point sizes based on frequency
    point_counts = (
        merged_df.groupby(["severity_rating_gemini", "severity_rating_deepseek"])
        .size()
        .reset_index(name="count")
    )

    # Normalize bubble sizes for better visualization
    max_count = point_counts["count"].max()
    point_counts["size"] = (
        50 + (point_counts["count"] / max_count) * 200
    )  # Scale sizes between 50 and 250

    # Plot points with different colors based on agreement/disagreement
    for _, row in point_counts.iterrows():
        x_val = row["severity_rating_gemini"]
        y_val = row["severity_rating_deepseek"]
        size = row["size"]

        if x_val == y_val:
            ax.scatter(
                x_val, y_val, s=size, color="green", alpha=0.7, label="Agreement"
            )
        elif x_val > y_val:
            ax.scatter(
                x_val, y_val, s=size, color="blue", alpha=0.7, label="Gemini > DeepSeek"
            )
        else:
            ax.scatter(
                x_val,
                y_val,
                s=size,
                color="orange",
                alpha=0.7,
                label="DeepSeek > Gemini",
            )

    # Draw perfect agreement line
    min_val = min(
        merged_df["severity_rating_gemini"].min(),
        merged_df["severity_rating_deepseek"].min(),
    )
    max_val = max(
        merged_df["severity_rating_gemini"].max(),
        merged_df["severity_rating_deepseek"].max(),
    )
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Agreement")

    # Set axis labels and title
    ax.set_xlabel("Gemini Severity Rating")
    ax.set_ylabel("DeepSeek Severity Rating")
    ax.set_title("Correlation between Model Ratings")

    # Set integer ticks
    ax.set_xticks(range(int(min_val), int(max_val) + 1))
    ax.set_yticks(range(int(min_val), int(max_val) + 1))

    # Remove duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best")

    return fig


def display_dataset_statistics(df, title, filename):
    """
    Display basic statistics and distribution of severity ratings for a dataset.
    """
    st.subheader(title)
    st.write(f"Total samples: {len(df)}")

    # Distribution of severity ratings
    st.subheader("Distribution of Severity Ratings")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="severity_rating", ax=ax, palette="viridis")
    ax.set_title(f"{title} Severity Rating Distribution")
    ax.set_xlabel("Severity Rating (0-4)")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    download_figure(fig, filename)

    # Calculate statistics
    st.subheader("Statistics")
    stats = {
        "Mean Severity": df["severity_rating"].mean(),
        "Median Severity": df["severity_rating"].median(),
        "Standard Deviation": df["severity_rating"].std(),
    }
    st.write(pd.DataFrame([stats]))


def merge_datasets(gemini_df, deepseek_df):
    """
    Merge Gemini and DeepSeek datasets on text for comparison.
    """
    return pd.merge(
        gemini_df[["text", "severity_rating"]],
        deepseek_df[["text", "severity_rating"]],
        on="text",
        suffixes=("_gemini", "_deepseek"),
    )


def plot_side_by_side_comparison(gemini_df, deepseek_df):
    """
    Generate a bar chart comparing severity ratings between Gemini and DeepSeek.
    """
    severity_counts_gemini = gemini_df["severity_rating"].value_counts().sort_index()
    severity_counts_deepseek = (
        deepseek_df["severity_rating"].value_counts().sort_index()
    )

    all_ratings = sorted(
        set(severity_counts_gemini.index) | set(severity_counts_deepseek.index)
    )
    complete_gemini = pd.Series(
        {r: severity_counts_gemini.get(r, 0) for r in all_ratings}, name="Gemini"
    )
    complete_deepseek = pd.Series(
        {r: severity_counts_deepseek.get(r, 0) for r in all_ratings}, name="DeepSeek"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    x = np.arange(len(all_ratings))

    ax.bar(x - bar_width / 2, complete_gemini, bar_width, label="Gemini")
    ax.bar(x + bar_width / 2, complete_deepseek, bar_width, label="DeepSeek")

    ax.set_xlabel("Severity Rating")
    ax.set_ylabel("Count")
    ax.set_title("Comparison of Severity Ratings Between Models")
    ax.set_xticks(x)
    ax.set_xticklabels(all_ratings)
    ax.legend()

    return fig


def overall_overview_tab(all_data):
    st.header("Model Overviews & Statistics")
    st.markdown("Basic statistics and severity distributions for each loaded model.")

    model_names = list(all_data.keys())
    num_models = len(model_names)

    # Use columns for better layout if many models
    cols_per_row = 2  # Adjust as needed
    num_rows = math.ceil(num_models / cols_per_row)

    model_iter = iter(model_names)

    for _ in range(num_rows):
        cols = st.columns(cols_per_row)
        for i in range(cols_per_row):
            model_name = next(model_iter, None)
            if model_name:
                with cols[i]:
                    st.subheader(f"--- {model_name} ---")
                    df = all_data[model_name]
                    display_dataset_statistics(
                        df, model_name, f"{model_name}_severity_distribution.png"
                    )
            else:
                break  # No more models
        st.markdown("---")  # Separator between rows


def overview_tab(gemini_df, deepseek_df):
    st.header("Dataset Overview")

    # Display basic statistics for both datasets
    col1, col2 = st.columns(2)

    with col1:
        display_dataset_statistics(
            gemini_df, "Gemini-Flash-2.0 Results", "gemini_severity_distribution.png"
        )

    with col2:
        display_dataset_statistics(
            deepseek_df, "DeepSeek-v3 Results", "deepseek_severity_distribution.png"
        )

    # Comparison between models
    st.header("Model Comparison")
    merged_df = merge_datasets(gemini_df, deepseek_df)

    # Calculate agreement metrics
    exact_agreement = (
        merged_df["severity_rating_gemini"] == merged_df["severity_rating_deepseek"]
    ).mean() * 100

    # Display agreement metrics
    st.subheader("Agreement Metrics")
    st.write(f"Exact Agreement: {exact_agreement:.2f}%")

    # Allow user to select visualization type
    st.subheader("Select Visualization Type")
    visualization_type = st.selectbox(
        "Choose a visualization to display the correlation:",
        options=["Scatter Plot", "Heatmap", "Joint Plot", "Violin Plot"],
    )

    # Generate the selected visualization
    if visualization_type == "Scatter Plot":
        fig = plot_scatter_correlation(merged_df)
        st.pyplot(fig)
        download_figure(fig, "model_rating_correlation_scatter.png")
    elif visualization_type == "Heatmap":
        fig = plot_correlation_heatmap(merged_df)
        st.pyplot(fig)
        download_figure(fig, "model_rating_correlation_heatmap.png")
    elif visualization_type == "Joint Plot":
        fig = plot_joint_density(merged_df)
        st.pyplot(fig.fig)  # Joint plot requires `.fig` to render in Streamlit
        download_figure(fig.fig, "model_rating_correlation_joint.png")
    elif visualization_type == "Violin Plot":
        fig = plot_violin_distribution(merged_df)
        st.pyplot(fig)
        download_figure(fig, "model_rating_correlation_violin.png")


def pairwise_comparison_tab(all_data):
    st.header("Pairwise Model Comparison")
    model_names = list(all_data.keys())

    if len(model_names) < 2:
        st.warning("Need at least two models loaded to perform pairwise comparison.")
        return

    # --- Model Selection ---
    col1, col2 = st.columns(2)
    with col1:
        model_a_name = st.selectbox(
            "Select Model A:", model_names, index=0, key="model_a_select"
        )
    with col2:
        # Ensure Model B options don't include Model A
        available_b = [m for m in model_names if m != model_a_name]
        if not available_b:
            st.warning("Only one model available.")
            return
        model_b_name = st.selectbox(
            "Select Model B:", available_b, index=0, key="model_b_select"
        )

    if not model_a_name or not model_b_name or model_a_name == model_b_name:
        st.warning("Please select two different models.")
        return

    st.subheader(f"Comparing: {model_a_name} vs. {model_b_name}")

    # --- Merge Selected Data ---
    df_a = all_data[model_a_name][["text", "severity_rating"]].copy()
    df_b = all_data[model_b_name][["text", "severity_rating"]].copy()

    # Rename columns before merge to avoid suffix issues if names were identical
    df_a = df_a.rename(columns={"severity_rating": "severity_rating_a"})
    df_b = df_b.rename(columns={"severity_rating": "severity_rating_b"})

    # Merge on text
    merged_pair_df = pd.merge(
        df_a, df_b, on="text", how="inner"
    )  # Inner join ensures text exists in both

    if merged_pair_df.empty:
        st.warning(
            f"No common 'text' entries found between {model_a_name} and {model_b_name}."
        )
        return

    # Drop rows with NaN ratings after merge
    merged_pair_df = merged_pair_df.dropna(
        subset=["severity_rating_a", "severity_rating_b"]
    )
    if merged_pair_df.empty:
        st.warning(
            f"No valid rating pairs after merging and cleaning NaNs for {model_a_name} and {model_b_name}."
        )
        return

    # --- Agreement Metrics ---
    exact_agreement = (
        merged_pair_df["severity_rating_a"] == merged_pair_df["severity_rating_b"]
    ).mean() * 100
    st.metric(label="Exact Agreement", value=f"{exact_agreement:.2f}%")

    # --- Visualization Selection ---
    st.subheader("Select Comparison Visualization")
    visualization_type = st.selectbox(
        "Choose a visualization:",
        options=["Scatter Plot", "Heatmap", "Joint Plot", "Violin Plot"],
        key=f"viz_select_{model_a_name}_{model_b_name}",  # Unique key per pair
    )

    # Adapt plotting functions to use generic column names 'severity_rating_a', 'severity_rating_b'
    # We might need small wrapper functions or modify the originals carefully

    # Example adaptation for scatter plot (modify plot_scatter_correlation or create a wrapper)
    def plot_scatter_correlation_pair(df, col_a, col_b, name_a, name_b):
        fig, ax = plt.subplots(figsize=(8, 8))
        point_counts = df.groupby([col_a, col_b]).size().reset_index(name="count")
        max_count = point_counts["count"].max()
        point_counts["size"] = 50 + (point_counts["count"] / max_count) * 200

        for _, row in point_counts.iterrows():
            x_val, y_val, size = row[col_a], row[col_b], row["size"]
            if x_val == y_val:
                color, label = "green", "Agreement"
            elif x_val > y_val:
                color, label = "blue", f"{name_a} > {name_b}"
            else:
                color, label = "orange", f"{name_b} > {name_a}"
            ax.scatter(x_val, y_val, s=size, color=color, alpha=0.7, label=label)

        min_val = min(df[col_a].min(), df[col_b].min())
        max_val = max(df[col_a].max(), df[col_b].max())
        ax.plot(
            [min_val, max_val], [min_val, max_val], "r--", label="Perfect Agreement"
        )
        ax.set_xlabel(f"{name_a} Severity Rating")
        ax.set_ylabel(f"{name_b} Severity Rating")
        ax.set_title(f"Correlation: {name_a} vs. {name_b}")
        ax.set_xticks(np.arange(int(min_val), int(max_val) + 1))
        ax.set_yticks(np.arange(int(min_val), int(max_val) + 1))
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="best")
        return fig

    # Similar adaptations needed for heatmap, jointplot, violinplot if used directly
    # Or modify the original functions to accept column names as arguments

    if visualization_type == "Scatter Plot":
        fig = plot_scatter_correlation_pair(
            merged_pair_df,
            "severity_rating_a",
            "severity_rating_b",
            model_a_name,
            model_b_name,
        )
        st.pyplot(fig)
        download_figure(fig, f"{model_a_name}_vs_{model_b_name}_scatter.png")
    # Add elif blocks for other plot types, calling adapted functions
    elif visualization_type == "Heatmap":
        # Adapt plot_correlation_heatmap or create wrapper
        st.info("Heatmap visualization needs adaptation for selected pair.")
        # fig = plot_correlation_heatmap_pair(merged_pair_df, 'severity_rating_a', 'severity_rating_b', model_a_name, model_b_name)
        # st.pyplot(fig)
        # download_figure(fig, f"{model_a_name}_vs_{model_b_name}_heatmap.png")
    elif visualization_type == "Joint Plot":
        st.info("Joint Plot visualization needs adaptation for selected pair.")
        # fig = plot_joint_density_pair(...)
        # st.pyplot(fig.fig)
        # download_figure(fig.fig, f"{model_a_name}_vs_{model_b_name}_joint.png")
    elif visualization_type == "Violin Plot":
        st.info("Violin Plot visualization needs adaptation for selected pair.")
        # fig = plot_violin_distribution_pair(...)
        # st.pyplot(fig)
        # download_figure(fig, f"{model_a_name}_vs_{model_b_name}_violin.png")


def text_analysis_tab_modular(all_data):
    st.header("Text Sample Analysis")
    model_names = list(all_data.keys())

    # --- Merge All DataFrames ---
    # Select essential columns and add model name suffix
    dfs_to_merge = []
    label_df = None  # Keep track of df with label column if exists
    for name, df in all_data.items():
        cols_to_keep = ["text", "severity_rating", "explanation"]
        # Check if label exists and keep it from the first df that has it
        if "label" in df.columns and label_df is None:
            cols_to_keep.append("label")
            label_df = df[["text", "label"]].copy().drop_duplicates(subset=["text"])

        # Ensure columns exist before selecting
        present_cols = [col for col in cols_to_keep if col in df.columns]
        if "text" not in present_cols:
            continue  # Skip if no text column

        df_renamed = df[present_cols].rename(
            columns={
                "severity_rating": f"rating_{name}",
                "explanation": f"explanation_{name}",
                # label is handled separately
            }
        )
        dfs_to_merge.append(
            df_renamed.drop(columns=["label"], errors="ignore")
        )  # Drop label if it was temporarily selected

    if not dfs_to_merge:
        st.warning("No dataframes available for merging.")
        return

    # Merge all using functools.reduce
    try:
        # Start with the first dataframe, then merge others onto it
        merged_all_df = reduce(
            lambda left, right: pd.merge(left, right, on="text", how="outer"),
            dfs_to_merge,
        )
        # Merge the label column back if it was found
        if label_df is not None:
            merged_all_df = pd.merge(merged_all_df, label_df, on="text", how="left")
            # Convert label to integer, handle potential NaNs from outer merge
            if "label" in merged_all_df.columns:
                merged_all_df["label"] = pd.to_numeric(
                    merged_all_df["label"], errors="coerce"
                )  # Coerce first
                # merged_all_df['label'] = merged_all_df['label'].fillna(-1).astype(int) # Fill NaN maybe? Or handle later
    except Exception as e:
        st.error(f"Failed to merge all dataframes: {e}")
        st.info(
            "This might happen if 'text' columns have incompatible values or due to memory issues."
        )
        return

    # --- Filtering ---
    st.subheader("Filter Samples")
    col_filter1, col_filter2, col_filter3 = st.columns(3)

    # Filter by Label (if label column exists)
    label_filter = None
    if "label" in merged_all_df.columns:
        with col_filter1:
            label_filter = st.selectbox(
                "Filter by Label",
                options=[None, 0, 1],  # Assuming binary label
                format_func=lambda x: (
                    "All"
                    if x is None
                    else ("Not Depressed (0)" if x == 0 else "Depressed (1)")
                ),
                key="label_filter_modular",
            )

    # Filter by a specific model's rating
    with col_filter2:
        filter_model_name = st.selectbox(
            "Filter by Model Rating:", ["None"] + model_names, key="filter_model_select"
        )

    rating_filter_value = None
    if filter_model_name != "None":
        rating_col = f"rating_{filter_model_name}"
        if rating_col in merged_all_df.columns:
            # Ensure ratings are numeric before getting unique values
            valid_ratings = sorted(
                merged_all_df[rating_col].dropna().unique().astype(int)
            )
            with col_filter3:
                rating_filter_value = st.multiselect(
                    f"Select {filter_model_name} Rating(s):",
                    options=valid_ratings,
                    key=f"rating_filter_{filter_model_name}",
                )
        else:
            st.warning(
                f"Rating column for {filter_model_name} not found in merged data."
            )

    # Apply filters
    filtered_df = merged_all_df.copy()

    if label_filter is not None and "label" in filtered_df.columns:
        # Handle potential NaNs in label before filtering
        filtered_df = filtered_df.dropna(subset=["label"])
        filtered_df["label"] = filtered_df["label"].astype(int)  # Ensure int type
        filtered_df = filtered_df[filtered_df["label"] == label_filter]

    if filter_model_name != "None" and rating_filter_value is not None:
        rating_col = f"rating_{filter_model_name}"
        if rating_col in filtered_df.columns:
            # Ensure column is numeric before filtering
            filtered_df[rating_col] = pd.to_numeric(
                filtered_df[rating_col], errors="coerce"
            )
            filtered_df = filtered_df.dropna(
                subset=[rating_col]
            )  # Drop NaNs in this column
            filtered_df = filtered_df[
                filtered_df[rating_col].astype(int).isin(rating_filter_value)
            ]

    # --- Pagination and Display ---
    items_per_page_options = [
        5,
        10,
        15,
    ]  # Reduced default options for potentially wider content
    items_per_page = st.selectbox(
        "Items per page:",
        options=items_per_page_options,
        index=1,  # Default to 10
        key="items_per_page_modular",
    )

    total_items = len(filtered_df)
    st.subheader(f"Filtered Samples ({total_items} samples)")

    if total_items > 0:
        total_pages = math.ceil(total_items / items_per_page)

        if "text_analysis_page_modular" not in st.session_state:
            st.session_state.text_analysis_page_modular = 1

        if st.session_state.text_analysis_page_modular > total_pages:
            st.session_state.text_analysis_page_modular = 1

        current_page = st.session_state.text_analysis_page_modular

        start_idx = (current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)

        paginated_df = filtered_df.iloc[start_idx:end_idx]

        for index, row in paginated_df.iterrows():  # Use index from iloc
            # Create a dynamic expander label - maybe just use index or text snippet
            expander_label = f"Sample (Index: {index}) - Text: {row['text'][:50]}..."
            with st.expander(expander_label):
                st.write(f"**Text:** {row['text']}")
                if "label" in row and pd.notna(row["label"]):
                    st.write(
                        f"**True Label:** {'Depressed (1)' if int(row['label']) == 1 else 'Not Depressed (0)'}"
                    )
                else:
                    st.write("**True Label:** N/A")

                st.markdown("---")
                st.markdown("##### Model Analyses:")

                # Display analysis for each model
                analysis_cols = st.columns(len(model_names))  # One column per model
                for i, name in enumerate(model_names):
                    with analysis_cols[i]:
                        st.markdown(f"**{name}**")
                        rating_col = f"rating_{name}"
                        expl_col = f"explanation_{name}"

                        rating = row.get(rating_col, "N/A")
                        explanation = str(row.get(expl_col, "N/A"))

                        # Display rating (handle potential NaN/None)
                        if pd.notna(rating):
                            st.write(f"Severity: {int(rating)}")
                        else:
                            st.write("Severity: N/A")

                        # Display explanation
                        st.write(
                            f"Explanation: {explanation.replace('**Explanation:**', '').strip()}"
                        )

        # --- Pagination Controls (reuse previous logic) ---
        st.write("---")
        col_page1, col_page2, col_page3 = st.columns([1, 2, 1])
        with col_page1:
            container = st.container()
            prev_disabled = current_page <= 1
            if container.button(
                "⬅️ Previous", disabled=prev_disabled, key="prev_btn_modular"
            ):
                st.session_state.text_analysis_page_modular -= 1
                st.rerun()
        with col_page2:
            st.markdown(
                f"<div style='text-align: center;'>Page {current_page} of {total_pages}</div>",
                unsafe_allow_html=True,
            )
        with col_page3:
            container = st.container()
            next_disabled = current_page >= total_pages
            if container.button(
                "Next ➡️", disabled=next_disabled, key="next_btn_modular"
            ):
                st.session_state.text_analysis_page_modular += 1
                st.rerun()

    else:
        st.write("No samples match the current filters.")


def model_performance_tab(all_data):
    st.header("Individual Model Classification Performance")
    st.markdown(
        "Confusion Matrix, Accuracy, and F1 Score for each model based on a severity threshold (>= 2 indicates 'Depressed'). Use the buttons below to cycle through models."
    )

    model_names = list(all_data.keys())
    num_models = len(model_names)

    if num_models == 0:
        st.warning("No models loaded to display performance.")
        return

    # --- Calculate Performance Data for ALL models first ---
    # This is needed for the overall comparison chart at the end
    performance_data = []
    all_figs = {} # Store figures for download later if needed within carousel

    for model_name in model_names:
        df = all_data[model_name].copy() # Work on a copy
        if "label" in df.columns:
            try:
                # Ensure label is int and handle potential NaNs
                df["label"] = pd.to_numeric(df["label"], errors="coerce")
                # Keep only rows where label is not NaN
                df = df.dropna(subset=["label"])
                df["label"] = df["label"].astype(int)

                df_clean = df.dropna(subset=["severity_rating", "label"])

                if not df_clean.empty:
                    fig, accuracy, f1 = plot_confusion_matrix(
                        df_clean, model_name
                    )
                    plt.close(fig) # Close the figure immediately after generating to save memory
                    all_figs[model_name] = fig # Store the figure object
                    performance_data.append(
                        {
                            "Model": model_name,
                            "Accuracy": accuracy,
                            "F1 Score": f1,
                        }
                    )
                else:
                     # Add placeholder if no valid data, so comparison chart doesn't break
                     performance_data.append({"Model": model_name, "Accuracy": np.nan, "F1 Score": np.nan})


            except Exception as e:
                st.error(f"Error calculating performance for {model_name}: {e}")
                performance_data.append({"Model": model_name, "Accuracy": np.nan, "F1 Score": np.nan})
        else:
            # Add placeholder if no label column
            performance_data.append({"Model": model_name, "Accuracy": np.nan, "F1 Score": np.nan})
            st.warning(f"Cannot calculate performance for {model_name}: 'label' column missing.")


    # --- Carousel Logic ---
    # Initialize the session state for the carousel index if it doesn't exist
    if 'performance_model_index' not in st.session_state:
        st.session_state.performance_model_index = 0

    # Ensure index is valid (e.g., if models were removed)
    if st.session_state.performance_model_index >= num_models:
        st.session_state.performance_model_index = 0

    current_index = st.session_state.performance_model_index

    # --- Display Carousel Controls ---
    st.write("---") # Separator
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        # Disable button if on the first model
        prev_disabled = current_index <= 0
        if st.button("⬅️ Previous", disabled=prev_disabled, key="perf_prev"):
            st.session_state.performance_model_index -= 1
            st.rerun() # Rerun to update the displayed model

    with col2:
        # Display the name of the current model being shown
        current_model_name = model_names[current_index]
        st.markdown(f"<h4 style='text-align: center; color: #2c3e50;'>Displaying: {current_model_name} ({current_index + 1} of {num_models})</h4>", unsafe_allow_html=True)


    with col3:
        # Disable button if on the last model
        next_disabled = current_index >= num_models - 1
        if st.button("Next ➡️", disabled=next_disabled, key="perf_next"):
            st.session_state.performance_model_index += 1
            st.rerun() # Rerun to update the displayed model
    st.write("---") # Separator

    # --- Display Performance for the SELECTED Model ---
    selected_model_name = model_names[current_index]
    st.subheader(f"Performance Details for: {selected_model_name}")

    if selected_model_name in all_figs:
        # Display the pre-generated figure for the current model
        st.pyplot(all_figs[selected_model_name])
        download_figure(all_figs[selected_model_name], f"{selected_model_name}_confusion_matrix.png")

        # Display metrics for this specific model
        model_perf = next((item for item in performance_data if item["Model"] == selected_model_name), None)
        if model_perf and pd.notna(model_perf['Accuracy']):
             st.write(f"**Accuracy:** {model_perf['Accuracy']:.3f}")
             st.write(f"**F1 Score:** {model_perf['F1 Score']:.3f}")
        else:
             st.write("Metrics could not be calculated (check data and 'label' column).")

    else:
        # This case might happen if performance calculation failed earlier
        st.warning(f"Performance data or plot not available for {selected_model_name}.")
        df_check = all_data[selected_model_name]
        if "label" not in df_check.columns:
             st.warning("'label' column is missing.")
        else:
             st.warning("Check if 'severity_rating' and 'label' columns have sufficient valid, non-missing data.")


    # --- Overall Performance Comparison (Stays the same) ---
    st.write("---") # Separator before overall comparison
    st.header("Overall Performance Comparison")

    # Filter out models where performance couldn't be calculated for the chart
    valid_performance_data = [p for p in performance_data if pd.notna(p.get('Accuracy')) and pd.notna(p.get('F1 Score'))]

    if valid_performance_data:
        comparison_df = pd.DataFrame(valid_performance_data)

        if not comparison_df.empty:
            # Plotting comparison
            fig_comp, ax_comp = plt.subplots(
                figsize=(max(6, len(comparison_df) * 1.5), 6) # Dynamic width based on valid models
            )
            x = np.arange(len(comparison_df))
            width = 0.35

            rects1 = ax_comp.bar(
                x - width / 2, comparison_df["Accuracy"], width, label="Accuracy"
            )
            rects2 = ax_comp.bar(
                x + width / 2, comparison_df["F1 Score"], width, label="F1 Score"
            )

            ax_comp.set_ylabel("Score")
            ax_comp.set_title("Model Performance Metrics Comparison")
            ax_comp.set_xticks(x)
            ax_comp.set_xticklabels(
                comparison_df["Model"], rotation=45, ha="right"
            )
            ax_comp.legend()
            ax_comp.bar_label(rects1, padding=3, fmt="%.2f")
            ax_comp.bar_label(rects2, padding=3, fmt="%.2f")
            fig_comp.tight_layout()

            st.pyplot(fig_comp)
            download_figure(fig_comp, "all_models_performance_comparison.png")
            st.dataframe(comparison_df.round(3))
        else:
             st.info("No valid performance metrics available to display comparison chart.")

    else:
        st.info(
            "No performance metrics could be calculated for any model (requires 'label' column in data)."
        )



def plot_confusion_matrix(df, model_name):
    df = df.copy()
    df["predicted_label"] = (df["severity_rating"] >= 2).astype(int)
    cm = confusion_matrix(df["label"], df["predicted_label"])
    accuracy = accuracy_score(df["label"], df["predicted_label"])
    f1 = f1_score(df["label"], df["predicted_label"])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Not Depressed", "Depressed"],
        yticklabels=["Not Depressed", "Depressed"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(
        f"{model_name} Confusion Matrix\nAccuracy: {accuracy:.2f}, F1 Score: {f1:.2f}"
    )

    return fig, accuracy, f1

def confusion_matrix_tab(gemini_df, deepseek_df):
    st.header("Depression Classification Performance")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gemini Classification Performance")
        gemini_fig, gemini_acc, gemini_f1 = plot_confusion_matrix(gemini_df, "Gemini")
        st.pyplot(gemini_fig)
        download_figure(gemini_fig, "gemini_confusion_matrix.png")

    with col2:
        st.subheader("DeepSeek Classification Performance")
        deepseek_fig, deepseek_acc, deepseek_f1 = plot_confusion_matrix(
            deepseek_df, "DeepSeek"
        )
        st.pyplot(deepseek_fig)
        download_figure(deepseek_fig, "deepseek_confusion_matrix.png")

    st.subheader("Model Performance Comparison")
    comparison_data = {
        "Model": ["Gemini Flash 2.0", "DeepSeek v3"],
        "Accuracy": [gemini_acc, deepseek_acc],
        "F1 Score": [gemini_f1, deepseek_f1],
    }

    comparison_df = pd.DataFrame(comparison_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comparison_data["Model"]))
    width = 0.35

    ax.bar(x - width / 2, comparison_data["Accuracy"], width, label="Accuracy")
    ax.bar(x + width / 2, comparison_data["F1 Score"], width, label="F1 Score")

    ax.set_ylabel("Score")
    ax.set_title("Model Performance Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_data["Model"])
    ax.legend()

    st.pyplot(fig)
    download_figure(fig, "performance_comparison.png")
    st.write(comparison_df)


# Run the app
if __name__ == "__main__":
    main()
