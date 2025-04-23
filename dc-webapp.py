import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import math
import io
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt

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
        mime="image/png"
    )

# Load the data files
@st.cache_data
def load_data():
    gemini_df = pd.read_csv("data/gemini-flash-2.0_results-525.csv")
    deepseek_df = pd.read_csv("data/deepseek-v3_results-525.csv")
    
    # Convert severity ratings to numeric
    gemini_df['severity_rating'] = pd.to_numeric(gemini_df['severity_rating'], errors='coerce')
    deepseek_df['severity_rating'] = pd.to_numeric(deepseek_df['severity_rating'], errors='coerce')
    
    return gemini_df, deepseek_df

# Create custom theme function
def set_custom_theme():
    st.markdown("""
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
    """, unsafe_allow_html=True)

# Main dashboard
def main():
    set_custom_theme() # Apply custom theme
    
    # Set title and description
    st.title("Depression Severity Detection Comparison")
    st.markdown("### Comparing Gemini Flash 2.0 vs DeepSeek v3 for Depression Assessment")
    
    try:
        gemini_df, deepseek_df = load_data()
        
        # Check data loading
        if gemini_df.empty or deepseek_df.empty:
            st.error("Error: One or both datasets are empty.")
            return

        # Create tabs for different sections 
        tab1, tab2, tab3 = st.tabs(["Overview & Statistics", "Text Analysis", "Confusion Matrix"])
        
        with tab1:
            overview_tab(gemini_df, deepseek_df)
        
        with tab2:
            text_analysis_tab(gemini_df, deepseek_df)
            
        with tab3:
            confusion_matrix_tab(gemini_df, deepseek_df)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.warning("Please make sure the data files exist in the correct location.")

def plot_correlation_heatmap(merged_df):
    """
    Generate a heatmap showing the correlation between Gemini and DeepSeek severity ratings.
    """
    # Create a pivot table for counts
    heatmap_data = merged_df.groupby(['severity_rating_gemini', 'severity_rating_deepseek']).size().unstack(fill_value=0)

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
        x='severity_rating_gemini',
        y='severity_rating_deepseek',
        kind='scatter',
        marginal_kws=dict(bins=10, fill=True),
        joint_kws=dict(alpha=0.7)
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
        x='severity_rating_gemini',
        y='severity_rating_deepseek',
        palette='muted',
        ax=ax
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
    point_counts = merged_df.groupby(['severity_rating_gemini', 'severity_rating_deepseek']).size().reset_index(name='count')

    # Normalize bubble sizes for better visualization
    max_count = point_counts['count'].max()
    point_counts['size'] = 50 + (point_counts['count'] / max_count) * 200  # Scale sizes between 50 and 250

    # Plot points with different colors based on agreement/disagreement
    for _, row in point_counts.iterrows():
        x_val = row['severity_rating_gemini']
        y_val = row['severity_rating_deepseek']
        size = row['size']

        if x_val == y_val:
            ax.scatter(x_val, y_val, s=size, color='green', alpha=0.7, label='Agreement')
        elif x_val > y_val:
            ax.scatter(x_val, y_val, s=size, color='blue', alpha=0.7, label='Gemini > DeepSeek')
        else:
            ax.scatter(x_val, y_val, s=size, color='orange', alpha=0.7, label='DeepSeek > Gemini')

    # Draw perfect agreement line
    min_val = min(merged_df['severity_rating_gemini'].min(), merged_df['severity_rating_deepseek'].min())
    max_val = max(merged_df['severity_rating_gemini'].max(), merged_df['severity_rating_deepseek'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Agreement')

    # Set axis labels and title
    ax.set_xlabel('Gemini Severity Rating')
    ax.set_ylabel('DeepSeek Severity Rating')
    ax.set_title('Correlation between Model Ratings')

    # Set integer ticks
    ax.set_xticks(range(int(min_val), int(max_val) + 1))
    ax.set_yticks(range(int(min_val), int(max_val) + 1))

    # Remove duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')

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
    sns.countplot(data=df, x='severity_rating', ax=ax, palette='viridis')
    ax.set_title(f'{title} Severity Rating Distribution')
    ax.set_xlabel('Severity Rating (0-4)')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    download_figure(fig, filename)

    # Calculate statistics
    st.subheader("Statistics")
    stats = {
        "Mean Severity": df['severity_rating'].mean(),
        "Median Severity": df['severity_rating'].median(),
        "Standard Deviation": df['severity_rating'].std()
    }
    st.write(pd.DataFrame([stats]))


def merge_datasets(gemini_df, deepseek_df):
    """
    Merge Gemini and DeepSeek datasets on text for comparison.
    """
    return pd.merge(
        gemini_df[['text', 'severity_rating']],
        deepseek_df[['text', 'severity_rating']],
        on='text',
        suffixes=('_gemini', '_deepseek')
    )


def plot_side_by_side_comparison(gemini_df, deepseek_df):
    """
    Generate a bar chart comparing severity ratings between Gemini and DeepSeek.
    """
    severity_counts_gemini = gemini_df['severity_rating'].value_counts().sort_index()
    severity_counts_deepseek = deepseek_df['severity_rating'].value_counts().sort_index()

    all_ratings = sorted(set(severity_counts_gemini.index) | set(severity_counts_deepseek.index))
    complete_gemini = pd.Series({r: severity_counts_gemini.get(r, 0) for r in all_ratings}, name='Gemini')
    complete_deepseek = pd.Series({r: severity_counts_deepseek.get(r, 0) for r in all_ratings}, name='DeepSeek')

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    x = np.arange(len(all_ratings))

    ax.bar(x - bar_width / 2, complete_gemini, bar_width, label='Gemini')
    ax.bar(x + bar_width / 2, complete_deepseek, bar_width, label='DeepSeek')

    ax.set_xlabel('Severity Rating')
    ax.set_ylabel('Count')
    ax.set_title('Comparison of Severity Ratings Between Models')
    ax.set_xticks(x)
    ax.set_xticklabels(all_ratings)
    ax.legend()

    return fig


def overview_tab(gemini_df, deepseek_df):
    st.header("Dataset Overview")

    # Display basic statistics for both datasets
    col1, col2 = st.columns(2)

    with col1:
        display_dataset_statistics(gemini_df, "Gemini-Flash-2.0 Results", "gemini_severity_distribution.png")

    with col2:
        display_dataset_statistics(deepseek_df, "DeepSeek-v3 Results", "deepseek_severity_distribution.png")

    # Comparison between models
    st.header("Model Comparison")
    merged_df = merge_datasets(gemini_df, deepseek_df)

    # Calculate agreement metrics
    exact_agreement = (merged_df['severity_rating_gemini'] == merged_df['severity_rating_deepseek']).mean() * 100

    # Display agreement metrics
    st.subheader("Agreement Metrics")
    st.write(f"Exact Agreement: {exact_agreement:.2f}%")

    # Allow user to select visualization type
    st.subheader("Select Visualization Type")
    visualization_type = st.selectbox(
        "Choose a visualization to display the correlation:",
        options=["Scatter Plot", "Heatmap", "Joint Plot", "Violin Plot"]
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



def text_analysis_tab(gemini_df, deepseek_df):
    st.header("Text Sample Analysis")

    # Merge dataframes to compare explanations
    merged_df = pd.merge(
        gemini_df[['text', 'label', 'severity_rating', 'explanation']],
        deepseek_df[['text', 'severity_rating', 'explanation']],
        on='text',
        suffixes=('_gemini', '_deepseek')
    )

    # Convert to numeric
    merged_df['label'] = merged_df['label'].astype(int)

    # Create a searchable/filterable table
    st.subheader("Filter Samples")

    # Create filter options
    col_filter1, col_filter2, col_filter3 = st.columns(3)

    with col_filter1:
        label_filter = st.selectbox(
            "Filter by Label",
            options=[None, 0, 1],
            format_func=lambda x: "All" if x is None else ("Not Depressed (0)" if x == 0 else "Depressed (1)"),
            key="label_filter"
        )

    with col_filter2:
        rating_diff_filter = st.slider(
            "Filter by Rating Difference (>=)",
            min_value=0, max_value=4, value=0,
            key="rating_diff_filter"
        )

    with col_filter3:
        valid_gemini_ratings = sorted(merged_df['severity_rating_gemini'].dropna().unique().astype(int))
        rating_gemini_filter = st.multiselect(
            "Filter by Gemini Rating",
            options=valid_gemini_ratings,
            key="rating_gemini_filter"
        )

    filtered_df = merged_df.copy()

    if label_filter is not None:
        filtered_df = filtered_df[filtered_df['label'] == label_filter]

    filtered_df = filtered_df.dropna(subset=['severity_rating_gemini', 'severity_rating_deepseek'])

    if rating_diff_filter > 0:
        filtered_df['severity_rating_gemini'] = pd.to_numeric(filtered_df['severity_rating_gemini'], errors='coerce')
        filtered_df['severity_rating_deepseek'] = pd.to_numeric(filtered_df['severity_rating_deepseek'], errors='coerce')
        filtered_df = filtered_df.dropna(subset=['severity_rating_gemini', 'severity_rating_deepseek'])
        if not filtered_df.empty:
             filtered_df = filtered_df[abs(filtered_df['severity_rating_gemini'] - filtered_df['severity_rating_deepseek']) >= rating_diff_filter]

    if rating_gemini_filter:
        filtered_df = filtered_df[filtered_df['severity_rating_gemini'].isin(rating_gemini_filter)]

    if not filtered_df.empty:
         if 'severity_rating_gemini' in filtered_df.columns and 'severity_rating_deepseek' in filtered_df.columns:
              filtered_df['rating_difference'] = abs(filtered_df['severity_rating_gemini'] - filtered_df['severity_rating_deepseek'])
         else:
              filtered_df['rating_difference'] = pd.Series(dtype='float')
    else:
         filtered_df['rating_difference'] = pd.Series(dtype='float')

    items_per_page_options = [10, 15, 25]
    items_per_page = st.selectbox(
        "Items per page:",
        options=items_per_page_options,
        index=0
    )
    
    total_items = len(filtered_df)
    st.subheader(f"Filtered Samples ({total_items} samples)")

    if total_items > 0:
        total_pages = math.ceil(total_items / items_per_page)

        if 'text_analysis_page' not in st.session_state:
            st.session_state.text_analysis_page = 1

        if st.session_state.text_analysis_page > total_pages:
             st.session_state.text_analysis_page = 1

        current_page = st.session_state.text_analysis_page

        start_idx = (current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)

        paginated_df = filtered_df.iloc[start_idx:end_idx]

        for i, row in paginated_df.iterrows():
            rating_g = int(row['severity_rating_gemini']) if pd.notna(row['severity_rating_gemini']) else 'N/A'
            rating_d = int(row['severity_rating_deepseek']) if pd.notna(row['severity_rating_deepseek']) else 'N/A'
            rating_diff = int(row['rating_difference']) if pd.notna(row['rating_difference']) else 'N/A'

            expander_label = (
                f"Sample (Index: {i}) - "
                f"Severity: G:{rating_g} vs "
                f"D:{rating_d} "
                f"(Diff: {rating_diff})"
            )
            with st.expander(expander_label):
                st.write(f"**Text:** {row['text']}")
                st.write(f"**True Label:** {'Depressed (1)' if row['label'] == 1 else 'Not Depressed (0)'}")

                col_exp1, col_exp2 = st.columns(2)

                with col_exp1:
                    st.markdown("#### **Gemini Analysis:**")
                    st.write(f"**Severity Rating:** {rating_g}")
                    explanation_g = str(row.get('explanation_gemini', 'N/A'))
                    st.write(f"**Explanation:** {explanation_g.replace('**Explanation:**', '').strip()}")

                with col_exp2:
                    st.markdown("#### **DeepSeek Analysis:**")
                    st.write(f"**Severity Rating:** {rating_d}")
                    explanation_d = str(row.get('explanation_deepseek', 'N/A'))
                    st.write(f"**Explanation:** {explanation_d.replace('**Explanation:**', '').strip()}")

        st.write("---")
        col_page1, col_page2, col_page3 = st.columns([1, 2, 1])

        with col_page1:
            container = st.container()
            # container.markdown("<div style='display: flex; justify-content: flex-end;'>", unsafe_allow_html=True)
            prev_disabled = current_page <= 1
            if container.button("⬅️ Previous", disabled=prev_disabled, key="prev_btn"):
                st.session_state.text_analysis_page -= 1
                st.rerun()
            container.markdown("</div>", unsafe_allow_html=True)

        with col_page2:
            st.markdown(
                f"<div style='display: flex; justify-content: center; align-items: center; height: 38px;'>"
                f"<span style='font-size: 16px;'>Page {current_page} of {total_pages}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        with col_page3:
            container = st.container()
            # container.markdown("<div style='display: flex; justify-content: flex-start;'>", unsafe_allow_html=True)
            next_disabled = current_page >= total_pages
            if container.button("Next ➡️", disabled=next_disabled, key="next_btn"):
                st.session_state.text_analysis_page += 1
                st.rerun()
            container.markdown("</div>", unsafe_allow_html=True)

    else:
        st.write("No samples match the current filters.")

def confusion_matrix_tab(gemini_df, deepseek_df):
    st.header("Depression Classification Performance")
    
    col1, col2 = st.columns(2)
    
    def plot_confusion_matrix(df, model_name):
        df = df.copy()
        df['predicted_label'] = (df['severity_rating'] >= 2).astype(int)
        cm = confusion_matrix(df['label'], df['predicted_label'])
        accuracy = accuracy_score(df['label'], df['predicted_label'])
        f1 = f1_score(df['label'], df['predicted_label'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Not Depressed', 'Depressed'],
                   yticklabels=['Not Depressed', 'Depressed'],
                   ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'{model_name} Confusion Matrix\nAccuracy: {accuracy:.2f}, F1 Score: {f1:.2f}')
        
        return fig, accuracy, f1
    
    with col1:
        st.subheader("Gemini Classification Performance")
        gemini_fig, gemini_acc, gemini_f1 = plot_confusion_matrix(gemini_df, "Gemini")
        st.pyplot(gemini_fig)
        download_figure(gemini_fig, "gemini_confusion_matrix.png")
    
    with col2:
        st.subheader("DeepSeek Classification Performance")
        deepseek_fig, deepseek_acc, deepseek_f1 = plot_confusion_matrix(deepseek_df, "DeepSeek")
        st.pyplot(deepseek_fig)
        download_figure(deepseek_fig, "deepseek_confusion_matrix.png")
    
    st.subheader("Model Performance Comparison")
    comparison_data = {
        "Model": ["Gemini Flash 2.0", "DeepSeek v3"],
        "Accuracy": [gemini_acc, deepseek_acc],
        "F1 Score": [gemini_f1, deepseek_f1]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comparison_data["Model"]))
    width = 0.35
    
    ax.bar(x - width/2, comparison_data["Accuracy"], width, label='Accuracy')
    ax.bar(x + width/2, comparison_data["F1 Score"], width, label='F1 Score')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_data["Model"])
    ax.legend()
    
    st.pyplot(fig)
    download_figure(fig, "performance_comparison.png")
    st.write(comparison_df)

# Run the app
if __name__ == "__main__":
    main()
