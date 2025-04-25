import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
import io
import glob
import os
import re
import krippendorff
from functools import reduce
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
)

st.set_page_config(page_title="Depression Detection Comparison", layout="wide")


# --- Theme Colors (Define globally for consistency) ---
LIGHT_THEME = {
    "text_color": "#2c3e50",  # Match h1,h2,h3 color
    "face_color": "#f8f9fa",  # Match main background
    "grid_color": "#D3D3D3",  # Light grey for grid
    "palette_sequential": "viridis",
    "palette_heatmap": "Reds",
    "palette_binary": ["#1f77b4", "#ff7f0e"],  # Default blue/orange
    "scatter_agree": "green",
    "scatter_a_gt_b": "blue",
    "scatter_b_gt_a": "orange",
    "perfect_line": "red",
}

DARK_THEME = {
    "text_color": "#FAFAFA",  # Light text for dark background
    "face_color": "#0E1117",  # Default Streamlit dark bg
    "grid_color": "#555555",  # Darker grey for grid
    "palette_sequential": "plasma",  # Different palette for dark
    "palette_heatmap": "hot",  # Different heatmap palette
    "palette_binary": ["#2ECC71", "#FFA500"],  # Green/Orange maybe?
    "scatter_agree": "#2ECC71",  # Brighter green
    "scatter_a_gt_b": "#3498DB",  # Brighter blue
    "scatter_b_gt_a": "#F39C12",  # Brighter orange
    "perfect_line": "#E74C3C",  # Brighter red
}


# --- Helper Functions ---


def download_figure(fig, filename):
    # (Keep this function as is)
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor()
    )  # Ensure facecolor is saved
    buf.seek(0)
    st.download_button(
        label="Download Figure",
        data=buf.getvalue(),
        file_name=filename,
        mime="image/png",
    )


@st.cache_data
def load_all_data(pattern="data/*_results-525.csv"):
    # (Keep this function as is)
    data_files = glob.glob(pattern)
    all_data = {}
    if not data_files:
        st.warning(f"No data files found: {pattern}")
        return all_data
    for f in data_files:
        try:
            model_name = os.path.basename(f).replace("_results-525.csv", "")
            model_name = model_name.replace("-", " ").replace("_", " ").title()
            df = pd.read_csv(f)
            if "severity_rating" not in df.columns or "text" not in df.columns:
                st.error(f"File '{f}' missing required columns. Skipping.")
                continue
            if "label" not in df.columns:
                st.warning(f"File '{f}' missing 'label' column.")
            df["severity_rating"] = pd.to_numeric(
                df["severity_rating"], errors="coerce"
            )
            all_data[model_name] = df
            print(f"Loaded {model_name} data with {len(df)} samples.")
        except Exception as e:
            st.error(f"Error loading file {f}: {str(e)}")
    return all_data


def set_custom_theme():
    # (Keep this function as is, maybe adjust max-width later if needed)
    st.markdown(
        """
    <style>
    .main { background-color: #f8f9fa; }
    .stApp { max-width: 1400px; margin: 0 auto; } /* Slightly wider max */
    h1, h2, h3 { color: #2c3e50; }
    .stAlert { border-radius: 8px; }
    /* Attempt to style plot background - might be overridden by matplotlib */
    .stpyplot { background-color: transparent !important; }
    </style>
    """,
        unsafe_allow_html=True,
    )


# --- Plotting Functions (Modified for Theme) ---


def plot_confusion_matrix(df, model_name, theme_colors):
    df = df.copy()
    df["label"] = df["label"].astype(int)
    df["predicted_label"] = (df["severity_rating"] >= 2).astype(int)
    cm = confusion_matrix(df["label"], df["predicted_label"])
    accuracy = accuracy_score(df["label"], df["predicted_label"])
    f1 = f1_score(df["label"], df["predicted_label"])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor(theme_colors["face_color"])  # Set figure background
    ax.set_facecolor(theme_colors["face_color"])  # Set axes background

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,  # Keep Blues for CM usually
        xticklabels=["Not Depressed", "Depressed"],
        yticklabels=["Not Depressed", "Depressed"],
        ax=ax,
        annot_kws={"color": theme_colors["text_color"]},  # Adjust annotation color
    )
    ax.set_xlabel("Predicted Label", color=theme_colors["text_color"])
    ax.set_ylabel("True Label", color=theme_colors["text_color"])
    ax.set_title(
        f"{model_name} Confusion Matrix\nAccuracy: {accuracy:.2f}, F1 Score: {f1:.2f}",
        color=theme_colors["text_color"],
    )
    # Set tick colors
    ax.tick_params(axis="x", colors=theme_colors["text_color"])
    ax.tick_params(axis="y", colors=theme_colors["text_color"])
    # Set border colors
    for spine in ax.spines.values():
        spine.set_edgecolor(theme_colors["grid_color"])

    fig.tight_layout()
    return fig, accuracy, f1


def display_dataset_statistics(df, title, filename, theme_colors):
    st.subheader(title)
    st.write(f"Total samples: {len(df)}")

    st.subheader("Distribution of Severity Ratings")
    fig, ax = plt.subplots(figsize=(7, 4.5))  # Slightly wider
    fig.patch.set_facecolor(theme_colors["face_color"])
    ax.set_facecolor(theme_colors["face_color"])

    sns.countplot(
        data=df, x="severity_rating", ax=ax, palette=theme_colors["palette_sequential"]
    )
    ax.set_title(
        f"{title} Severity Rating Distribution", color=theme_colors["text_color"]
    )
    ax.set_xlabel("Severity Rating (0-4)", color=theme_colors["text_color"])
    ax.set_ylabel("Count", color=theme_colors["text_color"])
    ax.grid(axis="y", color=theme_colors["grid_color"], linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_edgecolor(theme_colors["grid_color"])
    ax.spines["bottom"].set_edgecolor(theme_colors["grid_color"])
    ax.tick_params(axis="x", colors=theme_colors["text_color"])
    ax.tick_params(axis="y", colors=theme_colors["text_color"])

    # Make bar labels readable
    for container in ax.containers:
        ax.bar_label(container, color=theme_colors["text_color"], fontsize=9, padding=3)

    st.image(fig_to_buffer(fig), width=600)  # Use buffer and st.image
    plt.close(fig)
    download_figure(fig, filename)  # Download still uses original fig

    st.subheader("Statistics")
    stats = {
        "Mean Severity": df["severity_rating"].mean(),
        "Median Severity": df["severity_rating"].median(),
        "Standard Deviation": df["severity_rating"].std(),
    }
    st.dataframe(pd.DataFrame([stats]).round(3))  # Use dataframe for better formatting


def plot_correlation_heatmap_pair(df, col_a, col_b, name_a, name_b, theme_colors):
    if col_a not in df.columns or col_b not in df.columns:
        st.error("Rating columns missing.")
        return plt.figure()
    df[col_a] = pd.to_numeric(df[col_a], errors="coerce")
    df[col_b] = pd.to_numeric(df[col_b], errors="coerce")
    df_clean = df.dropna(subset=[col_a, col_b]).astype({col_a: int, col_b: int})
    if df_clean.empty:
        st.warning("No valid data for heatmap.")
        return plt.figure()

    try:
        all_ratings = np.arange(0, 5)
        heatmap_data = pd.crosstab(df_clean[col_a], df_clean[col_b])
        heatmap_data = heatmap_data.reindex(
            index=all_ratings, columns=all_ratings, fill_value=0
        )
    except Exception as e:
        st.error(f"Heatmap data error: {e}")
        return plt.figure()

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(theme_colors["face_color"])
    ax.set_facecolor(theme_colors["face_color"])

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt="d",
        cmap=theme_colors["palette_heatmap"],
        cbar=True,
        ax=ax,
        square=True,
        linewidths=0.5,
        linecolor=theme_colors["face_color"],
        annot_kws={"color": theme_colors["text_color"]},
        cbar_kws={"label": "Frequency"},
    )
    ax.figure.axes[-1].yaxis.label.set_color(
        theme_colors["text_color"]
    )  # Colorbar label
    ax.tick_params(colors=theme_colors["text_color"])  # Colorbar ticks

    ax.set_xlabel(f"{name_b} Severity Rating", color=theme_colors["text_color"])
    ax.set_ylabel(f"{name_a} Severity Rating", color=theme_colors["text_color"])
    ax.set_title(
        f"Rating Frequency Heatmap: {name_a} vs. {name_b}",
        color=theme_colors["text_color"],
    )
    ax.set_xticks(np.arange(len(all_ratings)) + 0.5)
    ax.set_yticks(np.arange(len(all_ratings)) + 0.5)
    ax.set_xticklabels(all_ratings, color=theme_colors["text_color"])
    ax.set_yticklabels(all_ratings, color=theme_colors["text_color"], rotation=0)
    fig.tight_layout()
    return fig


def plot_violin_distribution_pair(df, col_a, col_b, name_a, name_b, theme_colors):
    if col_a not in df.columns or col_b not in df.columns:
        st.error("Rating columns missing.")
        return plt.figure()
    df[col_a] = pd.to_numeric(df[col_a], errors="coerce")
    df[col_b] = pd.to_numeric(df[col_b], errors="coerce")
    df_clean = df.dropna(subset=[col_a, col_b]).astype({col_a: int})
    if df_clean.empty:
        st.warning("No valid data for violin plot.")
        return plt.figure()

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(theme_colors["face_color"])
    ax.set_facecolor(theme_colors["face_color"])

    try:
        sns.violinplot(
            data=df_clean,
            x=col_a,
            y=col_b,
            palette=theme_colors["palette_sequential"],
            inner="quartile",
            ax=ax,
            saturation=0.7,
        )
        sns.stripplot(
            data=df_clean,
            x=col_a,
            y=col_b,
            color=theme_colors["text_color"],
            alpha=0.2,
            size=3,
            ax=ax,
            jitter=True,  # Use text color for points
        )
        ax.set_xlabel(f"{name_a} Severity Rating", color=theme_colors["text_color"])
        ax.set_ylabel(f"{name_b} Severity Rating", color=theme_colors["text_color"])
        ax.set_title(
            f"Distribution of {name_b} Ratings per {name_a} Rating",
            color=theme_colors["text_color"],
        )
        ax.set_yticks(np.arange(0, 5))
        ax.grid(
            axis="y", color=theme_colors["grid_color"], linestyle="--", linewidth=0.5
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_edgecolor(theme_colors["grid_color"])
        ax.spines["bottom"].set_edgecolor(theme_colors["grid_color"])
        ax.tick_params(axis="x", colors=theme_colors["text_color"])
        ax.tick_params(axis="y", colors=theme_colors["text_color"])
        fig.tight_layout()
    except Exception as e:
        st.error(f"Error creating violin plot: {e}")
    return fig


def plot_scatter_correlation_pair(df, col_a, col_b, name_a, name_b, theme_colors):
    fig, ax = plt.subplots(figsize=(8, 7))  # Adjusted size
    fig.patch.set_facecolor(theme_colors["face_color"])
    ax.set_facecolor(theme_colors["face_color"])

    if col_a not in df.columns or col_b not in df.columns:
        st.error("Rating columns missing.")
        return fig
    df[col_a] = pd.to_numeric(df[col_a], errors="coerce")
    df[col_b] = pd.to_numeric(df[col_b], errors="coerce")
    df_clean = df.dropna(subset=[col_a, col_b])
    if df_clean.empty:
        st.warning("No valid data for scatter plot.")
        return fig

    jitter_strength = 0.15
    x_jitter = df_clean[col_a] + np.random.normal(
        0, jitter_strength, size=len(df_clean)
    )
    y_jitter = df_clean[col_b] + np.random.normal(
        0, jitter_strength, size=len(df_clean)
    )

    colors = np.where(
        df_clean[col_a] == df_clean[col_b],
        theme_colors["scatter_agree"],
        np.where(
            df_clean[col_a] > df_clean[col_b],
            theme_colors["scatter_a_gt_b"],
            theme_colors["scatter_b_gt_a"],
        ),
    )

    plotted_labels = set()
    for i in range(len(df_clean)):
        x_val, y_val = x_jitter.iloc[i], y_jitter.iloc[i]
        original_x, original_y = df_clean[col_a].iloc[i], df_clean[col_b].iloc[i]
        color = colors[i]
        if original_x == original_y:
            label = "Agreement"
        elif original_x > original_y:
            label = f"{name_a} > {name_b}"
        else:
            label = f"{name_b} > {name_a}"
        if label not in plotted_labels:
            ax.scatter(x_val, y_val, color=color, alpha=0.7, label=label, s=50)
            plotted_labels.add(label)
        else:
            ax.scatter(x_val, y_val, color=color, alpha=0.7, s=50)

    min_val, max_val = 0, 4
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color=theme_colors["perfect_line"],
        linestyle="--",
        label="Perfect Agreement",
        lw=1.5,
    )

    ax.set_xlabel(f"{name_a} Severity Rating", color=theme_colors["text_color"])
    ax.set_ylabel(f"{name_b} Severity Rating", color=theme_colors["text_color"])
    ax.set_title(
        f"Rating Comparison: {name_a} vs. {name_b}", color=theme_colors["text_color"]
    )
    ax.set_xticks(np.arange(min_val, max_val + 1))
    ax.set_yticks(np.arange(min_val, max_val + 1))
    ax.set_xlim(min_val - 0.5, max_val + 0.5)
    ax.set_ylim(min_val - 0.5, max_val + 0.5)
    ax.tick_params(axis="x", colors=theme_colors["text_color"])
    ax.tick_params(axis="y", colors=theme_colors["text_color"])
    ax.grid(color=theme_colors["grid_color"], linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_edgecolor(theme_colors["grid_color"])
    ax.spines["right"].set_edgecolor(theme_colors["grid_color"])
    ax.spines["left"].set_edgecolor(theme_colors["grid_color"])
    ax.spines["bottom"].set_edgecolor(theme_colors["grid_color"])

    # Update legend properties
    legend = ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False
    )
    for text in legend.get_texts():
        text.set_color(theme_colors["text_color"])

    fig.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to prevent legend overlap
    return fig


# --- Helper to convert fig to buffer for st.image ---
def fig_to_buffer(fig):
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor()
    )
    buf.seek(0)
    return buf


# --- Tab Functions (Modified to pass theme) ---


def overall_overview_tab(all_data, theme_colors):
    st.header("Model Overviews & Statistics")
    st.markdown("Basic statistics and severity distributions for each loaded model.")
    model_names = list(all_data.keys())
    num_models = len(model_names)
    cols_per_row = 2
    num_rows = math.ceil(num_models / cols_per_row)
    model_iter = iter(model_names)
    for _ in range(num_rows):
        cols = st.columns(cols_per_row)
        for i in range(cols_per_row):
            model_name = next(model_iter, None)
            if model_name:
                with cols[i]:
                    df = all_data[model_name]
                    # Pass theme_colors to display function
                    display_dataset_statistics(
                        df,
                        model_name,
                        f"{model_name}_severity_distribution.png",
                        theme_colors,
                    )
            else:
                break
        st.markdown("---")


def model_performance_tab(all_data, theme_colors):
    st.header("Individual Model Classification Performance")
    st.markdown(
        "Confusion Matrix, Accuracy, and F1 Score per model (Severity >= 2 -> Depressed)."
    )  # Concise
    model_names = list(all_data.keys())
    num_models = len(model_names)
    if num_models == 0:
        st.warning("No models loaded.")
        return

    performance_data = []
    all_figs = {}
    for model_name in model_names:
        df = all_data[model_name].copy()
        if "label" in df.columns:
            try:
                df["label"] = pd.to_numeric(df["label"], errors="coerce")
                df = df.dropna(subset=["label"])
                df["label"] = df["label"].astype(int)
                df_clean = df.dropna(subset=["severity_rating", "label"])
                if not df_clean.empty:
                    # Pass theme_colors to plot function
                    fig, accuracy, f1 = plot_confusion_matrix(
                        df_clean, model_name, theme_colors
                    )
                    all_figs[model_name] = fig
                    performance_data.append(
                        {"Model": model_name, "Accuracy": accuracy, "F1 Score": f1}
                    )
                else:
                    performance_data.append(
                        {"Model": model_name, "Accuracy": np.nan, "F1 Score": np.nan}
                    )
            except Exception as e:
                st.error(f"Perf. calc error for {model_name}: {e}")
                performance_data.append(
                    {"Model": model_name, "Accuracy": np.nan, "F1 Score": np.nan}
                )
        else:
            performance_data.append(
                {"Model": model_name, "Accuracy": np.nan, "F1 Score": np.nan}
            )

    if "performance_model_index" not in st.session_state:
        st.session_state.performance_model_index = 0
    if st.session_state.performance_model_index >= num_models:
        st.session_state.performance_model_index = 0
    if num_models > 0 and st.session_state.performance_model_index < 0:
        st.session_state.performance_model_index = 0
    current_index = st.session_state.performance_model_index

    st.write("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        prev_disabled = current_index <= 0
        if st.button(
            "⬅️ Previous",
            disabled=prev_disabled,
            key="perf_prev",
            use_container_width=True,
        ):
            st.session_state.performance_model_index -= 1
            st.rerun()
    with col2:
        if num_models > 0:
            current_model_name = model_names[current_index]
            st.markdown(
                f"<div style='text-align: center; margin-top: 5px; color: {theme_colors['text_color']};'><b>Displaying: {current_model_name}</b><br>({current_index + 1} of {num_models})</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='text-align: center; margin-top: 5px;'>No Models</div>",
                unsafe_allow_html=True,
            )
    with col3:
        next_disabled = current_index >= num_models - 1
        if st.button(
            "Next ➡️", disabled=next_disabled, key="perf_next", use_container_width=True
        ):
            st.session_state.performance_model_index += 1
            st.rerun()
    st.write("---")

    if num_models > 0:
        selected_model_name = model_names[current_index]
        with st.container():
            st.subheader(f"Performance Details for: {selected_model_name}")
            if selected_model_name in all_figs:
                fig_to_display = all_figs[selected_model_name]
                st.image(fig_to_buffer(fig_to_display), width=550)
                plt.close(fig_to_display)
                download_figure(
                    all_figs[selected_model_name],
                    f"{selected_model_name}_confusion_matrix.png",
                )
                model_perf = next(
                    (
                        item
                        for item in performance_data
                        if item["Model"] == selected_model_name
                    ),
                    None,
                )
                if model_perf and pd.notna(model_perf["Accuracy"]):
                    metric_col1, metric_col2 = st.columns(2)
                    metric_col1.metric(
                        label="Accuracy", value=f"{model_perf['Accuracy']:.3f}"
                    )
                    metric_col2.metric(
                        label="F1 Score", value=f"{model_perf['F1 Score']:.3f}"
                    )
                else:
                    st.info("Metrics N/A.")
            else:
                st.warning(f"Plot not available for {selected_model_name}.")
                if all_data[selected_model_name].get("label") is None:
                    st.warning("'label' column missing.")
                else:
                    st.warning("Check data validity.")
    else:
        st.info("No models loaded.")

    st.write("---")
    st.header("Overall Performance Comparison")
    valid_performance_data = [
        p
        for p in performance_data
        if pd.notna(p.get("Accuracy")) and pd.notna(p.get("F1 Score"))
    ]
    if valid_performance_data:
        comparison_df = pd.DataFrame(valid_performance_data)
        if not comparison_df.empty:
            fig_comp, ax_comp = plt.subplots(
                figsize=(max(5, len(comparison_df) * 1.2), 5)
            )
            fig_comp.patch.set_facecolor(theme_colors["face_color"])
            ax_comp.set_facecolor(theme_colors["face_color"])
            x = np.arange(len(comparison_df))
            width = 0.35
            rects1 = ax_comp.bar(
                x - width / 2,
                comparison_df["Accuracy"],
                width,
                label="Accuracy",
                color=theme_colors["palette_binary"][0],
            )
            rects2 = ax_comp.bar(
                x + width / 2,
                comparison_df["F1 Score"],
                width,
                label="F1 Score",
                color=theme_colors["palette_binary"][1],
            )
            ax_comp.set_ylabel("Score", color=theme_colors["text_color"])
            ax_comp.set_title(
                "Model Performance Metrics Comparison", color=theme_colors["text_color"]
            )
            ax_comp.set_xticks(x)
            ax_comp.set_xticklabels(
                comparison_df["Model"],
                rotation=45,
                ha="right",
                color=theme_colors["text_color"],
            )
            legend = ax_comp.legend()
            for text in legend.get_texts():
                text.set_color(theme_colors["text_color"])
            ax_comp.bar_label(
                rects1, padding=3, fmt="%.2f", color=theme_colors["text_color"]
            )
            ax_comp.bar_label(
                rects2, padding=3, fmt="%.2f", color=theme_colors["text_color"]
            )
            ax_comp.grid(
                axis="y",
                color=theme_colors["grid_color"],
                linestyle="--",
                linewidth=0.5,
            )
            ax_comp.spines["top"].set_visible(False)
            ax_comp.spines["right"].set_visible(False)
            ax_comp.spines["left"].set_edgecolor(theme_colors["grid_color"])
            ax_comp.spines["bottom"].set_edgecolor(theme_colors["grid_color"])
            ax_comp.tick_params(axis="x", colors=theme_colors["text_color"])
            ax_comp.tick_params(axis="y", colors=theme_colors["text_color"])
            fig_comp.tight_layout()

            st.image(fig_to_buffer(fig_comp), width=700)
            plt.close(fig_comp)
            download_figure(fig_comp, "all_models_performance_comparison.png")
            st.dataframe(comparison_df.round(3))
        else:
            st.info("No valid metrics for comparison chart.")
    else:
        st.info("No performance metrics calculated.")
    for fig in all_figs.values():
        plt.close(fig)


def pairwise_comparison_tab(all_data, theme_colors):
    st.header("Pairwise Model Comparison")
    model_names = list(all_data.keys())
    if len(model_names) < 2:
        st.warning("Need at least two models.")
        return

    col1, col2 = st.columns(2)
    with col1:
        model_a_name = st.selectbox(
            "Select Model A:", model_names, index=0, key="model_a_select"
        )
    with col2:
        available_b = [m for m in model_names if m != model_a_name]
        if not available_b:
            st.warning("Only one model available.")
            return
        default_b_index = 0 if available_b else -1
        model_b_name = st.selectbox(
            "Select Model B:", available_b, index=default_b_index, key="model_b_select"
        )

    if not model_a_name or not model_b_name or model_a_name == model_b_name:
        st.warning("Please select two different models.")
        return

    st.subheader(f"Comparing: {model_a_name} vs. {model_b_name}")
    df_a = (
        all_data[model_a_name][["text", "severity_rating"]]
        .copy()
        .rename(columns={"severity_rating": "severity_rating_a"})
    )
    df_b = (
        all_data[model_b_name][["text", "severity_rating"]]
        .copy()
        .rename(columns={"severity_rating": "severity_rating_b"})
    )
    merged_pair_df = pd.merge(df_a, df_b, on="text", how="inner")
    if merged_pair_df.empty:
        st.warning(f"No common 'text' entries found.")
        return
    merged_pair_df = merged_pair_df.dropna(
        subset=["severity_rating_a", "severity_rating_b"]
    )
    if merged_pair_df.empty:
        st.warning(f"No valid rating pairs after cleaning NaNs.")
        return

    try:
        ratings_a = merged_pair_df["severity_rating_a"].astype(int)
        ratings_b = merged_pair_df["severity_rating_b"].astype(int)
    except ValueError:
        st.error("Could not convert ratings to integers.")
        return

    st.subheader("Agreement Metrics")
    col_metric1, col_metric2 = st.columns(2)
    exact_agreement = (ratings_a == ratings_b).mean() * 100
    col_metric1.metric(label="Exact Agreement", value=f"{exact_agreement:.2f}%")
    try:
        weighted_kappa = cohen_kappa_score(ratings_a, ratings_b, weights="quadratic")
        col_metric2.metric(
            label="Weighted Kappa (Quadratic)", value=f"{weighted_kappa:.3f}"
        )
        kappa_interp = ""
        if weighted_kappa < 0:
            kappa_interp = "Poor agreement"
        elif weighted_kappa < 0.2:
            kappa_interp = "Slight agreement"
        elif weighted_kappa < 0.4:
            kappa_interp = "Fair agreement"
        elif weighted_kappa < 0.6:
            kappa_interp = "Moderate agreement"
        elif weighted_kappa < 0.8:
            kappa_interp = "Substantial agreement"
        else:
            kappa_interp = "Almost perfect agreement"
        st.caption(f"Interpretation: {kappa_interp}")
    except Exception as e:
        col_metric2.error(f"Kappa Error: {e}")

    st.subheader("Select Comparison Visualization")
    visualization_type = st.selectbox(
        "Choose a visualization:",
        options=["Heatmap", "Violin Plot", "Scatter Plot"],
        key=f"viz_select_{model_a_name}_{model_b_name}",
    )

    # Pass theme_colors to plotting functions
    if visualization_type == "Heatmap":
        fig = plot_correlation_heatmap_pair(
            merged_pair_df.copy(),
            "severity_rating_a",
            "severity_rating_b",
            model_a_name,
            model_b_name,
            theme_colors,
        )
        st.image(fig_to_buffer(fig), width=600)  # Use st.image
        plt.close(fig)
        download_figure(fig, f"{model_a_name}_vs_{model_b_name}_heatmap.png")
    elif visualization_type == "Violin Plot":
        fig = plot_violin_distribution_pair(
            merged_pair_df.copy(),
            "severity_rating_a",
            "severity_rating_b",
            model_a_name,
            model_b_name,
            theme_colors,
        )
        st.image(fig_to_buffer(fig), width=650)  # Use st.image
        plt.close(fig)
        download_figure(fig, f"{model_a_name}_vs_{model_b_name}_violin.png")
    elif visualization_type == "Scatter Plot":
        fig = plot_scatter_correlation_pair(
            merged_pair_df.copy(),
            "severity_rating_a",
            "severity_rating_b",
            model_a_name,
            model_b_name,
            theme_colors,
        )
        st.image(fig_to_buffer(fig), width=700)  # Use st.image
        plt.close(fig)
        download_figure(fig, f"{model_a_name}_vs_{model_b_name}_scatter_jitter.png")


def multi_model_agreement_tab(all_data):
    # (Keep this function as is, using krippendorff)
    st.markdown(
        "Calculates Krippendorff's Alpha (α) for agreement among 3+ models (Ordinal Scale)."
    )
    model_names = list(all_data.keys())
    num_models = len(model_names)
    if num_models < 3:
        st.warning("Requires at least three models.")
        return

    st.subheader("Select Models for Group Agreement Analysis")
    selected_models = st.multiselect(
        "Choose 3 or more models:",
        options=model_names,
        default=model_names[:3] if num_models >= 3 else [],
        key="multi_model_select",
    )
    if len(selected_models) < 3:
        st.info("Please select at least 3 models.")
        return
    st.markdown(f"**Calculating agreement for:** {', '.join(selected_models)}")

    st.subheader("Data Preparation & Calculation")
    rating_cols = [f"rating_{name}" for name in selected_models]
    relevant_dfs = []
    for name in selected_models:
        if name in all_data:
            df = all_data[name][["text", "severity_rating"]].copy()
            df["severity_rating"] = pd.to_numeric(
                df["severity_rating"], errors="coerce"
            )
            relevant_dfs.append(
                df.rename(columns={"severity_rating": f"rating_{name}"})
            )
        else:
            st.warning(f"Data for '{name}' not found.")
            return
    try:
        merged_selected_df = reduce(
            lambda left, right: pd.merge(left, right, on="text", how="inner"),
            relevant_dfs,
        )
    except Exception as e:
        st.error(f"Failed to merge data: {e}")
        return

    reliability_data_df = merged_selected_df[rating_cols].copy()
    for col in rating_cols:
        reliability_data_df[col] = pd.to_numeric(
            reliability_data_df[col], errors="coerce"
        )
    reliability_data_clean = reliability_data_df.dropna()
    if reliability_data_clean.empty:
        st.warning("No samples found where all selected models provided valid ratings.")
        return
    try:
        reliability_data_clean = reliability_data_clean.astype(float)
    except ValueError:
        st.error("Could not convert ratings to numeric.")
        return

    with st.expander("View Cleaned Ratings Data (Items x Raters) Used"):
        st.dataframe(reliability_data_clean)

    data_for_alpha = reliability_data_clean.values
    try:
        num_samples_used = len(data_for_alpha)
        st.write(f"Calculating Alpha using {num_samples_used} samples...")
        alpha_score = krippendorff.alpha(
            reliability_data=data_for_alpha.T,
            level_of_measurement="ordinal",
            value_domain=list(range(5)),
        )
        print(f"Krippendorff's Alpha: {alpha_score}")
        if pd.isna(alpha_score):
            st.warning("Alpha calculation resulted in NaN (likely no variation).")
            row_var_sum = reliability_data_clean.var(axis=1).sum()
            col_var_sum = reliability_data_clean.var(axis=0).sum()
            if row_var_sum == 0 and col_var_sum == 0:
                st.metric(label="Krippendorff's Alpha (Ordinal)", value="1.000*")
                st.caption("*NaN result interpreted as 1.0 (perfect agreement).")
            else:
                st.metric(label="Krippendorff's Alpha (Ordinal)", value="NaN")
        else:
            st.metric(
                label="Krippendorff's Alpha (Ordinal)", value=f"{alpha_score:.3f}"
            )
            alpha_interp = ""
            if alpha_score < 0:
                alpha_interp = "Poor agreement"
            elif alpha_score < 0.2:
                alpha_interp = "Slight agreement"
            elif alpha_score < 0.4:
                alpha_interp = "Fair agreement"
            elif alpha_score < 0.67:
                alpha_interp = "Moderate agreement"
            elif alpha_score < 0.8:
                alpha_interp = "Substantial agreement"
            else:
                alpha_interp = "High agreement"
            st.caption(f"Interpretation: {alpha_interp}")
        st.caption(f"Based on {num_samples_used} samples.")
    except Exception as e:
        st.error(f"Could not calculate Alpha: {e}")
        st.dataframe(reliability_data_clean.head())


def text_analysis_tab_modular(all_data):
    # (Keep this function as is - reverted to simple explanation display)
    st.header("Text Sample Analysis")
    model_names = list(all_data.keys())
    num_models = len(model_names)
    dfs_to_merge = []
    label_df = None
    for name, df in all_data.items():
        cols_to_keep = ["text", "severity_rating", "explanation"]
        if "label" in df.columns and label_df is None:
            if "text" in df.columns:
                cols_to_keep.append("label")
                label_df = df[["text", "label"]].copy().drop_duplicates(subset=["text"])
            else:
                st.warning(f"Model {name} has 'label' but missing 'text'.")
        present_cols = [col for col in cols_to_keep if col in df.columns]
        if "text" not in present_cols:
            st.warning(f"Model {name} missing 'text'. Skipping.")
            continue
        df_renamed = df[present_cols].rename(
            columns={
                "severity_rating": f"rating_{name}",
                "explanation": f"explanation_{name}",
            }
        )
        dfs_to_merge.append(df_renamed.drop(columns=["label"], errors="ignore"))
    if not dfs_to_merge:
        st.warning("No dataframes for merging.")
        return
    try:
        merged_all_df = reduce(
            lambda left, right: pd.merge(left, right, on="text", how="outer"),
            dfs_to_merge,
        )
        if label_df is not None and "text" in merged_all_df.columns:
            merged_all_df = pd.merge(merged_all_df, label_df, on="text", how="left")
            if "label" in merged_all_df.columns:
                merged_all_df["label"] = pd.to_numeric(
                    merged_all_df["label"], errors="coerce"
                )
    except Exception as e:
        st.error(f"Failed to merge: {e}")
        return

    st.subheader("Filter Samples")
    search_query = st.text_input(
        "Search Text:", key="text_search_query", placeholder="Enter keywords..."
    )
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    label_filter = None
    if "label" in merged_all_df.columns:
        with col_filter1:
            unique_labels = merged_all_df["label"].dropna().unique()
            label_options = [None] + sorted(
                [int(l) for l in unique_labels if l in [0, 1]]
            )
            label_filter = st.selectbox(
                "Label:",
                options=label_options,
                format_func=lambda x: (
                    "All" if x is None else ("Not Dep. (0)" if x == 0 else "Dep. (1)")
                ),
                key="label_filter_modular",
            )  # Shorter labels
    with col_filter2:
        filter_model_name = st.selectbox(
            "Model:", ["None"] + model_names, key="filter_model_select"
        )
    rating_filter_value = None
    if filter_model_name != "None":
        rating_col = f"rating_{filter_model_name}"
        if rating_col in merged_all_df.columns:
            valid_ratings = sorted(
                merged_all_df[rating_col].dropna().unique().astype(int)
            )
            if valid_ratings:
                with col_filter3:
                    rating_filter_value = st.multiselect(
                        f"{filter_model_name} Rating(s):",
                        options=valid_ratings,
                        key=f"rating_filter_{filter_model_name}",
                    )
            else:
                with col_filter3:
                    st.info(f"No ratings.")
        else:
            st.warning(f"Rating column missing.")

    filtered_df = merged_all_df.copy()
    if label_filter is not None and "label" in filtered_df.columns:
        filtered_df = filtered_df.dropna(subset=["label"])
        if not filtered_df.empty:
            filtered_df["label"] = filtered_df["label"].astype(int)
            filtered_df = filtered_df[filtered_df["label"] == label_filter]
    if filter_model_name != "None" and rating_filter_value:
        rating_col = f"rating_{filter_model_name}"
        if rating_col in filtered_df.columns:
            filtered_df[rating_col] = pd.to_numeric(
                filtered_df[rating_col], errors="coerce"
            )
            filtered_df = filtered_df.dropna(subset=[rating_col])
            if not filtered_df.empty:
                filtered_df = filtered_df[
                    filtered_df[rating_col].astype(int).isin(rating_filter_value)
                ]
    if search_query:
        if "text" in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df["text"]
                .astype(str)
                .str.contains(search_query, case=False, na=False)
            ]
        else:
            st.warning("Cannot search: 'text' column missing.")

    items_per_page_options = [5, 10, 15, 25]
    items_per_page = st.selectbox(
        "Items/Page:",
        options=items_per_page_options,
        index=1,
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
        if st.session_state.text_analysis_page_modular < 1:
            st.session_state.text_analysis_page_modular = 1
        current_page = st.session_state.text_analysis_page_modular
        start_idx = (current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        paginated_df = filtered_df.iloc[start_idx:end_idx]
        for index, row in paginated_df.iterrows():
            expander_label = f"Sample {index}: {str(row.get('text', 'N/A'))[:60]}..."  # Slightly more text
            with st.expander(expander_label):
                st.markdown(f"**Text:**")
                st.text_area(
                    "Text",
                    value=row.get("text", "N/A"),
                    height=100,
                    disabled=True,
                    label_visibility="collapsed",
                )  # Use text_area for scrollable text
                if "label" in row and pd.notna(row["label"]):
                    st.markdown(
                        f"**True Label:** {'Depressed (1)' if int(row['label']) == 1 else 'Not Depressed (0)'}"
                    )
                else:
                    st.markdown("**True Label:** N/A")
                st.markdown("---")
                st.markdown("##### Model Analyses:")
                remove_pattern = r"^\s*(\*{0,2})?Explanation(\*{0,2})?\s*:\s*\n?"
                models_per_row = min(num_models, 3)
                for row_idx in range(0, len(model_names), models_per_row):
                    row_models = model_names[row_idx : row_idx + models_per_row]
                    row_cols = st.columns(len(row_models))
                    for col_idx, name in enumerate(row_models):
                        with row_cols[col_idx]:
                            st.markdown(f"**{name}**")
                            rating_col = f"rating_{name}"
                            expl_col = f"explanation_{name}"
                            rating = row.get(rating_col, "N/A")
                            explanation_raw = str(row.get(expl_col, "N/A"))
                            explanation = (
                                re.sub(
                                    remove_pattern,
                                    "",
                                    explanation_raw,
                                    flags=re.IGNORECASE,
                                )
                                .strip()
                                .replace("**", "")
                            )
                            if pd.notna(rating) and str(rating).lower() != "n/a":
                                try:
                                    st.markdown(f"**Severity**: {int(float(rating))}")
                                except ValueError:
                                    st.markdown(f"**Severity**: {rating} (Invalid)")
                            else:
                                st.markdown("**Severity**: N/A")
                            # Use text_area for scrollable explanations
                            st.text_area(
                                f"Explanation_{name}_{index}",
                                value=(
                                    explanation
                                    if explanation and explanation.lower() != "n/a"
                                    else "N/A"
                                ),
                                height=150,
                                disabled=True,
                                label_visibility="collapsed",
                            )
                    if row_idx + models_per_row < len(model_names):
                        st.markdown("---")
        st.write("---")
        col_page1, col_page2, col_page3 = st.columns([1, 2, 1])
        with col_page1:
            prev_disabled = current_page <= 1
            if st.button(
                "⬅️ Previous",
                disabled=prev_disabled,
                key="prev_btn_modular",
                use_container_width=True,
            ):
                st.session_state.text_analysis_page_modular -= 1
                st.rerun()
        with col_page2:
            st.markdown(
                f"<div style='text-align: center; margin-top: 5px;'>Page {current_page} of {total_pages}</div>",
                unsafe_allow_html=True,
            )
        with col_page3:
            next_disabled = current_page >= total_pages
            if st.button(
                "Next ➡️",
                disabled=next_disabled,
                key="next_btn_modular",
                use_container_width=True,
            ):
                st.session_state.text_analysis_page_modular += 1
                st.rerun()
    else:
        st.write("No samples match the current filters.")


# --- NEW Function for Evaluations Tab ---
def evaluations_tab(all_data, theme_colors):  # Pass theme_colors
    st.header("Model Evaluation Metrics")
    model_names = list(all_data.keys())
    sub_tab_titles = ["Individual Performance", "Pairwise Comparison"]
    if len(model_names) >= 3:
        sub_tab_titles.append("Multi-Model Agreement")
    sub_tabs = st.tabs(sub_tab_titles)
    sub_tab_map = {title: tab for title, tab in zip(sub_tab_titles, sub_tabs)}

    with sub_tab_map["Individual Performance"]:
        st.subheader("Single Model Performance Metrics")
        model_performance_tab(all_data, theme_colors)  # Pass theme_colors

    if "Pairwise Comparison" in sub_tab_map:
        with sub_tab_map["Pairwise Comparison"]:
            st.subheader("Two-Model Comparison & Agreement")
            if len(model_names) >= 2:
                pairwise_comparison_tab(all_data, theme_colors)  # Pass theme_colors
            else:
                st.info("Requires at least two models.")

    if "Multi-Model Agreement" in sub_tab_map:
        with sub_tab_map["Multi-Model Agreement"]:
            st.subheader("Group Agreement (3+ Models)")
            if len(model_names) >= 3:
                multi_model_agreement_tab(
                    all_data
                )  # This tab doesn't have plots currently
            else:
                st.info("Requires at least three models.")


# --- Main Function (Modified for Theme Toggle) ---
def main():
    set_custom_theme()
    st.title("Depression Severity Detection Comparison")

    # --- Theme Toggle ---
    # Place it prominently, maybe in sidebar or near top
    # Using sidebar here for less clutter in main area
    with st.sidebar:
        st.header("Settings")
        use_dark_theme = st.checkbox(
            "Use Dark Theme for Plots", key="dark_theme_plots", value=False
        )  # Default to light

    # Determine theme colors based on toggle state
    theme_colors = DARK_THEME if use_dark_theme else LIGHT_THEME

    st.markdown("### Comparing LLM Performance for Depression Assessment")

    try:
        all_data = load_all_data()
        if not all_data:
            st.error("Error: No valid model data loaded.")
            return
        model_names = list(all_data.keys())
        print(f"Loaded models: {', '.join(model_names)}")

        tab_titles = ["Overall Overview", "Text Sample Analysis", "Evaluations"]
        tabs = st.tabs(tab_titles)
        tab_map = {title: tab for title, tab in zip(tab_titles, tabs)}

        with tab_map["Overall Overview"]:
            overall_overview_tab(all_data, theme_colors)  # Pass theme

        with tab_map["Text Sample Analysis"]:
            text_analysis_tab_modular(
                all_data
            )  # Text analysis doesn't use plots currently

        with tab_map["Evaluations"]:
            evaluations_tab(all_data, theme_colors)  # Pass theme

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        import traceback

        st.error("Traceback:")
        st.code(traceback.format_exc())
        st.warning("Check data files and formats.")


# Run the app
if __name__ == "__main__":
    main()
