"""Reusable Streamlit-compatible visualizations."""
import matplotlib.pyplot as plt

def class_distribution_figure(predictions):
    counts=predictions["predicted_class"].value_counts().reindex(["ham","spam"],fill_value=0)
    figure,axis=plt.subplots(figsize=(6.5,4))
    axis.bar(counts.index,counts.values)
    axis.set_title("Predicted SMS Class Distribution")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Messages")
    for index,value in enumerate(counts.values):
        axis.text(index,value,str(int(value)),ha="center",va="bottom")
    figure.tight_layout()
    return figure
