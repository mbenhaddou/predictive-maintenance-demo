# pages/introduction.py

import streamlit as st

def display():
    """Display the Introduction page."""
    st.header("Introduction")
    st.write("""
        Welcome to the **Predictive Maintenance using LSTM** application.
        This tool leverages Long Short-Term Memory (LSTM) neural networks to predict potential engine failures based on historical sensor data.
    """)
    st.write("""
        **Key Features:**
        - **Data Exploration:** Visualize and understand the dataset.
        - **Model Configuration:** Customize LSTM model parameters.
        - **Model Training:** Train the LSTM model with real-time feedback.
        - **Model Evaluation:** Assess model performance using various metrics.
        - **Prediction Simulation:** Simulate streaming data and observe real-time predictions.
    """)
    st.sidebar.info("Navigate through the sidebar to explore different functionalities.")
    # Optionally, add an image or diagram
    # st.image("path_to_image.jpg", use_column_width=True)
