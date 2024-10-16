# pages/data_exploration.py
import os
import pandas as pd
import streamlit as st
import plotly.express as px
from utils.helpers import plot_sensor_readings, plot_confusion_matrix

def display(config, train_df, test_df, motors_df, engine_motor_mapping, sensors_df):
    """Display the Data Exploration page."""
    st.header("Data Exploration")
    st.subheader("Select an Engine ID to Explore")
    engine_ids = train_df['id'].unique()
    selected_id = st.selectbox("Engine ID", engine_ids)

    # Filter data for the selected engine ID
    engine_data = train_df[train_df['id'] == selected_id]

    # Get the motor assigned to this engine
    motor_id = engine_motor_mapping[selected_id]
    motor_spec = motors_df[motors_df['Manufacturer Part Number'] == motor_id].iloc[0]

    st.write(f"Showing data for Engine ID: {selected_id}")

    # Display motor specifications and image together
    st.write("### Motor Information:")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("#### Specifications")
        st.dataframe(motor_spec.to_frame())

    with col2:
        st.write("#### Motor Image")
        motor_name = motor_spec['Brand']
        image_extensions = ['.png']
        image_found = False
        for ext in image_extensions:
            image_path = f'Dataset/images/{motor_name}{ext}'
            if os.path.exists(image_path):
                st.image(image_path, caption=motor_name, use_column_width=True)
                image_found = True
                break
        if not image_found:
            st.write(f"No image found for motor '{motor_name}'.")

    st.write("### Engine Data Preview:")
    st.dataframe(engine_data.head())

    # Time Series Plot of Sensors
    st.subheader("Time Series Plot of Sensors for Selected Engine")
    sensor_options = [col for col in train_df.columns if col in sensors_df['Sensor Name'].values]
    sensor_display_options = [f"{sensor} - {sensors_df.loc[sensors_df['Sensor Name'] == sensor, 'Description'].values[0]}" for sensor in sensor_options]

    selected_sensors_display = st.multiselect(
        "Select Sensors to Plot",
        options=sensor_display_options,
        default=sensor_display_options[:3]
    )

    selected_sensors = [option.split(' - ')[0] for option in selected_sensors_display]

    if selected_sensors:
        fig = plot_sensor_readings(engine_data, engine_data['cycle'], selected_sensors, dict(zip(sensors_df['Sensor Name'], sensors_df['Description'])))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select at least one sensor to plot.")

    # Distribution of Sensor Readings
    st.subheader("Distribution of Sensor Readings")
    selected_sensor_hist_option = st.selectbox("Select Sensor for Histogram", sensor_display_options)
    selected_sensor_hist = selected_sensor_hist_option.split(' - ')[0]

    fig_hist = px.histogram(train_df, x=selected_sensor_hist, nbins=50,
                            title=f"Distribution of {sensors_df.loc[sensors_df['Sensor Name'] == selected_sensor_hist, 'Description'].values[0]}")
    fig_hist.update_layout(
        xaxis_title=f"{sensors_df.loc[sensors_df['Sensor Name'] == selected_sensor_hist, 'Description'].values[0]} ({selected_sensor_hist})",
        yaxis_title='Frequency'
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Feature Selection Based on Correlation with Output
    st.header("Feature Selection Based on Output Correlation")

    # Output Type Selection
    output_options = config.OUTPUT_TYPES_OPTIONS
    selected_output = st.selectbox("Select Output Variable", output_options)

    # Calculate correlations with the selected output
    if config.OUTPUT_TYPE == "regression":
        # Use Pearson correlation for continuous output
        correlations = train_df[sensor_options + [selected_output]].corr()[selected_output].drop(selected_output)
    else:
        # Use point-biserial correlation for categorical outputs
        correlations = {}
        for feature in sensor_options:
            corr = train_df[feature].corr(train_df[selected_output], method='pearson')
            correlations[feature] = abs(corr)
        correlations = pd.Series(correlations)

    # Display correlations in a bar plot
    fig_corr_plot = px.bar(correlations, title=f"Correlation of Features with {selected_output}",
                           labels={'index': 'Feature', 'value': 'Correlation'})
    st.plotly_chart(fig_corr_plot)

    # Set a correlation threshold
    corr_threshold = st.slider("Correlation Threshold", min_value=0.0, max_value=1.0, value=0.3)
    selected_features_corr = correlations[correlations >= corr_threshold].index.tolist()
    st.write(f"Features selected based on correlation with {selected_output}: {selected_features_corr}")

    if selected_features_corr:
        # Filter data for selected features and add target for correlation matrix
        df_filtered_corr = train_df[selected_features_corr + [selected_output]]

        # Display correlation matrix including target label
        st.subheader(f"Correlation Matrix for Selected Features and {selected_output}")
        corr_matrix = df_filtered_corr.corr()

        fig_corr_matrix = plot_confusion_matrix(corr_matrix)
        st.pyplot(fig_corr_matrix)
    else:
        st.warning("No features meet the correlation threshold.")
