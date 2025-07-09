import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Fault Detection System", layout="wide")
st.title("⚡ FAULT DETECTION IN POWER SYSTEM USING AI ")

# Load your trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("Fault Detection/fault_detection_model.pkl")
        return model, True
    except Exception as e:
        return None, False

model, model_loaded = load_model()

if not model_loaded:
    st.error("⚠️ Model could not be loaded. Make sure 'fault_detection_model.pkl' exists in the app folder.")

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 Data Upload & Predictions", "📈 Waveform Analysis", "📋 Summary Report"])

with tab1:
    st.header("Data Upload and AI Predictions")
    
    uploaded_file = st.file_uploader("Upload Feature CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ CSV uploaded successfully!")
            st.subheader("📊 Uploaded Data Preview")
            st.dataframe(df.head())
            
            # Define the expected feature columns
            feature_cols = ["Va", "Vb", "Vc", "Ia", "Ib", "Ic"]
            
            # Check if all feature columns exist in uploaded CSV
            if all(col in df.columns for col in feature_cols):
                features = df[feature_cols]
                
                if model_loaded:
                    predictions = model.predict(features)
                    df["Predicted Fault Type"] = predictions
                    
                    st.subheader("🧠 AI Predictions")
                    st.dataframe(df[feature_cols + ["Predicted Fault Type"]])
                    
                    # Store data in session state for other tabs
                    st.session_state.df = df
                    st.session_state.feature_cols = feature_cols
                    st.session_state.predictions = predictions
                else:
                    st.warning("Model not available. Prediction skipped.")
            else:
                missing = list(set(feature_cols) - set(df.columns))
                st.error(f"❌ Missing required feature columns in CSV: {missing}")
        except Exception as e:
            st.error(f"❌ Error processing the uploaded file: {e}")

with tab2:
    st.header("Waveform Analysis")
    
    if 'df' in st.session_state:
        df = st.session_state.df
        feature_cols = st.session_state.feature_cols
        
        # Waveform display options
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Time range selection
            max_samples = len(df)
            start_sample = st.slider("Start Sample", 0, max_samples-100, 0)
            end_sample = st.slider("End Sample", start_sample+10, max_samples, min(start_sample+200, max_samples))
            
        with col2:
            # Waveform type selection
            waveform_type = st.selectbox(
                "Select Waveform Type",
                ["All", "Voltages Only", "Currents Only", "Individual Signals"]
            )
            
            if waveform_type == "Individual Signals":
                selected_signal = st.selectbox("Select Signal", feature_cols)
        
        # Create time axis (assuming sampling frequency)
        sampling_freq = st.number_input("Sampling Frequency (Hz)", value=1000, min_value=1)
        time_axis = np.arange(start_sample, end_sample) / sampling_freq
        
        # Plot waveforms based on selection
        if waveform_type == "All":
            # Create subplots for voltages and currents
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Voltage Waveforms', 'Current Waveforms'),
                vertical_spacing=0.1
            )
            
            # Voltage waveforms
            colors_v = ['red', 'green', 'blue']
            for i, col in enumerate(['Va', 'Vb', 'Vc']):
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=df[col].iloc[start_sample:end_sample],
                        mode='lines',
                        name=f'{col}',
                        line=dict(color=colors_v[i])
                    ),
                    row=1, col=1
                )
            
            # Current waveforms
            colors_i = ['orange', 'purple', 'brown']
            for i, col in enumerate(['Ia', 'Ib', 'Ic']):
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=df[col].iloc[start_sample:end_sample],
                        mode='lines',
                        name=f'{col}',
                        line=dict(color=colors_i[i])
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(height=600, title_text="Power System Waveforms")
            fig.update_xaxes(title_text="Time (seconds)")
            fig.update_yaxes(title_text="Amplitude", row=1, col=1)
            fig.update_yaxes(title_text="Amplitude", row=2, col=1)
            
        elif waveform_type == "Voltages Only":
            fig = go.Figure()
            colors = ['red', 'green', 'blue']
            for i, col in enumerate(['Va', 'Vb', 'Vc']):
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=df[col].iloc[start_sample:end_sample],
                        mode='lines',
                        name=f'{col}',
                        line=dict(color=colors[i])
                    )
                )
            fig.update_layout(title="Voltage Waveforms", xaxis_title="Time (seconds)", yaxis_title="Voltage")
            
        elif waveform_type == "Currents Only":
            fig = go.Figure()
            colors = ['orange', 'purple', 'brown']
            for i, col in enumerate(['Ia', 'Ib', 'Ic']):
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=df[col].iloc[start_sample:end_sample],
                        mode='lines',
                        name=f'{col}',
                        line=dict(color=colors[i])
                    )
                )
            fig.update_layout(title="Current Waveforms", xaxis_title="Time (seconds)", yaxis_title="Current")
            
        else:  # Individual Signals
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=df[selected_signal].iloc[start_sample:end_sample],
                    mode='lines',
                    name=selected_signal,
                    line=dict(width=2)
                )
            )
            fig.update_layout(
                title=f"{selected_signal} Waveform",
                xaxis_title="Time (seconds)",
                yaxis_title="Amplitude"
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Fault detection overlay
        st.subheader("🔍 Fault Detection Overlay")
        
        if 'predictions' in st.session_state:
            predictions = st.session_state.predictions[start_sample:end_sample]
            
            # Show fault detection results on timeline
            fault_fig = go.Figure()
            
            # Create a binary signal for fault detection
            fault_signal = [1 if pred != 'Normal' else 0 for pred in predictions]
            
            fault_fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=fault_signal,
                    mode='lines+markers',
                    name='Fault Detection',
                    line=dict(color='red', width=3),
                    fill='tozeroy'
                )
            )
            
            fault_fig.update_layout(
                title="Fault Detection Timeline",
                xaxis_title="Time (seconds)",
                yaxis_title="Fault Status (0=Normal, 1=Fault)",
                yaxis=dict(tickvals=[0, 1], ticktext=['Normal', 'Fault'])
            )
            
            st.plotly_chart(fault_fig, use_container_width=True)
            
            # Show detailed fault information
            fault_counts = pd.Series(predictions).value_counts()
            st.subheader("📊 Fault Distribution in Selected Range")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write("**Fault Type Counts:**")
                for fault_type, count in fault_counts.items():
                    percentage = (count / len(predictions)) * 100
                    st.write(f"- {fault_type}: {count} ({percentage:.1f}%)")
            
            with col2:
                # Pie chart of fault distribution
                pie_fig = go.Figure(data=[go.Pie(
                    labels=fault_counts.index,
                    values=fault_counts.values,
                    hole=0.3
                )])
                pie_fig.update_layout(title="Fault Distribution")
                st.plotly_chart(pie_fig, use_container_width=True)
    
    else:
        st.info("📤 Please upload a CSV file in the 'Data Upload & Predictions' tab first.")

with tab3:
    st.header("Summary Report")
    
    if 'df' in st.session_state:
        df = st.session_state.df
        predictions = st.session_state.predictions
        
        # Overall statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(df))
        
        with col2:
            normal_count = sum(1 for pred in predictions if pred == 'Normal')
            st.metric("Normal Samples", normal_count)
        
        with col3:
            fault_count = len(predictions) - normal_count
            st.metric("Fault Samples", fault_count)
        
        with col4:
            fault_percentage = (fault_count / len(predictions)) * 100
            st.metric("Fault Percentage", f"{fault_percentage:.1f}%")
        
        # Detailed analysis
        st.subheader("📈 Statistical Analysis")
        
        # Signal statistics
        stats_df = df[st.session_state.feature_cols].describe()
        st.write("**Signal Statistics:**")
        st.dataframe(stats_df)
        
        # Correlation matrix
        st.subheader("🔗 Signal Correlation Matrix")
        corr_matrix = df[st.session_state.feature_cols].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1
        ))
        fig_corr.update_layout(title="Signal Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Export options
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Download Results as CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="fault_detection_results.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📈 Generate Report"):
                report = f"""
                # Fault Detection Report
                
                ## Summary
                - Total Samples: {len(df)}
                - Normal Samples: {normal_count}
                - Fault Samples: {fault_count}
                - Fault Percentage: {fault_percentage:.1f}%
                
                ## Fault Type Distribution
                {pd.Series(predictions).value_counts().to_string()}
                
                ## Signal Statistics
                {stats_df.to_string()}
                """
                
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name="fault_detection_report.md",
                    mime="text/markdown"
                )
    
    else:
        st.info("📤 Please upload a CSV file in the 'Data Upload & Predictions' tab first.")

# Sidebar information
st.sidebar.title("ℹ️ System Information")
st.sidebar.info("""
**Features:**
- AI-powered fault detection
- Interactive waveform visualization
- Real-time analysis
- Statistical reporting
- Export capabilities

**Supported Signals:**
- Va, Vb, Vc (Voltages)
- Ia, Ib, Ic (Currents)

**Model Status:** """ + ("✅ Loaded" if model_loaded else "❌ Not Available"))

st.sidebar.markdown("---")
st.sidebar.markdown("**How to Use:**")
st.sidebar.markdown("""
1. Upload your CSV file with power system data
2. View AI predictions in the first tab
3. Analyze waveforms in the second tab
4. Generate reports in the third tab
""")
