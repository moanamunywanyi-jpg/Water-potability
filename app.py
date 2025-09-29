import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
import os
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Water Potability Predictor",
    page_icon="💧",
    layout="wide"
)

# Title and description
st.title("💧 Water Potability Prediction")
st.markdown("""
This app predicts whether water is safe to drink based on its chemical properties.
Enter the water quality parameters below to check potability.
""")

# Load or train model
@st.cache_resource
def load_model():
    try:
        # Check if model file exists
        if os.path.exists('water_potability_model.joblib'):
            model = joblib.load('water_potability_model.joblib')
            st.sidebar.success("✅ Model loaded successfully!")
            return model
        else:
            st.sidebar.info("📝 No pre-trained model found. Training new model...")
            return None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        return None

def train_new_model():
    try:
        # Generate synthetic data based on typical water quality ranges
        np.random.seed(42)
        n_samples = 2000
        
        data = {
            'ph': np.random.uniform(6.0, 8.5, n_samples),
            'Hardness': np.random.uniform(100, 300, n_samples),
            'Solids': np.random.uniform(200, 40000, n_samples),
            'Chloramines': np.random.uniform(0.5, 10, n_samples),
            'Sulfate': np.random.uniform(100, 400, n_samples),
            'Conductivity': np.random.uniform(200, 800, n_samples),
            'Organic_carbon': np.random.uniform(2, 20, n_samples),
            'Trihalomethanes': np.random.uniform(0, 120, n_samples),
            'Turbidity': np.random.uniform(0.5, 6.5, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create synthetic target based on reasonable thresholds
        conditions = (
            (df['ph'].between(6.5, 8.5)) &
            (df['Hardness'] <= 500) &
            (df['Solids'] <= 50000) &  # Fixed: was 500, should be higher
            (df['Chloramines'] <= 4) &
            (df['Sulfate'] <= 250) &
            (df['Conductivity'] <= 800) &
            (df['Organic_carbon'] <= 10) &
            (df['Trihalomethanes'] <= 80) &
            (df['Turbidity'] <= 5)
        )
        
        df['Potability'] = conditions.astype(int)
        
        # Split data
        X = df.drop('Potability', axis=1)
        y = df['Potability']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(model, 'water_potability_model.joblib')
        
        # Calculate accuracy
        accuracy = model.score(X_test, y_test)
        st.sidebar.success(f"✅ Model trained successfully! Accuracy: {accuracy:.2%}")
        
        return model, X_test, y_test
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None

# Sidebar for input parameters
st.sidebar.header("🔬 Water Quality Parameters")

def user_input_features():
    ph = st.sidebar.slider('pH', 0.0, 14.0, 7.0, 0.1)
    hardness = st.sidebar.slider('Hardness (mg/L)', 0, 500, 150)
    solids = st.sidebar.slider('Solids (ppm)', 0, 50000, 10000)
    chloramines = st.sidebar.slider('Chloramines (ppm)', 0.0, 15.0, 4.0, 0.1)
    sulfate = st.sidebar.slider('Sulfate (mg/L)', 0, 500, 200)
    conductivity = st.sidebar.slider('Conductivity (μS/cm)', 0, 1000, 400)
    organic_carbon = st.sidebar.slider('Organic Carbon (ppm)', 0.0, 30.0, 10.0, 0.1)
    trihalomethanes = st.sidebar.slider('Trihalomethanes (μg/L)', 0.0, 150.0, 40.0, 0.1)
    turbidity = st.sidebar.slider('Turbidity (NTU)', 0.0, 10.0, 3.0, 0.1)
    
    data = {
        'ph': ph,
        'Hardness': hardness,
        'Solids': solids,
        'Chloramines': chloramines,
        'Sulfate': sulfate,
        'Conductivity': conductivity,
        'Organic_carbon': organic_carbon,
        'Trihalomethanes': trihalomethanes,
        'Turbidity': turbidity
    }
    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# Main panel
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Input Parameters")
    st.dataframe(input_df.T.rename(columns={0: 'Value'}))

# Load or train model
model = load_model()
if model is None:
    with st.spinner('Training model for the first time... This may take a few seconds.'):
        model, X_test, y_test = train_new_model()

# Debug information (you can remove this after testing)
with st.sidebar:
    st.subheader("🔧 Debug Info")
    if model is not None:
        st.write(f"Model type: {type(model)}")
        st.write(f"Has predict method: {hasattr(model, 'predict')}")
    else:
        st.error("Model is None!")

# Prediction
if st.button('🔍 Predict Potability'):
    if model is None:
        st.error("❌ Model is not available. Please check the training process.")
    else:
        try:
            # Verify model has predict method
            if not hasattr(model, 'predict'):
                st.error("❌ Loaded object is not a valid model (missing predict method)")
                st.write(f"Object type: {type(model)}")
            else:
                prediction = model.predict(input_df)
                prediction_proba = model.predict_proba(input_df)
                
                with col2:
                    st.subheader("🎯 Prediction Result")
                    
                    if prediction[0] == 1:
                        st.success("✅ **POTABLE** - Water is safe to drink!")
                        st.balloons()
                    else:
                        st.error("❌ **NOT POTABLE** - Water is not safe for drinking!")
                    
                    st.metric("Confidence", f"{max(prediction_proba[0]):.2%}")
                    
                    # Show probability breakdown
                    prob_df = pd.DataFrame({
                        'Class': ['Not Potable', 'Potable'],
                        'Probability': prediction_proba[0]
                    })
                    st.bar_chart(prob_df.set_index('Class')['Probability'])
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            # Add more detailed error information
            st.write("**Debug Information:**")
            st.write(f"- Model type: {type(model)}")
            st.write(f"- Input shape: {input_df.shape}")
            st.write(f"- Input columns: {list(input_df.columns)}")

# Feature importance and guidelines
st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    st.subheader("📈 Feature Importance")
    if model is not None and hasattr(model, 'feature_importances_'):
        try:
            feature_importance = pd.DataFrame({
                'feature': input_df.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            st.bar_chart(feature_importance.set_index('feature')['importance'])
        except:
            st.info("Feature importance not available")
    else:
        st.info("Feature importance will be available after model training")

with col4:
    st.subheader("💡 Water Quality Guidelines")
    st.markdown("""
    **Typical Safe Ranges:**
    - **pH**: 6.5 - 8.5
    - **Hardness**: < 500 mg/L
    - **Chloramines**: < 4 ppm
    - **Sulfate**: < 250 mg/L
    - **Trihalomethanes**: < 80 μg/L
    - **Turbidity**: < 5 NTU
    
    *Note: These are general guidelines. Local regulations may vary.*
    """)

# Data upload section (optional)
st.markdown("---")
st.subheader("📁 Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=['csv'])

if uploaded_file is not None and model is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        required_columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                          'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
        
        if all(col in batch_data.columns for col in required_columns):
            batch_predictions = model.predict(batch_data[required_columns])
            batch_probabilities = model.predict_proba(batch_data[required_columns])
            
            results_df = batch_data.copy()
            results_df['Potability_Prediction'] = batch_predictions
            results_df['Potability_Probability'] = np.max(batch_probabilities, axis=1)
            results_df['Status'] = np.where(batch_predictions == 1, 'Potable', 'Not Potable')
            
            st.success(f"✅ Processed {len(results_df)} samples")
            st.dataframe(results_df)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name="water_potability_predictions.csv",
                mime="text/csv"
            )
        else:
            st.error(f"CSV file must contain these columns: {', '.join(required_columns)}")
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <style>
    .footer {
        text-align: center;
        padding: 10px;
        color: #666;
    }
    </style>
    <div class="footer">
        Made with ❤️ using Streamlit | Water Quality Prediction App
    </div>
    """,
    unsafe_allow_html=True
)