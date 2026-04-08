import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import shap
import numpy as np
import matplotlib.pyplot as plt

# পেজ কনফিগারেশন (ওয়াইড লেআউট)
st.set_page_config(page_title="Sleep Disorder Analysis", page_icon="🌙", layout="wide")

# কাস্টম CSS - UI সুন্দর করার জন্য
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# ১. মডেল এবং ডেটা লোড করা
@st.cache_resource
def load_model_and_explainer():
    try:
        data = joblib.load('best_sleep_model.pkl')
        model = data['model']
        scaler = data['scaler']
        encoders = data['label_encoders']
        features = data['feature_names']
        
        if 'train_sample' in data:
            background_data = data['train_sample']
        else:
            background_data = shap.sample(scaler.transform(pd.DataFrame(np.zeros((10, len(features))), columns=features)), 5)
        
        explainer = shap.KernelExplainer(model.predict, background_data)
        return model, scaler, encoders, features, explainer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None

model, scaler, encoders, features, explainer = load_model_and_explainer()

# --- হেডলাইন ---
st.title("🌙 Advanced Sleep Disorder Dashboard")
st.markdown("Provide your details below to analyze sleep health using Machine Learning.")
st.markdown("---")

# --- সেকশন ১: ইউজার ইনপুট (উপরে থাকবে ৩টি কলামে) ---
st.markdown("### ⚙️ 1. Enter Your Details")

col_in1, col_in2, col_in3 = st.columns(3)

with col_in1:
    st.markdown("#### 👤 Personal & Academic")
    age = st.number_input("Age", min_value=10, max_value=100, value=22, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    dept = st.selectbox("Department", ["CSE", "EEE","ECE","Civil Engineering","Mechanical Engineering","DVM", "BBA", "Pharmacy", "Agriculture","Sociology", "Mathematics","English","Development Studies","Physics","Statistics","FPE","Economics","Botany","zoology","Chemistry","Finance and Banking"])
    level = st.selectbox("Academic Level", ["Level-1", "Level-2", "Level-3", "Level-4"])
    uni = st.selectbox("University",["HSTU", "DU", "JnU", "RU", "CU", "KU" ,"MBSTU"])

with col_in2:
    st.markdown("#### 🛌 Lifestyle & Stress")
    s_dur = st.slider("Sleep Duration (Hours)", 1.0, 12.0, 7.0, 0.5)
    qual = st.select_slider("Quality of Sleep", options=["Poor", "Fair", "Good", "Excellent"], value="Good")
    act = st.slider("Physical Activity Level (0-100)", 0, 100, 50)
    stress = st.select_slider("Stress Level", options=["Very Low", "Low", "Moderate", "High", "Very High"], value="Moderate")

with col_in3:
    st.markdown("#### 🩺 Health Metrics")
    bmi = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese", "Underweight"])
    hr = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=72)
    steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=5000, step=500)
    
    col_bp1, col_bp2 = st.columns(2)
    with col_bp1:
        sys = st.number_input("Systolic BP", value=120)
    with col_bp2:
        dia = st.number_input("Diastolic BP", value=80)

# বাটনটি একটু ফাঁকা দিয়ে মাঝখানে রাখা
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_button = st.button("🚀 Analyze Now", use_container_width=True)

st.markdown("---")

# --- সেকশন ২: আউটপুট এবং অ্যানালাইসিস (নিচে থাকবে) ---
if predict_button and model is not None:
    st.markdown("### 🎯 2. Analysis Results")
    
    # ইনপুট প্রসেসিং
    input_dict = {
        'Department': dept, 'Gender': gender, 'Age': age, 'Sleep Duration': s_dur,
        'Quality of Sleep': qual, 'Physical Activity Level': act, 'Stress Level': stress,
        'BMI Category': bmi, 'Heart Rate (bpm)': hr, 'Daily Steps': steps,
        'Academic Level': level, 'University': uni, 'Systolic': sys, 'Diastolic': dia
    }
    df_in = pd.DataFrame([input_dict])[features]

    for col in df_in.columns:
        if col in encoders:
            try:
                df_in[col] = encoders[col].transform(df_in[col].astype(str))
            except:
                df_in[col] = 0
                
    scaled = scaler.transform(df_in)
    prediction = model.predict(scaled)[0]
    result_text = encoders['Sleep Disorder'].inverse_transform([prediction])[0]
    probs = model.predict_proba(scaled)[0]
    class_names = encoders['Sleep Disorder'].classes_

    # ১. প্রেডিকশন রেজাল্ট (বড় বক্স)
    is_healthy = (result_text == "No Sleep Disorder" or result_text == "None")
    bg_color = "#e8f5e9" if is_healthy else "#ffebee"
    border_color = "#4CAF50" if is_healthy else "#f44336"
    text_color = "#2E7D32" if is_healthy else "#c62828"
    icon = "✅" if is_healthy else "⚠️"
    
    st.markdown(f"""
        <div class='prediction-box' style='background-color: {bg_color}; border: 2px solid {border_color};'>
            <h1 style='color: {text_color}; margin: 0; font-size: 40px;'>{icon} {result_text}</h1>
            <p style='color: #555; font-size: 18px; margin-top: 10px;'>This prediction is based on your provided lifestyle and health metrics.</p>
        </div>
    """, unsafe_allow_html=True)

    # ২. গ্রাফিক্স (Confidence & Radar পাশাপাশি)
    col_out1, col_out2 = st.columns(2)
    
    with col_out1:
        st.markdown("#### 📊 Confidence / Probability")
        fig_bar = go.Figure(go.Bar(
            x=probs, y=class_names, orientation='h',
            marker=dict(color=['#2ecc71', '#e74c3c', '#f1c40f'][0:len(class_names)])
        ))
        fig_bar.update_layout(xaxis_title="Probability Chance", yaxis_title="",
                              xaxis=dict(range=[0, 1]), height=350, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_out2:
        st.markdown("#### 🕸️ Lifestyle Radar Profile")
        categories = ['Sleep Duration', 'Quality', 'Activity', 'Stress', 'Heart Rate']
        radar_values = [
            s_dur / 12, 
            (4 if qual=="Excellent" else 3 if qual=="Good" else 2 if qual=="Fair" else 1) / 4,
            act / 100,
            (1 if stress=="Very Low" else 2 if stress=="Low" else 3 if stress=="Moderate" else 4 if stress=="High" else 5) / 5,
            (100 - (hr-40)) / 100 if hr > 40 else 1 
        ]
        fig_radar = go.Figure(go.Scatterpolar(r=radar_values, theta=categories, fill='toself', 
                                              line=dict(color='#3498db'), fillcolor="rgba(52, 152, 219, 0.4)"))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])), 
                                showlegend=False, height=350, margin=dict(l=40, r=40, t=20, b=0))
        st.plotly_chart(fig_radar, use_container_width=True)

    # ৩. SHAP Force Plot (নিচে সম্পূর্ণ প্রশস্ততায়)
    st.markdown("---")
    st.markdown("### 🧬 3. AI Decision Interpretation (SHAP Force Plot)")
    st.markdown("*(Red elements push the model towards higher risk, blue elements push towards lower risk)*")
    
    with st.spinner("Generating interpretation plot..."):
        shap_values = explainer.shap_values(scaled)
        
        if isinstance(shap_values, list):
            current_shap = shap_values[prediction]
            expected_val = explainer.expected_value[prediction]
        else:
            current_shap = shap_values[0]
            expected_val = explainer.expected_value

        # Matplotlib-এর আগের ক্যাশে ক্লিয়ার করা
        plt.clf()
        
        # সরাসরি SHAP প্লট কল করা (এটি নিজে থেকে ফিগার তৈরি করবে)
        shap.force_plot(
            expected_val, 
            current_shap, 
            df_in.iloc[0], 
            matplotlib=True, 
            show=False,
            text_rotation=0
        )
        
        # SHAP যে ফিগারটি তৈরি করেছে সেটি ক্যাপচার করা
        fig = plt.gcf()
        fig.set_size_inches(16, 4) # ছবিটির অনুপাত ঠিক রাখার জন্য
        
        # ফিগারটি স্ট্রিমলিটে দেখানো
        st.pyplot(fig, use_container_width=True)
        
        # মেমরি লিক এড়াতে ফিগারটি ক্লিয়ার করা
        plt.clf()

elif not predict_button:
    st.info("👆 Please fill the inputs above and click **Analyze Now** to see the prediction and charts.")