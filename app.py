import gradio as gr
import pandas as pd
import joblib
import plotly.graph_objects as go
import shap
import numpy as np
import matplotlib.pyplot as plt

# ১. মডেল এবং ডেটা লোড করার ফাংশন
def load_resources():
    try:
        data = joblib.load('best_sleep_model.pkl')
        return data['model'], data['scaler'], data['label_encoders'], data['feature_names'], data.get('train_sample')
    except Exception as e:
        print(f"Resource Loading Error: {e}")
        return None, None, None, None, None

model, scaler, encoders, features, train_sample = load_resources()

# ২. মেইন অ্যানালাইসিস ফাংশন (Arguments order must match input_list exactly)
def predict_and_analyze(dept, gender, age, s_dur, qual, act, stress, bmi, hr, steps, level, uni, sys, dia):
    try:
        if model is None:
            return "### ❌ Model or Scaler not found! Check your .pkl file.", None, None

        # ইনপুট ডিকশনারি (মডেলের ট্রেনিংয়ের সময়কার ফিচার নেম অনুযায়ী)
        # নিশ্চিত হোন যে এই কী-গুলো আপনার মডেলে যা ছিল ঠিক তাই আছে
        input_dict = {
            'Department': dept, 'Gender': gender, 'Age': age, 'Sleep Duration': s_dur,
            'Quality of Sleep': qual, 'Physical Activity Level': act, 'Stress Level': stress,
            'BMI Category': bmi, 'Heart Rate (bpm)': hr, 'Daily Steps': steps,
            'Academic Level': level, 'University': uni, 'Systolic': sys, 'Diastolic': dia
        }
        
        # DataFrame তৈরি এবং ফিচারের ক্রম ঠিক করা
        df = pd.DataFrame([input_dict])
        df = df[features] 
        
        # এনকোডিং প্রসেস
        df_encoded = df.copy()
        for col in df_encoded.columns:
            if col in encoders:
                try:
                    df_encoded[col] = encoders[col].transform(df_encoded[col].astype(str))
                except:
                    df_encoded[col] = 0 
                
        # স্কেলিং এবং প্রেডিকশন
        scaled_data = scaler.transform(df_encoded)
        pred_idx = model.predict(scaled_data)[0]
        result = encoders['Sleep Disorder'].inverse_transform([pred_idx])[0]
        probs = model.predict_proba(scaled_data)[0]
        classes = encoders['Sleep Disorder'].classes_

        # --- রেজাল্ট কার্ড ---
        is_healthy = "None" in str(result) or "Healthy" in str(result) or "No" in str(result)
        color = "#00ff88" if is_healthy else "#ff4b4b"
        icon = "✅" if is_healthy else "⚠️"
        
        res_html = f"""
        <div style="background: rgba(255, 255, 255, 0.05); border: 2px solid {color}; border-radius: 20px; padding: 25px; text-align: center; box-shadow: 0 0 20px {color}44;">
            <h3 style="color: white; margin: 0; font-size: 1.1rem; opacity: 0.7;">AI Diagnostic Analysis</h3>
            <h1 style="color: {color}; margin: 10px 0; font-size: 2.8rem; font-weight: 800;">{icon} {result}</h1>
        </div>
        """

        # --- ২. কনফিডেন্স চার্ট ---
        fig_prob = go.Figure(data=[go.Pie(labels=classes, values=probs, hole=.6, marker=dict(colors=['#00ff88', '#ff4b4b', '#ffaa00']))])
        fig_prob.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"), height=300, showlegend=True, margin=dict(t=10, b=10, l=10, r=10))

        # --- ৩. SHAP Plot ---
        plt.close('all')
        bg = shap.sample(train_sample, 10) if train_sample is not None else scaled_data
        explainer = shap.KernelExplainer(model.predict, bg)
        shap_v = explainer.shap_values(scaled_data)
        
        ev = explainer.expected_value[pred_idx] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        sv = shap_v[pred_idx] if isinstance(shap_v, list) else shap_v[0]
        
        shap.force_plot(ev, sv, df_encoded.iloc[0], matplotlib=True, show=False)
        fig_shap = plt.gcf()
        fig_shap.set_facecolor('none')
        fig_shap.set_size_inches(12, 3)

        return res_html, fig_prob, fig_shap

    except Exception as e:
        # কোনো এরর হলে সেটা সরাসরি আউটপুটে দেখাবে
        return f"<div style='color:red; padding:20px;'>Error: {str(e)}</div>", None, None

# ৩. CSS
css = """
footer {display: none !important;}
.gradio-container { background: linear-gradient(135deg, #050505 0%, #1a1a2e 100%) !important; }
.glass-card { background: rgba(255, 255, 255, 0.03) !important; backdrop-filter: blur(15px) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; border-radius: 20px; padding: 20px; }
.main-btn { background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%) !important; color: black !important; font-weight: bold !important; border-radius: 12px !important; }
label { color: #4facfe !important; }
"""

with gr.Blocks(theme=gr.themes.Default(), css=css) as demo:
    gr.HTML("<div style='text-align: center; padding: 20px;'><h1 style='color: white; font-size: 2.5rem;'>🌙 SleepInsight AI</h1></div>")

    with gr.Row(elem_classes="glass-card"):
        with gr.Column():
            age = gr.Number(label="Age", value=22)
            gender = gr.Radio(["Male", "Female"], label="Gender", value="Male")
            dept = gr.Dropdown(["CSE", "EEE","ECE","Civil Engineering","Mechanical Engineering","DVM", "BBA", "Pharmacy", "Agriculture","Sociology", "Mathematics","English","Development Studies","Physics","Statistics","FPE","Economics","Botany","zoology","Chemistry","Finance and Banking"], label="Department", value="CSE")
            level = gr.Dropdown(["Level-1", "Level-2", "Level-3", "Level-4"], label="Level", value="Level-4")
            uni = gr.Dropdown(["HSTU", "DU", "JnU", "RU", "CU", "KU" ,"MBSTU"], label="University", value="HSTU")

        with gr.Column():
            s_dur = gr.Slider(1, 12, value=7, label="Sleep Hours")
            qual = gr.Dropdown(["Poor", "Average", "Good", "Excellent"], label="Sleep Quality", value="Good")
            act = gr.Slider(0, 100, value=50, label="Physical Activity")
            stress = gr.Dropdown(["Very Low", "Low", "Moderate", "High", "Very High"], label="Stress", value="Moderate")

        with gr.Column():
            bmi = gr.Dropdown(["Normal", "Overweight", "Obese", "Underweight"], label="BMI", value="Normal")
            hr = gr.Number(label="Heart Rate", value=72)
            steps = gr.Number(label="Daily Steps", value=5000)
            sys = gr.Number(label="Systolic BP", value=120)
            dia = gr.Number(label="Diastolic BP", value=80)

    submit_btn = gr.Button("🚀 ANALYZE MY DATA", elem_classes="main-btn")
    
    # আউটপুট এরিয়া
    out_html = gr.HTML()
    with gr.Row(elem_classes="glass-card"):
        out_plot_prob = gr.Plot(label="Confidence")
        out_plot_shap = gr.Plot(label="AI Logic")

    # ইনপুট লিস্টের সিকোয়েন্স ফাংশনের আর্গুমেন্ট সিকোয়েন্সের সাথে মিলাতে হবে
    input_list = [dept, gender, age, s_dur, qual, act, stress, bmi, hr, steps, level, uni, sys, dia]
    
    submit_btn.click(
        fn=predict_and_analyze,
        inputs=input_list,
        outputs=[out_html, out_plot_prob, out_plot_shap]
    )

if __name__ == "__main__":
    demo.launch()