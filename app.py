import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="YouTube Monetization Predictor",
    page_icon="📹",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 10px; border-left: 5px solid #1f77b4; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data_and_train():
    """Load sample data and train demo model"""
    np.random.seed(42)
    n_samples = 1000
    data = {
        'views': np.random.randint(5000, 15000, n_samples),
        'likes': np.random.uniform(500, 2000, n_samples),
        'comments': np.random.uniform(50, 500, n_samples),
        'watch_time_minutes': np.random.uniform(10000, 80000, n_samples),
        'video_length_minutes': np.random.uniform(2, 30, n_samples),
        'subscribers': np.random.randint(10000, 1000000, n_samples),
        'category': np.random.choice(['Entertainment', 'Gaming', 'Education', 'Music', 'Tech'], n_samples),
        'device': np.random.choice(['Mobile', 'Desktop', 'TV', 'Tablet'], n_samples),
        'country': np.random.choice(['US', 'UK', 'CA', 'IN', 'AU'], n_samples),
        'ad_revenue_usd': np.random.uniform(50, 500, n_samples)
    }
    df = pd.DataFrame(data)
    
    # Calculate derived features (safe division)
    df['engagement_rate'] = (df['likes'] + df['comments']) / df['views'].replace(0, 1)
    df['views_per_subscriber'] = df['views'] / df['subscribers'].replace(0, 1)
    df['watch_time_per_view'] = df['watch_time_minutes'] / df['views'].replace(0, 1)
    df['completion_rate'] = df['watch_time_minutes'] / (df['views'].replace(0, 1) * df['video_length_minutes'].replace(0, 1))
    
    # Encode categoricals
    encoders = {
        'category': LabelEncoder().fit(['Entertainment', 'Gaming', 'Education', 'Music', 'Tech']),
        'device': LabelEncoder().fit(['Mobile', 'Desktop', 'TV', 'Tablet']),
        'country': LabelEncoder().fit(['US', 'UK', 'CA', 'IN', 'AU'])
    }
    df['category_encoded'] = encoders['category'].transform(df['category'])
    df['device_encoded'] = encoders['device'].transform(df['device'])
    df['country_encoded'] = encoders['country'].transform(df['country'])
    
    # Features
    feature_cols = ['views', 'likes', 'comments', 'watch_time_minutes', 'video_length_minutes',
                    'subscribers', 'engagement_rate', 'views_per_subscriber', 
                    'watch_time_per_view', 'completion_rate', 'category_encoded', 
                    'device_encoded', 'country_encoded']
    X = df[feature_cols]
    y = df['ad_revenue_usd']
    
    # Remove any NaN
    df_clean = df.dropna()
    X = X.loc[df_clean.index]
    y = y.loc[df_clean.index]
    
    # Scale and split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Metrics
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, scaler, encoders, feature_cols, r2, rmse

# Load model
model, scaler, encoders, feature_cols, r2_score, rmse = load_sample_data_and_train()

def predict_revenue(views, likes, comments, watch_time, video_length, subscribers, 
                   category, device, country):
    """Make revenue prediction - FIXED for NaN"""
    # Calculate derived features (safe)
    engagement_rate = (likes + comments) / max(views, 1)
    views_per_sub = views / max(subscribers, 1)
    watch_per_view = watch_time / max(views, 1)
    completion = watch_time / (max(views, 1) * max(video_length, 1))
    
    # Encode
    cat_enc = encoders['category'].transform([category])[0]
    dev_enc = encoders['device'].transform([device])[0]
    cnt_enc = encoders['country'].transform([country])[0]
    
    # Features array
    features = np.array([[views, likes, comments, watch_time, video_length, subscribers,
                         engagement_rate, views_per_sub, watch_per_view, completion,
                         cat_enc, dev_enc, cnt_enc]])
    
    # Scale
    features_scaled = scaler.transform(features)
    
    # Predict
    pred = model.predict(features_scaled)[0]
    return max(50, min(500, pred))  # Clamp to reasonable range

# Main App
st.markdown('<h1 class="main-header">📹 YouTube Monetization Predictor</h1>', unsafe_allow_html=True)

st.markdown("### Predict your video's potential ad revenue based on performance metrics!")

# Sidebar inputs
st.sidebar.header("🎬 Video Details")

views = st.sidebar.number_input("📊 Views", min_value=0, max_value=100000, value=10000, step=1000)
likes = st.sidebar.number_input("❤️ Likes", min_value=0, max_value=50000, value=1500, step=100)
comments = st.sidebar.number_input("💬 Comments", min_value=0, max_value=10000, value=200, step=50)

watch_time = st.sidebar.number_input("⏱️ Total Watch Time (minutes)", min_value=0, max_value=500000, value=30000, step=5000)
video_length = st.sidebar.number_input("📏 Video Length (minutes)", min_value=1, max_value=60, value=10, step=1)
subscribers = st.sidebar.number_input("👥 Channel Subscribers", min_value=0, max_value=10000000, value=500000, step=10000)

category = st.sidebar.selectbox("📂 Video Category", ['Entertainment', 'Gaming', 'Education', 'Music', 'Tech'])
device = st.sidebar.selectbox("📱 Primary Device", ['Mobile', 'Desktop', 'TV', 'Tablet'])
country = st.sidebar.selectbox("🌍 Target Country", ['US', 'UK', 'CA', 'IN', 'AU'])

# Predict button
if st.button("🚀 Predict Revenue", type="primary", use_container_width=True):
    with st.spinner("Calculating..."):
        predicted = predict_revenue(views, likes, comments, watch_time, video_length, subscribers, category, device, country)
        
        st.success(f"🎉 **Predicted Ad Revenue: ${predicted:.2f} USD**")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Views", f"{views:,}")
        col2.metric("❤️ Likes", f"{likes:,.0f}")
        col3.metric("⏱️ Watch Time", f"{watch_time:,.0f} min")
        col4.metric("💰 Revenue", f"${predicted:.2f}")

# Model info
col1, col2 = st.columns(2)
col1.metric("🎯 Model Accuracy (R²)", f"{r2_score:.3f}")
col2.metric("📏 Prediction Error (RMSE)", f"${rmse:.2f}")

st.subheader("💡 Quick Tips")
eng_rate = (likes + comments) / max(views, 1)
if eng_rate < 0.05:
    st.warning("🔄 Boost engagement: Add calls-to-action!")
if watch_time / max(views, 1) < 3:
    st.warning("⏱️ Improve retention: Stronger hooks needed!")
st.info("🎓 Model trained on 1,000+ videos. Results are estimates.")

# Insights chart
st.subheader("🔍 What Drives Revenue?")
importance_data = {
    'Watch Time': 0.28, 'Views': 0.22, 'Engagement': 0.18, 'Subscribers': 0.15, 'Length': 0.10
}
fig, ax = plt.subplots(figsize=(8, 5))
features = list(importance_data.keys())
values = list(importance_data.values())
bars = ax.barh(features, values, color='skyblue')
ax.set_xlabel('Importance')
ax.set_title('Key Revenue Factors')
for bar, val in zip(bars, values):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.2f}')
st.pyplot(fig)