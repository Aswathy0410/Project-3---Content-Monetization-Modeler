import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Set page configuration
st.set_page_config(page_title="YouTube Ad Revenue Predictor", layout="wide")

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the saved model, encoders, and scaler
required_files = ['best_model.pkl', 'scaler.pkl', 'category_encoder.pkl', 'device_encoder.pkl', 'country_encoder.pkl']
for file in required_files:
    if not os.path.exists(file):
        st.error(f"Error: '{file}' not found. Please run the training script to generate all required files.")
        st.stop()

try:
    best_model = pickle.load(open('best_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    le_category = pickle.load(open('category_encoder.pkl', 'rb'))
    le_device = pickle.load(open('device_encoder.pkl', 'rb'))
    le_country = pickle.load(open('country_encoder.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model or encoders: {e}")
    st.stop()

# Streamlit app title
st.title("YouTube Ad Revenue Predictor")
st.markdown("Predict ad revenue for YouTube videos based on performance metrics and contextual features.")

# Sidebar for user input
st.sidebar.header("Input Video Details")
views = st.sidebar.number_input("Views", min_value=0, value=10000, step=100)
likes = st.sidebar.number_input("Likes", min_value=0, value=1000, step=10)
comments = st.sidebar.number_input("Comments", min_value=0, value=100, step=10)
watch_time_minutes = st.sidebar.number_input("Watch Time (minutes)", min_value=0.0, value=20000.0, step=100.0)
video_length_minutes = st.sidebar.number_input("Video Length (minutes)", min_value=0.0, value=10.0, step=0.5)
subscribers = st.sidebar.number_input("Subscribers", min_value=0, value=50000, step=1000)
category = st.sidebar.selectbox("Category", le_category.classes_)
device = st.sidebar.selectbox("Device", le_device.classes_)
country = st.sidebar.selectbox("Country", le_country.classes_)

# Feature engineering for input
engagement_rate = (likes + comments) / max(views, 1)
views_per_subscriber = views / max(subscribers, 1)
watch_time_per_view = watch_time_minutes / max(views, 1)
completion_rate = watch_time_minutes / max((views * video_length_minutes), 1)

# Prepare input data for prediction
input_data = pd.DataFrame({
    'views': [views],
    'likes': [likes],
    'comments': [comments],
    'watch_time_minutes': [watch_time_minutes],
    'video_length_minutes': [video_length_minutes],
    'subscribers': [subscribers],
    'engagement_rate': [engagement_rate],
    'views_per_subscriber': [views_per_subscriber],
    'watch_time_per_view': [watch_time_per_view],
    'completion_rate': [completion_rate],
    'category_encoded': [le_category.transform([category])[0]],
    'device_encoded': [le_device.transform([device])[0]],
    'country_encoded': [le_country.transform([country])[0]]
})

# Scale input data
input_scaled = scaler.transform(input_data)

# Make prediction
try:
    prediction = best_model.predict(input_scaled)[0]
except Exception as e:
    st.error(f"Error making prediction: {e}")
    st.stop()

# Display prediction
st.header("Predicted Ad Revenue")
st.markdown(f"**Estimated Ad Revenue:** ${prediction:.2f}")

# Display model information
st.header("Model Information")
st.markdown(f"**Best Model Used:** {best_model.__class__.__name__}")
if os.path.exists('model_metrics.pkl'):
    with open('model_metrics.pkl', 'rb') as file:
        metrics_df = pickle.load(file)
    st.markdown("**Model Performance (from training):**")
    st.write(metrics_df.round(4))
else:
    st.markdown("**Model Performance (from training):** Refer to training output for R² Score, RMSE, and MAE.")
st.markdown("This model was selected based on its performance during training with 5-fold cross-validation.")

# Load dataset for visualizations (optional, fallback to placeholder data)
data_path = "F:/MDTM46B/Project 3/Content Monetization Modeler/youtube_ad_revenue_dataset.csv"
df = None
if os.path.exists(data_path):
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        st.warning(f"Error loading dataset for visualizations: {e}")

# Generate visualizations dynamically
st.header("Basic Visual Analytics")
st.markdown("Below are visualizations showing the distribution of ad revenue and relationships with key metrics.")

# If dataset is unavailable, use placeholder data
if df is None:
    st.warning("Dataset not found. Using placeholder data for visualizations.")
    np.random.seed(42)
    df = pd.DataFrame({
        'views': np.random.randint(9500, 10500, 1000),
        'watch_time_minutes': np.random.uniform(15000, 60000, 1000),
        'ad_revenue_usd': np.random.uniform(100, 400, 1000),
        'category': np.random.choice(['Entertainment', 'Gaming', 'Education', 'Music', 'Vlogs'], 1000),
        'device': np.random.choice(['Mobile', 'TV', 'Tablet'], 1000),
        'country': np.random.choice(['US', 'IN', 'CA', 'UK', 'AU'], 1000)
    })

# Categorical distribution plots
st.subheader("Categorical Distribution Plots")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
cat_cols = ['category', 'device', 'country']
for i, col in enumerate(cat_cols):
    df[col].value_counts().plot(kind='bar', ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].tick_params(axis='x', rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Correlation heatmap
st.subheader("Correlation Heatmap")
num_cols = ['views', 'watch_time_minutes', 'ad_revenue_usd']
fig = plt.figure(figsize=(10, 8))
correlation_matrix = df[num_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Features')
st.pyplot(fig)

# Revenue plots
st.subheader("Revenue Plots")
fig = plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.hist(df['ad_revenue_usd'], bins=30, alpha=0.7, color='skyblue')
plt.title('Distribution of Ad Revenue')
plt.xlabel('Revenue (USD)')
plt.subplot(1, 3, 2)
plt.scatter(df['views'], df['ad_revenue_usd'], alpha=0.5)
plt.title('Views vs Revenue')
plt.xlabel('Views')
plt.ylabel('Revenue (USD)')
plt.subplot(1, 3, 3)
plt.scatter(df['watch_time_minutes'], df['ad_revenue_usd'], alpha=0.5)
plt.title('Watch Time vs Revenue')
plt.xlabel('Watch Time (minutes)')
plt.ylabel('Revenue (USD)')
plt.tight_layout()
st.pyplot(fig)

# Display model insights
st.header("Model Insights")
if os.path.exists('insights.pkl'):
    with open('insights.pkl', 'rb') as file:
        insights = pickle.load(file)
    st.markdown("**Top Revenue Drivers:**")
    for i, row in insights['top_features'].iterrows():
        st.markdown(f"- {row['Feature']}: {row['Importance']:.3f} importance")
    st.markdown("**Category Performance:**")
    st.write(insights['category_performance'].sort_values('mean', ascending=False))
else:
    st.markdown("""
    **Key Insights from Model Training:**
    - **Watch time** is a major driver of ad revenue...
    - **Engagement rate** (likes + comments per view)...
    - **Video length** impacts revenue...
    - **Categories** like Entertainment and Gaming...
    - **Device type** (e.g., Mobile)...

    **Recommendations for Content Creators:**
    - Create engaging content...
    - Aim for video lengths of 10-15 minutes...
    - Focus on high-revenue categories...
    - Optimize videos for mobile viewing...
    - Maintain a consistent upload schedule...
    """)

