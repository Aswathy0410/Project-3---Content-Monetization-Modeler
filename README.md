# YouTube Content Monetization Modeler

Predict YouTube ad revenue using machine learning! Built with Python, Scikit-learn, and Streamlit.

## 📊 Overview
This project builds a regression model to estimate ad revenue (`ad_revenue_usd`) from YouTube video metrics like views, likes, watch time, and category. 

- **Dataset**: ~122K rows of synthetic/real YouTube performance data.
- **Models**: 5 regressors (Random Forest best at R² ~0.85).
- **App**: Interactive Streamlit dashboard for predictions and insights.
- **Skills**: EDA, Feature Engineering, Regression, Streamlit.
  
📁 Files

main_analysis.py: Full ML pipeline (EDA → Models → Insights).
app.py: Interactive Streamlit predictor.
requirements.txt: Dependencies.
data/youtube_ad_revenue_dataset.csv: Dataset (add your CSV here).

🛠️ Key Features

Preprocessing: Duplicate removal, NaN handling (likes/comments=0, watch_time=median).
Features: engagement_rate = (likes + comments) / views + 3 more engineered.
Models: Linear Regression, Decision Tree, Random Forest, SVR, KNN.
Evaluation: R², RMSE, MAE (Random Forest: R²=0.85, RMSE=$24).
Insights: Watch time (28%) is #1 driver; Entertainment category best.

🤝 Contributing

Fork the repo.
Add features (e.g., XGBoost, real-time API).
PR with tests.

📄 License
MIT License - Feel free to use/modify.

👤 Author
[Aswathy B] 
