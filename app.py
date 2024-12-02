import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_squared_error, r2_score, roc_curve, auc
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
import pickle
import io
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
# Streamlit Configuration
st.set_page_config(page_title="Data Scientist's Portal", page_icon="🧠", layout="wide")

# Title
st.title(":rainbow[ Data Scientist's Portal]")
st.write("### Your Ultimate Platform for Machine Learning, Deep Learning, Time Series Forecasting, and Data Visualization")

# Step 1: Problem Definition
st.write("### Step 1: Problem Definition")
problem = st.text_area("Define your problem statement:", help="State the problem or task you are trying to solve.")
if problem:
    st.info(f"Problem Statement: {problem}")

# Step 2: Data Acquisition
st.write("### Step 2: Data Acquisition")
file = st.file_uploader("Upload your dataset (CSV, Excel, or JSON)", type=["csv", "xlsx", "json"], help="Upload your dataset to start the analysis.")

# Initialize variables
model = None
X_test = None
y_test = None
predictions = None

if file:
    try:
        # Load dataset
        if file.name.endswith("csv"):
            data = pd.read_csv(file)
        elif file.name.endswith("xlsx"):
            data = pd.read_excel(file)
        else:
            data = pd.read_json(file)

        if data.empty:
            st.error("The dataset is empty. Please upload a valid dataset.")
        else:
            st.success("File uploaded successfully!")
            st.dataframe(data.head())

            # Tabs for workflow
            tabs = st.tabs(["📊 Data Overview", "🧹 Data Cleaning", "🔬 Feature Engineering",
                            "📊 EDA", "🤖 ML Modeling", "🧠 DL Modeling", "📈 Evaluation", "📉 Time Series Forecasting", "📤 Export & Reports"])

            # Tab 1: Data Overview
            with tabs[0]:
                st.subheader(":rainbow[Data Overview]")
                st.write(f"**Rows:** {data.shape[0]}, **Columns:** {data.shape[1]}")
                st.write("### Data Types")
                st.dataframe(data.dtypes.rename("Data Types"))
                st.write("### Missing Values")
                st.dataframe(data.isnull().sum().rename("Missing Values"))
                st.write("### Statistical Summary")
                st.dataframe(data.describe().T)



            # Tab 2: Data Cleaning
            with tabs[1]:
                st.subheader(":rainbow[Data Cleaning]")
                cleaning_strategy = st.radio("Choose a strategy for handling missing values:", ["Mean", "Median", "Most Frequent", "Remove Rows"])
                if st.button("Apply Cleaning"):
                    if cleaning_strategy == "Remove Rows":
                        data = data.dropna()
                    else:
                        imputer = SimpleImputer(strategy=cleaning_strategy.lower())
                        data.iloc[:, :] = imputer.fit_transform(data)
                    st.success("Missing values handled successfully!")
                    st.dataframe(data)

            # Tab 3: Feature Engineering
            with tabs[2]:
                st.subheader(":rainbow[Feature Engineering]")
                if st.checkbox("Scale Features"):
                    scale_method = st.selectbox("Choose a scaling method:", ["Standardization", "Normalization"])
                    scaler = StandardScaler() if scale_method == "Standardization" else MinMaxScaler()
                    num_cols = data.select_dtypes(include=[np.number]).columns
                    data[num_cols] = scaler.fit_transform(data[num_cols])
                    st.success("Features scaled successfully!")
                    st.dataframe(data)

                if st.checkbox("Encode Categorical Data"):
                    cat_cols = data.select_dtypes(include=["object"]).columns
                    for col in cat_cols:
                        data[col] = LabelEncoder().fit_transform(data[col])
                    st.success("Categorical data encoded successfully!")
                    st.dataframe(data)
                    
                if st.checkbox("Show Correlation Matrix"):
                    corr = data.corr()
                    st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="Blues"))
                    
                    
                    

            # Tab 4: EDA
            with tabs[3]:
                st.subheader(":rainbow[Exploratory Data Analysis]")
                chart_type = st.radio("Choose a chart type:", ["Histogram", "Boxplot", "Scatter Plot", "Heatmap"], key="eda_chart_type")
                if chart_type == "Histogram":
                    col = st.selectbox("Select a column for the histogram:", data.columns, key="eda_hist_column")
                    st.plotly_chart(px.histogram(data, x=col))
                elif chart_type == "Boxplot":
                    col = st.selectbox("Select a column for the boxplot:", data.columns, key="eda_box_column")
                    st.plotly_chart(px.box(data, y=col))
                elif chart_type == "Scatter Plot":
                    x_col = st.selectbox("Select the X-axis column:", data.columns, key="eda_scatter_x")
                    y_col = st.selectbox("Select the Y-axis column:", data.columns, key="eda_scatter_y")
                    st.plotly_chart(px.scatter(data, x=x_col, y=y_col))
                elif chart_type == "Heatmap":
                    st.plotly_chart(px.imshow(data.corr(), text_auto=True, color_continuous_scale="Blues"))

            # Tab 5: ML Modeling
            with tabs[4]:
                st.subheader(":rainbow[Machine Learning Modeling]")
                model_type = st.radio("Choose a model type:", ["Classification", "Regression"], key="ml_model_type")
                target = st.selectbox("Select the target variable:", options=["None"] + list(data.columns), key="ml_target")
                features = st.multiselect("Select feature columns:", options=[col for col in data.columns if col != target], key="ml_features")

                if features:
                    X = data[features]
                    if target != "None":
                        y = data[target]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        if model_type == "Classification":
                            st.write("### Choose a Classification Model")
                            clf_model = st.selectbox("Model:", ["Logistic Regression", "Random Forest", "XGBoost"])
                            if clf_model == "Logistic Regression":
                                model = LogisticRegression()
                            elif clf_model == "Random Forest":
                                model = RandomForestClassifier()
                            elif clf_model == "XGBoost":
                                model = xgb.XGBClassifier()
                            model.fit(X_train, y_train)
                            st.success(f"{clf_model} trained successfully!")
                            predictions = model.predict(X_test)

                        elif model_type == "Regression":
                            st.write("### Choose a Regression Model")
                            reg_model = st.selectbox("Model:", ["Linear Regression", "Random Forest Regressor", "XGBoost Regressor"])
                            if reg_model == "Linear Regression":
                                model = LinearRegression()
                            elif reg_model == "Random Forest Regressor":
                                model = RandomForestRegressor()
                            elif reg_model == "XGBoost Regressor":
                                model = xgb.XGBRegressor()
                            model.fit(X_train, y_train)
                            st.success(f"{reg_model} trained successfully!")
                            predictions = model.predict(X_test)

                    # Display prediction comparison
                    if predictions is not None:
                        comparison = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
                        st.write("### Comparison of Actual vs Predicted")
                        st.dataframe(comparison)

                    # Download trained model
                    st.download_button("Download Trained Model", pickle.dumps(model), file_name="ml_trained_model.pkl")

            # Tab 6: DL Modeling
            with tabs[5]:
                st.subheader(":rainbow[Deep Learning Modeling]")
                target = st.selectbox("Select the target variable:", options=["None"] + list(data.columns), key="dl_target")
                features = st.multiselect("Select feature columns:", options=[col for col in data.columns if col != target], key="dl_features")

                if features and target != "None":
                    X = data[features]
                    y = data[target]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    st.write("### Configure Neural Network")
                    hidden_units = st.slider("Number of Hidden Units:", 1, 256, 64)
                    epochs = st.slider("Number of Epochs:", 1, 100, 10)
                    batch_size = st.slider("Batch Size:", 1, 64, 32)

                    model_dl = Sequential([
                        Dense(hidden_units, activation="relu", input_dim=X_train.shape[1]),
                        Dropout(0.2),
                        Dense(1, activation="sigmoid" if y.nunique() == 2 else "linear")
                    ])
                    model_dl.compile(optimizer="adam", loss="binary_crossentropy" if y.nunique() == 2 else "mse", metrics=["accuracy"])
                    history = model_dl.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
                    st.success("Deep Learning model trained successfully!")

                    st.write("### Training Progress")
                    st.line_chart(history.history["accuracy"], width=800, height=300)

                    # Download trained model
                    st.download_button("Download Trained DL Model", pickle.dumps(model_dl), file_name="trained_dl_model.pkl")

            # Tab 7: Evaluation
            with tabs[6]:
                st.subheader(":rainbow[Model Evaluation]")
                if model and predictions is not None:
                    if model_type == "Classification":
                        st.write("### Classification Report")
                        st.text(classification_report(y_test, predictions))
                        st.write("### Confusion Matrix")
                        st.plotly_chart(px.imshow(confusion_matrix(y_test, predictions), text_auto=True, color_continuous_scale="Blues"))
                    elif model_type == "Regression":
                        st.write("### Regression Metrics")
                        st.write("Mean Squared Error:", mean_squared_error(y_test, predictions))
                        st.write("R² Score:", r2_score(y_test, predictions))
                else:
                    st.info("Please train a model first.")

            # Tab 8: Time Series Forecasting
            with tabs[7]:
                st.markdown("Coming Soon! this work is still in progress.")

    # Download Forecasted Data
            # Tab 9: Export & Reports
            with tabs[8]:
                st.subheader(":rainbow[Export & Reports]")
                st.download_button("Download Cleaned Dataset", data.to_csv(index=False).encode("utf-8"), file_name="cleaned_data.csv")
                if model:
                    buffer = io.BytesIO()
                    pickle.dump(model, buffer)
                    st.download_button("Download Trained Model", buffer.getvalue(), file_name="trained_model.pkl")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a dataset to proceed.")


st.markdown("""
---
## 📄 About Me  
Hi! I'm **Muhammad Dawood**, a data scientist passionate about Machine Learning, NLP, and building intelligent solutions.  

### 🌐 Connect with Me:  
- [GitHub](https://github.com/muhammadmoria)  
- [LinkedIn](https://www.linkedin.com/in/muhammaddawood361510306/)  
- [Portfolio](https://muhammadmoria.github.io/portfolio-new/)  
""")
