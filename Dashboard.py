import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Streamlit Page Configuration
st.set_page_config(page_title="Customer Churn Dashboard", page_icon="📊", layout="wide")

# File Upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

def load_data(file):
    df = pd.read_csv(file)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    return df

# Load data
df = load_data(uploaded_file) if uploaded_file else load_data("churn_dataset.csv")

# Sidebar Filters
st.sidebar.header("Filters")
gender_filter = st.sidebar.multiselect("Filter by Gender", df["gender"].unique(), default=df["gender"].unique())
contract_filter = st.sidebar.multiselect("Filter by Contract", df["Contract"].unique(), default=df["Contract"].unique())

# Apply filters
filtered_df = df[(df["gender"].isin(gender_filter)) & (df["Contract"].isin(contract_filter))]

# Main Dashboard
st.title("📊 Customer Churn Dashboard")
st.markdown("### Interactive Insights for Churn Analysis")

# Churn Pie Chart using Matplotlib
st.subheader("Churn Distribution")
churn_counts = df["Churn"].value_counts()
fig, ax = plt.subplots()
ax.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', colors=['lightcoral', 'lightblue'])
ax.set_title("Churn Distribution")
st.pyplot(fig)

# Customer Spending Scatter Plot using Seaborn
st.subheader("Customer Spending Patterns")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="MonthlyCharges", y="TotalCharges", hue="Churn", ax=ax)
st.pyplot(fig)

# Churn Distribution
st.subheader("Churn Distribution")
fig, ax = plt.subplots()
sns.countplot(x="Churn", data=df, ax=ax)
st.pyplot(fig)

# Monthly Charges vs Churn
st.subheader("Monthly Charges vs Churn")
fig, ax = plt.subplots()
sns.boxplot(x="Churn", y="MonthlyCharges", data=df, ax=ax)
st.pyplot(fig)

# High-Value Customers
st.subheader("🏆 High-Value Customers")
high_value_customers = df[(df["tenure"] > 50) & (df["MonthlyCharges"] > 80)]
st.dataframe(high_value_customers)

# Display Filtered Data
st.subheader("📋 Filtered Data")
st.dataframe(filtered_df)

# Machine Learning Model: Churn Prediction
st.subheader("🤖 Churn Prediction Model")

# Prepare data
X = df.drop(columns=["customerID", "Churn"])
X = pd.get_dummies(X, drop_first=True)
Y = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

st.write(f"Model Accuracy: {accuracy:.2f}")
st.text("Classification Report:")
st.text(classification_report(Y_test, Y_pred))

# Download Insights Report
st.subheader("📥 Download Insights Report")
report = f"""
Customer Churn Insights Report
---------------------------------
Total Customers: {df.shape[0]}
Churn Rate: {df['Churn'].value_counts(normalize=True)['Yes']:.2%}
High-Value Customers: {high_value_customers.shape[0]}
Model Accuracy: {accuracy:.2f}
"""
st.download_button(label="Download Report", data=report, file_name="churn_insights.txt", mime="text/plain")
