import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="📊 EDA Dashboard",
    page_icon="📈",
    layout="wide"
)

# ===================== CSS Styling =====================
st.markdown("""
    <style>
        /* App background */
        .stApp {
            background-color: #F7F9FC;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Headings */
        h1, h2, h3 {
            text-align: center;
            font-weight: 700;
            color: #2C3E50;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #2C3E50;
            color: #ECF0F1;
            border-radius: 12px;
            padding: 16px;
        }
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #ECF0F1 !important;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(90deg, #3498DB, #2ECC71);
            color: white !important;
            border-radius: 10px;
            border: none;
            font-size: 15px;
            font-weight: 600;
            padding: 10px 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transition: 0.3s ease;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #2ECC71, #27AE60);
            transform: scale(1.05);
        }

        /* File uploader */
        .stFileUploader {
            border: 2px dashed #3498DB;
            border-radius: 12px;
            padding: 12px;
            background: white;
        }

        /* Data preview table */
        .dataframe {
            border: 1px solid #ddd;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ===================== Title =====================
st.markdown("<h1>📊 Exploratory Data Analysis (EDA) Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#555;'>Upload your dataset and explore insights visually.</p>", unsafe_allow_html=True)

# ===================== Sidebar =====================
st.sidebar.header("⚙️ Controls")
st.sidebar.markdown("Select your options to generate visualizations")

# ===================== Session State =====================
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'default_loaded' not in st.session_state:
    st.session_state.default_loaded = False

# ===================== File Upload =====================
file = st.sidebar.file_uploader("📂 Upload CSV/Excel", type=["csv", "xlsx"])
df = st.session_state.df

# ===================== Helper Functions =====================
def outli(se_co):
    va = df[se_co].values
    m = df[se_co].median()
    q1 = np.percentile(df[se_co], 25)
    q3 = np.percentile(df[se_co], 75)
    iqr = q3 - q1
    lb, ub = q1 - (1.5 * iqr), q3 + (1.5 * iqr)
    return np.where((va < lb) | (va > ub), m, df[se_co])

def chart(pl, se_co):
    if pl == 'bar':
        if df[se_co].nunique() <= 12:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(data=df, x=se_co, palette="Set2", ax=ax)
            ax.set_title(f"Bar Plot of {se_co}")
            st.pyplot(fig)

    elif pl == 'pie':
        data = df[se_co].value_counts()
        if len(data) <= 12:
            fig, ax = plt.subplots()
            ax.pie(data.values, labels=data.index, autopct='%1.1f%%',
                   explode=[0.05]*len(data), colors=sns.color_palette("pastel"))
            ax.set_title(f"Pie Chart of {se_co}")
            st.pyplot(fig)

    elif pl == 'hist':
        df['bal'] = outli(se_co)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(df[se_co], color='skyblue', edgecolor='black')
        axes[0].set_title(f"{se_co} (Before Outliers)")
        axes[1].hist(df['bal'], color='salmon', edgecolor='black')
        axes[1].set_title(f"{se_co} (After Outliers)")
        st.pyplot(fig)
        df.drop("bal", axis=1, inplace=True)

    elif pl == 'dist':
        df['bal'] = outli(se_co)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[se_co], kde=True, color="skyblue", ax=axes[0])
        axes[0].set_title(f"{se_co} (Before Outliers)")
        sns.histplot(df['bal'], kde=True, color="salmon", ax=axes[1])
        axes[1].set_title(f"{se_co} (After Outliers)")
        st.pyplot(fig)
        df.drop("bal", axis=1, inplace=True)

    elif pl == 'boxplot':
        df['bal'] = outli(se_co)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.boxplot(y=df[se_co], color="skyblue", ax=axes[0])
        axes[0].set_title(f"{se_co} (Before Outliers)")
        sns.boxplot(y=df['bal'], color="salmon", ax=axes[1])
        axes[1].set_title(f"{se_co} (After Outliers)")
        st.pyplot(fig)
        df.drop("bal", axis=1, inplace=True)

def heat():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

def cro(se_col1, se_col2):
    df1 = pd.crosstab(df[se_col1], df[se_col2])
    fig, ax = plt.subplots(figsize=(8, 4))
    df1.plot(kind="bar", ax=ax, color=['skyblue', 'salmon'])
    ax.set_title(f"{se_col1} vs {se_col2}")
    st.pyplot(fig)

# ===================== File Reading =====================
if file is not None:
    try:
        if file.name.endswith(".csv"):
            sep = st.sidebar.text_input("CSV Separator", ",")
            df = pd.read_csv(file, sep=sep)
        else:
            df = pd.read_excel(file)
        st.session_state.df = df

        st.subheader("📋 Data Preview")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

if st.sidebar.button("📌 Use Default Dataset"):
    if not st.session_state.default_loaded:
        df = pd.read_csv(r"C:\Users\Aliya Khan\Desktop\Data_Files\bank (1).csv", sep=";")
        st.session_state.df = df
        st.session_state.default_loaded = True
        st.success("✅ Default dataset loaded")
        st.dataframe(df.head())

# ===================== Plot Selection =====================
if not df.empty:
    catcol = df.select_dtypes(include="object").columns
    numcol = df.select_dtypes(exclude="object").columns

    st.sidebar.subheader("📊 Visualization Options")
    se_op = st.sidebar.selectbox("Choose Plot Type:", ["bar", "pie", "hist", "dist", "boxplot", "heatmap", "crosstab"])

    if se_op == "crosstab":
        se_col1 = st.sidebar.selectbox("Column 1:", catcol)
        se_col2 = st.sidebar.selectbox("Column 2:", catcol)
    elif se_op in ["bar", "pie"]:
        se_co = st.sidebar.selectbox("Categorical Column:", catcol)
    elif se_op in ["hist", "dist", "boxplot"]:
        se_co = st.sidebar.selectbox("Numerical Column:", numcol)

    if st.sidebar.button("🚀 Generate Plot"):
        with st.container():
            if se_op == "crosstab":
                cro(se_col1, se_col2)
            elif se_op == "heatmap":
                heat()
            else:
                chart(se_op, se_co)
else:
    st.info("👆 Upload a dataset or load the default one to start exploring.")

