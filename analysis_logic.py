import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import os

# Define the relative file path for the dataset
FILEPATH = "customer_shopping_data.csv"

def load_and_process_data():
    """Loads and preprocesses the customer shopping data."""
    if not os.path.exists(FILEPATH):
        raise FileNotFoundError(f"File not found: {FILEPATH}")
    df = pd.read_csv(FILEPATH)
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%d/%m/%Y')
    df['total_price'] = df['quantity'] * df['price']
    return df

def calculate_rfm(df):
    """Calculates RFM scores and segments customers."""
    snapshot_date = df['invoice_date'].max() + dt.timedelta(days=1)
    rfm = df.groupby('customer_id').agg({
        'invoice_date': lambda date: (snapshot_date - date.max()).days,
        'invoice_no': 'nunique',
        'total_price': 'sum'
    })
    rfm.rename(columns={'invoice_date': 'Recency', 'invoice_no': 'Frequency', 'total_price': 'Monetary'}, inplace=True)
    
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])
    
    seg_map = {
        r'[1-2][1-2]': 'Hibernating', r'[1-2][3-4]': 'At Risk',
        r'3[1-2]': 'About To Sleep', r'33': 'Need Attention',
        r'[3-4][3-4]': 'Loyal Customers', r'41': 'Promising',
        r'4[2-3]': 'Potential Loyalists', r'44': 'Champions'
    }
    rfm['Segment'] = (rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str)).replace(seg_map, regex=True)
    return rfm

def train_spending_model(df):
    """Trains a simple model and logs it to MLflow."""
    mlflow.set_tracking_uri("file:///tmp/mlruns")  # Always use local file store
    mlflow.set_experiment("Retail Spending Predictor")

    with mlflow.start_run():
        model_data = df[['age', 'total_price']].dropna()
        X = model_data[['age']]
        y = model_data['total_price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "spending-predictor-model")
        
        run_id = mlflow.active_run().info.run_id
        print(f"Training complete. Run ID: {run_id}, MSE: {mse}")

        return {"message": "Model training complete.", "run_id": run_id, "mse": mse}
