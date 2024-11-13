from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']

        # Load the dataset
        data = pd.read_csv(file)

        # Define features and target variable
        features = ['cpu_usage_percent', 'memory_usage_percent', 'disk_space_remaining_percent', 
                    'network_traffic_mbps', 'response_time_ms', 'error_rate_percent', 'server_health']
        X = data[features]
        y = data['time_to_failure']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build and train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return render_template('results.html', mse=mse, r2=r2)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)