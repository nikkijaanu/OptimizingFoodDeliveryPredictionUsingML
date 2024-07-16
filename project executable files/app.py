from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime

app = Flask(__name__)

# Load your regression model and scaler
model = pickle.load(open("rand.pkl", "rb"))
scaler = pickle.load(open("ss.pk1", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))  # Load label encoders

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/predict')
def predict():
    return render_template("predict.html")

@app.route('/service')
def service():
    return render_template("service.html")

@app.route('/submit', methods=["POST"])
def submit():
    if request.method == 'POST':
        try:
            # Get form data
            Delivery_person_Ratings = float(request.form['Delivery_person_Ratings'])
            Delivery_location_longitude = float(request.form['Delivery_location_longitude'])
            Delivery_location_latitude = float(request.form['Delivery_location_latitude'])
            Restaurant_longitude = float(request.form['restaurant_longitude'])  # Corrected name
            Restaurant_latitude = float(request.form['Restaurant_latitude'])
            Time_Orderd = request.form['Time_Orderd']
            Type_of_order = request.form['Type_of_order']
            Weatherconditions = request.form['Weatherconditions']
            Road_traffic_density = request.form['Road_traffic_density']
            Festival = request.form['Festival']
            City = request.form['City']  # Corrected name
        except KeyError as e:
            return f"Missing data for {e.args[0]}", 400

        # Parse Time_Orderd to extract hours and minutes
        try:
            time_ordered = datetime.strptime(Time_Orderd, '%H:%M:%S')
            Time_Orderd = time_ordered.hour + time_ordered.minute / 60.0  # Convert to fractional hour
        except ValueError:
            return "Incorrect time format, should be HH:MM:SS", 400

        # Create a dictionary to hold the data
        data = {
            'Delivery_person_Ratings': [Delivery_person_Ratings],
            'Restaurant_latitude': [Restaurant_latitude],
            'Restaurant_longitude': [Restaurant_longitude],
            'Delivery_location_latitude': [Delivery_location_latitude],
            'Delivery_location_longitude': [Delivery_location_longitude],
            'Time_Orderd': [Time_Orderd],  # Use the corrected Time_Orderd
            'Weatherconditions': [Weatherconditions],
            'Road_traffic_density': [Road_traffic_density],
            'Type_of_order': [Type_of_order],
            'Festival': [Festival],
            'City': [City]  # Corrected name
        }

        # Convert to DataFrame
        features_df = pd.DataFrame(data)

        # Handle previously unseen labels
        def encode_with_default(feature, label):
            if label not in label_encoders:
                return 0  # Default value for unknown labels
            try:
                return label_encoders[label].transform([feature])[0]
            except ValueError:
                if 'Unknown' in label_encoders[label].classes_:
                    return label_encoders[label].transform(['Unknown'])[0]
                else:
                    return label_encoders[label].transform([label_encoders[label].classes_[0]])[0]

        # Encode categorical variables using label encoder
        features_df['Type_of_order'] = encode_with_default(Type_of_order, 'Type_of_order')
        features_df['Festival'] = encode_with_default(Festival, 'Festival')
        features_df['City'] = encode_with_default(City, 'City')
        features_df['Weatherconditions'] = encode_with_default(Weatherconditions, 'Weatherconditions')
        features_df['Road_traffic_density'] = encode_with_default(Road_traffic_density, 'Road_traffic_density')

        # Reorder DataFrame columns to match training order
        features_df = features_df[['Delivery_person_Ratings', 'Restaurant_latitude', 'Restaurant_longitude',
                                   'Delivery_location_latitude', 'Delivery_location_longitude', 'Time_Orderd',
                                   'Weatherconditions', 'Road_traffic_density', 'Type_of_order', 'Festival', 'City']]

        # Scale numerical features
        scaled_features = scaler.transform(features_df)

        # Predict the outcome
        prediction = model.predict(scaled_features)
        predicted_time = round(prediction[0], 2)

        return render_template("submit.html", pred=predicted_time)

if __name__ == "__main__":
    app.run(debug=True)
