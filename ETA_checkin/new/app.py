from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import time

app = Flask(__name__)

data = pd.read_csv('C:\\Users\\Admin\\Downloads\\ETA_checkin\\ETA_checkin\\model_input.csv')

data['reservation_time'] = pd.to_datetime(data['reservation_time'])

data['month'] = data['reservation_time'].dt.month
data['season'] = (data['month'] % 12 + 3) // 3 

season_encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
season_encoded = season_encoder.fit_transform(data[['season']])
season_encoded_df = pd.DataFrame(season_encoded.toarray(), columns=season_encoder.get_feature_names_out(['season']), index=data.index)

month_encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
month_encoded = month_encoder.fit_transform(data[['month']])
month_encoded_df = pd.DataFrame(month_encoded.toarray(), columns=month_encoder.get_feature_names_out(['month']), index=data.index)

high_chair_encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
high_chair_encoded = high_chair_encoder.fit_transform(data[['high_chair_size']])
high_chair_encoded_df = pd.DataFrame(high_chair_encoded.toarray(), columns=high_chair_encoder.get_feature_names_out(['high_chair_size']), index=data.index)

data_encoded = pd.concat([data, season_encoded_df, month_encoded_df, high_chair_encoded_df], axis=1)

features = ['party_size', 'queue_size'] + list(season_encoded_df.columns) + list(month_encoded_df.columns) + list(high_chair_encoded_df.columns)
target = 'ETA'


def handle_negative_zero(df):
    df['ETA'] = df['ETA'].apply(lambda x: max(x, 0))
    return df

xgb_model = XGBRegressor(random_state=42)

df_processed = data_encoded.copy()
df_processed = handle_negative_zero(df_processed)
X = df_processed[features]
y = df_processed[target]

xgb_model.fit(X, y)

@app.route('/predict_eta', methods=['GET'])
def predict_eta():
    try:
        user_party_size = int(request.args.get('party_size'))
        user_queue_size = int(request.args.get('queue_size'))
        user_reservation_time = pd.to_datetime(request.args.get('reservation_time'))
        user_high_chair_size = int(request.args.get('high_chair_size'))

        user_month = user_reservation_time.month
        user_season = (user_month % 12 + 3) // 3

        user_input_df = pd.DataFrame({
            'party_size': [user_party_size],
            'queue_size': [user_queue_size],
            'reservation_time': [user_reservation_time],
            'month': [user_month],
            'season': [user_season],
            'high_chair_size': [user_high_chair_size]
        })

        user_season_encoded = season_encoder.transform(user_input_df[['season']])
        user_month_encoded = month_encoder.transform(user_input_df[['month']])
        user_high_chair_encoded = high_chair_encoder.transform(user_input_df[['high_chair_size']])

        user_input_encoded = pd.concat([user_input_df, 
                                        pd.DataFrame(user_season_encoded.toarray(), columns=season_encoder.get_feature_names_out(['season'])),
                                        pd.DataFrame(user_month_encoded.toarray(), columns=month_encoder.get_feature_names_out(['month'])),
                                        pd.DataFrame(user_high_chair_encoded.toarray(), columns=high_chair_encoder.get_feature_names_out(['high_chair_size']))
                                       ], axis=1)

        X_user = user_input_encoded[features]

        y_pred_user = xgb_model.predict(X_user)

        predicted_eta = y_pred_user[0] 

        predicted_time = user_reservation_time + pd.Timedelta(minutes=predicted_eta)

        formatted_predicted_eta = f"{predicted_eta:.2f}"

        response = {
            'predicted_eta': formatted_predicted_eta,
            'predicted_time': predicted_time.strftime("%Y-%m-%d %H:%M:%S")
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
