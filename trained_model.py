import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("/mnt/user-data/uploads/station_depart.csv")

# ---- Task 2.2 replica: parse times ----
df["Arrival_time"] = pd.to_datetime(df["Arrival_time"], format="%H:%M:%S")
df["Departure_Time"] = pd.to_datetime(df["Departure_Time"], format="%H:%M:%S")

# ---- journey duration per train ----
journey = (
    df.groupby("Train_No")
      .agg(Start_Time=("Departure_Time", "first"), End_Time=("Arrival_time", "last"))
      .reset_index()
)
journey["Journey_Duration"] = (journey["End_Time"] - journey["Start_Time"]).dt.total_seconds() / 3600
journey.loc[journey["Journey_Duration"] < 0, "Journey_Duration"] += 24

# ---- feature engineering ----
total_distance = df.groupby("Train_No")["Distance"].max().reset_index(name="Total_Distance")
total_stops = df.groupby("Train_No")["Station_Name"].count().reset_index(name="Total_Stops")
starting_station = df.groupby("Train_No")["Station_Name"].first().reset_index(name="Starting_Station")
ending_station = df.groupby("Train_No")["Station_Name"].last().reset_index(name="Ending_Station")

train_features = (
    total_distance
    .merge(total_stops, on="Train_No")
    .merge(starting_station, on="Train_No")
    .merge(ending_station, on="Train_No")
    .merge(journey[["Train_No", "Journey_Duration"]], on="Train_No")
)

# drop rows with zero/garbage duration or distance (basic sanity cleaning)
train_features = train_features[(train_features["Journey_Duration"] > 0) & (train_features["Total_Distance"] > 0)]

# ---- FIX: one LabelEncoder fit on the UNION of start+end station names ----
# (original notebook fit the same encoder twice, so the saved encoder only
#  reflected the Ending_Station classes -- Starting_Station codes were invalid)
station_encoder = LabelEncoder()
all_station_names = pd.concat([train_features["Starting_Station"], train_features["Ending_Station"]]).unique()
station_encoder.fit(all_station_names)

train_features["Starting_Station_enc"] = station_encoder.transform(train_features["Starting_Station"])
train_features["Ending_Station_enc"] = station_encoder.transform(train_features["Ending_Station"])

FEATURES = ["Total_Distance", "Total_Stops", "Starting_Station_enc", "Ending_Station_enc"]
X = train_features[FEATURES]
y = train_features["Journey_Duration"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")

# rename columns back to friendly names for saving / clarity
model.feature_names_in_ = np.array(["Total_Distance", "Total_Stops", "Starting_Station", "Ending_Station"])

joblib.dump(model, "/home/claude/work/journey_duration_model.pkl")
joblib.dump(station_encoder, "/home/claude/work/encoder.pkl")

# save test-set results for the "Actual vs Predicted" visualization in the app
test_results = X_test.copy()
test_results["Actual_Duration"] = y_test.values
test_results["Predicted_Duration"] = y_pred
test_results["Starting_Station"] = station_encoder.inverse_transform(test_results["Starting_Station_enc"])
test_results["Ending_Station"] = station_encoder.inverse_transform(test_results["Ending_Station_enc"])
test_results.to_csv("/home/claude/work/test_results.csv", index=False)

print("Saved model, encoder, and test_results.csv")
print(train_features.shape, "trains used for training")
