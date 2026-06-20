import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Create room recommendation dataset
data = {
    "budget": [
        2000, 2500, 3000, 3500, 4000,
        4500, 5000, 5500, 6000, 6500,
        7000, 8000, 8500, 9000, 10000,
        11000, 12000, 13000, 15000, 18000,
        20000, 22000, 25000
    ],
    "guests": [
        1, 1, 2, 2, 1,
        2, 2, 2, 2, 2,
        2, 3, 3, 3, 3,
        3, 2, 2, 2, 4,
        4, 4, 4
    ],
    "stay_days": [
        1, 2, 1, 2, 1,
        2, 3, 1, 2, 3,
        2, 2, 3, 2, 3,
        3, 2, 3, 2, 3,
        2, 3, 4
    ],
    "room_type": [
        "Standard Single",
        "Standard Single",
        "Standard Double",
        "Standard Double",
        "Deluxe Single",
        "Deluxe Double",
        "Deluxe Double",
        "Executive Room",
        "Business Room",
        "Business Room",
        "Family Room",
        "Family Room",
        "Garden View Room",
        "Garden View Room",
        "Suite Room",
        "Ocean View Room",
        "Luxury Suite",
        "Luxury Suite",
        "Honeymoon Suite",
        "Presidential Suite",
        "Presidential Suite",
        "Penthouse Suite",
        "Penthouse Suite"
    ]
}

df = pd.DataFrame(data)

# Save dataset also
df.to_csv("room_dataset.csv", index=False)

X = df[["budget", "guests", "stay_days"]]
y = df["room_type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

joblib.dump(model, "room_recommendation_model.pkl")

print("Room recommendation model trained successfully.")
print("Saved as room_recommendation_model.pkl")
print("Accuracy:", accuracy)