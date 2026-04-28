from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from ucimlrepo import fetch_ucirepo

wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features
y = wine_quality.data.targets.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(
    kernel="rbf",
    C=3,
    gamma=0.01,
    class_weight="balanced",
    decision_function_shape="ovr",
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

y_train_pred = model.predict(X_train_scaled)

print("--- WYNIKI NA ZBIORZE TESTOWYM (NOWE DANE) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, zero_division=0))

print("\n" + "=" * 50 + "\n")

print("--- WYNIKI NA ZBIORZE TRENINGOWYM (TO CO MODEL ZAPAMIĘTAŁ) ---")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.2f}")
print(classification_report(y_train, y_train_pred, zero_division=0))
