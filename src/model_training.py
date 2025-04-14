import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error
import joblib

# Cargar los datos preprocesados y balanceados
file_path = 'C:\\Users\\jimen\\Documentos\\codes\\AI\\modelo_prediccion_obesidad\\data\\resampled_data.csv'
data = pd.read_csv(file_path)

# Selección de características y variable objetivo
features = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'CALC']
X = data[features]
y = data['NObeyesdad']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento y evaluación de múltiples modelos
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'SVM': SVC()
}

results = {}

for model_name, model in models.items():
    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Guardar resultados
    results[model_name] = {
        'classification_report': report,
        'confusion_matrix': cm
    }

    # Guardar el modelo entrenado
    joblib.dump(model, f'C:\\Users\\jimen\\Documentos\\codes\\AI\\modelo_prediccion_obesidad\\results\\{model_name}_model.pkl')

# Guardar los resultados de la evaluación
with open('C:\\Users\\jimen\\Documentos\\codes\\AI\\modelo_prediccion_obesidad\\results\\model_performance.txt', 'w') as f:
    for model_name, result in results.items():
        f.write(f'Model: {model_name}\n')
        f.write(f'Classification Report:\n{result["classification_report"]}\n')
        f.write(f'Confusion Matrix:\n{result["confusion_matrix"]}\n')
        f.write('\n')

# Validación cruzada
cv_results = {}
for model_name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5)
    cv_results[model_name] = cv_scores

# Guardar los resultados de la validación cruzada
with open('C:\\Users\\jimen\\Documentos\\codes\\AI\\modelo_prediccion_obesidad\\results\\cv_performance.txt', 'w') as f:
    for model_name, scores in cv_results.items():
        f.write(f'Model: {model_name}\n')
        f.write(f'CV Scores: {scores}\n')
        f.write(f'Mean CV Score: {scores.mean()}\n')
        f.write('\n')
