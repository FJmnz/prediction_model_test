import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Cargar los datos preprocesados
file_path = 'C:\\Users\\jimen\\Documentos\\codes\\AI\\modelo_prediccion_obesidad\\data\\preprocessed_data.csv'
data = pd.read_csv(file_path)

# Selección de características
features = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC']
X = data[features]
y = data['NObeyesdad']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Guardar el modelo entrenado
joblib.dump(model, 'C:\\Users\\jimen\\Documentos\\codes\\AI\\modelo_prediccion_obesidad\\results\\obesity_model.pkl')

# Evaluación del modelo
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Guardar los resultados de la evaluación
with open('C:\\Users\\jimen\\Documentos\\codes\\AI\\modelo_prediccion_obesidad\\results\\model_performance.txt', 'w') as f:
    f.write(classification_report(y_test, y_pred))
    f.write('\n')
    f.write(str(confusion_matrix(y_test, y_pred)))
