import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el modelo entrenado
model = joblib.load('C:\\Users\\jimen\\Documentos\\codes\\AI\\modelo_prediccion_obesidad\\results\\obesity_model.pkl')

# Cargar los datos preprocesados
file_path = 'C:\\Users\\jimen\\Documentos\\codes\\AI\\modelo_prediccion_obesidad\\data\\preprocessed_data.csv'
data = pd.read_csv(file_path)

# Selección de características
features = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC']
X = data[features]

# Importancia de las variables
importances = model.feature_importances_
feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Guardar la importancia de las variables
feature_importances.to_csv('C:\\Users\\jimen\\Documentos\\codes\\AI\\modelo_prediccion_obesidad\\results\\feature_importances.csv', index=False)

# Visualización de la importancia de las variables
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.title('Importancia de las Variables')
plt.xlabel('Importancia')
plt.ylabel('Variable')
plt.savefig('C:\\Users\\jimen\\Documentos\\codes\\AI\\modelo_prediccion_obesidad\\results\\feature_importances.png')
