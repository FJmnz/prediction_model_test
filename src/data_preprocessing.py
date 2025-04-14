import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Cargar el archivo CSV
file_path = 'C:\\Users\\jimen\\Documentos\\codes\\AI\\modelo_prediccion_obesidad\\data\\ObesityDataSet.csv'
data = pd.read_csv(file_path)

# Manejo de valores faltantes
data = data.dropna()

# Codificación de variables categóricas
label_encoders = {}
categorical_vars = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'NObeyesdad']

for var in categorical_vars:
    label_encoders[var] = LabelEncoder()
    data[var] = label_encoders[var].fit_transform(data[var])

# Estandarización de variables numéricas
scaler = StandardScaler()
numeric_vars = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O']
data[numeric_vars] = scaler.fit_transform(data[numeric_vars])

# Selección de características y variable objetivo
features = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'CALC']
X = data[features]
y = data['NObeyesdad']

# Aplicar SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Guardar los datos balanceados
resampled_data = pd.concat([X_resampled, y_resampled], axis=1)
resampled_data.to_csv('C:\\Users\\jimen\\Documentos\\codes\\AI\\modelo_prediccion_obesidad\\data\\resampled_data.csv', index=False)

# Guardar los codificadores y el escalador
joblib.dump(label_encoders, 'C:\\Users\\jimen\\Documentos\\codes\\AI\\modelo_prediccion_obesidad\\results\\label_encoders.pkl')
joblib.dump(scaler, 'C:\\Users\\jimen\\Documentos\\codes\\AI\\modelo_prediccion_obesidad\\results\\scaler.pkl')
