import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Cargar el archivo CSV
file_path = 'C:\\Users\\jimen\\Documentos\\codes\\AI\\modelo_prediccion_obesidad\\data\\ObesityDataSet.csv'
data = pd.read_csv(file_path)

# Manejo de valores faltantes
data = data.dropna()

# Codificación de variables categóricas
label_encoders = {}
categorical_vars = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC']

for var in categorical_vars:
    label_encoders[var] = LabelEncoder()
    data[var] = label_encoders[var].fit_transform(data[var])

# Estandarización de variables numéricas
scaler = StandardScaler()
numeric_vars = ['Age', 'Height', 'Weight']
data[numeric_vars] = scaler.fit_transform(data[numeric_vars])

# Guardar los datos preprocesados
data.to_csv('C:\\Users\\jimen\\Documentos\\codes\\AI\\modelo_prediccion_obesidad\\data\\preprocessed_data.csv', index=False)


# Imprimir las primeras filas y los nombres de las columnas
print(data.head())
print(data.columns)