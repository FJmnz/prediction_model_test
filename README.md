# prediction_model_test

## Descripción
Esta actividad tiene como objetivo desarrollar un modelo predictivo para nivel de obesidad basado en hábitos alimenticios y condición física.

## Hipótesis
1. El consumo de alimentos calóricos tiene una correlación directa con el sobrepeso y la obesidad.
2. La actividad física regular reduce el riesgo de obesidad.
3. La ingesta de vegetales y frutas está inversamente relacionada con los niveles de obesidad.
4. El consumo de agua adecuado ayuda a mantener un peso saludable.



## Estrucura del proyecto
prediction_model_test/  
│  
├── data/  
│   ├── ObesityDataSet.csv  
│   └── resampled_data.csv
│  
├── notebooks/  
│   └── Obesity_Prediction_Analysis.ipynb  
│  
├── src/  
│   ├── data_preprocessing.py  
│   └── model_training.py  
│   
│  
├── results/   
│   ├── cv_performance.txt
│   ├── label_encoders.pkl 
│   ├── LogisticRegression_model.pkl
│   ├── model_performance.txt
│   ├── scaler.pkl
│   └── SVM_model.pkl 
│  
└── requirements.txt  



# Analisis y Conclusiones en el Desempeño del Modelo

#### Modelo de Regresión Logística
- **Precisión Global**: 90.24%
- **Precisión por Clase**:
  - **Insufficient_Weight**: 88.31%
  - **Normal_Weight**: 94.74%
  - **Obesity_Type_I**: 93.55%
  - **Obesity_Type_II**: 89.74%
  - **Obesity_Type_III**: 100%
  - **Overweight_Level_I**: 79.45%
  - **Overweight_Level_II**: 87.32%
- **Recall Global**: 90.32%
- **F1-Score Global**: 90.03%
- **Validación Cruzada (Mean CV Score)**: 89.79%

#### Modelo de SVM
- **Precisión Global**: 92.68%
- **Precisión por Clase**:
  - **Insufficient_Weight**: 95.71%
  - **Normal_Weight**: 92.86%
  - **Obesity_Type_I**: 96.83%
  - **Obesity_Type_II**: 93.24%
  - **Obesity_Type_III**: 100%
  - **Overweight_Level_I**: 82.09%
  - **Overweight_Level_II**: 87.84%
- **Recall Global**: 92.61%
- **F1-Score Global**: 92.58%
- **Validación Cruzada (Mean CV Score)**: 92.23%

### Comparación de Modelos
El modelo de SVM mostró un mejor desempeño general en comparación con el modelo de Regresión Logística, con una precisión global del 92.68% y un F1-score global del 92.58%. La matriz de confusión indica que el modelo de SVM es capaz de clasificar correctamente la mayoría de las instancias en las diferentes categorías de obesidad.

### Importancia de las Variables
Las variables más importantes para predecir el nivel de obesidad fueron:
1. **Weight**: 0.3158
2. **Age**: 0.1022
3. **Height**: 0.1019
4. **FCVC**: 0.0937
5. **NCP**: 0.0572
6. **Gender**: 0.0551
7. **TUE**: 0.0549
8. **CH2O**: 0.0498
9. **FAF**: 0.0457
10. **CALC**: 0.0335

Esto sugiere que estos factores tienen un impacto significativo en el nivel de obesidad de una persona.

### Recomendaciones
- **Intervenciones Personalizadas**: Basado en los resultados, se pueden diseñar intervenciones personalizadas que se enfoquen en los factores más influyentes, como la dieta y el ejercicio físico.
- **Políticas de Salud Pública**: Los resultados pueden informar políticas de salud pública que promuevan hábitos alimenticios saludables y actividad física regular para reducir los niveles de obesidad en la población.

