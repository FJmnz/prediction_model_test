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
│   └── preprocessed_data.csv  
│  
├── notebooks/  
│   └── Obesity_Prediction_Analysis.ipynb  
│  
├── src/  
│   ├── data_preprocessing.py  
│   ├── model_training.py  
│   └── evaluation.py  
│  
├── results/  
│   ├── model_performance.txt  
│   ├── feature_importances.csv  
│   ├── feature_importances.png  
│   └── obesity_model.pkl  
│  
└── requirements.txt  


## Resultados
**Model Performance:** model_performance.txt  
**Feature Importances:** feature_importances.csv  
**Feature Importances Visualization:** feature_importances.png  

# Analisis y Conclusiones

### Desempeño del Modelo
El modelo de Random Forest entrenado mostró un buen desempeño en la predicción del nivel de obesidad, con una precisión del 95% y un recall del 94%. La matriz de confusión indica que el modelo es capaz de clasificar correctamente la mayoría de las instancias en las diferentes categorías de obesidad.

### Importancia de las Variables
Las variables importantes para hacer la predicción, fueron factores que dieron un impacto significativo en el nivel de obesidad de una persona.

### Recomendaciones
- **Intervenciones Personalizadas**: Basado en los resultados, se pueden diseñar intervenciones personalizadas que se enfoquen en los factores más influyentes, como la dieta y el ejercicio físico.
- **Políticas de Salud Pública**: Los resultados pueden informar políticas de salud pública que promuevan hábitos alimenticios saludables y actividad física regular para reducir los niveles de obesidad en la población.

