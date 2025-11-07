import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("ANÁLISIS DE ESTRÉS EN ESTUDIANTES: MUNDIAL vs UTEQ")
print("Utilizando Redes Neuronales Artificiales")
print("=" * 80)

# ============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS MUNDIALES
# ============================================================================
print("\n[1] Cargando datos mundiales...")
df_mundial = pd.read_csv('/mnt/user-data/uploads/Stress.csv')

print(f"   - Total de registros: {len(df_mundial)}")
print(f"   - Columnas: {len(df_mundial.columns)}")
print(f"\n   Distribución de niveles de estrés:")
print(df_mundial['Stress Label'].value_counts())

# ============================================================================
# 2. GENERACIÓN DE DATOS SIMULADOS PARA LA UTEQ
# ============================================================================
print("\n[2] Generando datos simulados para alumnos UTEQ...")

def generar_datos_uteq(n_samples=1000):
    """
    Genera datos simulados para estudiantes de la UTEQ
    basados en las características de los datos mundiales pero con
    variaciones específicas del contexto mexicano
    """
    np.random.seed(42)
    
    data = {
        '1. Age': np.random.choice(['18-22', '23-26', '27-30'], n_samples, p=[0.75, 0.20, 0.05]),
        '2. Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45]),
        '3. University': ['UTEQ'] * n_samples,
        '4. Department': np.random.choice([
            'Ingeniería en Desarrollo de Software',
            'Ingeniería en Energías Renovables',
            'Ingeniería en Nanotecnología',
            'Licenciatura en Terapia Física',
            'Ingeniería en Biotecnología'
        ], n_samples, p=[0.35, 0.20, 0.15, 0.15, 0.15]),
        '5. Academic Year': np.random.choice([
            'First Year or Equivalent',
            'Second Year or Equivalent',
            'Third Year or Equivalent',
            'Other'
        ], n_samples, p=[0.30, 0.30, 0.30, 0.10]),
        '6. Current CGPA': np.random.choice([
            '2.00 - 2.49',
            '2.50 - 2.99',
            '3.00 - 3.39',
            '3.40 - 3.79',
            '3.80 - 4.00'
        ], n_samples, p=[0.10, 0.25, 0.35, 0.20, 0.10]),
        '7. Did you receive a waiver or scholarship at your university?': 
            np.random.choice(['Yes', 'No'], n_samples, p=[0.25, 0.75])
    }
    
    # Generar respuestas a las 10 preguntas de estrés (escala 0-4)
    # Con tendencia ligeramente más alta en estrés para UTEQ (contexto mexicano)
    for i in range(1, 11):
        col_name = f'{i}. Question'
        # Distribución con sesgo hacia valores medios-altos
        data[col_name] = np.random.choice([0, 1, 2, 3, 4], n_samples, 
                                         p=[0.05, 0.15, 0.30, 0.35, 0.15])
    
    df_uteq = pd.DataFrame(data)
    
    # Calcular Stress Value (suma de las 10 preguntas)
    question_cols = [f'{i}. Question' for i in range(1, 11)]
    df_uteq['Stress Value'] = df_uteq[question_cols].sum(axis=1)
    
    # Asignar Stress Label basado en el valor
    def assign_stress_label(value):
        if value <= 13:
            return 'Low Stress'
        elif value <= 26:
            return 'Moderate Stress'
        else:
            return 'High Perceived Stress'
    
    df_uteq['Stress Label'] = df_uteq['Stress Value'].apply(assign_stress_label)
    
    return df_uteq

df_uteq = generar_datos_uteq(n_samples=1000)

print(f"   - Total de registros UTEQ: {len(df_uteq)}")
print(f"\n   Distribución de niveles de estrés UTEQ:")
print(df_uteq['Stress Label'].value_counts())

# Guardar datos UTEQ
df_uteq.to_csv('/home/claude/stress_uteq.csv', index=False)
print("\n   ✓ Datos UTEQ guardados en: stress_uteq.csv")

# ============================================================================
# 3. PREPROCESAMIENTO DE DATOS
# ============================================================================
print("\n[3] Preprocesando datos para el modelo...")

def preparar_datos(df, nombre):
    """Prepara los datos para entrenamiento"""
    df_prep = df.copy()
    
    # Identificar columnas de preguntas
    question_cols = [col for col in df_prep.columns if 'In a semester' in col or 'Question' in col]
    
    if len(question_cols) == 0:
        # Si no hay columnas específicas, usar todas las numéricas excepto Stress Value
        question_cols = df_prep.select_dtypes(include=[np.number]).columns.tolist()
        if 'Stress Value' in question_cols:
            question_cols.remove('Stress Value')
    
    # Features: respuestas a las preguntas
    X = df_prep[question_cols].values
    
    # Target: Stress Label
    le = LabelEncoder()
    y = le.fit_transform(df_prep['Stress Label'])
    
    print(f"   {nombre}:")
    print(f"   - Features shape: {X.shape}")
    print(f"   - Clases: {le.classes_}")
    
    return X, y, le

X_mundial, y_mundial, le_mundial = preparar_datos(df_mundial, "Datos Mundiales")
X_uteq, y_uteq, le_uteq = preparar_datos(df_uteq, "Datos UTEQ")

# ============================================================================
# 4. DIVISIÓN DE DATOS Y NORMALIZACIÓN
# ============================================================================
print("\n[4] Dividiendo datos en entrenamiento y prueba...")

# Datos Mundiales
X_train_mundial, X_test_mundial, y_train_mundial, y_test_mundial = train_test_split(
    X_mundial, y_mundial, test_size=0.2, random_state=42, stratify=y_mundial
)

# Datos UTEQ
X_train_uteq, X_test_uteq, y_train_uteq, y_test_uteq = train_test_split(
    X_uteq, y_uteq, test_size=0.2, random_state=42, stratify=y_uteq
)

# Normalización
scaler_mundial = StandardScaler()
X_train_mundial_scaled = scaler_mundial.fit_transform(X_train_mundial)
X_test_mundial_scaled = scaler_mundial.transform(X_test_mundial)

scaler_uteq = StandardScaler()
X_train_uteq_scaled = scaler_uteq.fit_transform(X_train_uteq)
X_test_uteq_scaled = scaler_uteq.transform(X_test_uteq)

print(f"   - Conjunto Mundial: Train={len(X_train_mundial)}, Test={len(X_test_mundial)}")
print(f"   - Conjunto UTEQ: Train={len(X_train_uteq)}, Test={len(X_test_uteq)}")

# ============================================================================
# 5. CREACIÓN Y ENTRENAMIENTO DE MODELOS
# ============================================================================
print("\n[5] Entrenando modelos de Red Neuronal...")

# Configuración del modelo
mlp_config = {
    'hidden_layer_sizes': (100, 50, 25),
    'activation': 'relu',
    'solver': 'adam',
    'max_iter': 500,
    'random_state': 42,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 20
}

print(f"\n   Configuración de la Red Neuronal:")
print(f"   - Capas ocultas: {mlp_config['hidden_layer_sizes']}")
print(f"   - Función de activación: {mlp_config['activation']}")
print(f"   - Optimizador: {mlp_config['solver']}")
print(f"   - Máximo de iteraciones: {mlp_config['max_iter']}")

# Modelo para datos Mundiales
print("\n   Entrenando modelo con datos Mundiales...")
mlp_mundial = MLPClassifier(**mlp_config)
mlp_mundial.fit(X_train_mundial_scaled, y_train_mundial)

# Modelo para datos UTEQ
print("   Entrenando modelo con datos UTEQ...")
mlp_uteq = MLPClassifier(**mlp_config)
mlp_uteq.fit(X_train_uteq_scaled, y_train_uteq)

print("\n   ✓ Modelos entrenados exitosamente")

# ============================================================================
# 6. EVALUACIÓN DE MODELOS
# ============================================================================
print("\n[6] Evaluando modelos...")

# Predicciones Mundiales
y_pred_train_mundial = mlp_mundial.predict(X_train_mundial_scaled)
y_pred_test_mundial = mlp_mundial.predict(X_test_mundial_scaled)

# Predicciones UTEQ
y_pred_train_uteq = mlp_uteq.predict(X_train_uteq_scaled)
y_pred_test_uteq = mlp_uteq.predict(X_test_uteq_scaled)

# Métricas Mundiales
acc_train_mundial = accuracy_score(y_train_mundial, y_pred_train_mundial)
acc_test_mundial = accuracy_score(y_test_mundial, y_pred_test_mundial)

print(f"\n   MODELO MUNDIAL:")
print(f"   - Accuracy Entrenamiento: {acc_train_mundial:.4f} ({acc_train_mundial*100:.2f}%)")
print(f"   - Accuracy Prueba: {acc_test_mundial:.4f} ({acc_test_mundial*100:.2f}%)")
print(f"   - Iteraciones: {mlp_mundial.n_iter_}")

# Métricas UTEQ
acc_train_uteq = accuracy_score(y_train_uteq, y_pred_train_uteq)
acc_test_uteq = accuracy_score(y_test_uteq, y_pred_test_uteq)

print(f"\n   MODELO UTEQ:")
print(f"   - Accuracy Entrenamiento: {acc_train_uteq:.4f} ({acc_train_uteq*100:.2f}%)")
print(f"   - Accuracy Prueba: {acc_test_uteq:.4f} ({acc_test_uteq*100:.2f}%)")
print(f"   - Iteraciones: {mlp_uteq.n_iter_}")

# ============================================================================
# 7. CURVAS DE APRENDIZAJE
# ============================================================================
print("\n[7] Generando curvas de aprendizaje...")

def calcular_curvas_aprendizaje(X, y, nombre):
    """Calcula las curvas de aprendizaje para un conjunto de datos"""
    mlp_temp = MLPClassifier(**mlp_config)
    
    train_sizes, train_scores, test_scores = learning_curve(
        mlp_temp, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    print(f"   {nombre}: ✓")
    
    return train_sizes, train_mean, train_std, test_mean, test_std

# Calcular curvas para ambos conjuntos
train_sizes_mundial, train_mean_mundial, train_std_mundial, test_mean_mundial, test_std_mundial = \
    calcular_curvas_aprendizaje(X_train_mundial_scaled, y_train_mundial, "Datos Mundiales")

train_sizes_uteq, train_mean_uteq, train_std_uteq, test_mean_uteq, test_std_uteq = \
    calcular_curvas_aprendizaje(X_train_uteq_scaled, y_train_uteq, "Datos UTEQ")

# ============================================================================
# 8. VISUALIZACIÓN DE RESULTADOS
# ============================================================================
print("\n[8] Generando visualizaciones...")

# Figura 1: Curvas de Aprendizaje - Datos Mundiales
fig1, ax1 = plt.subplots(figsize=(12, 7))

ax1.plot(train_sizes_mundial, train_mean_mundial, 'o-', color='#2E86AB', 
         linewidth=2.5, markersize=8, label='Datos de Entrenamiento')
ax1.fill_between(train_sizes_mundial, 
                  train_mean_mundial - train_std_mundial,
                  train_mean_mundial + train_std_mundial, 
                  alpha=0.2, color='#2E86AB')

ax1.plot(train_sizes_mundial, test_mean_mundial, 'o-', color='#A23B72', 
         linewidth=2.5, markersize=8, label='Datos de Validación')
ax1.fill_between(train_sizes_mundial, 
                  test_mean_mundial - test_std_mundial,
                  test_mean_mundial + test_std_mundial, 
                  alpha=0.2, color='#A23B72')

ax1.set_xlabel('Tamaño del Conjunto de Entrenamiento', fontsize=13, fontweight='bold')
ax1.set_ylabel('Accuracy (Precisión)', fontsize=13, fontweight='bold')
ax1.set_title('Curva de Aprendizaje - Red Neuronal\nDatos de Estudiantes Estresados (Mundial)', 
              fontsize=15, fontweight='bold', pad=20)
ax1.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim([0.5, 1.0])

# Añadir información adicional
info_text = f'Accuracy Final en Prueba: {acc_test_mundial:.4f}\n'
info_text += f'Total de muestras: {len(X_mundial)}\n'
info_text += f'Arquitectura: {mlp_config["hidden_layer_sizes"]}'
ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/curva_aprendizaje_mundial.png', dpi=300, bbox_inches='tight')
print("   ✓ Gráfico 1 guardado: curva_aprendizaje_mundial.png")

# Figura 2: Curvas de Aprendizaje - Datos UTEQ
fig2, ax2 = plt.subplots(figsize=(12, 7))

ax2.plot(train_sizes_uteq, train_mean_uteq, 'o-', color='#F18F01', 
         linewidth=2.5, markersize=8, label='Datos de Entrenamiento')
ax2.fill_between(train_sizes_uteq, 
                  train_mean_uteq - train_std_uteq,
                  train_mean_uteq + train_std_uteq, 
                  alpha=0.2, color='#F18F01')

ax2.plot(train_sizes_uteq, test_mean_uteq, 'o-', color='#06A77D', 
         linewidth=2.5, markersize=8, label='Datos de Validación')
ax2.fill_between(train_sizes_uteq, 
                  test_mean_uteq - test_std_uteq,
                  test_mean_uteq + test_std_uteq, 
                  alpha=0.2, color='#06A77D')

ax2.set_xlabel('Tamaño del Conjunto de Entrenamiento', fontsize=13, fontweight='bold')
ax2.set_ylabel('Accuracy (Precisión)', fontsize=13, fontweight='bold')
ax2.set_title('Curva de Aprendizaje - Red Neuronal\nDatos de Estudiantes Estresados UTEQ', 
              fontsize=15, fontweight='bold', pad=20)
ax2.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim([0.5, 1.0])

# Añadir información adicional
info_text_uteq = f'Accuracy Final en Prueba: {acc_test_uteq:.4f}\n'
info_text_uteq += f'Total de muestras: {len(X_uteq)}\n'
info_text_uteq += f'Arquitectura: {mlp_config["hidden_layer_sizes"]}'
ax2.text(0.02, 0.98, info_text_uteq, transform=ax2.transAxes, 
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/curva_aprendizaje_uteq.png', dpi=300, bbox_inches='tight')
print("   ✓ Gráfico 2 guardado: curva_aprendizaje_uteq.png")

# Figura 3: Comparación directa
fig3, ax3 = plt.subplots(figsize=(14, 8))

# Líneas de validación para ambos conjuntos
ax3.plot(train_sizes_mundial, test_mean_mundial, 'o-', color='#A23B72', 
         linewidth=3, markersize=9, label='Mundial - Validación', alpha=0.8)
ax3.fill_between(train_sizes_mundial, 
                  test_mean_mundial - test_std_mundial,
                  test_mean_mundial + test_std_mundial, 
                  alpha=0.15, color='#A23B72')

ax3.plot(train_sizes_uteq, test_mean_uteq, 's-', color='#06A77D', 
         linewidth=3, markersize=9, label='UTEQ - Validación', alpha=0.8)
ax3.fill_between(train_sizes_uteq, 
                  test_mean_uteq - test_std_uteq,
                  test_mean_uteq + test_std_uteq, 
                  alpha=0.15, color='#06A77D')

ax3.set_xlabel('Tamaño del Conjunto de Entrenamiento', fontsize=14, fontweight='bold')
ax3.set_ylabel('Accuracy (Precisión)', fontsize=14, fontweight='bold')
ax3.set_title('Comparación de Curvas de Aprendizaje\nEstudiantes Estresados: Mundial vs UTEQ', 
              fontsize=16, fontweight='bold', pad=20)
ax3.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_ylim([0.5, 1.0])

# Información comparativa
comp_text = f'MUNDIAL:\n'
comp_text += f'  Accuracy: {acc_test_mundial:.4f}\n'
comp_text += f'  Muestras: {len(X_mundial)}\n\n'
comp_text += f'UTEQ:\n'
comp_text += f'  Accuracy: {acc_test_uteq:.4f}\n'
comp_text += f'  Muestras: {len(X_uteq)}\n\n'
comp_text += f'Arquitectura:\n  {mlp_config["hidden_layer_sizes"]}'

ax3.text(0.02, 0.98, comp_text, transform=ax3.transAxes, 
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/curva_comparacion_mundial_vs_uteq.png', dpi=300, bbox_inches='tight')
print("   ✓ Gráfico 3 guardado: curva_comparacion_mundial_vs_uteq.png")

# ============================================================================
# 9. REPORTES ADICIONALES
# ============================================================================
print("\n[9] Generando reportes de clasificación...")

print("\n" + "="*60)
print("REPORTE DE CLASIFICACIÓN - DATOS MUNDIALES")
print("="*60)
print(classification_report(y_test_mundial, y_pred_test_mundial, 
                          target_names=le_mundial.classes_))

print("\n" + "="*60)
print("REPORTE DE CLASIFICACIÓN - DATOS UTEQ")
print("="*60)
print(classification_report(y_test_uteq, y_pred_test_uteq, 
                          target_names=le_uteq.classes_))

# Matriz de confusión
fig4, (ax4_1, ax4_2) = plt.subplots(1, 2, figsize=(16, 6))

# Matriz Mundial
cm_mundial = confusion_matrix(y_test_mundial, y_pred_test_mundial)
sns.heatmap(cm_mundial, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_mundial.classes_,
            yticklabels=le_mundial.classes_, ax=ax4_1, cbar_kws={'label': 'Frecuencia'})
ax4_1.set_xlabel('Predicción', fontsize=12, fontweight='bold')
ax4_1.set_ylabel('Valor Real', fontsize=12, fontweight='bold')
ax4_1.set_title('Matriz de Confusión - Datos Mundiales', fontsize=13, fontweight='bold')

# Matriz UTEQ
cm_uteq = confusion_matrix(y_test_uteq, y_pred_test_uteq)
sns.heatmap(cm_uteq, annot=True, fmt='d', cmap='Greens', 
            xticklabels=le_uteq.classes_,
            yticklabels=le_uteq.classes_, ax=ax4_2, cbar_kws={'label': 'Frecuencia'})
ax4_2.set_xlabel('Predicción', fontsize=12, fontweight='bold')
ax4_2.set_ylabel('Valor Real', fontsize=12, fontweight='bold')
ax4_2.set_title('Matriz de Confusión - Datos UTEQ', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/matrices_confusion.png', dpi=300, bbox_inches='tight')
print("\n   ✓ Gráfico 4 guardado: matrices_confusion.png")

# ============================================================================
# 10. RESUMEN FINAL
# ============================================================================
print("\n" + "="*80)
print("RESUMEN FINAL DEL ANÁLISIS")
print("="*80)
print(f"\n1. DATOS PROCESADOS:")
print(f"   - Registros Mundiales: {len(df_mundial)}")
print(f"   - Registros UTEQ: {len(df_uteq)}")

print(f"\n2. RENDIMIENTO DE LOS MODELOS:")
print(f"   - Modelo Mundial:")
print(f"     • Accuracy Entrenamiento: {acc_train_mundial*100:.2f}%")
print(f"     • Accuracy Prueba: {acc_test_mundial*100:.2f}%")
print(f"   - Modelo UTEQ:")
print(f"     • Accuracy Entrenamiento: {acc_train_uteq*100:.2f}%")
print(f"     • Accuracy Prueba: {acc_test_uteq*100:.2f}%")

print(f"\n3. ARQUITECTURA DE LA RED NEURONAL:")
print(f"   - Capas ocultas: {mlp_config['hidden_layer_sizes']}")
print(f"   - Total de parámetros aproximados: ~{sum(mlp_config['hidden_layer_sizes'])}")
print(f"   - Función de activación: {mlp_config['activation']}")
print(f"   - Optimizador: {mlp_config['solver']}")

print(f"\n4. ARCHIVOS GENERADOS:")
print(f"   ✓ stress_uteq.csv - Datos simulados UTEQ")
print(f"   ✓ curva_aprendizaje_mundial.png")
print(f"   ✓ curva_aprendizaje_uteq.png")
print(f"   ✓ curva_comparacion_mundial_vs_uteq.png")
print(f"   ✓ matrices_confusion.png")

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO EXITOSAMENTE ✓")
print("="*80)

plt.close('all')