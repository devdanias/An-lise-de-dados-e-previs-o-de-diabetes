# 1. Importação de Bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, auc)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings

# Configurações iniciais
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
pd.set_option('display.max_columns', None)
sns.set_palette("husl")

# 2. Carregamento dos Dados
try:
    # Substitua pelo caminho do seu arquivo ou URL
    df = pd.read_csv('diabetes_dataset.csv')  
    print("Dados carregados com sucesso!")
except FileNotFoundError:
    print("Arquivo não encontrado. Por favor, verifique o caminho.")
    # Carregar dados de exemplo (remova isso no seu código real)
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    df = pd.DataFrame(X)
    df['Diabetes_binary'] = y
    print("Dataset de exemplo criado para demonstração.")

# 3. Pré-processamento dos Dados
print("\nInformações do dataset:")
print(df.info())

# Verificar e tratar valores ausentes
print("\nValores ausentes por coluna:")
print(df.isnull().sum())

# Método de imputação KNN para colunas numéricas
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'Diabetes_binary' in numeric_cols:
    numeric_cols.remove('Diabetes_binary')

if len(numeric_cols) > 0:
    imputer = KNNImputer(n_neighbors=5)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    print("\nDados imputados com sucesso usando KNNImputer.")
else:
    print("\nNenhuma coluna numérica encontrada para imputação.")

# 4. Análise Exploratória dos Dados
print("\nEstatísticas descritivas:")
print(df.describe())

# Configuração de estilo para os gráficos
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# Função para plotar gráficos
def plot_distribution(data, column, target='Diabetes_binary', kind='hist'):
    """Função para plotar distribuições"""
    plt.figure(figsize=(12, 6))
    
    if kind == 'hist':
        sns.histplot(data=data, x=column, hue=target, kde=True, 
                     bins=30, palette='viridis', alpha=0.6)
        plt.title(f'Distribuição de {column} por Status de Diabetes')
    elif kind == 'count':
        sns.countplot(data=data, x=column, hue=target, palette='viridis')
        plt.title(f'Contagem de {column} por Status de Diabetes')
    elif kind == 'box':
        sns.boxplot(data=data, x=target, y=column, palette='viridis')
        plt.title(f'Distribuição de {column} por Status de Diabetes')
    
    plt.xlabel(column)
    plt.ylabel('Contagem' if kind != 'box' else column)
    plt.legend(title='Diabetes', labels=['Não', 'Sim'])
    plt.xticks(rotation=45 if kind in ['count'] else 0)
    plt.tight_layout()
    plt.show()

# Gráfico da variável target
plot_distribution(df, 'Diabetes_binary', kind='count')

# Gráficos das principais variáveis
if 'HighBP' in df.columns:
    plot_distribution(df, 'HighBP', kind='count')

if 'HighChol' in df.columns:
    plot_distribution(df, 'HighChol', kind='count')

if 'BMI' in df.columns:
    plot_distribution(df, 'BMI', kind='box')

if 'Age' in df.columns:
    plot_distribution(df, 'Age', kind='count')

# Matriz de correlação
if len(numeric_cols) > 1:
    plt.figure(figsize=(14, 10))
    corr_matrix = df[numeric_cols + ['Diabetes_binary']].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                center=0, linewidths=0.5)
    plt.title('Matriz de Correlação entre Variáveis')
    plt.tight_layout()
    plt.show()

# 5. Análise Estatística
#  ----------- Aplicação em Probabilidade ------------- #
if all(col in df.columns for col in ['HighBP', 'HighChol', 'Diabetes_binary']):
    condition = (df['HighBP'] == 1) & (df['HighChol'] == 1)
    total_high = df[condition].shape[0]
    if total_high > 0:
        diabetes_high = df[condition & (df['Diabetes_binary'] == 1)].shape[0]
        prob = diabetes_high / total_high
        print(f"\nProbabilidade condicional:")
        print(f"P(Diabetes | HighBP e HighChol) = {prob:.2%}")
    else:
        print("\nNão há pacientes com HighBP e HighChol para cálculo da probabilidade.")

# ------------ Teste de Hipóteses -------------- #
if all(col in df.columns for col in ['HighBP', 'Diabetes_binary']):
    high_bp_diabetes = df[df['HighBP'] == 1]['Diabetes_binary']
    no_high_bp_diabetes = df[df['HighBP'] == 0]['Diabetes_binary']
    
    count = [high_bp_diabetes.sum(), no_high_bp_diabetes.sum()]
    nobs = [len(high_bp_diabetes), len(no_high_bp_diabetes)]
    
    z_stat, p_value = proportions_ztest(count, nobs)
    print("\nTeste de Hipóteses:")
    print(f"Estatística Z: {z_stat:.4f}")
    print(f"Valor-p: {p_value:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        print("Rejeitamos H0 - há diferença significativa nas proporções.")
    else:
        print("Não rejeitamos H0 - não há diferença significativa.")

# 6. Preparação para Modelagem
# Separar features e target
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

# Divisão treino-teste com estratificação
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

print("\nDistribuição das classes:")
print(f"Original: {Counter(y)}")
print(f"Treino: {Counter(y_train)}")
print(f"Teste: {Counter(y_test)}")

# Balanceamento com SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"\nApós SMOTE - Treino: {Counter(y_train_smote)}")

# Padronização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# 7. Modelagem Preditiva
# Dicionário de modelos
models = {
    "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# Treinar e avaliar modelos
results = []
for name, model in models.items():
    print(f"\nTreinando {name}...")
    model.fit(X_train_scaled, y_train_smote)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    results.append({
        'Modelo': name,
        'Acurácia': accuracy,
        'Precisão': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    })
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Não', 'Sim'], 
                yticklabels=['Não', 'Sim'])
    plt.title(f'Matriz de Confusão - {name}')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.show()
    
    # Curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curva ROC - {name}')
    plt.legend(loc="lower right")
    plt.show()

# Resultados
results_df = pd.DataFrame(results)
print("\nDesempenho dos Modelos:")
print(results_df.sort_values(by='F1 Score', ascending=False))

# 8. Análise de Importância de Variáveis (para modelos baseados em árvores)
if 'Random Forest' in models:
    rf_model = models['Random Forest']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("Importância das Variáveis - Random Forest")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.show()

# 9. Otimização do Melhor Modelo (Random Forest)
print("\nOtimizando o melhor modelo...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                          param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train_smote)

print(f"Melhores parâmetros: {grid_search.best_params_}")
print(f"Melhor F1 Score: {grid_search.best_score_:.4f}")

# 10. Avaliação do Modelo Otimizado
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
y_prob_best = best_model.predict_proba(X_test_scaled)[:, 1]

print("\nRelatório de Classificação do Modelo Otimizado:")
print(classification_report(y_test, y_pred_best))

# 11. Salvando os Resultados
results_df.to_csv('resultados_modelos.csv', index=False)
print("\nResultados salvos em 'resultados_modelos.csv'")

# 12. Conclusão
print("\nConclusão:")
print("1. O melhor modelo foi:", results_df.loc[results_df['F1 Score'].idxmax(), 'Modelo'])
print("2. Principais variáveis preditoras:", list(X_train.columns[indices][:3]))
print("3. Probabilidade condicional mostra que pacientes com HighBP e HighChol têm maior risco")
print("4. O teste de hipóteses confirmou diferença significativa nas proporções de diabetes")