# Análise de Dados e Previsão de Diabetes em Pacientes

**Autores**: Daniele de Almeida Silva
**Data**: 14/07/2025

## Objetivo do Trabalho

Este projeto tem como objetivo analisar um conjunto de dados de pacientes com e sem diabetes, identificar os principais fatores de risco e desenvolver um modelo preditivo para classificação de diabetes. As análises incluirão:

1.  Análise exploratória das variáveis do dataset
2.  Tratamento de dados ausentes com métodos avançados
3.  Aplicações de probabilidade e inferência estatística
4.  Desenvolvimento e comparação de modelos de machine learning
5.  Seleção do melhor modelo com base em métricas adequadas

O dataset utilizado é o "Diabetes Health Indicators Dataset" disponível no Kaggle, que contém informações sobre saúde, hábitos e condições médicas de pacientes.

## 1. Importação de Bibliotecas e Carregamento dos Dados

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from imblearn.over_sampling import SMOTE
from collections import Counter

# Configuração de estilo para os gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
palette = sns.color_palette("husl", 8)

# Carregar os dados
url = "https://raw.githubusercontent.com/yourusername/yourrepository/main/diabetes_dataset.csv"
df = pd.read_csv(url)

# Visualizar as primeiras linhas
print(df.head())
print("\nInformações do dataset:")
print(df.info())
```

## 2. Descrição do Dataset e Variáveis

O dataset contém as seguintes variáveis:

1.  **Diabetes_binary**: Variável alvo (0 = não diabético, 1 = diabético)
2.  **HighBP**: Pressão arterial alta (0 = não, 1 = sim)
3.  **HighChol**: Colesterol alto (0 = não, 1 = sim)
4.  **CholCheck**: Verificação de colesterol nos últimos 5 anos (0 = não, 1 = sim)
5.  **BMI**: Índice de Massa Corporal
6.  **Smoker**: Fumou pelo menos 100 cigarros na vida (0 = não, 1 = sim)
7.  **Stroke**: Já teve um derrame (0 = não, 1 = sim)
8.  **HeartDiseaseorAttack**: Doença coronariana ou infarto (0 = não, 1 = sim)
9.  **PhysActivity**: Atividade física nos últimos 30 dias (0 = não, 1 = sim)
10. **Fruits**: Consome frutas regularmente (0 = não, 1 = sim)
11. **Veggies**: Consome vegetais regularmente (0 = não, 1 = sim)
12. **HvyAlcoholConsump**: Consumo pesado de álcool (0 = não, 1 = sim)
13. **AnyHealthcare**: Tem algum plano de saúde (0 = não, 1 = sim)
14. **NoDocbcCost**: Não foi ao médico nos últimos 12 meses por causa do custo (0 = não, 1 = sim)
15. **GenHlth**: Autoavaliação da saúde (1 = excelente, 5 = ruim)
16. **MentHlth**: Dias com saúde mental ruim nos últimos 30 dias
17. **PhysHlth**: Dias com saúde física ruim nos últimos 30 dias
18. **DiffWalk**: Dificuldade para caminhar ou subir escadas (0 = não, 1 = sim)
19. **Sex**: Sexo (0 = feminino, 1 = masculino)
20. **Age**: Faixa etária (1 = 18-24, ..., 13 = 80+)
21. **Education**: Nível educacional (1 = nunca estudou, ..., 6 = ensino superior)
22. **Income**: Renda (1 = <10k, ..., 8 = >75k)

**Objetivo final**: Desenvolver um modelo preditivo para identificar pacientes com maior risco de diabetes com base em seus indicadores de saúde.

## 3. Pré-processamento e Tratamento de Dados Ausentes

```python
# Verificar dados ausentes
print("Dados ausentes por coluna:")
print(df.isnull().sum())

# Método de imputação escolhido: KNN Imputer (método avançado)
# Justificativa: O KNN Imputer considera a similaridade entre observações para imputar valores ausentes,
# preservando melhor a estrutura dos dados do que métodos simples como média ou mediana.

# Preparar os dados para imputação
cols_to_impute = ['BMI', 'MentHlth', 'PhysHlth']
df_to_impute = df[cols_to_impute].copy()

# Aplicar KNN Imputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df_to_impute)

# Atualizar o dataframe com os valores imputados
df_imputed = pd.DataFrame(df_imputed, columns=cols_to_impute)
for col in cols_to_impute:
    df[col] = df_imputed[col]

# Verificar se ainda há dados ausentes
print("\nDados ausentes após imputação:")
print(df.isnull().sum())
```

## 4. Análise Descritiva das Variáveis

```python
# Estatísticas descritivas
print("Estatísticas descritivas das variáveis numéricas:")
print(df.describe())

# Distribuição da variável alvo
plt.figure(figsize=(8, 6))
sns.countplot(x='Diabetes_binary', data=df, palette=palette)
plt.title('Distribuição da Variável Alvo (Diabetes)')
plt.xlabel('Diabetes (0 = Não, 1 = Sim)')
plt.ylabel('Contagem')
plt.show()

# Proporção de casos de diabetes
diabetes_prop = df['Diabetes_binary'].mean()
print(f"\nProporção de pacientes com diabetes: {diabetes_prop:.2%}")
```

### Análise das Principais Variáveis

```python
# Relação entre diabetes e pressão alta
plt.figure(figsize=(10, 6))
sns.countplot(x='HighBP', hue='Diabetes_binary', data=df, palette=palette)
plt.title('Relação entre Pressão Alta e Diabetes')
plt.xlabel('Pressão Alta (0 = Não, 1 = Sim)')
plt.ylabel('Contagem')
plt.legend(title='Diabetes', labels=['Não', 'Sim'])
plt.show()

# Relação entre diabetes e colesterol alto
plt.figure(figsize=(10, 6))
sns.countplot(x='HighChol', hue='Diabetes_binary', data=df, palette=palette)
plt.title('Relação entre Colesterol Alto e Diabetes')
plt.xlabel('Colesterol Alto (0 = Não, 1 = Sim)')
plt.ylabel('Contagem')
plt.legend(title='Diabetes', labels=['Não', 'Sim'])
plt.show()

# Distribuição de BMI por status de diabetes
plt.figure(figsize=(10, 6))
sns.boxplot(x='Diabetes_binary', y='BMI', data=df, palette=palette)
plt.title('Distribuição de IMC por Status de Diabetes')
plt.xlabel('Diabetes (0 = Não, 1 = Sim)')
plt.ylabel('Índice de Massa Corporal (BMI)')
plt.show()

# Correlação entre variáveis
plt.figure(figsize=(14, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Matriz de Correlação entre Variáveis')
plt.show()
```

## 5. Aplicação em Probabilidade

**Problema**: Qual a probabilidade de um paciente ter diabetes dado que tem pressão alta e colesterol alto?

```python
# Probabilidade condicional
# P(Diabetes | HighBP e HighChol)

# Filtar pacientes com HighBP e HighChol
condition = (df['HighBP'] == 1) & (df['HighChol'] == 1)
total_high = df[condition].shape[0]
diabetes_high = df[condition & (df['Diabetes_binary'] == 1)].shape[0]
prob = diabetes_high / total_high
print(f"Probabilidade de diabetes dado pressão alta e colesterol alto: {prob:.2%}")

# Comparar com probabilidade geral
print(f"Probabilidade geral de diabetes: {diabetes_prop:.2%}")
```

## 6. Aplicação em Inferência Estatística

**Teste de Hipóteses**: Verificar se a proporção de diabetes é significativamente maior em pacientes com pressão alta.

```python
# Separar os grupos
high_bp_diabetes = df[df['HighBP'] == 1]['Diabetes_binary']
no_high_bp_diabetes = df[df['HighBP'] == 0]['Diabetes_binary']

# Contagem de sucessos (diabetes) e tamanho das amostras
count = [high_bp_diabetes.sum(), no_high_bp_diabetes.sum()]
nobs = [len(high_bp_diabetes), len(no_high_bp_diabetes)]

# Teste Z para proporções
z_stat, p_value = proportions_ztest(count, nobs)
print(f"Estatística Z: {z_stat:.4f}")
print(f"Valor-p: {p_value:.4f}")

# Interpretação
alpha = 0.05
if p_value < alpha:
    print("Rejeitamos a hipótese nula - há diferença significativa nas proporções de diabetes entre os grupos.")
else:
    print("Não rejeitamos a hipótese nula - não há diferença significativa.")
```

## 7. Técnica de Amostragem

**Método escolhido**: Amostragem estratificada
**Justificativa**: Como a proporção de pacientes com diabetes é baixa (desequilíbrio de classes), a amostragem estratificada garante que ambas as classes (diabéticos e não-diabéticos) estejam adequadamente representadas na amostra de treino e teste.

```python
# Dividir os dados em treino e teste com estratificação
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                     stratify=y,
                                                     random_state=42)

print("\nDistribuição original:")
print(y.value_counts(normalize=True))
print("\nDistribuição no treino:")
print(y_train.value_counts(normalize=True))
print("\nDistribuição no teste:")
print(y_test.value_counts(normalize=True))
```

## 8. Modelagem Preditiva

```python
# Balanceamento de classes com SMOTE
print("Contagem de classes antes do SMOTE:", Counter(y_train))
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("Contagem de classes após SMOTE:", Counter(y_train_smote))

# Padronização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)
```

### Treinamento e Comparação de Modelos

```python
# Inicializar modelos
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# Treinar e avaliar modelos
results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train_smote)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    })

# Resultados
results_df = pd.DataFrame(results)
print("\nDesempenho dos Modelos:")
print(results_df.sort_values(by='F1 Score', ascending=False))
```
```

