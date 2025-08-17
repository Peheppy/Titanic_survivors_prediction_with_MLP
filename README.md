# Predição de Sobrevivência no Titanic com MLP

## Redes Neurais 2025.1  

Este projeto realiza a predição da sobrevivência dos passageiros do Titanic utilizando **MLPs (Multi-Layer Perceptrons)** implementadas em **PyTorch**, com experimentos comparativos entre diferentes arquiteturas, funções de ativação e otimizadores.

---

## 📂 Estrutura do Projeto

- `train.csv` : Conjunto de dados de treinamento com rótulos de sobrevivência (`Survived`).  
- `test.csv` : Conjunto de dados de teste sem rótulos de sobrevivência.  
- `results/` : Pasta gerada pelo projeto contendo:
  - Modelos treinados (`*.pth`)  
  - Históricos de treinamento (`*_history.json`)  
  - Comparativo final dos modelos (`model_comparison.csv`)  

- `notebook.ipynb` : Notebook principal com pré-processamento, treino, avaliação e visualização dos experimentos.  

---

## 🛠 Tecnologias Utilizadas

- Python 3.x  
- Pandas, Numpy  
- Matplotlib, Seaborn  
- Scikit-learn (pré-processamento e métricas)  
- PyTorch (MLP, treino e avaliação)  
- tqdm (barra de progresso)

---

## 🔍 Pipeline do Projeto

1. **Pré-processamento de dados**  
   - Tratamento de valores ausentes (`Age`, `Fare`, `Embarked`)  
   - Criação de novas features (`FamilySize`, `IsAlone`, `Title`)  
   - Conversão de variáveis categóricas e remoção de colunas irrelevantes (`PassengerId`, `Name`, `Ticket`, `Cabin`)  

2. **Divisão em conjuntos**  
   - Separação de treino, validação e teste  

3. **Criação de DataLoaders**  
   - Preparação dos datasets em batches para treino e validação  

4. **Definição e treinamento dos modelos MLP**  
   - Arquiteturas comparativas: `Baseline`, `DeeperNet` e `WideNet`  
   - Funções de ativação: ReLU, LeakyReLU, Tanh  
   - Otimizadores: Adam, RMSprop  
   - Dropout e LayerNorm para regularização  

5. **Monitoramento de métricas**  
   - Loss (CrossEntropy) e acurácia para treino e validação  
   - Early stopping para evitar overfitting  
   - Scheduler de learning rate (`ReduceLROnPlateau`)  

6. **Avaliação final e comparação de modelos**  
   - Comparativo de loss e acurácia  
   - Evolução do learning rate  
   - Salvamento de modelos e históricos  

---

## ⚙ Configuração dos Modelos

Exemplo de configuração de um modelo:

```python
{
    "name": "Baseline",
    "input_size": 9,
    "num_classes": 2,
    "layer_sizes": [64, 32],
    "activation_fn": nn.ReLU(),
    "dropout_rate": 0.3,
    "learning_rate": 1e-3,
    "optimizer": torch.optim.Adam,
    "n_epochs": 100,
    "bs_train": 32,
    "bs_val_test": 64,
    "loss_fn": nn.CrossEntropyLoss(),
    "patience": 10
}
