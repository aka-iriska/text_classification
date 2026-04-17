# Text Classification: Сравнение TF-IDF, FastText, BERT

**Задача**: бинарная классификация веб-контента (safe / NSFW) по URL и заголовку страницы.

**Датасет**: [Porn Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/dulinaira/porn-detection-dataset) — распакуй `train.csv`, `test.csv` в `/data`.

---


## Настройка
Перед началом необходимо установить нужные библиотеки. Моя установка происходит на ОС Windows при помощи менеджера пакетов uv.

```shell
uv venv --python 3.12
```
```shell
.\.venv\Scripts\activate.ps1
```
```shell
uv pip install pandas scikit-learn fasttext-wheel nltk jupyterlab sentence-transformers torch matplotlib seaborn pymorphy3 scipy-stubs ipywidgets datasets
```

---

## Структура ноутбука

### 0. EDA ✅
- Размер датасета, распределение классов (несбалансированность!)
- Примеры строк, длина заголовков, топ доменов

---

### 1. Препроцессинг — у каждого метода СВОЙ

Ключевой момент: **TF-IDF, FastText и BERT хотят разный вид данных**.

| | TF-IDF | FastText | BERT |
|---|---|---|---|
| URL + title | ❌ **два отдельных векторизатора** → `hstack` | ✅ объединить строкой | ✅ объединить через `[SEP]` |
| Нижний регистр | ✅ | ✅ | ✅ |
| Убрать спецсимволы | ✅ | ✅ | ✅ |
| Стемминг | ✅ **нужен** | ⚠️ необязателен | ❌ **нельзя** — ломает токенизацию BERT |
| Стоп-слова | ✅ убрать | ⚠️ необязательно | ❌ не трогать |
| Ручная токенизация | ✅ да | ✅ по пробелу | ❌ BERT сам токенизирует |

**Почему BERT нельзя стеммировать?**
BERT обучен на нормальных словах — «играет», «играл», «играть». Если ты передашь «игра» (стемм), он не поймёт контекст. Его сила именно в том, что он понимает морфологию сам.

**URL и title как отдельные фичи:**

- **TF-IDF** — два независимых векторизатора с разной токенизацией, результат объединяется через `hstack`:
  ```python
  from scipy.sparse import hstack

  # URL: разбиваем домен на части (уже есть токенизатор)
  vec_url = TfidfVectorizer(tokenizer=url_tokenizer, ngram_range=(1, 2))
  # title: обычный текст со стеммингом
  vec_title = TfidfVectorizer(tokenizer=title_tokenizer, ngram_range=(1, 2))

  X_train = hstack([vec_url.fit_transform(df_train['url']),
                    vec_title.fit_transform(df_train['title'])])
  X_test  = hstack([vec_url.transform(df_test['url']),
                    vec_title.transform(df_test['title'])])
  ```
  Плюс: модель знает, что `fanserials` — это домен, а не слово из заголовка.

- **FastText** — нет удобного способа принять две фичи; просто объединяем строкой:
  `url + " " + title` (как сейчас).

- **BERT** — объединяем через разделитель, который он понимает:
  `url + " [SEP] " + title`

---

### 2. TF-IDF + классификатор

```
Текст → [кастомный токенизатор со стеммингом] → TfidfVectorizer → вектор → LogisticRegression → предсказание
```

- Используй `TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1,2))`
- Классификатор: `LogisticRegression(class_weight='balanced')` — важно из-за дисбаланса
- Метрики: `classification_report` + ROC-AUC

---

### 3. FastText supervised ✅ (уже есть)

```
Текст → [лёгкий токенизатор] → .txt файл с __label__ → fasttext.train_supervised → предсказание
```

- Oversampling уже реализован
- Метрики: Precision / Recall / F1

---

### 4. BERT (sentence-transformers + классификатор)

Полный fine-tuning BERT — сложно и долго. Проще: использовать готовые эмбеддинги.

```
Текст → [просто str, без стемминга] → SentenceTransformer.encode() → вектор 768d → LogisticRegression → предсказание
```

```python
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

# Многоязычная модель — понимает русский
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

X_train_bert = model.encode(X_train_raw.tolist(), show_progress_bar=True)
X_test_bert  = model.encode(X_test_raw.tolist(),  show_progress_bar=True)

clf = LogisticRegression(class_weight='balanced', max_iter=1000)
clf.fit(X_train_bert, y_train)
```

- Preprocessing: только `url + " " + title`, нижний регистр, убрать лишние символы — **без стемминга**
- Метрики: те же

---

### 5. Итоговое сравнение

Таблица результатов на одной и той же тестовой выборке:

| Метод | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| TF-IDF + LogReg | | | | |
| FastText supervised | | | | |
| BERT + LogReg | | | | |

**Вывод**: какой метод лучше, почему, где разница в качестве.

---

### 6. (Опционально) Визуализация embeddings через t-SNE

Если хочешь красивую картинку для GitHub:

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Берём BERT эмбеддинги (или TF-IDF, уменьшив через TruncatedSVD сначала)
tsne = TSNE(n_components=2, random_state=42)
coords = tsne.fit_transform(X_test_bert[:500])  # 500 точек достаточно

plt.figure(figsize=(8, 6))
plt.scatter(coords[:, 0], coords[:, 1], c=y_test[:500], cmap='RdYlGn', alpha=0.6, s=15)
plt.title('BERT embeddings (t-SNE)')
plt.colorbar(label='0=safe, 1=NSFW')
plt.savefig('tsne_bert.png', dpi=150)
```

Это покажет — насколько хорошо BERT «разделяет» два класса визуально.