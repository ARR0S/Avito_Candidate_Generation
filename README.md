# Candidate Generation для поиска услуг Авито

Решение тестового задания на кандидатогенерацию для поиска услуг.

**Итоговый benchmark Recall@50: `0.818206`.**

## Постановка задачи

Для каждого поискового запроса нужно вернуть не более 50 объявлений из заданного корпуса. На этом этапе особенно важно не потерять релевантное объявление: если оно не попало в candidate pool, последующий ranking уже не сможет его вернуть.

Метрика — **Recall@50**:

$$
Recall@50 = \frac{1}{N}\sum_q \frac{|Top50_q \cap Relevant_q|}{|Relevant_q|}
$$

Порядок элементов внутри финальных 50 кандидатов на значение метрики не влияет.

## Кратко о решении

Итоговый pipeline объединяет lexical, semantic и географические сигналы, после чего обучаемый ранкер выбирает 50 кандидатов из объединённого пула.

```text
                         query
                           |
          +----------------+----------------+
          |                |                |
   geo-aware BM25    global semantic    geo-semantic
      top-500           top-1000       top-200 / top-100
          |                |                |
          +----------------+----------------+
                           |
                    union candidates
                    ~1500 на запрос
                           |
                    LightGBM ranker
                           |
                         top-50
```

## Данные

Используются три файла:

- `train.parquet` — 497 673 положительных query–item пары;
- `benchmark_queries.parquet` — 2 452 запроса без разметки;
- `benchmark_items.parquet` — 189 212 объявлений.

Основные признаки запроса: текст, локация, категория и текст фильтров. У объявления доступны title, description, параметры, категория, microcat, локация, rating, price и несколько служебных признаков.

## Валидация

Random split строк здесь приводит к утечке: одни и те же тексты запросов много раз повторяются в train.

Поэтому использован **grouped holdout по нормализованному `search_query`**:

- lowercase;
- `ё → е`;
- нормализация пробелов;
- validation-тексты полностью исключаются из fit.

Для сравнения решений был зафиксирован primary-срез из 3 000 новых текстов. Финальный ранкер дополнительно проверялся на отдельном fresh confirmation-срезе ещё из 3 000 текстов, которые не использовались при обучении.

## 1. Полевой BM25

Начальный baseline — интерпретируемый lexical retrieval. Поля объявления получают разные веса:

| Поле | Вес |
|---|---:|
| `item_title_raw` | 4.0 |
| `item_infm_params_text` | 1.0 |
| `item_description_raw` | 0.4 |

Параметры BM25: `k1=1.2`, `b=0.75`. Текст фильтров запроса добавляется с меньшим весом.

**Primary Recall@50: `0.320390`.**

## 2. География

Error analysis показал, что у большого числа пропущенных positives локация совпадает с запросом, но такие объявления вытесняются более сильными лексическими совпадениями из других локаций.

При этом hard filter по location использовать нельзя: в train есть систематические релевантные пары с разными `search_location_id` и `item_location_id`.

Поэтому география учитывается мягко:

- exact-location умножает score на `×3`;
- устойчивые переходы между location IDs оцениваются по train со сглаживанием;
- transition bonus ограничен и применяется только для поддержанных связей;
- географические сигналы применяются **до отсечения top-k**.

После переноса географии внутрь candidate generation:

- Recall@50: `0.536322`;
- Recall@500: `0.702944`.

## 3. Semantic retrieval

BM25 ограничен буквальными совпадениями. Для перефразирований и синонимов добавлен `intfloat/multilingual-e5-small` — компактный multilingual bi-encoder с 384-мерными embeddings.

Текст объявления:

```text
passage: <title>. <первые 600 символов params>
```

Текст запроса:

```text
query: <search_query>. <первые 200 символов filters>
```

Используются три semantic-канала:

1. global semantic top-1000 по всему корпусу;
2. exact-location semantic top-200;
3. transition-location semantic top-100 внутри до трёх наиболее надёжных связанных локаций.

Глобальный semantic-канал сохраняется всегда, поэтому geography-conditioned retrieval не является hard filter.

## 4. Candidate pool

Финальный pool:

```text
geo-BM25-500
∪ global-semantic-1000
∪ exact-location-semantic-200
∪ transition-location-semantic-100
```

| Срез | Candidate recall |
|---|---:|
| Primary | `0.915611` |
| Fresh confirmation | `0.925850` |

После этого главным bottleneck стал уже не поиск новых кандидатов, а выбор 50 лучших из примерно 1.5 тыс. объектов.

## 5. Supervised reranking

Для финального выбора используется `LightGBM LGBMRanker` с `lambdarank`.

На каждую пару query–candidate строятся 39 признаков:

- BM25 score / relative score / rank / reciprocal rank;
- semantic cosine и ranks разных semantic-каналов;
- exact-location и transition signal;
- число retrieval-каналов, в которых найден кандидат;
- совпадение категории;
- token overlap запроса с title, params и description;
- overlap фильтров с params;
- длины текстов;
- rating, число reviews, price и contact flags.

`query_id` и `item_id` как признаки модели не используются.

## 6. Hard-negative mining

Первая версия ranker обучалась на сильных retrieval-кандидатах. Затем frozen v1 использовался для поиска **hard negatives** — нерелевантных кандидатов, которым сама модель ошибочно присваивает высокий score.

Для v2 на каждый training query используются:

- все positives, найденные в candidate pool;
- 200 negatives с максимальным score v1;
- RRF-дополнение сильными retrieval-кандидатами до 250 строк на query.

Архитектура и 39 признаков не менялись: улучшение получилось именно за счёт более сложных обучающих примеров.

## Результаты

### Локальная проверка

| Этап | Primary Recall@50 | Fresh Recall@50 |
|---|---:|---:|
| BM25 baseline | `0.320390` | — |
| Geography-aware BM25 | `0.536322` | `0.534521` |
| Reranker v1 | `0.773538` | `0.788533` |
| Reranker v2 + hard negatives | **`0.788886`** | **`0.813367`** |

### Benchmark

| Submission | Recall@50 |
|---|---:|
| Geography-aware lexical pipeline | `0.519964` |
| **Hybrid retrieval + LightGBM v2** | **`0.818206`** |

Fresh validation (`0.813367`) и hidden benchmark (`0.818206`) отличаются меньше чем на 0.5 п.п., поэтому выбранная grouped validation хорошо отражала качество финального решения.

## Что оказалось наиболее важным

1. **Географию нужно применять до candidate cutoff.** Поздний rerank уже не помогает объявлениям, которые не попали в пул.
2. **Semantic retrieval дополняет, а не заменяет BM25.** Каналы хорошо находят разные типы positives.
3. **Semantic search внутри локации** существенно повышает coverage по сравнению с одним global semantic top-k.
4. При candidate recall выше 0.91 основным ограничением становится ranking.
5. **Hard-negative mining** улучшил ranking без усложнения inference.

## Воспроизведение

Минимальные зависимости:

```bash
pip install -r requirements.txt
```

Структура данных:

```text
data/
├── train.parquet
├── benchmark_queries.parquet
└── benchmark_items.parquet
```

Полный запуск:

```bash
python solution.py --data-dir data --output answer.csv
```

Первый запуск последовательно строит BM25-индексы, E5 embeddings, обучает ranker v1, выполняет hard-negative mining, обучает v2 и формирует benchmark submission. Промежуточные результаты сохраняются в `artifacts/repro`, поэтому прерванный/повторный запуск переиспользует готовые этапы.

### Semantic model

По умолчанию `transformers` использует pinned revision модели `intfloat/multilingual-e5-small`.

Если модель уже скачана локально и запуск должен быть полностью offline:

```bash
python solution.py \
  --data-dir data \
  --model-path /path/to/multilingual-e5-small \
  --output answer.csv
```

После загрузки весов внешние API решению не нужны.

## Файлы

- `solution.py` — end-to-end pipeline от parquet до `answer.csv`;
- `solution.ipynb` — компактное объяснение решения и результатов;
- `requirements.txt` — зависимости;
- `answer.csv` — финальный submission.

## С какими проблемами столкнулся

- **Утечка при random split.** Повторяющиеся тексты запросов делали оценку слишком оптимистичной, поэтому validation строилась по группам `search_query`.

- **География сильно влияла на выдачу.** Нужные объявления часто терялись до top-k, поэтому географические сигналы стали применяться до candidate cutoff.

- **BM25 не покрывал перефразирования.** Для запросов без явного совпадения слов пришлось добавить semantic retrieval.

- **Высокого candidate recall оказалось недостаточно.** После достижения покрытия >90% главным bottleneck стал выбор top-50, поэтому в финальное решение добавили LightGBM reranker и hard-negative mining.

## Возможные улучшения

Текущий candidate pool уже покрывает около 92% релевантных объявлений, поэтому основной резерв качества находится в reranking.

Основные направления улучшения:

- **Cross-encoder для top-100/200 кандидатов** после LightGBM — более точное semantic сравнение query–item.
- **Fine-tuning bi-encoder** на train query–item парах с hard negatives вместо zero-shot E5.
- **Раздельные semantic-признаки** для query ↔ title, params и description вместо одного общего embedding-сигнала.
- **Более точный hard-negative mining** — например, отдельно подбирать сложные negatives из правильной локации или похожей услуги.
- **Более богатые географические признаки**: расстояние, соседние регионы, разные гео-сигналы для разных типов услуг.

При дальнейшем развитии я бы в первую очередь улучшал второй этап ранжирования, а не расширял candidate pool.