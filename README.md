# Система идентификации личности по голосу

Учебный прототип КП (тема №5): **MFCC → голосовой слепок (mean-vector) → cosine similarity**.

Связь с курсом «Мультимедийные системы»: оцифровка (лек. 3), тембр/спектр (лек. 4), WAV и FFT (лек. 13), этапы проекта (лек. 17).

## Быстрый старт

```bash
cd "/Users/vladimirpocepaj/Study/ПИД ЖИГАЛОВ"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# демо-датасет (синтетические «голоса» с разным тембром)
python scripts/generate_demo_dataset.py

# регистрация слепков
python -m src.enroll --all

# идентификация файла
python -m src.identify data/raw/speaker_01/test_01.wav

# метрики + таблица порогов
python -m src.metrics
python -m src.metrics --sweep --out docs/metrics_table.csv

# графики для записки
python scripts/plots.py

# GUI
python app/gui.py
```

## Структура

| Путь | Назначение |
|------|------------|
| `src/audio_io.py` | чтение/запись WAV, ресемплинг |
| `src/preprocess.py` | нормализация, energy VAD, pre-emphasis |
| `src/features.py` | FFT → Mel → MFCC, слепок, cosine |
| `src/enroll.py` | регистрация профилей `.npy` |
| `src/identify.py` | идентификация 1:N + порог |
| `src/metrics.py` | Accuracy / FAR / FRR |
| `app/gui.py` | простой интерфейс |
| `docs/` | ТЗ, оглавление записки, черновик, презентация |
| `data/raw/` | WAV по спикерам |

## Параметры (не менять без правки записки)

- `fs = 16000`, окно 25 мс, hop 10 мс  
- 26 mel-фильтров, 13 MFCC (без C0)  
- порог cosine по умолчанию `0.92` (`src/config.py`; подбирается через `metrics --sweep`)

## Реальные голоса бригады

См. [docs/RECORDING_PROTOCOL.md](docs/RECORDING_PROTOCOL.md).  
Синтетический набор нужен, чтобы прототип работал «из коробки»; на защите лучше показать свои записи.

## Документы

- [ТЗ](docs/TZ.md) — требования и метрики  
- [Оглавление записки](docs/REPORT_OUTLINE.md)  
- [Черновик записки](docs/REPORT_DRAFT.md)  
- [Презентация (тезисы слайдов)](docs/PRESENTATION.md)  
- [Сценарий защиты](docs/DEFENSE_SCRIPT.md)  
