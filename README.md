# aci-note-generation

Generación de notas clínicas a partir de conversaciones médico-paciente (ACI-Bench) con Qwen3-4B-Instruct-2507:
prompting zero-shot, few-shot y fine-tuning LoRA, y evaluación de la fidelidad (ROUGE, BERTScore, BLEURT, omisiones/alucinaciones con NER).

## Estructura

```
data/raw/aci-bench/   conversaciones originales del artículo (JSON) + SOURCE.md
src/acinotes/         código propio (data, naming, provenance; después generación y métricas)
scripts/              programas ejecutables (import_data...)
predictions/          notas generadas, una ejecución por fichero
metrics/              métricas de cada predicción
tests/                pruebas
```

## Convención de nombres

Predicción: `<split>__<system>__<model>__<variant>__<stamp>.jsonl`
Métrica: `<nombre-de-la-predicción>__<metric>.json`

Detalles en `predictions/README.md`, `metrics/README.md` y `src/acinotes/naming.py`.

## Puesta en marcha

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python scripts/import_data.py --src <ruta>/aci-bench/data/challenge_data_json   # solo si data/raw está vacío
python -m unittest discover tests
```

Para la GPU: `pip install -e ".[ml,eval]"`.
