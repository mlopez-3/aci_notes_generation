# metrics/

Métricas calculadas sobre cada predicción. El nombre de la predicción va entero y se le añade la métrica:

```
<nombre-de-la-predicción>__<metric>.json
```

Ejemplo:
`valid__zeroshot__qwen3-4b-instruct-2507__4bit-bf16__20261116T1702Z__entities-v4.json`

Métricas previstas: `surface-v1` (ROUGE, BERTScore), `bleurt-elron-base-512-v1`, `entities-v4` (omisiones y alucinaciones por NER).
Con `acinotes.naming.metrics_for(pid)` se listan todas las métricas de una predicción.
