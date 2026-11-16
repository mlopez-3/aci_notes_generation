# predictions/

Notas generadas por los modelos. Una ejecución = un `.jsonl` + un `.meta.json`.

```
<split>__<system>__<model>__<variant>__<stamp>.jsonl
<split>__<system>__<model>__<variant>__<stamp>.meta.json
```

Ejemplo:
`valid__zeroshot__qwen3-4b-instruct-2507__4bit-bf16__20261116T1702Z.jsonl`

| campo | significado |
|---|---|
| split | `valid`, `test1`... conjunto de conversaciones usado |
| system | `zeroshot`, `fewshot`, `lora` |
| model | modelo base |
| variant | configuración (cuantización, nº de ejemplos, hiperparámetros) |
| stamp | fecha y hora UTC de la generación |

El `.meta.json` guarda la configuración exacta y el commit del código. Nunca se sobrescribe un fichero existente.
Generar los nombres con `acinotes.naming`, no a mano.
