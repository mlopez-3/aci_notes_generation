# ACI-Bench

Conversaciones médico-paciente con su nota clínica de referencia.

- Artículo: Yim et al., "ACI-BENCH: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation", Scientific Data, 2023.
- Repositorio: https://github.com/wyim/aci-bench
- Licencia: CC BY 4.0
- Ficheros copiados de `aci-bench/data/challenge_data_json/` (ver `CHECKSUMS.md`).

| split | fichero | uso |
|---|---|---|
| train | train_full.json | 67 ejemplos para few-shot y LoRA |
| valid | valid_full.json | 20, ajustar prompts y hiperparámetros |
| test1 | clinicalnlp_taskB_test1_full.json | 40, test oficial Tarea B (se usa una sola vez) |
| test2 | clinicalnlp_taskC_test2_full.json | 40 |
| test3 | clef_taskC_test3_full.json | 40 |

Campos: `src` (transcripción), `tgt` (nota de referencia), `file` (id, p. ej. `D2N071-virtassist`).
