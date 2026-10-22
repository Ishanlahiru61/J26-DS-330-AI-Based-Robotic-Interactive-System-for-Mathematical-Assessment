# Source of the raw data

The four Excel files in this folder are the public dataset of:

> Decarli, G., Paris, E., Tencati, C., Nardelli, C., Vescovi, M., Surian, L., & Piazza, M. (2020).
> Impaired large numerosity estimation and intact subitizing in developmental dyscalculia.
> *PLoS ONE, 15*(12), e0244578. https://doi.org/10.1371/journal.pone.0244578

- Data repository: https://osf.io/gjpb3/
- Licence: CC BY 4.0 (cite the paper when using the data)

## Sample

- 64 Italian children: 32 with developmental dyscalculia (DD), 32 typically developing controls (`contr`)
- Age 7.4–14 years (DD mean 9.51, controls mean 9.79). No age column in the files.
- DD criterion: at or below the 5th percentile on standard maths batteries (AC-MT or BDE-2), with IQ in the normal range

## Column meanings (`Complete_Dataframe_def.xlsx`)

| Column | Meaning |
|---|---|
| `Similarities_IQ`, `MatrixReas_IQ` | WISC subtest scaled scores |
| `w_fraction_def`, `Mean_ACC_ANS`, `Mean_RTs_ANS` | Non-symbolic dot comparison (ANS): Weber fraction, accuracy %, RT ms |
| `Mean_ACC_13`, `MeanRT_13`, `CoV_13`, `EfficiencyScore_13` | Exact enumeration, subitizing range (1–3 dots) |
| `Mean_ACC_47`, `MeanRT_47`, `CoV_47`, `EfficiencyScore_47` | Exact enumeration, counting range (4–7 dots) |
| `wm_K` | Visuo-spatial short-term memory, Cowan's K (colour change detection) |
| `Avanti`, `Indietro`, `Average_Corsi` | Corsi block span forward, backward, average |
| `Accuracy_SymbolicComp`, `RTs_SymbolicComp` | Symbolic digit comparison (1–9): accuracy, RT ms |
| `group` | `DD` or `contr` |

The other three files hold per-condition results (by numerosity or distance) for the same 64 children.

## Limitation for this project

The children are older (mean about 9.6) than our target group (6–8), so the values cannot be used as age-6–8 norms directly. They are used to estimate **how much each skill differs between DD and typical children** (effect sizes). Age-specific norms must come from other sources.
