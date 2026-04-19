# Table 1: Overall Model Performance

| Model | Experiment | Accuracy | Macro F1 | Weighted F1 | Samples |
|-------|------------|----------|----------|-------------|----------|
| MuRIL-Large | Exp1_Supervised_Baseline | 0.5992 | 0.5887 | 0.5887 | 252 |
| MuRIL-Large | Exp2_Hindi_Monolingual | 0.5476 | 0.4595 | 0.4595 | 84 |
| MuRIL-Large | Exp2_CodeMixed_Monolingual | 0.5833 | 0.5619 | 0.5619 | 84 |
| MuRIL-Large | Exp2_English_Monolingual | 0.7262 | 0.6975 | 0.6975 | 84 |
| MuRIL-Large | Exp3_ZeroShot_Hindi_CodeMixed_to_English | 0.7125 | 0.6797 | 0.6791 | 400 |
| MuRIL-Large | Exp3_ZeroShot_English_CodeMixed_to_Hindi | 0.5450 | 0.4758 | 0.4748 | 400 |
| MuRIL-Large | Exp3_ZeroShot_Hindi_English_to_CodeMixed | 0.6400 | 0.6159 | 0.6163 | 400 |
| MuRIL-Large | Exp4_FewShot_5_Hindi_CodeMixed_to_English | 0.7089 | 0.6786 | 0.6751 | 395 |
| MuRIL-Large | Exp4_FewShot_10_Hindi_CodeMixed_to_English | 0.7051 | 0.6775 | 0.6710 | 390 |
| MuRIL-Large | Exp4_FewShot_20_Hindi_CodeMixed_to_English | 0.6974 | 0.6749 | 0.6624 | 380 |
| MuRIL-Large | Exp4_FewShot_50_Hindi_CodeMixed_to_English | 0.6743 | 0.6663 | 0.6363 | 350 |
| XLM-RoBERTa-Large | Exp1_Supervised_Baseline | 0.8849 | 0.8852 | 0.8852 | 252 |
| XLM-RoBERTa-Large | Exp2_Hindi_Monolingual | 0.9524 | 0.9524 | 0.9524 | 84 |
| XLM-RoBERTa-Large | Exp2_CodeMixed_Monolingual | 0.8095 | 0.8021 | 0.8021 | 84 |
| XLM-RoBERTa-Large | Exp2_English_Monolingual | 0.8929 | 0.8900 | 0.8900 | 84 |
| XLM-RoBERTa-Large | Exp3_ZeroShot_Hindi_CodeMixed_to_English | 0.8500 | 0.8425 | 0.8422 | 400 |
| XLM-RoBERTa-Large | Exp3_ZeroShot_English_CodeMixed_to_Hindi | 0.9425 | 0.9427 | 0.9426 | 400 |
| XLM-RoBERTa-Large | Exp3_ZeroShot_Hindi_English_to_CodeMixed | 0.8150 | 0.8115 | 0.8114 | 400 |
| XLM-RoBERTa-Large | Exp4_FewShot_5_Hindi_CodeMixed_to_English | 0.8481 | 0.8425 | 0.8402 | 395 |
| XLM-RoBERTa-Large | Exp4_FewShot_10_Hindi_CodeMixed_to_English | 0.8462 | 0.8425 | 0.8381 | 390 |
| XLM-RoBERTa-Large | Exp4_FewShot_20_Hindi_CodeMixed_to_English | 0.8421 | 0.8425 | 0.8339 | 380 |
| XLM-RoBERTa-Large | Exp4_FewShot_50_Hindi_CodeMixed_to_English | 0.8286 | 0.8425 | 0.8196 | 350 |
