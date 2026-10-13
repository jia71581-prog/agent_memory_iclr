"""Adversarial test of 'no surface rule can separate abstention from answerable'.
Upper bound: fit a supervised surface classifier with CV. If even a fitted
bag-of-words/char model cannot separate, the negative claim is strong.
"""
import json, sys, re, numpy as np
sys.path.insert(0,"/root/agent_memory_iclr/code")
D=json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
qs=[r["question"] for r in D]; y=np.array([1 if r["question_id"].endswith("_abs") else 0 for r in D])
print("prevalence", y.mean())
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    for name, vec in [("word1-2", TfidfVectorizer(ngram_range=(1,2), min_df=2)),
                      ("char3-5", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2))]:
        pipe=make_pipeline(vec, LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))
        cv=StratifiedKFold(5, shuffle=True, random_state=0)
        p=cross_val_predict(pipe, qs, y, cv=cv, method="predict_proba")[:,1]
        ap=average_precision_score(y,p); auc=roc_auc_score(y,p)
        # precision at top-k
        order=np.argsort(-p)
        prec30=y[order[:30]].sum()/30; prec60=y[order[:60]].sum()/60
        print(f"[E] {name}: CV AP={ap:.3f} (chance {y.mean():.3f}) AUC={auc:.3f} "
              f"P@30={prec30:.3f} P@60={prec60:.3f} recall@30={y[order[:30]].sum()}/30")
except ImportError as e:
    print("sklearn missing:", e)

# also: length / surface statistics
import statistics
la=[len(r["question"].split()) for r in D if r["question_id"].endswith("_abs")]
ln=[len(r["question"].split()) for r in D if not r["question_id"].endswith("_abs")]
print(f"[E] mean question length abs={statistics.mean(la):.1f} ans={statistics.mean(ln):.1f}")
