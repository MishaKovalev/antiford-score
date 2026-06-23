import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, confusion_matrix, precision_score,
                             recall_score, f1_score, brier_score_loss)
 
RNG = 42
OUT = r"C:\Users\smith\Documents\Study\Дипломная работа\docs\kovalev\figures"
import os; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "figure.dpi": 150})
 
COST_FN = 20.0
COST_FP = 1.0
 
FEATURES = {
    "fns_match":"Соответствие паспорта и ИНН (ФНС)","pfr_match":"Соответствие СНИЛС (ПФР)",
    "passport_valid":"Действительность паспорта (МВД)","dob_mismatch":"Несоответствие даты рождения",
    "name_diff":"Расхождения в ФИО между источниками","addr_mismatch":"Несоответствие адреса",
    "fns_resp_h":"Время ответа ФНС (СМЭВ3), ч","pfr_resp_h":"Время ответа ПФР (СМЭВ3), ч",
    "passport_resp_s":"Время ответа паспорта (СМЭВ4), с","num_reverif":"Число повторных верификаций",
    "num_prior_apps":"Число прошлых заявок","days_to_verif":"Срок до верификации, дн",
    "loan_amount_m":"Сумма кредита, млн руб.","loan_term":"Срок кредита, мес.",
    "borrower_age":"Возраст заёмщика","property_new":"Новостройка (1)/вторичка (0)",
    "region_risk":"Риск-индекс региона","doc_anomaly":"Аномальность документов",
    "device_change":"Смена устройства","night_app":"Подача заявки ночью",
}
FEAT = list(FEATURES.keys())
 
def generate_dataset(n=24000, fraud_rate=0.05, seed=RNG):
    rng = np.random.default_rng(seed)
    fns_match=rng.binomial(1,0.985,n); pfr_match=rng.binomial(1,0.985,n)
    passport_valid=rng.binomial(1,0.99,n); dob_mismatch=rng.binomial(1,0.02,n)
    name_diff=rng.poisson(0.15,n); addr_mismatch=rng.binomial(1,0.08,n)
    fns_resp_h=np.clip(rng.gamma(2.0,3.0,n),0.05,96); pfr_resp_h=np.clip(rng.gamma(2.0,2.6,n),0.05,96)
    passport_resp_s=np.clip(rng.gamma(2.2,1.3,n),0.2,60); num_reverif=rng.poisson(0.3,n)
    num_prior_apps=rng.poisson(0.8,n); days_to_verif=np.clip(rng.normal(2.0,1.0,n),0.1,10)
    loan_amount_m=np.clip(rng.lognormal(1.2,0.5,n),0.5,30); loan_term=rng.choice([60,120,180,240,300],n,p=[.05,.15,.3,.3,.2])
    borrower_age=np.clip(rng.normal(38,9,n),21,70).astype(int); property_new=rng.binomial(1,0.45,n)
    region_risk=np.clip(rng.beta(2,6,n),0,1); doc_anomaly=np.clip(rng.beta(2,9,n),0,1)
    device_change=rng.binomial(1,0.07,n); night_app=rng.binomial(1,0.12,n)
    lin = (2.0*(1-fns_match)+1.9*(1-pfr_match)+1.6*(1-passport_valid)
           +1.7*dob_mismatch+0.5*name_diff+0.7*addr_mismatch
           +3.0*doc_anomaly+1.2*device_change+0.6*night_app
           +0.5*num_reverif+0.15*num_prior_apps+1.3*region_risk
           +0.05*(loan_amount_m-3)-0.012*(borrower_age-38)
           +5.5*((1-fns_match)*(1-pfr_match))
           +5.5*((doc_anomaly>0.30).astype(float)*device_change)
           +4.0*((num_reverif>=2).astype(float)*addr_mismatch)
           +3.6*((region_risk>0.40).astype(float)*night_app)
           +3.4*((doc_anomaly>0.25).astype(float)*(1-passport_valid))
           +2.6*(dob_mismatch*addr_mismatch))
    lin = lin + rng.normal(0,0.10,n)
    lo,hi=-18.0,6.0
    for _ in range(90):
        mid=(lo+hi)/2
        if np.mean(1/(1+np.exp(-(lin+mid))))>fraud_rate: hi=mid
        else: lo=mid
    y=rng.binomial(1,1/(1+np.exp(-(lin+(lo+hi)/2))))
    df=pd.DataFrame({k:v for k,v in zip(FEAT,[fns_match,pfr_match,passport_valid,dob_mismatch,name_diff,
        addr_mismatch,fns_resp_h,pfr_resp_h,passport_resp_s,num_reverif,num_prior_apps,days_to_verif,
        loan_amount_m,loan_term,borrower_age,property_new,region_risk,doc_anomaly,device_change,night_app])})
    df["is_fraud"]=y; return df
 
def best_threshold_by_cost(y, p):
    ths=np.linspace(0.01,0.99,99); best=(0.5,1e18)
    for t in ths:
        yp=(p>=t).astype(int); cm=confusion_matrix(y,yp)
        fp=cm[0,1]; fn=cm[1,0]; cost=COST_FP*fp+COST_FN*fn
        if cost<best[1]: best=(t,cost)
    return best[0]
 
def main():
    df=generate_dataset(); X=df[FEAT].values; y=df["is_fraud"].values
    print(f"Выборка {len(df)}, мошеннических {y.mean():.2%} ({y.sum()})")
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.30,stratify=y,random_state=RNG)
    skf=StratifiedKFold(5,shuffle=True,random_state=RNG)
 
    best_params={"n_estimators":600,"max_depth":3,"learning_rate":0.03,
                 "subsample":0.8,"min_samples_leaf":20}
    print("Гиперпараметры бустинга (подбор):",best_params)
    gb_best=GradientBoostingClassifier(random_state=RNG,**best_params)
 
    models={
        "Логистическая регрессия":Pipeline([("sc",StandardScaler()),
            ("clf",LogisticRegression(max_iter=1000,class_weight="balanced"))]),
        "Случайный лес":RandomForestClassifier(n_estimators=400,max_depth=12,
            class_weight="balanced",random_state=RNG,n_jobs=-1),
        "Градиентный бустинг":gb_best,
    }
    from sklearn.model_selection import cross_val_score
    results={}; proba={}
    for name,m in models.items():
        cv=cross_val_score(m,Xtr,ytr,cv=skf,scoring="roc_auc",n_jobs=-1)
        m.fit(Xtr,ytr); pr=m.predict_proba(Xte)[:,1]; proba[name]=pr
        prec,rec,thr=precision_recall_curve(yte,pr); f1s=2*prec*rec/(prec+rec+1e-9)
        t=thr[min(int(np.nanargmax(f1s)),len(thr)-1)]; yp=(pr>=t).astype(int)
        results[name]={"cv_auc_mean":float(cv.mean()),"cv_auc_std":float(cv.std()),
            "test_auc":float(roc_auc_score(yte,pr)),"test_ap":float(average_precision_score(yte,pr)),
            "threshold_f1":float(t),"precision":float(precision_score(yte,yp)),
            "recall":float(recall_score(yte,yp)),"f1":float(f1_score(yte,yp)),
            "brier":float(brier_score_loss(yte,pr)),"confusion":confusion_matrix(yte,yp).tolist()}
        print(f"{name:24s} CV-AUC={cv.mean():.3f}±{cv.std():.3f} AUC={results[name]['test_auc']:.3f} "
              f"AP={results[name]['test_ap']:.3f} P={results[name]['precision']:.3f} R={results[name]['recall']:.3f} F1={results[name]['f1']:.3f}")
    best=max(results,key=lambda k:results[k]["test_ap"]); print("Лучшая модель:",best)
 
    cal=CalibratedClassifierCV(gb_best,method="isotonic",cv=skf)
    cal.fit(Xtr,ytr); pcal=cal.predict_proba(Xte)[:,1]
    proba["Бустинг (калиброванный)"]=pcal
    results["Бустинг (калиброванный)"]={"test_auc":float(roc_auc_score(yte,pcal)),
        "test_ap":float(average_precision_score(yte,pcal)),"brier":float(brier_score_loss(yte,pcal))}
    print(f"Калибровка: AUC={results['Бустинг (калиброванный)']['test_auc']:.3f} "
          f"AP={results['Бустинг (калиброванный)']['test_ap']:.3f} Brier {results[best]['brier']:.4f}->{results['Бустинг (калиброванный)']['brier']:.4f}")
 
    t_cost=best_threshold_by_cost(yte,pcal)
    ops=[]
    for label,t in [("Высокая полнота",0.20),("Стоимостно-оптимальная",t_cost),("Высокая точность",0.60)]:
        yp=(pcal>=t).astype(int); cm=confusion_matrix(yte,yp); tn,fp,fn,tp=cm.ravel()
        ops.append({"label":label,"threshold":float(t),
            "precision":float(tp/(tp+fp+1e-9)),"recall":float(tp/(tp+fn+1e-9)),
            "fraud_caught":float(tp/(tp+fn)),"manual_share":float((tp+fp)/len(yte)),
            "tp":int(tp),"fp":int(fp),"fn":int(fn),"tn":int(tn)})
        print(f"  Рабочая точка [{label}] t={t:.2f}: P={ops[-1]['precision']:.3f} R={ops[-1]['recall']:.3f} "
              f"поймано {ops[-1]['fraud_caught']:.1%}, на ручную {ops[-1]['manual_share']:.1%}")
 
    pi=permutation_importance(gb_best,Xte,yte,scoring="average_precision",n_repeats=10,random_state=RNG,n_jobs=-1)
 
    def roc():
        plt.figure(figsize=(6,5))
        for nm in ["Логистическая регрессия","Случайный лес","Градиентный бустинг"]:
            f,t,_=roc_curve(yte,proba[nm]); plt.plot(f,t,lw=2,label=f"{nm} (AUC={results[nm]['test_auc']:.3f})")
        plt.plot([0,1],[0,1],"k--",lw=1); plt.xlabel("FPR (доля ложноположительных)")
        plt.ylabel("TPR (доля истинноположительных)"); plt.title("ROC-кривые моделей антифрод-скоринга")
        plt.legend(loc="lower right",fontsize=9); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(f"{OUT}/ml_roc.png"); plt.close()
    def pr():
        plt.figure(figsize=(6,5))
        for nm in ["Логистическая регрессия","Случайный лес","Градиентный бустинг"]:
            p_,r_,_=precision_recall_curve(yte,proba[nm]); plt.plot(r_,p_,lw=2,label=f"{nm} (AP={results[nm]['test_ap']:.3f})")
        plt.axhline(yte.mean(),color="gray",ls=":",label=f"Базовая доля ({yte.mean():.2f})")
        plt.xlabel("Полнота (Recall)"); plt.ylabel("Точность (Precision)")
        plt.title("Кривые «точность–полнота»"); plt.legend(loc="upper right",fontsize=9)
        plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{OUT}/ml_pr.png"); plt.close()
    def cm_fig():
        yp=(pcal>=t_cost).astype(int); cm=confusion_matrix(yte,yp)
        plt.figure(figsize=(4.6,4.2)); plt.imshow(cm,cmap="Blues")
        for (i,j),v in np.ndenumerate(cm):
            plt.text(j,i,str(v),ha="center",va="center",color="white" if v>cm.max()/2 else "black",fontsize=13,fontweight="bold")
        plt.xticks([0,1],["Легитимный","Мошеннический"]); plt.yticks([0,1],["Легитимный","Мошеннический"])
        plt.xlabel("Прогноз"); plt.ylabel("Факт"); plt.title("Матрица ошибок (калиброванный бустинг,\nстоимостно-оптимальный порог)",fontsize=10)
        plt.tight_layout(); plt.savefig(f"{OUT}/ml_cm.png"); plt.close()
    def imp():
        order=np.argsort(pi.importances_mean)[::-1][:12]
        plt.figure(figsize=(8,5.2))
        plt.barh(range(len(order))[::-1],pi.importances_mean[order],
                 xerr=pi.importances_std[order],color="#3b6fb5",ecolor="#888")
        plt.yticks(range(len(order))[::-1],[FEATURES[FEAT[i]] for i in order],fontsize=8.5)
        plt.xlabel("Перестановочная важность (по AP)"); plt.title("Значимость признаков модели (перестановочная важность)")
        plt.grid(axis="x",alpha=.3); plt.tight_layout(); plt.savefig(f"{OUT}/ml_importance.png"); plt.close()
    def calib():
        plt.figure(figsize=(5.6,5))
        for nm,pp in [("Бустинг (до калибровки)",proba["Градиентный бустинг"]),("Бустинг (калиброванный)",pcal)]:
            fop,mpv=calibration_curve(yte,pp,n_bins=10,strategy="quantile"); plt.plot(mpv,fop,"o-",lw=1.6,ms=4,label=nm)
        plt.plot([0,1],[0,1],"k--",lw=1,label="Идеальная калибровка")
        plt.xlabel("Средняя предсказанная вероятность"); plt.ylabel("Наблюдаемая доля")
        plt.title("Калибровочные кривые"); plt.legend(fontsize=8); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f"{OUT}/ml_calibration.png"); plt.close()
    def cost():
        ths=np.linspace(0.01,0.99,99); costs=[]
        for t in ths:
            cm=confusion_matrix(yte,(pcal>=t).astype(int)); costs.append(COST_FP*cm[0,1]+COST_FN*cm[1,0])
        plt.figure(figsize=(6.5,4.4)); plt.plot(ths,costs,lw=2,color="#27548a")
        plt.axvline(t_cost,color="#c0504d",ls="--",lw=1.4,label=f"Оптимальный порог t={t_cost:.2f}")
        plt.xlabel("Порог принятия решения"); plt.ylabel("Ожидаемая стоимость ошибок (усл. ед.)")
        plt.title(f"Стоимость ошибок и выбор порога (C_FN:C_FP = {int(COST_FN)}:{int(COST_FP)})")
        plt.legend(fontsize=9); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{OUT}/ml_cost.png"); plt.close()
    def scoredist():
        plt.figure(figsize=(6.5,4.2))
        plt.hist(pcal[yte==0],bins=40,alpha=.6,density=True,label="Легитимные",color="#4c9f70")
        plt.hist(pcal[yte==1],bins=40,alpha=.6,density=True,label="Мошеннические",color="#c0504d")
        plt.axvline(t_cost,color="k",ls="--",lw=1.2,label="Порог решения")
        plt.xlabel("Оценка риска (калиброванная вероятность)"); plt.ylabel("Плотность")
        plt.title("Распределение оценок риска"); plt.legend(fontsize=9); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f"{OUT}/ml_scoredist.png"); plt.close()
    roc(); pr(); cm_fig(); imp(); calib(); cost(); scoredist()
 
    json.dump({"n":len(df),"fraud_rate":float(y.mean()),"best_model":best,
               "best_params":best_params,"cost_fn":COST_FN,"cost_fp":COST_FP,
               "threshold_cost":float(t_cost),"results":results,"operating_points":ops,
               "perm_importance":{FEAT[i]:float(pi.importances_mean[i]) for i in np.argsort(pi.importances_mean)[::-1]},
               "features":FEATURES},
              open(f"{OUT}/ml_metrics.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
    df.to_csv(f"{OUT}/dataset_sample.csv",index=False,encoding="utf-8")
    print("Готово. Графики и метрики в",OUT)
 
if __name__=="__main__":
    main()
