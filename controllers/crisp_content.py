import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")

PHASES = [
    {"num": 1, "slug": "business-and-data-understanding", "title": "Business and data understanding", "color": "#2e9e5b",
     "summary": "What problem we solve in Chía, for whom, which questions the model answers and how we will know it works."},
    {"num": 2, "slug": "data-engineering", "title": "Data engineering", "color": "#3a8fb7",
     "summary": "How the raw counts became clean, time-ordered training, validation and test partitions without leakage."},
    {"num": 3, "slug": "model-engineering", "title": "Model engineering", "color": "#e0a526",
     "summary": "Five algorithms, tuned with time-series cross-validation, with every experiment recorded."},
    {"num": 4, "slug": "model-evaluation", "title": "Model evaluation", "color": "#d64545",
     "summary": "Metrics, cross-validation, comparison, risks, expert review and the final result on an independent test set."},
    {"num": 5, "slug": "model-deployment", "title": "Model deployment", "color": "#4a63c9",
     "summary": "The saved model, a working demo, its inputs and outputs, and how it plugs into the dashboard and simulation."},
    {"num": 6, "slug": "monitoring-and-maintenance", "title": "Monitoring and maintenance", "color": "#6b7a72",
     "summary": "Indicators, drift detection, responsibilities, retraining rules and a labelled monitoring test."},
]

REFERENCES = [
    "Géron, A. (2022). <em>Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow</em> (3rd ed.). O'Reilly.",
    "Hyndman, R. J., &amp; Athanasopoulos, G. (2021). <em>Forecasting: Principles and practice</em> (3rd ed.). OTexts.",
    "KDnuggets. (2025). <em>The machine learning lifecycle</em>.",
    "ML-Ops.org. (n.d.). <em>CRISP-ML(Q): The ML lifecycle process</em>.",
    "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. <em>Journal of Machine Learning Research, 12</em>, 2825–2830.",
    "Schmid, S., Wurst, M., &amp; Wirth, R. (2020). <em>Towards CRISP-ML(Q): A machine learning process model with quality assurance methodology</em>.",
]

LABELS = {"seasonal_mean": "Seasonal mean (baseline)", "ridge": "Ridge regression", "random_forest": "Random Forest",
          "gradient_boosting": "Gradient Boosting", "mlp": "MLP (PyTorch)"}
ORDER = ["seasonal_mean", "ridge", "random_forest", "gradient_boosting", "mlp"]


def load(name):
    try:
        with open(os.path.join(ART, name), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def n(x, d=0):
    return f"{x:,.{d}f}"


def fig(file, caption):
    return {"file": file, "caption": caption}


def get_phase(num):
    for p in PHASES:
        if p["num"] == num:
            return p
    return None


def _params(p):
    return ", ".join(f"{k} = {tuple(v) if isinstance(v, list) else v}" for k, v in p.items()) or "no parameters"


def build_phase(num, inters):
    rep, mon, ctl = load("report.json"), load("monitoring.json"), load("control_evaluation.json")
    meta = dict(get_phase(num))
    if rep is None:
        meta["intro"] = "Run <code>python train.py</code> to generate the results shown in this phase."
        meta["sections"] = []
        return meta
    q, fam, fin, sp = rep["data_quality"], rep["families"], rep["final"], rep["splits"]
    sel = fin["selected"]
    test, base = fin["test"], fin["test_baseline_seasonal_mean"]
    names = {str(i["junction"]): i["address"] for i in inters}
    gain = 100 * (1 - test["mae"] / base["mae"])
    builders = {1: _phase1, 2: _phase2, 3: _phase3, 4: _phase4, 5: _phase5, 6: _phase6}
    meta.update(builders[num](rep, q, fam, fin, sp, sel, test, base, names, gain, mon, ctl))
    return meta


def _phase1(rep, q, fam, fin, sp, sel, test, base, names, gain, mon, ctl):
    b = q["before"]
    d = b["vehicles_describe"]
    ctl_txt = (f"{ctl['improvement_mean']}% on average (from {ctl['improvement_min']}% to {ctl['improvement_max']}% over 5 trainings; better in "
               f"{sum(1 for r in ctl['runs'] if r['improvement_pct'] > 0)} of 5)"
               if ctl else "see phase 4")
    return {"intro": "Most traffic lights in Chía work with fixed times: the green lasts the same at 3 a.m. and at 6 p.m. "
                     "The project uses machine learning to anticipate how many vehicles will arrive at an intersection "
                     "and to adapt the green light to that demand.",
            "sections": [
        {"title": "Problem, context, stakeholders and beneficiaries",
         "paragraphs": ["Chía (Cundinamarca) is a fast-growing municipality north of Bogotá. Its main corridors (Av. Pradilla, "
                        "Carrera 9, Av. Chilacos) carry commuter traffic towards Bogotá and the universities, and fixed-time signals "
                        "cannot adapt to the peaks. The approved problem is: <b>forecast the hourly traffic demand of four "
                        "intersections and use it to decide the green light</b>."],
         "tables": [{"head": ["Stakeholder", "Role", "Interest"], "rows": [
             ["Secretaría de Movilidad de Chía", "Owner and future operator", "Shorter queues, evidence for signal timing"],
             ["Drivers, bus users and pedestrians", "Beneficiaries", "Less waiting time and more predictable trips"],
             ["Residents near the corridors", "Beneficiaries", "Less congestion and noise"],
             ["Traffic engineers", "Domain experts", "Plans that respect minimum green and safety rules"],
             ["Project team and course professor", "Developers and reviewer", "A valid, documented and reproducible model"]]}]},
        {"title": "Business objectives translated into machine learning objectives",
         "tables": [{"head": ["Business objective", "Machine learning objective"], "rows": [
             ["Know in advance how much traffic each intersection will have", "Regression model that predicts vehicles per hour from date, hour and intersection"],
             ["Recognise peak and off-peak periods", "Unsupervised clustering (K-Means) of the 168 hours of the week"],
             ["Reduce the time vehicles wait at the signal", "Reinforcement learning agent (DQN) that decides keep or change, using the forecast"],
             ["Give operators a tool they can use", "Web application with dashboard, simulation and a prediction demo"]]}]},
        {"title": "Questions and learning task",
         "bullets": ["<b>How many vehicles per hour will arrive at intersection <i>j</i> on date <i>d</i> at hour <i>h</i>?</b> "
                     "Supervised learning, <b>regression</b>. This is the model documented in phases 2 to 6.",
                     "What kind of moment is it (off-peak, moderate, peak)? Unsupervised learning, clustering. Used as a feature.",
                     "Should the green be kept or changed this second? Reinforcement learning. Consumes the forecast in the simulator."]},
        {"title": "Data sources, variables, size and target",
         "tables": [{"head": ["Source", "Content", "Size"], "rows": [
             ["traffic.csv", "Hourly vehicle counts: DateTime, Junction, Vehicles, ID", f"{n(b['rows'])} rows, {b['date_from'][:10]} to {b['date_to'][:10]}"],
             ["Intersecciones_Chia.csv", "Address, number of phases and coordinates of the 4 intersections", "4 rows"],
             ["Red_Semaforica.csv", "Reference inventory of traffic signals (structure of the intersection data)", "1,747 rows, 26 columns"]]},
                    {"caption": "Intersections of the case study", "head": ["Junction", "Intersection", "Records"],
                     "rows": [[j, names.get(j, "–"), n(sum(sp[k]["rows_by_junction"].get(j, 0) for k in sp))] for j in sorted(names)]}],
         "paragraphs": [f"<b>Target variable:</b> <code>Vehicles</code>, vehicles per hour (mean {d['mean']}, median {d['50%']}, "
                        f"maximum {int(d['max'])})."]},
        {"title": "Data quality, relevance and sufficiency",
         "bullets": [f"Quality: {sum(b['missing_values'].values())} missing values, {b['invalid_dates']} invalid dates, "
                     f"{b['negative_counts']} negative counts and {b['duplicated_date_junction']} duplicated (date, junction) pairs.",
                     "Outliers: isolated spikes up to 180 vehicles per hour at junction 3 (the 99.5th percentile is about 64).",
                     "Relevance: the counts describe exactly the target (hourly demand per intersection).",
                     "Sufficiency: 20 months of hourly data cover every hour of the week many times; junction 4 only has data from January 2017.",
                     "Trend: traffic grew strongly during the period (junction 1 went from about 20 to 73 vehicles per hour), so the date must be part of the model."],
         "figures": [fig("monthly_trend.png", "Monthly average per intersection: a strong upward trend."),
                     fig("hourly_profile.png", "Average per hour of day: low at dawn, high from 10:00 to 22:00.")]},
        {"title": "Feasibility, limitations and risks",
         "bullets": ["<b>Technical:</b> the data fits in memory, models train on CPU in seconds or minutes, and the app runs on a free Render instance.",
                     "<b>Organisational:</b> there is no access to the real controllers, so control decisions are tested in a simulator.",
                     "<b>Limitation:</b> the counts come from a public hourly dataset assigned to four Chía intersections as a case study; they are not field measurements from Chía.",
                     "<b>Limitation:</b> hourly granularity; second-by-second arrivals are generated by the simulator.",
                     "<b>Risk:</b> tree models do not extrapolate trends, so the model must be retrained as traffic grows (phase 6)."]},
        {"title": "Measurable success criteria",
         "tables": [{"head": ["Criterion", "Target", "Result", ""], "rows": [
             ["R² on the independent test set (requested by the course professor)", "≥ 0.80", f"{test['r2']}", "✓" if test["r2"] >= 0.8 else "✗"],
             ["MAE reduction vs the seasonal-mean baseline", "≥ 30%", f"{gain:.0f}% ({base['mae']} → {test['mae']} veh/h)", "✓" if gain >= 30 else "✗"],
             ["Queue reduction of the agent vs a fixed 30 s signal (simulation)", "≥ 10%", ctl_txt, "✓" if (ctl and ctl["improvement_mean"] >= 10) else "–"],
             ["Prediction available through the web demo and API", "working", "/demo and POST /v1/forecast", "✓"]]}]},
    ]}


def _phase2(rep, q, fam, fin, sp, sel, test, base, names, gain, mon, ctl):
    b, a = q["before"], q["after"]
    out = a["outliers_by_junction"]
    raw_rows = [[str(r.get("DateTime")), r.get("Junction"), r.get("Vehicles"), r.get("ID")] for r in b["sample"]]
    fn = a["feature_names"]
    feat_rows = [[r["datetime"], r["junction"], r["vehicles"]] for r in a["sample_raw_for_features"]]
    xs = a["sample_features"]
    return {"intro": "All the steps below are code (<code>models/data_loader.py</code> and <code>ml/features.py</code>) "
                     "and run the same way every time with <code>python train.py</code>.",
            "sections": [
        {"title": "Selection and justification of sources and features",
         "bullets": ["From traffic.csv: DateTime (when), Junction (where) and Vehicles (target). The ID column is a row identifier and is not used as a feature.",
                     "From Intersecciones_Chia.csv: address and phases, used to name and describe each intersection.",
                     "Features are chosen from domain knowledge: traffic depends on the hour, the day of the week, weekends, the type of moment, the intersection and the long-term growth."]},
        {"title": "Missing values, duplicates, inconsistencies and outliers",
         "bullets": ["Dates are parsed; rows with invalid dates would be dropped (0 found).",
                     "Vehicles are converted to numbers; empty or negative values would be dropped (0 found).",
                     "Duplicated (date, junction) records would be removed (0 found).",
                     "<b>Outliers:</b> spikes above the 99.5th percentile of each intersection are capped in the training target only. "
                     "The caps are computed on the training partition, and evaluation always uses the real, uncapped values."],
         "tables": [{"caption": "Outlier capping (computed on the training partition)", "head": ["Junction", "Cap (P99.5)", "Rows capped", "Max before", "Max after"],
                     "rows": [[j, v["cap_p99_5"], v["rows_above_cap"], v["max_before"], v["max_after"]] for j, v in sorted(out.items())]}],
         "figures": [fig("outliers_before_after.png", "Training target before and after capping the spikes.")]},
        {"title": "Encoding, scaling and feature engineering",
         "tables": [{"head": ["Feature", "Built from", "Why"], "rows": [
             ["hour, dayofweek", "DateTime", "Daily and weekly pattern"],
             ["is_weekend", "dayofweek", "Weekends carry about 40% less traffic at noon"],
             ["hour_sin, hour_cos", "hour", "Cyclic encoding: 23:00 is next to 00:00"],
             ["cluster", "K-Means on the training profile", "Type of moment (off-peak, moderate, peak) per intersection"],
             ["days_since_start, month", "DateTime", "Long-term growth and seasonality"],
             ["junction_1 … junction_4", "Junction", "One-hot encoding: each intersection has its own level"]]}],
         "paragraphs": ["Scaling: Ridge and the MLP standardize their inputs with a scaler fitted on the training data inside the model; "
                        "tree models do not need scaling."]},
        {"title": "Evidence: data before and after processing",
         "tables": [{"caption": "Before: first rows of traffic.csv", "head": ["DateTime", "Junction", "Vehicles", "ID"], "rows": raw_rows},
                    {"caption": "After: the same kind of rows as model inputs",
                     "head": ["datetime", "junction", "vehicles"] + fn[:8],
                     "rows": [r + [x[k] for k in fn[:8]] for r, x in zip(feat_rows, xs)]}],
         "paragraphs": [f"Target before capping (training): mean {a['train_target_describe_before_cap']['mean']}, max {a['train_target_describe_before_cap']['max']}; "
                        f"after capping: mean {a['train_target_describe_after_cap']['mean']}, max {a['train_target_describe_after_cap']['max']}. "
                        "The four one-hot intersection columns follow the eight columns shown."]},
        {"title": "Documented transformations",
         "tables": [{"head": ["Step", "Where", "Fitted on"], "rows": [
             ["Parse dates, drop invalid, negative and duplicated rows", "models/data_loader.py", "– (rule based)"],
             ["Merge with the intersection inventory", "models/data_loader.py", "– (rule based)"],
             ["Time features, cyclic hour, one-hot intersection", "ml/features.py", "– (deterministic)"],
             ["K-Means cluster table per intersection", "ml/features.py", "training partition"],
             ["Outlier caps (P99.5 per intersection)", "ml/features.py", "training partition"],
             ["Standardization (Ridge, MLP)", "ml/pipeline.py, ml/regressors.py", "training partition"]]}]},
        {"title": "Training, validation and test partitions (no leakage)",
         "paragraphs": ["The data is a time series, so random splits would let the model see the future. The partitions follow time:"],
         "tables": [{"head": ["Partition", "Period", "Rows", "Use"], "rows": [
             ["Train", "Nov 2015 – Feb 2017", n(sp["train"]["rows"]), "Fit preprocessing and models; 5-fold time-series cross-validation for tuning"],
             ["Validation", "Mar – Apr 2017", n(sp["validation"]["rows"]), "Compare the tuned models and select the best"],
             ["Test", "May – Jun 2017", n(sp["test"]["rows"]), "Final, independent result (used once)"]]}],
         "figures": [fig("splits_timeline.png", "Time-ordered partitions.")],
         "note": "Every preprocessing step that learns from data (K-Means clusters, outlier caps, scalers) is fitted only on the training "
                 "data of each fold. The dashboard's Group card describes the whole history and is not used to train the forecast."},
    ]}


def _phase3(rep, q, fam, fin, sp, sel, test, base, names, gain, mon, ctl):
    grids = rep["grids"]
    rows = []
    for f in ORDER:
        if f not in fam:
            continue
        g = grids.get(f, {})
        rows.append([LABELS[f], ", ".join(f"{k}: {v}" for k, v in g.items()) or "–", _params(fam[f]["best_params"]),
                     f"{fam[f]['cv_rmse_mean']} ± {fam[f]['cv_rmse_std']}"])
    return {"intro": "The forecasting problem is a tabular regression with strong non-linear interactions (hour × day × intersection) "
                     "and a trend. We compared five algorithms of increasing flexibility under exactly the same protocol.",
            "sections": [
        {"title": "Algorithms compared and why",
         "tables": [{"head": ["Algorithm", "Why it fits the problem"], "rows": [
             ["Seasonal mean (baseline)", "Average per intersection, day and hour: the simplest reasonable forecast"],
             ["Ridge regression", "Linear model: tests whether a linear combination of the features is enough"],
             ["Random Forest", "Many decision trees: captures interactions and is robust to outliers"],
             ["Gradient Boosting (histogram)", "Trees that correct each other: usually the strongest on tabular data"],
             ["MLP neural network (PyTorch)", "Learns non-linear functions; can extrapolate smooth trends better than trees"]]}]},
        {"title": "Training: features, parameters and configuration",
         "bullets": [f"Inputs: {len(rep['data_quality']['after']['feature_names'])} features (phase 2). Target: vehicles per hour, capped at P99.5 for training.",
                     "Seed 42 for every model. Code in <code>ml/pipeline.py</code> and <code>train.py</code>.",
                     "MLP: Adam optimizer, MSE loss, dropout 0.05, weight decay 1e-5, 40 epochs, batches of 256, standardized inputs and target."]},
        {"title": "Validation and hyperparameter tuning",
         "paragraphs": ["Each configuration of the grid was evaluated with <b>5-fold time-series cross-validation</b> on the training partition "
                        "(each fold trains on the past and validates on the next block). The configuration with the lowest mean RMSE was kept."],
         "tables": [{"head": ["Model", "Grid searched", "Best configuration", "CV RMSE (mean ± std)"], "rows": rows}],
         "figures": [fig("cv_results.png", "Cross-validation RMSE of the best configuration of each model (error bars: standard deviation across folds).")]},
        {"title": "Experiment log",
         "paragraphs": [f"All {sum(1 for _ in _experiments())} runs are stored in <code>artifacts/experiments.csv</code> (model, parameters, CV RMSE, MAE, R² and time). "
                        "The table shows each run."],
         "tables": [{"head": ["Model", "Parameters", "CV RMSE", "CV MAE", "CV R²", "Seconds"], "rows": _experiments()}]},
        {"title": "How domain knowledge informed the modelling",
         "bullets": ["Traffic is periodic: cyclic hour encoding, day of week and weekend flag.",
                     "Intersections have very different volumes: one-hot intersection instead of a single numeric code.",
                     "Peaks follow regimes: the K-Means cluster is given to the model as a feature.",
                     "Traffic grows over the years: date trend features, time-ordered validation and monthly retraining.",
                     "Counting spikes (events, sensor errors) should not drive the model: training target capped at P99.5.",
                     "In the simulator: minimum green 8 s, maximum 60 s and yellow 3 s, as in traffic engineering practice."]},
    ]}


def _experiments():
    path = os.path.join(ART, "experiments.csv")
    rows = []
    try:
        import csv
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append([r["model"], r["params"].replace('"', ""), r["cv_rmse_mean"], r["cv_mae_mean"], r["cv_r2_mean"], r["seconds"]])
    except OSError:
        pass
    return rows


def _phase4(rep, q, fam, fin, sp, sel, test, base, names, gain, mon, ctl):
    comp = [[LABELS[f] + (" ✓" if f == sel else ""), fam[f]["cv_rmse_mean"], fam[f]["validation"]["mse"], fam[f]["validation"]["rmse"],
             fam[f]["validation"]["mae"], fam[f]["validation"]["r2"]] for f in ORDER if f in fam]
    over = [[LABELS[f], fam[f]["train"]["rmse"], fam[f]["validation"]["rmse"], round(fam[f]["validation"]["rmse"] - fam[f]["train"]["rmse"], 2),
             fam[f]["train"]["r2"], fam[f]["validation"]["r2"]] for f in ORDER if f in fam]
    tj = fin["test_by_junction"]
    imp = sorted(fin["feature_importance"].items(), key=lambda kv: -kv[1])[:5]
    sections = [
        {"title": "Metrics and why",
         "bullets": ["<b>MSE</b> (selection criterion): penalizes large errors, which are the ones that create long queues.",
                     "<b>RMSE</b>: the same error in vehicles per hour, easier to read.",
                     "<b>MAE</b>: average absolute error in vehicles per hour; what an operator can expect.",
                     "<b>R²</b>: share of the variation explained; the course professor set 0.80 as the minimum to use a model."]},
        {"title": "Cross-validation suited to the data",
         "paragraphs": ["Because the data is ordered in time, we used <code>TimeSeriesSplit</code> with 5 folds on the training partition: "
                        "each fold always validates on a period after the one it trained on. Its RMSE (phase 3) is higher than on the "
                        "validation set because the first folds train with only a few months of data and must forecast a later, busier period."]},
        {"title": "Comparison of the models",
         "tables": [{"caption": "Validation set (Mar – Apr 2017), models trained on the training partition with the tuned parameters",
                     "head": ["Model", "CV RMSE", "Val. MSE", "Val. RMSE", "Val. MAE", "Val. R²"], "rows": comp,
                     "highlight": ORDER.index(sel)}]},
        {"title": "Overfitting, underfitting, bias and risks",
         "tables": [{"head": ["Model", "Train RMSE", "Val. RMSE", "Gap", "Train R²", "Val. R²"], "rows": over}],
         "figures": [fig("overfitting.png", "Train vs validation error."), fig("test_by_junction.png", "Test error by intersection.")],
         "bullets": ["<b>Underfitting:</b> the seasonal mean and Ridge stay below R² 0.80 in validation; they cannot model the interactions and the growth.",
                     f"<b>Overfitting:</b> the tree models fit training better than validation (gap of about 2 vehicles per hour); "
                     f"it is moderate and controlled by depth limits, minimum leaf size and early comparison on validation.",
                     f"<b>Bias by intersection:</b> the test MAE is {tj['1']['mae']} at junction 1 (it grew fastest and the model underestimates it), "
                     f"{tj['2']['mae']} at 2, {tj['3']['mae']} at 3 and {tj['4']['mae']} at 4. Inside a single intersection the R² is lower "
                     f"({', '.join(tj[j]['r2'].__str__() for j in sorted(tj))}) because much of the global R² comes from the difference between intersections.",
                     "<b>Trend risk:</b> trees keep the last traffic level they saw; the permutation importance of <i>days_since_start</i> is 0 on "
                     "validation because those dates are all later than the training data. Monthly retraining handles this (phase 6).",
                     "<b>Data risk:</b> the counts are a public dataset used as a case study for Chía, not field measurements."]},
        {"title": "Review with a domain-knowledgeable person",
         "tables": [{"head": ["Reviewer", "Feedback", "Action taken"], "rows": [
             ["Course professor (Machine Learning)", "Compare the models with the same metrics (MSE, MAE, R²) in a table.", "Comparison tables in the dashboard and in this phase."],
             ["Course professor (Machine Learning)", "A model must reach R² ≥ 0.80 to be usable; the first version reached about 0.68.",
              "Added the date trend and the intersection, rebuilt the protocol with time-ordered partitions; the selected model reaches R² " + str(test["r2"]) + " on the independent test set."]]}],
         "note": "Pending: a review with the mobility office of Chía before any field use."},
        {"title": "Model selection",
         "paragraphs": [f"Criterion: lowest MSE on the validation set among the tuned models. <b>Selected: {LABELS[sel]}</b> "
                        f"({_params(fin['params'])}), validation MSE {fam[sel]['validation']['mse']} and R² {fam[sel]['validation']['r2']}."],
         "figures": [fig("feature_importance.png", "Permutation importance on validation. Most useful: " +
                         ", ".join(f"{k}" for k, _ in imp) + ".")]},
        {"title": "Final performance on the independent test set",
         "paragraphs": [f"The selected model was refitted on train + validation and evaluated <b>once</b> on May – June 2017, "
                        "a period never used for tuning or selection."],
         "tables": [{"head": ["Model", "MSE", "RMSE", "MAE (veh/h)", "R²"], "rows": [
             [LABELS[sel] + " ✓", test["mse"], test["rmse"], test["mae"], test["r2"]],
             ["Seasonal mean (baseline, for reference)", base["mse"], base["rmse"], base["mae"], base["r2"]]], "highlight": 0},
                    {"caption": "Test by intersection", "head": ["Junction", "Intersection", "MAE", "RMSE", "R²"],
                     "rows": [[j, names.get(j, "–"), tj[j]["mae"], tj[j]["rmse"], tj[j]["r2"]] for j in sorted(tj)]}],
         "figures": [fig("test_actual_vs_pred.png", "Independent test: actual vs predicted for one week.")]},
    ]
    if ctl:
        sections.append({"title": "Downstream check: control agent in the simulator",
                         "paragraphs": [f"With the selected forecast as demand, the DQN agent reduced the average queue by "
                                        f"<b>{ctl['improvement_mean']}%</b> against a fixed 30 s signal (mean of 5 trainings, range "
                                        f"{ctl['improvement_min']}% to {ctl['improvement_max']}%), serving {ctl['dqn']['throughput']} vehicles "
                                        f"per episode against {ctl['fixed']['throughput']}. The agent beat the fixed plan in "
                                        f"{sum(1 for r in ctl['runs'] if r['improvement_pct'] > 0)} of {len(ctl['runs'])} trainings; because one "
                                        "training can be worse, every new agent is evaluated against the fixed plan before it is used. "
                                        "Simulation results, not field measurements."],
                         "tables": [{"head": ["Junction", "Intersection", "Avg. queue DQN", "Avg. queue fixed"],
                                     "rows": [[j, names.get(j, "–"), v["dqn"], v["fixed"]] for j, v in sorted(ctl["by_intersection"].items())]}]})
    return {"intro": f"The selected model is <b>{LABELS[sel]}</b>: R² {test['r2']} and MAE {test['mae']} vehicles per hour on the "
                     f"independent test set, {gain:.0f}% less error than the seasonal baseline.", "sections": sections}


def _phase5(rep, q, fam, fin, sp, sel, test, base, names, gain, mon, ctl):
    return {"intro": f"The selected {LABELS[sel]} and its preprocessing are saved in one file and loaded by the web application, "
                     "which already works as the information system of Activity 2.",
            "sections": [
        {"title": "Saved model and preprocessing components",
         "tables": [{"head": ["File", "Content"], "rows": [
             ["artifacts/model.joblib", f"Fitted TrafficFeaturizer (K-Means tables, caps, encoding) + {LABELS[sel]} + version and test metrics"],
             ["artifacts/report.json", "Data quality, partitions, cross-validation, validation, selection and test results"],
             ["artifacts/experiments.csv", "Every hyperparameter configuration tried"],
             ["artifacts/monitoring.json", "Summary of the monitoring test (simulated operation)"],
             ["artifacts/control_evaluation.json", "Agent vs fixed-time evaluation (5 trainings)"]]}],
         "code": "python train.py      # rebuilds every artifact and figure (about 4 minutes on a laptop)\npython monitor.py    # monitoring test with simulated operational data"},
        {"title": "Functional demonstration",
         "paragraphs": ["The <a href=\"/demo\">Demo page</a> accepts a new date, hour and intersection and returns the forecast, "
                        "the type of moment and the 24-hour curve of that day. It is public, so it can be tested without logging in."],
         "figures": [fig("demo_evidence.png", "Deployment evidence: the demo answering a new input with the saved model.")]
         if os.path.exists(os.path.join(ROOT, "static", "img", "crisp", "demo_evidence.png")) else []},
        {"title": "Inputs, outputs, dependencies and execution",
         "tables": [{"head": ["Item", "Detail"], "rows": [
             ["Input", "date (YYYY-MM-DD), hour (0–23), junction (1–4) — or datetime"],
             ["Output", "vehicles_per_hour, group, hourly_curve (24 values), model, model_version, notes"],
             ["API", "POST /v1/forecast (JSON), POST /v1/feedback, GET /v1/monitoring"],
             ["Dependencies", "Python 3.12, Flask, gunicorn, pandas, NumPy, scikit-learn 1.9.1 (pinned for the saved model), PyTorch (CPU), joblib, matplotlib (training only)"],
             ["Run locally", "pip install -r requirements.txt → python app.py → http://localhost:5000"],
             ["Production", "gunicorn app:app on Render (render.yaml)"]]}],
         "code": "curl -X POST https://<app>/v1/forecast -H \"Content-Type: application/json\" \\\n     -d '{\"date\": \"2017-06-12\", \"hour\": 8, \"junction\": 1}'"},
        {"title": "Initial user instructions",
         "bullets": ["Open the Demo, choose date, hour and intersection, press Predict.",
                     "Read the expected vehicles per hour and the group; the bar chart shows the whole day.",
                     "If the value looks wrong, use the Feedback form (type Incident) and describe what you observed.",
                     "Team members log in to use the Dashboard (training and comparison) and the Simulation."]},
        {"title": "Integration with the information system (Activity 2)",
         "bullets": ["Done: the Dashboard and the Simulation load the same artifacts/model.joblib through <code>PredictionService</code>; "
                     "the simulator uses the forecast as the arrival rate of the DQN agent.",
                     "Done: the JSON API lets other modules (or the mobility office) request forecasts.",
                     "Next: a monthly job runs train.py with the new counts, stores the new artifact with its version and replaces the "
                     "current one only if it is better on the latest month (phase 6).",
                     "Next: replace the public dataset with the counts of the Chía detectors when they are available."]},
        {"title": "Monitoring, incident reporting and user feedback",
         "bullets": ["Every prediction is logged (time, input, output, model version, latency) in logs/predictions.jsonl.",
                     "Errors are logged in logs/errors.jsonl; feedback and incidents in logs/feedback.jsonl.",
                     "Incidents are reviewed within one working day by the model owner; if the model is wrong for an intersection, "
                     "the dashboard can fall back to the historical average while it is retrained.",
                     "GET /v1/monitoring summarizes counts, errors, latency and the latest records."]},
    ]}


def _phase6(rep, q, fam, fin, sp, sel, test, base, names, gain, mon, ctl):
    secs = [
        {"title": "Indicators", "tables": [{"head": ["Indicator", "How it is measured", "Threshold"], "rows": [
            ["MAE (7-day window)", "Forecast vs observed counts per day", f"> 1.5 × test MAE ({round(1.5 * test['mae'], 2)}) → retrain"],
            ["Bias (7-day window)", "Mean of observed − predicted", "Sustained positive bias → traffic grew"],
            ["PSI of the volume per intersection", "7-day window vs the last 8 training weeks", "> 0.25 → drift warning"],
            ["Latency p95", "Logged per prediction", "> 500 ms → check the server"],
            ["Error rate and feedback", "logs/errors.jsonl and logs/feedback.jsonl", "Any incident → review within one working day"]]}]},
        {"title": "Detecting data drift and performance degradation",
         "bullets": ["Data drift: the Population Stability Index compares the distribution of the latest counts with the training reference.",
                     "Performance degradation: as soon as real counts arrive they are compared with the forecasts (MAE and bias).",
                     "Because traffic keeps growing, volume drift appears often; retraining is triggered by the error, and drift is a warning that explains it."]},
        {"title": "Review frequency, responsibilities and retraining criteria",
         "tables": [{"head": ["Task", "Frequency", "Responsible"], "rows": [
             ["Check indicators and incidents", "Weekly", "Edward Gil (forecast model owner)"],
             ["Data quality of new counts", "On every new batch", "Erick (data and clustering)"],
             ["Retrain and compare the forecast", "Monthly, or when MAE 7d crosses the threshold", "Edward Gil"],
             ["Retrain and evaluate the DQN agent", "After each new forecast model", "Santiago (control agent)"],
             ["Deploy the new version and keep the app running", "After approval", "Ivan (application and deployment)"]]}],
         "bullets": ["Retrain when: MAE 7d > 1.5 × test MAE, or a month has passed, or a new intersection is added.",
                     "Promote the new model only if its MAE on the latest month is lower than the current model's."]},
        {"title": "Risks, quality controls and update procedure",
         "bullets": ["Risk: traffic growth beyond the training range → monthly retraining.",
                     "Risk: sensor failures or special events → caps on the target and incident reports.",
                     "Risk: a bad model in production → versioned artifacts and the historical-average fallback.",
                     "Quality controls: fixed seeds, the same partitions and metrics in every run, and the experiment log.",
                     "Update procedure: 1) add the new data, 2) python train.py, 3) compare with the current model on the latest month, "
                     "4) commit the new artifact through a pull request, 5) Render redeploys, 6) python monitor.py to confirm."]},
    ]
    if mon:
        secs.append({"title": "Initial monitoring test (simulated operational data)",
                     "paragraphs": [f"<span class=\"sim-label\">SIMULATED OPERATIONAL DATA</span> {mon['description']}",
                                    f"Result: {mon['predictions_logged']:,} predictions logged over {mon['days']} days. The 7-day MAE stayed at "
                                    f"{mon['mae_7d_before_drift']} before the injected change; the first <b>retraining alert</b> was raised on "
                                    f"<b>{mon['first_retrain_alert_day']}</b>, three days after the change, and the last week reached "
                                    f"{mon['mae_7d_last_week']}. Each daily record is in <code>{mon['log_file']}</code> with <code>\"simulated\": true</code>."],
                     "figures": [fig("monitoring_test.png", "SIMULATED: 7-day MAE with the retraining threshold (top) and PSI per intersection (bottom).")]})
    return {"intro": "Traffic changes: new housing, road works and growth alter the patterns the model learned. "
                     "This phase defines how we notice it and what we do.", "sections": secs}
