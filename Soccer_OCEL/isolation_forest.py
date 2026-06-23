import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import pickle as pickle
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from pm4py.algo.transformation.log_to_features import algorithm as log_to_features
from pm4py import convert_to_event_log
from collections import Counter

def encode_non_numeric(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ["case:concept:name", "time:timestamp", "concept:name"]
    
    df = df.copy()
    dict_columns = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, dict)).any()]
    if len(dict_columns)>0:
        dropped = []
        for col in dict_columns:
            if col in df.columns:
                df.drop(columns=col, inplace=True)
                dropped.append(col)
        print("Dropped dict columns:", dropped)

    encoders = {
        "label": {},
        "multilabel": {}
    }

    for col in df.columns:

        if col in exclude_cols:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        col_series = df[col]

        if col_series.apply(lambda x: isinstance(x, tuple)).any():
            # Convert NaN → tuple of (nan, nan)
            df[[f"{col}_0", f"{col}_1"]] = col_series.apply(
                lambda v: pd.Series(v if isinstance(v, tuple) else (np.nan, np.nan))
            )

            df = df.drop(columns=[col])
            continue

        if col_series.apply(lambda x: isinstance(x, list)).any():
            cleaned = col_series.apply(lambda x: x if isinstance(x, list) else [])

            mlb = MultiLabelBinarizer()
            encoded = mlb.fit_transform(cleaned)

            encoded_df = pd.DataFrame(
                encoded,
                columns=[f"{col}_{c}" for c in mlb.classes_],
                index=df.index
            )

            df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
            encoders["multilabel"][col] = mlb
            continue

        le = LabelEncoder()

        cleaned = col_series.fillna("<<MISSING>>")

        encoded = le.fit_transform(cleaned)

        df[col] = encoded
        encoders["label"][col] = le

    return df, encoders

def find_anom(log_df, variant=None, n_estimators=300):
    log = convert_to_event_log(log_df)
    if not variant:
        log_features, feature_names_log = log_to_features.apply(log, parameters={"add_case_identifier_column": True})
    elif variant=='trace':
        from pm4py.algo.transformation.log_to_features.variants import trace_based
        log_features, feature_names_log = log_to_features.apply(log, variant=trace_based, parameters={"add_case_identifier_column": True})
    #log_features, feature_names_log = log_to_features.apply(log, parameters={'STR_EVENT_ATTRIBUTES':str_columns, 'NUM_EVENT_ATTRIBUTES': num_columns})
    log_features_df = pd.DataFrame(log_features, columns=feature_names_log)
    log_features_df = log_features_df.fillna(0)
    case_idlist=log_features_df['@@case_id_column'].tolist()
    log_features_df.drop(columns='@@case_id_column', inplace=True)
    model = IsolationForest(n_estimators=n_estimators)
    model.fit(log_features_df)

    log_features_df["scores"] = model.decision_function(log_features_df)
    results = dict()
    results["avg"] = log_features_df["scores"].mean()
    count_traces = log_features_df["scores"].count() + 1
    anonmalies = log_features_df[log_features_df.scores <= 0].shape[0]
    results["anonmaly_relative_frequency"] = anonmalies/count_traces
    print(results)
    ano_cases=[]
    for idx, row in log_features_df[log_features_df.scores <= 0].iterrows():
        ano_cases.append(case_idlist[idx])
    ano_log=log_df[log_df['case:concept:name'].isin(ano_cases)]
    print(f"Found {len(ano_cases)} traces")
    ano_events,ano_freq=np.unique(ano_log['case:concept:name'], return_counts=True)

    events_per_case_ano = ano_log.groupby("case:concept:name")["concept:name"].unique()
    case_count_ano = Counter()
    for events in events_per_case_ano:
        case_count_ano.update(set(events)) 
    case_count_ano_df = pd.DataFrame.from_dict(case_count_ano, orient="index", columns=["ano_traces_with_event"])

    event_count_ano = ano_log["concept:name"].value_counts()
    case_count_ano_df["event_count_in_ano_log"] = event_count_ano

    event_count_all = log_df["concept:name"].value_counts()
    case_count_ano_df["event_count_in_log"] = event_count_all

    events_per_case_all = log_df.groupby("case:concept:name")["concept:name"].unique()
    case_count_all = Counter()
    for events in events_per_case_all:
        case_count_all.update(set(events))
    case_count_ano_df["trace_fraction_in_anomalies"] = (
        case_count_ano_df["ano_traces_with_event"] / pd.Series(case_count_all)
    )

    case_count_ano_df = case_count_ano_df.fillna(0)

    case_count_ano_df = case_count_ano_df.sort_values("trace_fraction_in_anomalies", ascending=False)

    print(case_count_ano_df)
    return ano_log, log_features_df, case_idlist, case_count_ano_df

