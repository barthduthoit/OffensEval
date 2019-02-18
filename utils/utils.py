def make_submission(y_pred, mapping, df, filename):
    df_sub = df.copy().drop(columns=["tweet", "clean_tweets"])
    df_sub["task_a"] = y_pred
    df_sub["task_a"] = df_sub["task_a"].map(mapping)
    df_sub.to_csv(filename, index=False, header=False)