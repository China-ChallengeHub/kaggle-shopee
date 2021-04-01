from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def get_stratified_kfold(df, splits=12, seed=42):
    df["enc_label_group"] = LabelEncoder().fit_transform(trn_df["label_group"])
    skf = StratifiedKFold(n_splits=splits, random_state=seed, shuffle=True)
    folds = [-1 for _ in range(len(trn_df))]
    for fold, (_, tst_index) in enumerate(skf.split(trn_df, trn_df["enc_label_group"])):
        for ix in tst_index:
            folds[ix] = fold

    fold = pd.DataFrame({"fold": folds})
    assert (fold["fold"] >= 0).all()
    assert len(fold) == len(df)

    return fold
