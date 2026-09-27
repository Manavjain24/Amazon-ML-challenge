import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = "./data/training_pairs.tsv"
MODEL_PATH = "./data/entity_resolution_model.joblib"

ID_COLUMNS = [
    "source1_entity_id",
    "matched_entity_id",
    "label",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("             ENTITY RESOLUTION MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load features
    # --------------------------------------------------------

    print("\nLoading training features...")

    df = pd.read_csv(
        INPUT_PATH,
        sep="\t",
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    print(
        "Label distribution:"
    )

    print(
        df["label"].value_counts()
    )

    # --------------------------------------------------------
    # Separate X / y
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column not in ID_COLUMNS
    ]

    X = df[feature_columns]
    y = df["label"]

    print(
        f"\nFeatures: {len(feature_columns)}"
    )

    for feature in feature_columns:
        print(
            f"  {feature}"
        )

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print()
    print(
        f"Training rows: {len(X_train):,}"
    )

    print(
        f"Validation rows: {len(X_valid):,}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nTraining HistGradientBoostingClassifier...")

    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\nEvaluating validation set...")

    predictions = model.predict(
        X_valid
    )

    probabilities = model.predict_proba(
        X_valid
    )[:, 1]

    print("\nClassification report:")
    print(
        classification_report(
            y_valid,
            predictions,
            digits=4,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_valid,
            predictions,
        )
    )

    print(
        f"\nROC-AUC: "
        f"{roc_auc_score(y_valid, probabilities):.4f}"
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print(
        "\nFeature importance:"
    )

    if hasattr(model, "feature_importances_"):

        importance = sorted(
            zip(
                feature_columns,
                model.feature_importances_,
            ),
            key=lambda x: x[1],
            reverse=True,
        )

        for feature, value in importance:

            print(
                f"{feature:30s} "
                f"{value:.6f}"
            )

    else:

        print(
            "HistGradientBoostingClassifier does not "
            "provide feature_importances_."
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
        },
        MODEL_PATH,
    )

    print()
    print(
        f"Model saved to: {MODEL_PATH}"
    )

    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()