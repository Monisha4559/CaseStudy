# 👇 Paste the entire Streamlit code I gave you here
import streamlit as st
from pathlib import Path
import zipfile, shutil, os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Currency ML Pipeline + PCA", layout="wide")
st.title("ML Pipeline with PCA for Currency Dataset")

# -------------------- Utilities --------------------
def extract_if_zip(path: Path):
    """Extract zip if needed and return dataset folder"""
    if path.suffix == ".zip":
        extract_dir = Path("dataset_extracted")
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(extract_dir)
        # handle possible nested folder
        subdirs = [p for p in extract_dir.iterdir() if p.is_dir()]
        if len(subdirs) == 1 and any(f.is_dir() for f in subdirs[0].iterdir()):
            return subdirs[0]
        return extract_dir
    return path

def load_image_paths(data_dir: Path):
    """Recursively collect image paths"""
    X_paths, y = [], []
    for f in data_dir.rglob("*"):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            X_paths.append(f)
            y.append(f.parent.name)
    return X_paths, np.array(y)

def image_to_feature(path: Path, size=(64,32)):
    try:
        img = Image.open(path).convert('L').resize(size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.flatten()
    except Exception as e:
        st.warning(f"Failed: {path}, {e}")
        return None

# -------------------- Sidebar: Dataset --------------------
st.sidebar.header("Dataset")
local_path = st.sidebar.text_input("Path to dataset folder or ZIP", "demo_currency_dataset.zip")
DATA_DIR = Path(local_path)

if not DATA_DIR.exists():
    st.error(f"Path not found: {DATA_DIR}")
    st.stop()

DATA_DIR = extract_if_zip(DATA_DIR)

X_paths, y = load_image_paths(DATA_DIR)
if len(X_paths) == 0:
    st.error("No images found in dataset")
    st.stop()

st.sidebar.success(f"Found {len(np.unique(y))} classes, {len(y)} images")

# -------------------- Feature Extraction --------------------
st.header("1) Feature Extraction")
resize_w = st.number_input("Resize width", value=64, min_value=8)
resize_h = st.number_input("Resize height", value=32, min_value=8)
feature_size = (resize_w, resize_h)

progress = st.progress(0)
features, labels = [], []
for i, p in enumerate(X_paths):
    vec = image_to_feature(p, size=feature_size)
    if vec is not None:
        features.append(vec)
        labels.append(y[i])
    progress.progress(int((i+1)/len(X_paths)*100))

X = np.vstack(features)
y = np.array(labels)
st.write(f"Extracted features shape: {X.shape}")

# -------------------- Train/Test Split --------------------
st.header("2) Train / Test Split")
test_size = st.slider("Test size", 0.1, 0.5, 0.2)
random_state = st.number_input("Random state", value=42, min_value=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=int(random_state)
)
st.write(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# -------------------- Pipeline --------------------
st.header("3) Pipeline & Model")
pca_components = st.number_input("PCA components", min_value=2, max_value=min(X.shape[1],200), value=50)
classifier_name = st.selectbox("Classifier", ["LogisticRegression", "SVC", "RandomForest"])

if classifier_name == "LogisticRegression":
    clf = LogisticRegression(max_iter=1000)
elif classifier_name == "SVC":
    clf = SVC(probability=True)
else:
    clf = RandomForestClassifier(n_estimators=100)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=int(pca_components))),
    ("clf", clf)
])

if st.button("Train model"):
    with st.spinner("Training..."):
        pipeline.fit(X_train, y_train)
    st.success("Training complete")

    # Evaluation
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.metric("Test Accuracy", f"{acc:.4f}")

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).T)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=np.unique(y), yticklabels=np.unique(y), ax=ax)
    st.pyplot(fig)

    st.subheader("Cross-validation (5-fold)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(random_state))
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    st.write(scores)
    st.write(f"Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")

    st.subheader("ROC Curves")
    classes = np.unique(y)
    y_test_bin = label_binarize(y_test, classes=classes)
    try:
        y_score = pipeline.predict_proba(X_test)
    except:
        try:
            y_score = pipeline.decision_function(X_test)
        except:
            y_score = None
    if y_score is not None and y_test_bin.shape[1] > 1:
        fig2, ax2 = plt.subplots(figsize=(6,5))
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")
        ax2.plot([0,1],[0,1],'k--')
        ax2.legend()
        st.pyplot(fig2)

    # Save pipeline
    joblib.dump(pipeline, "trained_pipeline.joblib")
    with open("trained_pipeline.joblib", "rb") as f:
        st.download_button("Download trained pipeline", f, file_name="trained_pipeline.joblib")

# -------------------- PCA Scatter --------------------
st.header("4) PCA (2D Projection)")
pca2 = PCA(n_components=2)
proj = pca2.fit_transform(StandardScaler().fit_transform(X))
df = pd.DataFrame(dict(x=proj[:,0], y=proj[:,1], label=y))
fig3, ax3 = plt.subplots(figsize=(7,5))
for lbl in df['label'].unique():
    subset = df[df['label']==lbl]
    ax3.scatter(subset['x'], subset['y'], label=lbl, alpha=0.7)
ax3.legend()
st.pyplot(fig3)


