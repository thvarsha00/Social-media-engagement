"""
Social Media Engagement Segment Predictor — v2
--------------------------------------------------
Three views:
  1. Predict Single  — manual input, one prediction
  2. Batch Predict    — upload a CSV, get predictions for every row + download
  3. Dashboard         — explore the cluster profiles from training data

RUN:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Engagement Segment Predictor", page_icon="📊", layout="wide")

FEATURES = [
    'likes_count', 'shares_count', 'comments_count',
    'impressions', 'engagement_rate', 'sentiment_score',
    'user_engagement_growth'
]

SEGMENT_COLORS = {
    "Low Engagement": "#EF553B",
    "Moderate Engagement": "#FFA15A",
    "High Engagement": "#00CC96",
    "Viral / Outlier": "#AB63FA",
}

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: rgba(28, 131, 225, 0.08);
        border-radius: 10px;
        padding: 12px 16px;
    }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load("engagement_cluster_classifier.pkl")
    profile = pd.read_csv("cluster_profile.csv", index_col=0)
    return model, profile


def label_clusters(profile: pd.DataFrame) -> dict:
    """Rank clusters by mean engagement_rate so labels stay correct even if
    cluster numbering shifts between training runs."""
    ranked = profile["engagement_rate"].sort_values().index.tolist()
    generic_labels = ["Low Engagement", "Moderate Engagement", "High Engagement", "Viral / Outlier"]
    return {cluster_id: (generic_labels[i] if i < len(generic_labels) else f"Segment {cluster_id}")
            for i, cluster_id in enumerate(ranked)}


def predict_single_tab(model, profile, cluster_labels):
    st.subheader("Enter post / user activity metrics")
    col1, col2 = st.columns(2)

    with col1:
        likes = st.number_input("Likes count", min_value=0, max_value=5000, value=2500, step=50)
        shares = st.number_input("Shares count", min_value=0, max_value=2000, value=1000, step=25)
        comments = st.number_input("Comments count", min_value=0, max_value=1000, value=500, step=10)
        impressions = st.number_input("Impressions", min_value=0, max_value=100000, value=50000, step=500)

    with col2:
        engagement_rate = st.number_input(
            "Engagement rate", min_value=0.0, max_value=35.0, value=0.15, step=0.01,
            help="Typical range is 0.05–0.5; a handful of viral outliers go much higher."
        )
        sentiment_score = st.slider("Sentiment score", -1.0, 1.0, 0.0, 0.01)
        engagement_growth = st.slider("User engagement growth", -0.5, 0.5, 0.0, 0.01)

    if st.button("Predict Segment", type="primary"):
        input_df = pd.DataFrame([{
            "likes_count": likes, "shares_count": shares, "comments_count": comments,
            "impressions": impressions, "engagement_rate": engagement_rate,
            "sentiment_score": sentiment_score, "user_engagement_growth": engagement_growth,
        }])[FEATURES]

        pred_cluster = model.predict(input_df)[0]
        pred_proba = model.predict_proba(input_df)[0]
        label = cluster_labels.get(pred_cluster, f"Segment {pred_cluster}")
        color = SEGMENT_COLORS.get(label, "#636EFA")

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Segment", label)
        m2.metric("Model Confidence", f"{pred_proba.max():.1%}")
        m3.metric("Segment Size (training data)", f"{int(profile.loc[pred_cluster, 'num_posts'])} posts")

        st.subheader(f"How this compares to the '{label}' segment average")
        cluster_avg = profile.loc[pred_cluster, FEATURES]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Your input", x=FEATURES, y=input_df.iloc[0].values, marker_color="#636EFA"))
        fig.add_trace(go.Bar(name=f"{label} avg", x=FEATURES, y=cluster_avg.values, marker_color=color))
        fig.update_layout(barmode="group", height=420, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)


def batch_predict_tab(model, profile, cluster_labels):
    st.subheader("Upload a dataset to predict segments for every row")
    st.caption(
        "Your CSV needs these columns: " + ", ".join(f"`{c}`" for c in FEATURES)
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to get predictions for every row.")
        return

    try:
        data = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Couldn't read that file: {e}")
        return

    missing = [c for c in FEATURES if c not in data.columns]
    if missing:
        st.error(f"Your file is missing these required columns: {', '.join(missing)}")
        st.write("Columns found in your file:", list(data.columns))
        return

    predict_df = data[FEATURES].copy()
    # Drop rows with missing values in required columns rather than crash
    valid_mask = predict_df.notna().all(axis=1)
    if (~valid_mask).sum() > 0:
        st.warning(f"Skipping {(~valid_mask).sum()} row(s) with missing values in required columns.")

    predict_df = predict_df[valid_mask]
    results = data[valid_mask].copy()

    preds = model.predict(predict_df)
    probs = model.predict_proba(predict_df).max(axis=1)

    results["predicted_cluster"] = preds
    results["predicted_segment"] = [cluster_labels.get(p, f"Segment {p}") for p in preds]
    results["confidence"] = probs

    st.success(f"Predicted segments for {len(results)} rows.")

    seg_counts = results["predicted_segment"].value_counts().reset_index()
    seg_counts.columns = ["segment", "count"]

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.pie(
            seg_counts, names="segment", values="count", hole=0.45,
            color="segment", color_discrete_map=SEGMENT_COLORS,
            title="Predicted segment distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        avg_conf = results.groupby("predicted_segment")["confidence"].mean().reset_index()
        fig2 = px.bar(
            avg_conf, x="predicted_segment", y="confidence",
            color="predicted_segment", color_discrete_map=SEGMENT_COLORS,
            title="Average model confidence per segment"
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Full results")
    st.dataframe(results, use_container_width=True)

    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions as CSV", csv, "predicted_segments.csv", "text/csv")


def dashboard_tab(profile, cluster_labels):
    st.subheader("Training data: segment overview")

    display_profile = profile.copy()
    display_profile.insert(0, "segment", [cluster_labels[c] for c in display_profile.index])
    total_posts = display_profile["num_posts"].sum()

    cols = st.columns(len(display_profile))
    for col, (idx, row) in zip(cols, display_profile.iterrows()):
        share = row["num_posts"] / total_posts
        col.metric(row["segment"], f"{int(row['num_posts'])} posts", f"{share:.1%} of data")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        fig = px.pie(
            display_profile, names="segment", values="num_posts", hole=0.45,
            color="segment", color_discrete_map=SEGMENT_COLORS,
            title="Segment sizes in training data"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.bar(
            display_profile, x="segment", y="engagement_rate",
            color="segment", color_discrete_map=SEGMENT_COLORS,
            title="Average engagement rate per segment"
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Feature comparison across segments")
    melted = display_profile.melt(
        id_vars="segment", value_vars=FEATURES, var_name="feature", value_name="avg_value"
    )
    fig3 = px.bar(
        melted, x="feature", y="avg_value", color="segment", barmode="group",
        color_discrete_map=SEGMENT_COLORS, title="Average metric values by segment"
    )
    fig3.update_layout(xaxis_tickangle=-30, height=450)
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Raw cluster profile table"):
        st.dataframe(display_profile, use_container_width=True)


def main():
    st.title("📊 Social Media Engagement Segment Predictor")
    st.caption(
        "Built on the Social Media User Behavior Analysis project — Gradient Boosting "
        "classifier (98% accuracy) predicting engagement segments from post activity metrics."
    )

    try:
        model, profile = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Missing model files. Run the export cell in your notebook and place "
            "`engagement_cluster_classifier.pkl` and `cluster_profile.csv` in this folder."
        )
        return

    cluster_labels = label_clusters(profile)

    tab1, tab2, tab3 = st.tabs(["🔮 Predict Single", "📁 Batch Predict (Upload Dataset)", "📊 Dashboard"])

    with tab1:
        predict_single_tab(model, profile, cluster_labels)
    with tab2:
        batch_predict_tab(model, profile, cluster_labels)
    with tab3:
        dashboard_tab(profile, cluster_labels)


if __name__ == "__main__":
    main()
