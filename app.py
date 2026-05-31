import streamlit as st
import seaborn as sns
import pandas as pd
import re
import zipfile
from collections import Counter
import emoji
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import helper

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="WhatsApp Chat Analyzer 📱",
    page_icon="📊",
    layout="wide"
)

PRIMARY = "#FF4B4B"
BG_DARK = "#0f172a"
CARD = "#111827"

# ---------- GLOBAL STYLE ----------
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BG_DARK};
        color: #e5e7eb;
        font-family: "SF Pro Text", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 3rem;
    }}
    .metric-card {{
        background: {CARD};
        padding: 1rem 1.2rem;
        border-radius: 0.75rem;
        border: 1px solid #1f2937;
    }}
    .metric-label {{
        font-size: 0.9rem;
        color: #9ca3af;
        margin-bottom: 0.2rem;
    }}
    .metric-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: #f9fafb;
    }}
    .section-title {{
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- PREPROCESS WHATSAPP CHAT ----------
def preprocess(data: str) -> pd.DataFrame:
    pattern = (
        r"\[(\d{1,2}/\d{1,2}/\d{2}, "
        r"\d{1,2}:\d{2}:\d{2}(?:\u202F|\s)?[AP]M)\] "
        r"([\w\W]+?): ([\w\W]+?)(?=\n\[|\Z)"
    )

    rows = []
    for m in re.finditer(pattern, data, flags=re.DOTALL):
        date_str = m.group(1)
        user = m.group(2).strip()
        message = m.group(3).strip()
        rows.append((date_str, user, message))

    if not rows:
        return pd.DataFrame(columns=[
            "date", "user", "message",
            "year", "month", "day", "hour", "minute", "second"
        ])

    df = pd.DataFrame(rows, columns=["date_raw", "user", "message"])

    # IMPORTANT: correct format & no 1900-01-01 fallback
    df["date"] = pd.to_datetime(
        df["date_raw"],
        format="%d/%m/%y, %I:%M:%S %p",   # note space before %p
        errors="coerce",
    )

    # build time parts
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute
    df["second"] = df["date"].dt.second

    return df[[
        "date", "user", "message",
        "year", "month", "day", "hour", "minute", "second"
    ]]


# ---------- STATS & HELPERS ----------
def fetch_stats(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    num_messages = len(df)
    words = sum(len(str(m).split()) for m in df["message"])
    num_media = df["message"].str.contains("media omitted", case=False, na=False).sum()
    num_links = df["message"].str.contains("http", na=False).sum()
    return num_messages, words, num_media, num_links

def most_busy_users(df):
    counts = df["user"].value_counts().head(8)
    percent = (df["user"].value_counts(normalize=True) * 100).round(2).reset_index()
    percent.columns = ["name", "percent"]
    return counts, percent
def most_busy_day(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    day_counts = (
        df["date"].dt.day_name()
        .value_counts()
        .reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
        )
        .fillna(0)
        .reset_index()
    )
    day_counts.columns = ["day", "count"]
    return day_counts

def most_busy_month(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    month_counts = (
        df["date"].dt.month_name()
        .value_counts()
        .reindex(
            ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"]
        )
        .fillna(0)
        .reset_index()
    )
    month_counts.columns = ["month", "count"]
    return month_counts

def activity_heatmap(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    df = df.copy()
    df["day_name"] = df["date"].dt.day_name()
    df["hour"] = df["date"].dt.hour

    pivot = (
        df.pivot_table(
            index="day_name",
            columns="hour",
            values="message",
            aggfunc="count"
        )
        .reindex(["Monday","Tuesday","Wednesday","Thursday",
                  "Friday","Saturday","Sunday"])
        .fillna(0)
    )
    return pivot




def create_wordcloud(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]
    text = " ".join(df["message"].astype(str))
    wc = WordCloud(
        width=800, height=800, background_color="white", min_font_size=10
    ).generate(text)
    return wc

def most_common_words(selected_user, df):
    stop_words = set([
        "the","is","in","at","of","and","to","a","for","you",
        "are","that","this","but","not","with","was","from"
    ])
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    temp = df[df["user"] != "group_notification"]
    temp = temp[~temp["message"].str.contains("media omitted", case=False, na=False)]

    words = []
    for msg in temp["message"]:
        for w in str(msg).lower().split():
            if w not in stop_words and len(w) > 2:
                words.append(w)

    common = pd.DataFrame(Counter(words).most_common(12), columns=["word", "count"])
    return common

def emoji_helper(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    temp = df[df["user"] != "group_notification"]
    temp = temp[~temp["message"].str.contains("media omitted", case=False, na=False)]

    emojis_list = []
    for msg in temp["message"]:
        for item in emoji.emoji_list(str(msg)):
            emojis_list.append(item["emoji"])

    emoji_df = pd.DataFrame(Counter(emojis_list).most_common(20),
                            columns=["emoji", "count"])
    return emoji_df

def user_emoji_count(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]
    temp = df[df["user"] != "group_notification"]
    emojis_list = []
    for msg in temp["message"]:
        for item in emoji.emoji_list(str(msg)):
            emojis_list.append(item["emoji"])
    return len(emojis_list)
def build_timelines(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    daily = (
        df.groupby(df["date"].dt.date)["message"]
        .count()
        .reset_index()
        .rename(columns={"date": "day", "message": "count"})
    )

    monthly = (
        df.groupby(df["date"].dt.to_period("M"))["message"]
        .count()
        .reset_index()
        .rename(columns={"date": "month", "message": "count"})
    )
    monthly["month"] = monthly["month"].astype(str)

    yearly = (
        df.groupby(df["date"].dt.year)["message"]
        .count()
        .reset_index()
        .rename(columns={"date": "year", "message": "count"})
    )

    return daily, monthly, yearly


def most_linked_urls(selected_user, df, top_n=10):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]
    urls = df["message"].str.extractall(r'(https?://\S+)')[0]
    if urls.empty:
        return pd.DataFrame(columns=["url", "count"])
    url_counts = urls.value_counts().head(top_n).reset_index()
    url_counts.columns = ["url", "count"]
    return url_counts

def response_time_stats(selected_user, df):
    """
    Returns:
      stats_df: per-user avg/median reply time (in minutes)
      overall_avg: overall average reply time (minutes)
      overall_median: overall median reply time (minutes)
    """
    if df.empty:
        return pd.DataFrame(columns=["user", "avg_minutes", "median_minutes"]), 0, 0

    # sort by time
    df = df.sort_values("date").reset_index(drop=True)

    # time diff from previous message
    df["time_diff"] = df["date"].diff()

    # only keep diffs where sender changed (actual replies)
    df["prev_user"] = df["user"].shift(1)
    mask_reply = df["user"] != df["prev_user"]
    reply_df = df[mask_reply & df["time_diff"].notna()].copy()

    if selected_user != "Overall":
        reply_df = reply_df[reply_df["user"] == selected_user]

    if reply_df.empty:
        return pd.DataFrame(columns=["user", "avg_minutes", "median_minutes"]), 0, 0

    # convert to minutes
    reply_df["minutes"] = reply_df["time_diff"].dt.total_seconds() / 60.0

    # remove huge gaps (e.g., > 1 day) so they don't skew stats
    reply_df = reply_df[reply_df["minutes"] <= 24 * 60]

    # per-user stats
    stats = (
        reply_df.groupby("user")["minutes"]
        .agg(["mean", "median"])
        .reset_index()
        .rename(columns={"mean": "avg_minutes", "median": "median_minutes"})
    )

    overall_avg = reply_df["minutes"].mean()
    overall_median = reply_df["minutes"].median()

    return stats, overall_avg, overall_median


# ---------- SIDEBAR ----------
st.sidebar.markdown("### WhatsApp Chat 📱")
st.sidebar.caption("Visual insights from your conversations")

uploaded_file = st.sidebar.file_uploader(
    "Upload WhatsApp chat (.txt or .zip)", type=["txt", "zip"]
)
analyze_button = st.sidebar.button("✨ Run Analysis")

# ---------- MAIN HEADER ----------
st.markdown(
    "<h1>WhatsApp Chat Dashboard <span style='font-size:1.8rem'>📊💬</span></h1>",
    unsafe_allow_html=True,
)
st.write("Analyze message activity, vocabulary, and emoji usage from WhatsApp chats.")

df = None
media_files = []

# ---------- LOAD DATA ----------
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".txt"):
            data = uploaded_file.getvalue().decode("utf-8-sig")
            df = preprocess(data)
        elif uploaded_file.name.endswith(".zip"):
            with zipfile.ZipFile(uploaded_file) as z:
                chat_file = [f for f in z.namelist() if f.endswith(".txt")][0]
                with z.open(chat_file) as f:
                    data = f.read().decode("utf-8-sig")
                df = preprocess(data)
                media_files = [f for f in z.namelist() if not f.endswith(".txt")]
    except Exception as e:
        st.error(f"Could not read this file: {e}")
else:
    st.info("Upload a WhatsApp chat (.txt or .zip) and click 'Run Analysis'.")


# ---------- MAIN LOGIC ----------
if df is None or df.empty:
    st.info("Upload a chat file or enable the sample chat in the sidebar, then click **Run Analysis**.")
else:
    # user selection
    user_list = df["user"].unique().tolist()
    if "group_notification" in user_list:
        user_list.remove("group_notification")
    user_list.sort()
    user_list.insert(0, "Overall")

    top_row = st.columns([2, 1])
    with top_row[0]:
        st.markdown(
            f"**Loaded:** {len(df)} messages from `{df['user'].nunique()}` participants"
        )
    with top_row[1]:
        selected_user = st.selectbox("Focus on user", user_list, index=0)

    if not analyze_button:
        st.caption("Press **Run Analysis** in the sidebar to generate visualizations.")
    else:
        # ---------- USER HEADER ----------
        display_name = selected_user if selected_user != "Overall" else "All participants"
        st.markdown(
            f"<h2 style='margin-top:0.5rem'>{display_name}</h2>",
            unsafe_allow_html=True,
        )
        st.caption(f"Showing activity summary for {display_name.lower()}.")

        # ---------- METRICS ----------
        num_messages, words, num_media, num_links = fetch_stats(selected_user, df)
        total_emojis = user_emoji_count(selected_user, df)

        c1, c2, c3, c4 = st.columns(4)
        for col, label, value, icon in [
            (c1, "Messages", num_messages, "💬"),
            (c2, "Words", words, "✍️"),
            (c3, "Emojis sent", total_emojis, "😄"),
            (c4, "Links shared", num_links, "🔗"),
        ]:
            with col:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-label'>{icon} {label}</div>"
                    f"<div class='metric-value'>{value}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # ---------- TABS ----------
        tab_overview, tab_words, tab_emojis = st.tabs(
            ["Overview 🔍", "Words ☁️", "Emojis 😄"]
        )

        # ===== OVERVIEW TAB =====
        with tab_overview:
            # ... Active Participants + sample messages ...

            # ----- TIME‑SERIES ACTIVITY -----

            st.markdown(
                "<div class='section-title'>Active participants 👥</div>",
                unsafe_allow_html=True,
            )

            counts, percent = most_busy_users(df)

            o1, o2 = st.columns([1.5, 1])
            with o1:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(counts.index, counts.values, color="#38bdf8")
                ax.set_ylabel("Messages")
                ax.set_title("Top active users")
                ax.tick_params(axis="x", rotation=45)
                fig.tight_layout()
                st.pyplot(fig)

            with o2:
                st.caption("Share of messages per user (%)")
                st.dataframe(
                    percent,  # columns: name, percent
                    use_container_width=True,
                )

            st.markdown(
                "<div class='section-title'>Activity over time 📈</div>",
                unsafe_allow_html=True,
            )

            daily, monthly, yearly = build_timelines(selected_user, df)

            t1, t2 = st.columns(2)
            with t1:
                st.caption("Messages per day")
                fig_d, ax_d = plt.subplots(figsize=(6, 3))
                ax_d.plot(daily["day"], daily["count"], color="#22c55e")

                # show only every 5th date label
                ax_d.set_xticks(daily["day"][::5])
                ax_d.set_xlabel("Date")
                ax_d.set_ylabel("Messages")
                ax_d.tick_params(axis="x", rotation=45)
                st.pyplot(fig_d)

            with t2:
                st.caption("Messages per month")
                fig_m, ax_m = plt.subplots(figsize=(6, 3))
                ax_m.plot(monthly["month"], monthly["count"], marker="o", color="#f97316")
                ax_m.set_xlabel("Year‑Month")
                ax_m.set_ylabel("Messages")
                ax_m.tick_params(axis="x", rotation=60)
                st.pyplot(fig_m)

                st.caption("Messages per year")
                if len(yearly) > 1:
                    fig_y, ax_y = plt.subplots(figsize=(6, 3))
                    ax_y.bar(yearly["year"], yearly["count"], color="#38bdf8")
                    ax_y.set_xlabel("Year")
                    ax_y.set_ylabel("Messages")
                    ax_y.set_xticks(yearly["year"])
                    st.pyplot(fig_y)
                else:
                    year = int(yearly["year"].iloc[0])
                    count = int(yearly["count"].iloc[0])
                    st.info(f"All messages in this chat are from {year} ({count} messages).")

            # ===== NEW: MOST BUSY DAY / MONTH / HEATMAP =====

            # Most busy day
            st.markdown(
                "<div class='section-title'>Most busy day 📊</div>",
                unsafe_allow_html=True,
            )
            day_counts = most_busy_day(selected_user, df)
            fig_bday, ax_bday = plt.subplots(figsize=(4, 3))
            ax_bday.bar(day_counts["day"], day_counts["count"], color="#22c55e")
            ax_bday.set_xlabel("Day of week")
            ax_bday.set_ylabel("Messages")
            ax_bday.tick_params(axis="x", rotation=45)
            st.pyplot(fig_bday)

            # Most busy month
            st.markdown(
                "<div class='section-title'>Most busy month 📊</div>",
                unsafe_allow_html=True,
            )
            month_counts = most_busy_month(selected_user, df)
            fig_bmon, ax_bmon = plt.subplots(figsize=(4, 3))
            ax_bmon.bar(month_counts["month"], month_counts["count"], color="#f97316")
            ax_bmon.set_xlabel("Month")
            ax_bmon.set_ylabel("Messages")
            ax_bmon.tick_params(axis="x", rotation=60)
            st.pyplot(fig_bmon)

            # Weekly activity heatmap
            st.markdown(
                "<div class='section-title'>Weekly activity heatmap 🔥</div>",
                unsafe_allow_html=True,
            )
            heatmap_df = activity_heatmap(selected_user, df)
            fig_h, ax_h = plt.subplots(figsize=(8, 4))
            sns.heatmap(heatmap_df, cmap="copper", ax=ax_h)
            ax_h.set_xlabel("Hour of day")
            ax_h.set_ylabel("Day of week")
            st.pyplot(fig_h)

            st.markdown(
                "<div class='section-title'>Response time analysis ⏱️</div>",
                unsafe_allow_html=True,
            )

            stats_rt, overall_avg, overall_median = response_time_stats(selected_user, df)

            if stats_rt.empty:
                st.info("Not enough data to compute reply times for this selection.")
            else:
                # keep only top 5 users by message count (or by number of replies)
                top5 = stats_rt.sort_values("median_minutes").head(5)

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Average reply time (overall)",
                              f"{overall_avg:.1f} min")
                with c2:
                    st.metric("Median reply time (overall)",
                              f"{overall_median:.1f} min")

                # bar chart: median reply time for top 5 users
                fig_rt, ax_rt = plt.subplots(figsize=(8, 4))
                ax_rt.bar(top5["user"], top5["median_minutes"], color="#6366f1")
                ax_rt.set_xlabel("User")
                ax_rt.set_ylabel("Median reply time (min)")
                ax_rt.set_xticklabels(top5["user"], rotation=30, ha="right")
                fig_rt.tight_layout()
                st.pyplot(fig_rt)

                st.caption("Per‑user reply time (minutes)")
                st.dataframe(
                    stats_rt.sort_values("median_minutes"),
                    use_container_width=True,
                )

        # ===== WORDS TAB =====
        with tab_words:
            st.markdown(
                "<div class='section-title'>Word cloud ☁️</div>",
                unsafe_allow_html=True,
            )
            wc = create_wordcloud(selected_user, df)
            fig_wc, ax_wc = plt.subplots()
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")
            st.pyplot(fig_wc)

            st.markdown(
                "<div class='section-title'>Most common words 📝</div>",
                unsafe_allow_html=True,
            )
            common_df = most_common_words(selected_user, df)
            fig_cw, ax_cw = plt.subplots()
            ax_cw.barh(common_df["word"], common_df["count"], color="#f97316")
            plt.gca().invert_yaxis()
            st.pyplot(fig_cw)
            st.dataframe(common_df, use_container_width=True)

        # ===== EMOJIS TAB =====
        with tab_emojis:
            st.markdown(
                "<div class='section-title'>Emoji usage 😄</div>",
                unsafe_allow_html=True,
            )

            emoji_df = emoji_helper(selected_user, df)

            e1, e2 = st.columns([1, 2])
            with e1:
                st.dataframe(emoji_df, use_container_width=True)

            with e2:
                if emoji_df.empty:
                    st.info("No emojis found for this selection.")
                else:
                    top5 = emoji_df.head(5)

                    plt.rcParams["font.family"] = [
                        "Apple Color Emoji",  # macOS
                        "Segoe UI Emoji",  # Windows
                        "Noto Color Emoji",  # Linux
                        "sans-serif",
                    ]

                    fig, ax = plt.subplots(figsize=(6, 6))
                    wedges, texts, autotexts = ax.pie(
                        top5["count"],
                        labels=None,
                        autopct="%1.1f%%",
                        textprops={"fontsize": 12},
                    )

                    for i, wedge in enumerate(wedges):
                        angle = (wedge.theta2 + wedge.theta1) / 2.0
                        x = 0.7 * np.cos(np.deg2rad(angle))
                        y = 0.7 * np.sin(np.deg2rad(angle))
                        ax.text(
                            x,
                            y,
                            top5["emoji"].iloc[i],
                            ha="center",
                            va="center",
                            fontsize=28,
                            fontfamily="Apple Color Emoji",
                        )

                    ax.axis("equal")
                    st.pyplot(fig)

            # ----- EMOJI CLOUD (ADD THIS BLOCK) -----
            # ----- EMOJI CLOUD (names instead of glyphs) -----
            from emoji import demojize

            if not emoji_df.empty:
                st.markdown(
                    "<div class='section-title'>Emoji cloud ☁️</div>",
                    unsafe_allow_html=True,
                )

                # Build frequency dict with readable labels like "smiling face"
                freq = {}
                for _, row in emoji_df.iterrows():
                    emoji_char = row["emoji"]
                    count = row["count"]
                    if pd.notna(emoji_char) and pd.notna(count):
                        try:
                            count_int = int(count)
                        except Exception:
                            continue
                        if count_int > 0:
                            label = demojize(str(emoji_char)).strip(":").replace("_", " ")
                            freq[label] = freq.get(label, 0) + count_int

                if len(freq) == 0:
                    st.info("Not enough emoji data to build a cloud.")
                else:
                    emoji_wc = WordCloud(
                        width=600,
                        height=400,
                        background_color="white",
                        collocations=False,
                    ).generate_from_frequencies(freq)

                    fig_ec, ax_ec = plt.subplots(figsize=(6, 4))
                    ax_ec.imshow(emoji_wc, interpolation="bilinear")
                    ax_ec.axis("off")
                    st.pyplot(fig_ec)

            # ----- MOST LINKED URLS (KEEP EXISTING CODE) -----
            st.markdown(
                "<div class='section-title'>Most linked URLs 🔗</div>",
                unsafe_allow_html=True,
            )
            url_df = most_linked_urls(selected_user, df)
            if url_df.empty:
                st.info("No URLs found in this selection.")
            else:
                st.dataframe(url_df, use_container_width=True)
