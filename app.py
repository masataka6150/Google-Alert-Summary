import os
import streamlit as st
import pandas as pd
from alert_processor import (
    extract_related_terms,
    save_terms_to_csv,
    extract_text_from_url,
    extract_real_url,
    is_semantically_related,
    summarize_text,
    save_article_summaries_to_csv,
    RELATED_TERMS_CSV
)
import feedparser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
import datetime

# Streamlit UI Config
st.set_page_config(page_title="Googleアラート記事要約フィルタ", page_icon="📰", layout="wide")

# --------------------
# セッション初期化
# --------------------
if "mode" not in st.session_state:
    st.session_state["mode"] = "default"
if "articles" not in st.session_state:
    st.session_state["articles"] = []
if "selected_articles" not in st.session_state:
    st.session_state["selected_articles"] = []
if "chat_articles" not in st.session_state:
    st.session_state["chat_articles"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "llm" not in st.session_state:
    st.session_state["llm"] = ChatOpenAI(model_name="gpt-4o", temperature=0.3)

# --------------------
# 通常モード
# --------------------
if st.session_state["mode"] != "conversation":
    st.title("📰 Googleアラート記事の自動要約＆関連性フィルタリング")

    st.sidebar.header("🔧 設定")
    rss_url = st.sidebar.text_input("RSSフィード URL", "GoogleアラートのRSSフィードURLを入力")
    user_keywords = st.sidebar.text_area("🔍 検索キーワード", "Googleアラートで設定しているキーワードを入力")
    csv_filename = st.sidebar.text_area("📃 要約記事ダウンロード時のファイル名", "記事をcsv出力する時のファイル名を入力。日付_設定したファイル名_summary.csvのファイル名で出力されます")

    if st.sidebar.button("▶ 要約と認識"):
        with st.spinner("関連語を抽出中..."):
            alert_keywords = [kw.strip() for kw in user_keywords.split(",") if kw.strip()]
            related_terms_str = extract_related_terms(alert_keywords)
            related_df = save_terms_to_csv(related_terms_str)

        st.session_state["related_df"] = related_df
        st.session_state["keywords"] = alert_keywords
        st.session_state["articles"] = []

        processed_titles = set()
        with st.spinner("RSSフィードを読み込み中..."):
            feed = feedparser.parse(rss_url)
            for entry in feed.entries:
                real_url = extract_real_url(entry.link)
                pub_date = entry.published if "published" in entry else ""
                title = entry.title.strip()

                if title in processed_titles:
                    continue

                text = extract_text_from_url(real_url)
                if not text:
                    continue

                if is_semantically_related(text, related_df, alert_keywords):
                    summary = summarize_text(text)
                    st.session_state["articles"].append({
                        "title": title,
                        "pub_date": pub_date,
                        "url": real_url,
                        "summary": summary
                    })
                    processed_titles.add(title)

    if st.session_state["articles"]:
        st.subheader("📥 関連ありと判断された記事")
        selected_articles = []

        for i, article in enumerate(st.session_state["articles"]):
            col1, col2, col3 = st.columns([0.05, 0.8, 0.15])
            with col1:
                checked = st.checkbox("", key=f"select_{i}")
            with col2:
                with st.expander(f"📰 {article['title']}（{article['pub_date']}）"):
                    st.markdown(article["summary"])
            if checked:
                selected_articles.append(article)

        if st.button("💬 選択した記事について質問する"):
            st.session_state["chat_articles"] = selected_articles
            st.session_state["mode"] = "conversation"
            st.rerun()
        
        if st.button("📝 記事の要約をCSV出力する"):
            if st.session_state["articles"]:
                # 日付（例：2025-06-21）
                today_str = datetime.date.today().isoformat()
                # 検索ワードを結合してファイル名用に加工（スペースを_に、カンマを-に）
                raw_keywords = ",".join(st.session_state.get("keywords", []))
                keywords_for_filename = csv_filename.replace(" ", "_").replace(",", "-")

                # ファイル名を組み立てる
                filename = f"{today_str}_{keywords_for_filename}_summary.csv"
                
                csv_path = save_article_summaries_to_csv(st.session_state["articles"])
                # st.success(f"CSVファイルを保存しました: {csv_path}")
            with open(csv_path, "rb") as f:
                st.download_button("⬇️ ダウンロード", f, file_name=filename)
                
            # ダウンロードボタンの直後でファイル削除（try-exceptで安全に）
            try:
                os.remove(csv_path)
            except Exception as e:
                st.warning(f"一時ファイル削除に失敗しました: {e}")
        # else:
        #     st.warning("要約済みの記事がありません。")


# --------------------
# 会話モード
# --------------------
# 会話モード
else:
    from langchain.chat_models import ChatOpenAI

    st.title("💬 選択した記事についてChatGPTに質問")

    if st.button("🔙 戻る", key="back_to_default"):
        st.session_state["mode"] = "default"
        st.rerun()  # 通常モードに戻る



    # 記事要約の表示
    for i, a in enumerate(st.session_state["chat_articles"]):
        st.markdown(f"**📌 記事{i+1}：{a['title']}**")
        st.markdown(a["summary"])

    # 初期化（セッション内で1回だけ）
    if "conv_memory" not in st.session_state:
        st.session_state["conv_memory"] = ConversationSummaryBufferMemory(
            llm=st.session_state["llm"],
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=1000
        )

        summaries = "\n".join([f"- {a['summary']}" for a in st.session_state["chat_articles"]])
        system_msg = f"""あなたは記事要約に基づいてユーザーの質問に丁寧に答えるアシスタントです。
以下の内容は、質問のベースとなる記事の要約です。

{summaries}
"""

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_msg),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])


        st.session_state["chat_chain"] = ConversationChain(
            llm=st.session_state["llm"],
            prompt=prompt_template,
            memory=st.session_state["conv_memory"],
            verbose=False
        )

    # # 会話履歴の表示
    # st.markdown("### 💬 会話履歴")
    # for m in st.session_state["conv_memory"].chat_memory.messages:
    #     role = "👤 ユーザー" if m.type == "human" else "🤖 ChatGPT"
    #     st.markdown(f"**{role}**: {m.content}")

    st.markdown("### 💬 会話履歴")

    for m in st.session_state["conv_memory"].chat_memory.messages:
        if m.type == "human":
            # ユーザーの質問（背景：#f0f0f0）
            st.markdown(
                f"""
                <div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
                    <strong>👤 ユーザー:</strong><br>{m.content}
                </div>
                """,
                unsafe_allow_html=True
            )
        elif m.type == "ai":
            # ChatGPTの回答（背景：#e0e0e0）少し濃いグレー
            st.markdown(
                f"""
                <div style='background-color: #e0e0e0; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
                    <strong>🤖 ChatGPT:</strong><br>{m.content}
                </div>
                """,
                unsafe_allow_html=True
            )



    # ユーザー入力
    # 初期化（チャット入力欄を空にするフラグ）
    if "clear_input" not in st.session_state:
        st.session_state["clear_input"] = False

    # ユーザー入力欄
    default_input = "" if st.session_state["clear_input"] else st.session_state.get("chat_input", "")
    user_input = st.text_input("あなたの質問を入力してください", key="chat_input", value=default_input)

    # 送信ボタン処理
    if st.button("送信") and user_input.strip():
        with st.spinner("🤖 回答を生成中..."):
            response = st.session_state["chat_chain"].invoke({"input": user_input})
            # レスポンス処理をここに（省略）

        # 次回の再描画で入力欄を空にするためのフラグを立てて rerun
        st.session_state["clear_input"] = True
        st.rerun()

    # rerun 後の1回だけ入力欄を初期化（描画前に state を変更）
    if st.session_state["clear_input"]:
        st.session_state["clear_input"] = True  # これが False に戻ると次回は保持される
