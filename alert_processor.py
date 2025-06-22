import os
import requests
import feedparser
import csv
import time
from urllib.parse import urlparse, parse_qs
from newspaper import Article
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory


# 環境変数の読み込み
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# 🔹 ユーザーが手動で設定したキーワード
alert_keywords = ["再生可能エネルギー", "省エネ", "カーボンニュートラル", "エネルギー政策"]

# 🔹 関連語CSVの保存パス
RELATED_TERMS_CSV = "related_terms.csv"

# 🔹 RSS URL（Googleアラートから取得）
RSS_FEED_URL = "https://www.google.com/alerts/feeds/xxxxxxxxxxxxxxxxxxxxxxxx"

# 関連語の抽出関数
def extract_related_terms(keywords, top_n=10):
    prompt = f"以下のキーワードに関連する重要な単語を、それぞれについて2つずつ挙げてください。\n\nキーワード:\n" + "\n".join(keywords)
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# 抽出結果をCSVに保存
def save_terms_to_csv(terms_str):
    terms = []
    for line in terms_str.splitlines():
        if ":" in line:
            keyword, related = line.split(":", 1)
            for word in related.strip().split("、"):
                terms.append([keyword.strip(), word.strip()])
    df = pd.DataFrame(terms, columns=["Keyword", "RelatedTerm"])
    df.to_csv(RELATED_TERMS_CSV, index=False, encoding="utf-8")
    return df

# 記事本文抽出
def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip()
    except Exception:
        return ""

# 関連性チェック（意味的）
def is_semantically_related(text, keywords_df, original_keywords):
    if text.strip() == "":
        return False
    keyword_block = ", ".join(original_keywords)
    related_terms_block = ", ".join(keywords_df["RelatedTerm"].tolist())
    check_prompt = f"""以下の記事本文と、検索キーワード「{keyword_block}」およびその関連語「{related_terms_block}」との意味的な関連性を評価してください。
関連性がある場合は「Yes」、関連性がない場合は「No」のみを返してください。

記事本文:
{text[:1500]}"""  # 最初の1500文字で評価

    messages = [{"role": "user", "content": check_prompt}]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0
    )
    result = response.choices[0].message.content.strip().lower()
    return "yes" in result

# 要約処理（箇条書き）
def summarize_text(text):
    prompt = f"""以下の文章を、1項目50〜100文字の箇条書き3点で要約してください。要約の文字数は必ず50〜100文字の間におさめてください。\n\n{text[:3000]}"""
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# URLデコード処理
def extract_real_url(google_url):
    parsed = urlparse(google_url)
    params = parse_qs(parsed.query)
    return params.get("url", [google_url])[0]

# アラート処理本体
def process_alerts():
    print("✅ 関連語を抽出中...")
    related_terms_str = extract_related_terms(alert_keywords)
    related_df = save_terms_to_csv(related_terms_str)
    print("✅ 抽出された関連語:\n", related_df)

    print("✅ RSSフィードを読み込み中...")
    feed = feedparser.parse(RSS_FEED_URL)

    for entry in feed.entries:
        google_url = entry.link
        real_url = extract_real_url(google_url)
        print(f"\n🔗 処理中: {real_url}")
        text = extract_text_from_url(real_url)
        if not text:
            print("❌ 本文取得失敗")
            continue
        print(f"✅ 本文取得成功（冒頭100字）: {text[:100]}")

        if is_semantically_related(text, related_df, alert_keywords):
            summary = summarize_text(text)
            print("📝 要約:\n", summary)
        else:
            print("🚫 関連性が低いためスキップ")

import pandas as pd

# alert_processor.py

import pandas as pd

def save_article_summaries_to_csv(articles: list[dict], file_path: str = "article_summaries.csv") -> str:
    """
    記事のタイトル・日付・URL・要約をCSVに保存する関数

    Parameters:
        articles (list[dict]): 各記事は {"title": ..., "pub_date": ..., "url": ..., "summary": ...} を含む辞書
        file_path (str): 保存ファイル名（デフォルト: article_summaries.csv）

    Returns:
        str: 保存したファイルパス
    """
    df = pd.DataFrame([
        {
            "タイトル": a.get("title", ""),
            "日付": a.get("pub_date", ""),
            "要約": a.get("summary", ""),
            "URL": a.get("url", ""),
        }
        for a in articles
    ])
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    return file_path



# 実行
if __name__ == "__main__":
    process_alerts()