import os
import io
import json
import uuid
import sqlite3
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from google import genai
from google.genai import types as genai_types


# ==============
# Configuration
# ==============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)

def read_key(name: str, default: str = "") -> str:
    v = os.getenv(name)
    if v:
        return v
    try:
        return st.secrets[name]  # secrets.toml が無い環境では except 側に落ちる
    except Exception:
        return default

OPENAI_MODEL = read_key("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = read_key("OPENAI_API_KEY", "")
DB_PATH = read_key("LIKES_DB_PATH", os.path.join(BASE_DIR, "likes.db"))


# ==============
# DB utilities
# ==============
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS likes (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            story_title TEXT,
            synopsis TEXT,
            details TEXT,
            prompt TEXT,
            character_image_path TEXT,
            image_urls TEXT
        )
        """
    )
    conn.commit()
    conn.close()

def add_like(entry: Dict[str, Any]):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO likes (id, created_at, story_title, synopsis, details, prompt, character_image_path, image_urls) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            entry.get("id"),
            entry.get("created_at"),
            entry.get("story_title"),
            entry.get("synopsis"),
            entry.get("details"),
            entry.get("prompt"),
            entry.get("character_image_path"),
            json.dumps(entry.get("image_urls", [])),
        ),
    )
    conn.commit()
    conn.close()

def list_likes() -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, created_at, story_title, synopsis, details, prompt, character_image_path, image_urls FROM likes ORDER BY created_at DESC"
    )
    rows = cur.fetchall()
    conn.close()
    items = []
    for r in rows:
        items.append({
            "id": r[0],
            "created_at": r[1],
            "story_title": r[2],
            "synopsis": r[3],
            "details": r[4],
            "prompt": r[5],
            "character_image_path": r[6],
            "image_urls": json.loads(r[7] or "[]"),
        })
    return items

def ensure_tmp_dir() -> str:
    tmp = os.path.join(BASE_DIR, ".cache", "uploads")
    os.makedirs(tmp, exist_ok=True)
    return tmp


# ==============
# Story ideation (OpenAI) — temperature は 1.0 固定
# ==============
def generate_story_options(user_prompt: str, character_desc: str) -> List[Dict[str, str]]:
    if not OPENAI_API_KEY:
        # フォールバック案
        base = []
        for i in range(1, 6):
            base.append({
                "title": f"コンセプト{i}",
                "synopsis": f"『{user_prompt}』と{character_desc}をベースにした方向性。",
                "details": "- 導入\n- 事件\n- 対立\n- 山場\n- 余韻",
            })
        return base

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_msg = {
        "role": "system",
        "content": (
            "あなたはプロの漫画編集者。日本語で異なる方向性を持つ5つの物語案を出す。"
            "必ず JSON で返す。キー: options (配列)。各案に title, synopsis, details を含む。"
            "synopsis は 200字以内。details は 3〜5個の短文の箇条書き。"
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f"キャラクター設定: {character_desc}\n\nテーマ: {user_prompt}\n\n"
            "出力は JSON のみ。説明文なし。"
        ),
    }

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[system_msg, user_msg],
        temperature=1.0,  # 固定
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    data = json.loads(content)
    options = data.get("options", [])
    normed = []
    for opt in options[:5]:
        det = opt.get("details", [])
        if isinstance(det, list):
            det_text = "\n".join([f"- {x}" for x in det])
        else:
            det_text = str(det)
        normed.append({
            "title": opt.get("title", "無題"),
            "synopsis": opt.get("synopsis", ""),
            "details": det_text,
        })
    while len(normed) < 5:
        idx = len(normed) + 1
        normed.append({
            "title": f"コンセプト{idx}",
            "synopsis": "プロンプトに基づく仮案。",
            "details": "- 導入\n- 事件\n- 対立\n- 山場\n- 余韻"
        })
    return normed


# ==============
# ChatGPT に画像生成プロンプト（構図＋セリフ込み）を作らせる
# ==============
def produce_image_prompt(story_title: str, story_details: str) -> str:
    """
    ユーザー選択は不要。バックエンドで最適案を選ぶ。
    シンプルに 1 案のみを JSON で返させ、そのまま採用する。
    失敗時は簡易テンプレにフォールバック。
    """
    if not OPENAI_API_KEY:
        return f"【タイトル】{story_title}\n【指示】コマ割りは3〜4。各コマに短い日本語セリフを入れる。主要キャラの表情が読み取れる構図。"

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_msg = {
        "role": "system",
        "content": (
            "あなたは漫画の構図・演出・セリフ設計に長けたプロ。"
            "与えられたタイトルと詳細（箇条書き）に基づき、画像生成用の単一プロンプトを日本語で作る。"
            "コマ割り・構図・カメラ距離・キャラ配置・吹き出し位置と日本語セリフ内容を含める。"
            "出力は JSON。キー: prompt_text（文字列のみ）。"
            "文字は短く明瞭に。1ページ内のセリフは合計50字以内に抑える。"
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f"タイトル: {story_title}\n"
            "詳細（箇条書き）:\n"
            f"{story_details}\n"
            "上記を元に、画像生成のための単一の最終プロンプトを JSON で返してください。"
        ),
    }

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[system_msg, user_msg],
            temperature=1.0,  # 固定
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        pt = data.get("prompt_text")
        if isinstance(pt, str) and pt.strip():
            return pt.strip()
    except Exception as e:
        st.warning(f"OpenAI prompt produce error: {e}")

    # フォールバック
    return (
        f"【タイトル】{story_title}\n"
        "【コマ割り】3コマ。1コマ目: 導入。2コマ目: 対立。3コマ目: 余韻。\n"
        "【セリフ】短い日本語のみ。各コマ10字以内。明瞭な吹き出し。"
    )


# ==============
# Gemini 画像生成（nano-banana） — ガード付き & 文字描画指示強化
# ==============
def nano_banana_generate_comics(character_image_path: str, prompt_text: str, num: int = 3) -> List[str]:
    """
    prompt_text は最終プロンプト（構図＋吹き出し＋短い日本語セリフ込み）を想定。
    """
    client = genai.Client()

    try:
        pil_img = None
        if character_image_path and os.path.exists(character_image_path):
            pil_img = Image.open(character_image_path).convert("RGB")

        helper = (
            "以下の指示どおりに漫画ページ風の画像を生成してください。\n"
            "画像内の日本語テキスト（セリフ／キャプション）は、読みやすいゴシック体風で、"
            "断筆や崩れのない正確な文字として描画すること。吹き出しを明瞭に。\n"
        )
        full_prompt = helper + prompt_text

        images = []
        for _ in range(num):
            config = genai_types.GenerateContentConfig(
                response_modalities=["Image"],
                image_config=genai_types.ImageConfig(aspect_ratio="3:4"),
            )
            contents = [full_prompt] if pil_img is None else [full_prompt, pil_img]

            resp = client.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=contents,
                config=config,
            )

            candidates = getattr(resp, "candidates", None)
            if not candidates:
                st.warning("DEBUG: resp.candidates is None")
                # フォールバック生成
                img = Image.new("RGB", (768, 1024), color=(230, 230, 230))
                tmp = ensure_tmp_dir()
                fname = os.path.join(tmp, f"placeholder_{uuid.uuid4().hex}.png")
                img.save(fname)
                images.append(fname)
                continue

            cand0 = candidates[0]
            content_obj = getattr(cand0, "content", None)
            parts = getattr(content_obj, "parts", None) if content_obj else None
            parts = parts or []

            saved = False
            for part in parts:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    data = inline.data
                    tmp = ensure_tmp_dir()
                    fname = os.path.join(tmp, f"gemini_{uuid.uuid4().hex}.png")
                    with open(fname, "wb") as f:
                        f.write(data)
                    images.append(fname)
                    saved = True

            if not saved:
                img = Image.new("RGB", (768, 1024), color=(230, 230, 230))
                tmp = ensure_tmp_dir()
                fname = os.path.join(tmp, f"placeholder_{uuid.uuid4().hex}.png")
                img.save(fname)
                images.append(fname)

        return images if images else [os.path.join(ensure_tmp_dir(), "empty.png")]

    except Exception as e:
        st.warning(f"Gemini generation error: {e}")
        fallback = []
        for i in range(num):
            img = Image.new("RGB", (768, 1024), color=(240 - i*20, 240 - i*10, 240))
            tmp = ensure_tmp_dir()
            fname = os.path.join(tmp, f"placeholder_{uuid.uuid4().hex}.png")
            img.save(fname)
            fallback.append(fname)
        return fallback


# ==============
# UI
# ==============
def render_generate_tab():
    st.header("漫画ストーリー・インスピレーター")
    st.caption("ストーリー案生成 → バックエンドで最終プロンプト生成 → 漫画画像生成（3枚）")

    col1, col2 = st.columns([1, 1])
    with col1:
        character_image = st.file_uploader(
            "キャラクター画像をアップロード",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
        )
        if character_image is not None:
            st.image(character_image, caption="プレビュー", use_column_width=True)

    with col2:
        character_desc = st.text_input(
            "キャラクターの説明（見た目・性格など）",
            value="17歳の高校生ヒーロー。人混みが苦手だが心は熱い。",
        )
        user_prompt = st.text_area(
            "プロンプト（テーマや展開案の種）",
            height=120,
            value="親友との別離と再会を軸にした青春バトル",
        )

        if "story_options" not in st.session_state:
            st.session_state.story_options = []
        if st.button("5つの展開案を生成"):
            with st.spinner("ストーリー案を生成中..."):
                st.session_state.story_options = generate_story_options(user_prompt, character_desc)

    st.divider()

    selected_idx = st.session_state.get("selected_option_idx")
    if st.session_state.story_options:
        st.subheader("展開案（5つ）")
        for idx, opt in enumerate(st.session_state.story_options):
            with st.expander(f"{idx+1}. {opt['title']}", expanded=(selected_idx == idx)):
                st.markdown(f"**あらすじ**\n\n{opt['synopsis']}")
                st.markdown("**詳細**")
                st.markdown(opt["details"])

                c1, c2, _ = st.columns([1, 1, 2])
                with c1:
                    if st.button(f"この案で生成", key=f"select_{idx}"):
                        st.session_state.selected_option_idx = idx
                with c2:
                    if st.button(f"この案を編集", key=f"edit_{idx}"):
                        st.session_state.edit_idx = idx

        edit_idx = st.session_state.get("edit_idx")
        if edit_idx is not None:
            st.subheader("選んだ案を編集")
            opt = st.session_state.story_options[edit_idx]
            new_title = st.text_input("タイトル", value=opt["title"], key=f"edit_title_{edit_idx}")
            new_synopsis = st.text_area("あらすじ", value=opt["synopsis"], key=f"edit_synopsis_{edit_idx}")
            new_details = st.text_area("詳細（箇条書き）", value=opt["details"], height=160, key=f"edit_details_{edit_idx}")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("更新"):
                    st.session_state.story_options[edit_idx] = {
                        "title": new_title,
                        "synopsis": new_synopsis,
                        "details": new_details,
                    }
                    st.session_state.edit_idx = None
            with c2:
                if st.button("キャンセル"):
                    st.session_state.edit_idx = None

        # 自動プロンプト生成→画像生成
        selected_idx = st.session_state.get("selected_option_idx")
        if selected_idx is not None:
            opt = st.session_state.story_options[selected_idx]

            # キャラ画像の保存
            char_img_path = None
            if st.session_state.get("character_image_path"):
                char_img_path = st.session_state["character_image_path"]
            elif character_image is not None:
                tmp = ensure_tmp_dir()
                char_img_path = os.path.join(tmp, f"char_{uuid.uuid4().hex}.png")
                image_bytes = character_image.read()
                Image.open(io.BytesIO(image_bytes)).convert("RGB").save(char_img_path)
                st.session_state["character_image_path"] = char_img_path

            st.subheader("バックエンドで最終プロンプトを生成し、画像化します")
            if st.button("漫画画像を3枚生成"):
                with st.spinner("プロンプト生成中..."):
                    final_prompt = produce_image_prompt(opt["title"], opt["details"])
                with st.spinner("画像生成中..."):
                    urls = nano_banana_generate_comics(char_img_path or "", final_prompt, num=3)
                    st.session_state.generated_images = urls
                    st.session_state.final_prompt_used = final_prompt

            urls = st.session_state.get("generated_images", [])
            if urls:
                st.write("生成された画像")
                for i, u in enumerate(urls, start=1):
                    st.image(u, caption=f"生成画像 {i}", use_column_width=True)

                if st.button("いいねに保存"):
                    entry = {
                        "id": uuid.uuid4().hex,
                        "created_at": datetime.utcnow().isoformat(),
                        "story_title": opt["title"],
                        "synopsis": opt["synopsis"],
                        "details": opt["details"],
                        "prompt": st.session_state.get("final_prompt_used", ""),
                        "character_image_path": st.session_state.get("character_image_path", ""),
                        "image_urls": urls,
                    }
                    add_like(entry)
                    st.success("いいね保存しました")
    else:
        st.info("キャラ説明とテーマを入力して、展開案生成を実行してください。")


def render_likes_tab():
    st.header("いいね一覧")
    items = list_likes()
    if not items:
        st.info("まだいいねはありません")
        return
    for it in items:
        with st.expander(f"{it['story_title']} | {it['created_at']}"):
            st.markdown(f"**あらすじ**\n\n{it['synopsis']}")
            st.markdown("**詳細**")
            st.markdown(it["details"])
            if it.get("image_urls"):
                for i, u in enumerate(it["image_urls"], start=1):
                    st.image(u, caption=f"画像 {i}", use_column_width=True)


def main():
    st.set_page_config(page_title="Manga Ideation Studio", page_icon="📚", layout="wide")
    init_db()
    tabs = st.tabs(["生成", "いいね"])
    with tabs[0]:
        render_generate_tab()
    with tabs[1]:
        render_likes_tab()


if __name__ == "__main__":
    main()
