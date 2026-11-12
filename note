---
title: 【実務用】Whisper文字起こしを1ファイルで：全行コメント付きソースコード徹底解説（OpenAI API / Azure / ローカル）
tags: Python Whisper OpenAI Azure fasterwhisper
---

## はじめに

会議の録音を文字起こしするたびに、スクリプトを書き換えていませんか。
この記事では、**1ファイルで3つの方式（ローカル / OpenAI API / Azure OpenAI）を切り替えられる、実務向けの文字起こしスクリプト**を紹介します。

**全行にコメントを付けた完全版**を載せ、そのあとで「なぜこう書いたのか」を設計判断ごとに解説します。

### このスクリプトでできること

| 機能 | 内容 |
|---|---|
| 方式の切り替え | `--engine local / openai / azure` |
| 外部送信なし | `local` は音声を外に送らない（機密音声向け） |
| 25MB 超の自動分割 | API 方式で、上限を超えたら ffmpeg で自動分割して連結 |
| フォルダ一括 | フォルダを指定すると、名前順に全ファイルを処理 |
| 誤認識対策 | 無音除去（VAD）と、専門用語ヒント（`--prompt`） |
| 認証 | Azure は API キー / キーレス（Entra ID）を選べる |
| 後始末 | 分割ファイルは一時フォルダに作り、終了時に自動削除 |

### 対象読者

- Python の基本が分かる方（`pip install` と `python xxx.py` ができる）
- 業務で音声の文字起こしを使いたい方

:::note warn
**動作確認の範囲（先に正直に書きます）**
サンドボックス（Linux / Python 3.12）で、**API クライアントとモデルを模擬（モック）した状態**で、次を確認しました。
分割（36.6MB → 4 分割）、フォルダ一括と見出し、Azure のデプロイ名の受け渡し、`--prompt` と `--no-vad` の受け渡し、各種エラー表示、クライアント生成。
**OpenAI / Azure への実送信、モデルによる実際の文字起こし、Windows 実機、GPU は未検証です。** 業務で使う前に、ご自身の環境で、練習用の音声で動作確認してください。
:::

---

## 先に使い方

### インストール（PowerShell）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install openai==3.49.0 python-dotenv==1.2.4 faster-whisper==1.2.1
```

上の版は、サンドボックスで `pip install` と import を確認した版です。Azure をキーレス（Entra ID）で使う場合は、`azure-identity==1.27.0` も入れます。
25MB 超の自動分割を使う場合は、ffmpeg が必要です。

```powershell
winget install Gyan.FFmpeg
```

入れたあとは、VSCode を再起動します。（faster-whisper の文字起こしだけなら、ffmpeg の別途インストールは不要です。音声の読み込みに使うライブラリに、FFmpeg が同梱されているためです）

### `.env`（API 方式のみ）

```text:.env
# OpenAI
OPENAI_API_KEY=sk-xxxxxxxx

# Azure OpenAI
AZURE_OPENAI_API_KEY=xxxxxxxx
AZURE_OPENAI_ENDPOINT=https://あなたのリソース名.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=（自分で付けたデプロイ名）
AZURE_OPENAI_API_VERSION=2024-02-01
```

`.env` は **Git に上げない**でください（`.gitignore` に追加）。

### 実行例

```powershell
# 外部に送らない（CPU / small）
python whisper_transcribe.py 会議.mp3 --engine local

# GPU + large-v3（要 CUDA 12 用ライブラリ）
python whisper_transcribe.py 会議.mp3 --engine local --model large-v3 --device cuda

# OpenAI API / Azure OpenAI
python whisper_transcribe.py 会議.mp3 --engine openai
python whisper_transcribe.py 会議.mp3 --engine azure

# フォルダ一括 + 専門用語ヒント
python whisper_transcribe.py ./rec --engine local --prompt "Azure, VSCode, Whisper"
```

結果は `out/` フォルダに、`入力名.txt` として保存されます。

---

## 全体の流れ

```mermaid
flowchart TD
    A["引数を解析<br/>target / engine / model ..."] --> B["load_dotenv<br/>.env を読む"]
    B --> C["collect_files<br/>処理対象を集める"]
    C --> D{"engine は？"}
    D -- "local" --> L["WhisperModel を読み込み<br/>1ファイルずつ transcribe_local"]
    D -- "openai / azure" --> E["build_api_client<br/>設定チェック + クライアント作成"]
    E --> F{"ファイルが<br/>25MB超？"}
    F -- "はい" --> G["split_audio<br/>ffmpeg で分割"]
    F -- "いいえ" --> H["そのまま送信"]
    G --> I["transcribe_api<br/>1ファイルずつ送信"]
    H --> I
    L --> J["結果をまとめる<br/>複数ファイルは見出し付き"]
    I --> J
    J --> K["out/入力名.txt に<br/>UTF-8 で保存"]
```

---

## 完全版ソースコード（全行コメント付き）

`whisper_transcribe.py` として保存してください。コードの行ごとに、右側にコメントを付けています。

```python:whisper_transcribe.py
"""whisper_transcribe.py ― 音声文字起こし 1ファイル版（実務用・全行コメント付き）
使い方:
  python whisper_transcribe.py 会議.mp3 --engine local               # 外部に送らない（CPU / small）
  python whisper_transcribe.py 会議.mp3 --engine openai              # OpenAI API
  python whisper_transcribe.py 会議.mp3 --engine azure               # Azure OpenAI
  python whisper_transcribe.py ./rec --engine local --model large-v3 --device cuda
"""
import argparse                                   # コマンドライン引数を解析する標準ライブラリ
import os                                         # 環境変数（APIキーなど）を読むための標準ライブラリ
import shutil                                     # ffmpeg がインストール済みか調べるために使う
import subprocess                                 # ffmpeg を外部コマンドとして呼び出すために使う
import sys                                        # エラーメッセージを出して終了（sys.exit）するために使う
import tempfile                                   # 分割した音声の置き場（一時フォルダ）を作るために使う
from pathlib import Path                          # ファイルパスを、文字列より安全に扱うためのクラス

from dotenv import load_dotenv                    # .env ファイルから環境変数を読み込む関数

MAX_BYTES = 25 * 1024 * 1024                      # API の1ファイル上限（25MB）をバイト数で定義
SPLIT_SECONDS = 600                               # 上限を超えたとき、何秒ごとに分割するか（10分）
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".mp4", ".webm", ".ogg", ".flac"}  # 対象とする音声の拡張子


def collect_files(target: str) -> list[Path]:     # 入力（ファイルまたはフォルダ）から処理対象を集める関数
    p = Path(target)                              # 文字列を Path オブジェクトに変換する
    if not p.exists():                            # 指定した場所が存在しない場合は、
        sys.exit(f"[エラー] 見つかりません: {p}")  # 理由を表示してプログラムを終了する
    if p.is_file():                               # 1つのファイルが指定された場合は、
        return [p]                                # そのファイルだけのリストを返す
    files = sorted(                               # フォルダの場合は、中の音声ファイルを名前順に並べる
        f for f in p.iterdir()                    # フォルダ内の全ての項目を1つずつ取り出し、
        if f.suffix.lower() in AUDIO_EXT          # 拡張子が音声のものだけを残す（大文字小文字は無視）
    )                                             # 名前順の並べ替えで、分割ファイルの順序が保たれる
    if not files:                                 # 音声ファイルが1つも無い場合は、
        sys.exit(f"[エラー] 音声ファイルがありません: {p}")  # 理由を表示して終了する
    return files                                  # 集めたファイルのリストを返す


def split_audio(src: Path, out_dir: Path) -> list[Path]:  # 25MB 超の音声を分割する関数（ffmpeg が必要）
    if shutil.which("ffmpeg") is None:            # ffmpeg が PATH に見つからない場合は、
        sys.exit("[エラー] ffmpeg がありません。winget install Gyan.FFmpeg の後、VSCode を再起動")  # 案内して終了
    out_dir.mkdir(parents=True, exist_ok=True)    # 分割ファイルの置き場フォルダを作る（既にあってもよい）
    subprocess.run(                               # ffmpeg を実行する
        [                                         # コマンドを「単語のリスト」で渡す（空白やスペースに強い）
            "ffmpeg", "-y", "-loglevel", "error", # 上書き許可、ログはエラーだけにする
            "-i", str(src),                       # 入力ファイルを指定する
            "-vn", "-ac", "1",                    # 映像を捨て、音声をモノラルにする（容量削減）
            "-c:a", "libmp3lame", "-b:a", "64k",  # mp3 の 64kbps に変換する（10分で約5MB）
            "-f", "segment",                      # 出力を複数ファイルに分割する方式にする
            "-segment_time", str(SPLIT_SECONDS),  # 指定秒数ごとに区切る
            "-reset_timestamps", "1",             # 各ファイルの時刻を 0 から始め直す
            str(out_dir / "part_%03d.mp3"),       # 出力名（part_000.mp3, part_001.mp3, ...）
        ],                                        # コマンドのリストはここまで
        check=True,                               # ffmpeg が失敗したら例外を出して気づけるようにする
    )                                             # subprocess.run の呼び出しはここまで
    parts = sorted(out_dir.glob("part_*.mp3"))    # 作られた分割ファイルを名前順に集める
    return [p for p in parts if p.stat().st_size >= 2048]  # 末尾にできる極小ファイルは除外して返す


def build_api_client(engine: str):                # API 用のクライアント（接続オブジェクト）を作る関数
    if engine == "openai":                        # OpenAI API を使う場合
        if not os.environ.get("OPENAI_API_KEY"):  # APIキーが設定されていなければ、
            sys.exit("[エラー] OPENAI_API_KEY がありません（.env を確認）")  # 案内して終了する
        from openai import OpenAI                 # 必要になった時点で OpenAI ライブラリを読み込む
        return OpenAI(timeout=120, max_retries=3) # 120秒でタイムアウト、失敗時は自動で3回まで再試行
    need = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]  # Azure で必須の設定名を並べる
    if os.environ.get("USE_ENTRA") != "1":        # キーレス認証（Entra ID）を使わない場合は、
        need.append("AZURE_OPENAI_API_KEY")       # APIキーも必須にする
    missing = [k for k in need if not os.environ.get(k)]  # 未設定の項目を洗い出す
    if missing:                                   # 1つでも足りなければ、
        sys.exit(f"[エラー] .env に未設定: {', '.join(missing)}")  # 足りない名前を表示して終了する
    from openai import AzureOpenAI                # Azure 用のクライアントクラスを読み込む
    common = dict(                                # 2つの認証方式で共通の設定をまとめる
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],  # 自分の Azure リソースのURL
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),  # API版（未設定なら既定値）
        timeout=120, max_retries=3,               # タイムアウトと自動再試行の設定
    )                                             # 共通設定はここまで
    if os.environ.get("USE_ENTRA") == "1":        # キーレス認証（Entra ID）を選んだ場合
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider  # 認証用の部品を読み込む
        provider = get_bearer_token_provider(     # 利用者のログイン情報から、アクセストークンの取得係を作る
            DefaultAzureCredential(),             # 環境に応じて、利用可能な認証方法を自動で選ぶ
            "https://cognitiveservices.azure.com/.default",  # トークンを使うサービスの範囲（スコープ）
        )                                         # トークン取得係の作成はここまで
        return AzureOpenAI(azure_ad_token_provider=provider, **common)  # キーの代わりにトークン取得係を渡す
    return AzureOpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"], **common)  # APIキー方式のクライアントを返す


def transcribe_api(client, model: str, path: Path, prompt: str | None) -> str:  # API で1ファイルを文字起こしする関数
    with path.open("rb") as fp:                   # 音声ファイルをバイナリ（rb）で開く。使い終わると自動で閉じる
        kwargs = {"model": model, "file": fp, "language": "ja"}  # 送る設定：モデル名・音声・言語（日本語）
        if prompt:                                # 用語ヒントが指定されていれば、
            kwargs["prompt"] = prompt             # それも一緒に送る（専門用語の誤認識を減らす）
        return client.audio.transcriptions.create(**kwargs).text  # API を呼び、結果の文字列だけを返す


def transcribe_local(model, path: Path, prompt: str | None, vad: bool) -> list[str]:  # 手元のPCで文字起こしする関数
    segments, _info = model.transcribe(           # モデルに音声を渡す（結果は区間ごとの「遅延リスト」）
        str(path), language="ja", beam_size=5,    # ファイルパス・日本語・探索幅5（精度と速度のバランス）
        vad_filter=vad,                           # 無音区間を除き、無音での誤認識（幻覚）を減らす
        initial_prompt=prompt,                    # 用語ヒント（None なら何も渡さない）
    )                                             # この時点では、まだ文字起こしは終わっていない
    return [                                      # 区間を1つずつ取り出して、文字列のリストにする
        f"[{s.start:7.1f}s - {s.end:7.1f}s] {s.text.strip()}"  # 「[開始秒 - 終了秒] テキスト」の形式
        for s in segments                         # ここで実際に文字起こしが進む（イテレーションで実行）
    ]                                             # 全区間を取り出し終えたら、リストを返す


def main() -> None:                               # プログラムの入口（全体の流れを組み立てる）
    ap = argparse.ArgumentParser(description="音声文字起こし（Whisper）")  # 引数の解析器を作る
    ap.add_argument("target", help="音声ファイル or フォルダ")  # 必須の引数：処理する対象
    ap.add_argument("--engine", choices=["local", "openai", "azure"], default="local",  # 使う方式（既定は local）
                    help="local=外部送信なし / openai / azure")  # --help に出す説明
    ap.add_argument("--model", default="small", help="local用: small / medium / large-v3 など")  # ローカルのモデル名
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])  # ローカルの実行装置（GPUはcuda）
    ap.add_argument("--prompt", default=None, help="専門用語のヒント（例: 'Azure, VSCode'）")  # 用語ヒント
    ap.add_argument("--no-vad", action="store_true", help="無音除去(VAD)を切る")  # 指定すると VAD を無効化
    args = ap.parse_args()                        # 実際のコマンドライン引数を解析して、args に入れる

    load_dotenv()                                 # .env があれば読み込み、環境変数として使えるようにする
    files = collect_files(args.target)            # 処理対象のファイルを集める
    sections: list[tuple[str, str]] = []          # 「(ファイル名, 文字起こし結果)」を貯めるリスト

    if args.engine == "local":                    # --- ローカル方式（音声を外部に送らない） ---
        from faster_whisper import WhisperModel   # 必要になった時点でライブラリを読み込む
        compute = "float16" if args.device == "cuda" else "int8"  # GPUなら float16、CPUなら int8（軽量）
        print(f"モデル読み込み: {args.model} / {args.device} / {compute}（初回はダウンロードあり）")  # 進捗表示
        model = WhisperModel(args.model, device=args.device, compute_type=compute)  # モデルを読み込む
        for f in files:                           # ファイルを1つずつ処理する
            print(f"処理中: {f.name}")            # どのファイルを処理中か表示する
            lines = transcribe_local(model, f, args.prompt, not args.no_vad)  # 文字起こしを実行する
            sections.append((f.name, "\n".join(lines)))  # 結果を、改行でつないで保存する
    else:                                         # --- API方式（音声が外部サービスに送信される） ---
        client = build_api_client(args.engine)    # 接続オブジェクトを作る（設定不足ならここで終了）
        model_name = "whisper-1" if args.engine == "openai" else os.environ["AZURE_OPENAI_DEPLOYMENT"]  # Azureはデプロイ名
        with tempfile.TemporaryDirectory() as tmp:  # 分割ファイル用の一時フォルダ（ブロックを出ると自動削除）
            for i, f in enumerate(files):         # 番号 i とファイル f を同時に取り出す
                print(f"処理中: {f.name}")        # どのファイルを処理中か表示する
                targets = [f]                     # 通常は、そのファイルをそのまま送る
                if f.stat().st_size > MAX_BYTES:  # 25MB を超えている場合は、
                    print("  25MBを超えるため分割します")  # 分割することを知らせて、
                    targets = split_audio(f, Path(tmp) / f"{i:03d}")  # ffmpeg で分割したファイルに置き換える
                texts = [transcribe_api(client, model_name, t, args.prompt) for t in targets]  # 各ファイルを順に文字起こし
                sections.append((f.name, "\n".join(texts)))  # 分割分を改行でつないで保存する

    body = "\n\n".join(                           # 全ファイルの結果を、空行区切りで1つの文字列にする
        (f"=== {name} ===\n{text}" if len(files) > 1 else text)  # 複数ファイルのときだけ見出しを付ける
        for name, text in sections                # 保存しておいた結果を1件ずつ取り出す
    )                                             # 結合はここまで
    out_dir = Path("out")                         # 出力先フォルダを「out」にする
    out_dir.mkdir(exist_ok=True)                  # フォルダを作る（既にあってもエラーにしない）
    out = out_dir / f"{Path(args.target).stem}.txt"  # 出力ファイル名（入力名.txt）を決める
    out.write_text(body, encoding="utf-8")        # UTF-8 で書き出す（Windowsの文字化け防止）
    print(f"完了 → {out}")                        # 保存先を表示して終了する


if __name__ == "__main__":                        # このファイルを直接実行したときだけ、
    main()                                        # main 関数を呼び出す（import されたときは動かさない）
```

---

## 設計の解説：なぜこう書いたのか

### 1. 使う方式のライブラリだけを、必要なときに読み込む

`from faster_whisper import WhisperModel` や `from openai import OpenAI` を、**関数の中**に書いています。

- `--engine local` のときは、openai ライブラリを読み込まない
- `--engine openai` のときは、faster-whisper（大きなライブラリ）を読み込まない
- 使わない方式のライブラリが未導入でも、動く

### 2. 25MB の壁は「止める」のではなく「自動で分割」する

OpenAI と Azure OpenAI は、1ファイル 25MB が上限です（Azure の公式クイックスタートにも記載があります）。会議の録音は、1 時間で上限を超えることが多くあります。

`split_audio` は、ffmpeg で次の変換をして分割します。

| 設定 | 狙い |
|---|---|
| `-ac 1`（モノラル） | 容量を半分にする。会話の文字起こしには、通常は十分 |
| `-b:a 64k` | 10 分で約 5MB。上限に余裕をもって収まる |
| `-segment_time 600` | 10 分ごとに区切る |
| `-reset_timestamps 1` | 各ファイルの時刻を 0 から始める |

:::note warn
**注意：** 時間で機械的に切るため、**区切り位置で単語が欠ける**ことがあります。また、音質を落とすため、聞き取りにくい音声では精度が下がることがあります。
Azure で大きな音声を扱う場合は、公式が案内している Azure AI Speech のバッチ文字起こしも、選択肢です。
:::

### 3. Azure の `model=` には「デプロイ名」を入れる

`model_name = ... os.environ["AZURE_OPENAI_DEPLOYMENT"]` の行が、最大のつまずきポイントへの対策です。Azure では、モデル名ではなく、**自分で付けたデプロイ名**を渡します。公式のクイックスタートにも、デプロイ名とモデル名が異なると、モデル名を入れたときにエラーになるという注意があります。

### 4. `segments` は「遅延評価」（faster-whisper の落とし穴）

`model.transcribe()` の戻り値は、**すぐには文字起こしを終えていません。** 戻ってくる `segments` を `for` で取り出したときに、はじめて処理が進みます（公式 README にも、`list(segments)` で実行されるという説明があります）。

このコードでは、リスト内包表記の中で `for s in segments` と取り出して、確実に実行されるようにしています。`transcribe()` を呼んだだけで「終わった」と思うと、処理時間の計測や例外処理を取り違えます。

### 5. タイムアウトと自動リトライ

```python
OpenAI(timeout=120, max_retries=3)
```

ネットワークの一時的な失敗で、長い処理を最初からやり直すのは、実務では痛手です。120 秒で諦め、一時的な失敗は 3 回まで自動で再試行します。数値は、お使いの環境に合わせて調整してください。

### 6. 分割ファイルは `TemporaryDirectory` で自動削除

分割した音声ファイルにも、元の音声と同じ機密性があります。`with tempfile.TemporaryDirectory() as tmp:` のブロックを出ると、**エラーで終わった場合も含めて**、一時フォルダごと自動で削除されます。消し忘れを、仕組みで防いでいます。

### 7. 日本語の文字化けを防ぐ：`encoding="utf-8"`

`out.write_text(body, encoding="utf-8")` と、エンコーディングを明示しています。Windows では、指定しないと既定の文字コードで書かれ、他のツールで開いたときに文字化けすることがあります。

### 8. エラーは「次に何をすればよいか」を書いて終了する

設定不足やファイルが無いときは、Python の長いエラー表示ではなく、`sys.exit("[エラー] ...")` で、**原因と対処を 1 行で**表示します。受講者や同僚に渡すスクリプトでは、これが問い合わせを減らします。

---

## 実務で使うときの注意

### 方式は、音声の機密度で選ぶ

| 音声の種類 | 目安 |
|---|---|
| 個人情報・顧客情報・機密を含む | `--engine local`（外部に送らない）。API を使う場合は、社内の承認と契約の確認が先 |
| 機密ではない社内会議 | 社内で許可されたサービスの範囲で、`openai` / `azure` も可 |
| 練習用の自作音声 | どれでも可 |

録音の同意、データの保持・学習利用の条件、処理リージョンは、**契約形態ごとの公式資料で確認し、確認日を残してください。** この記事のコードは、そのいずれも保証しません。

### 出力ファイルも、同じ機密レベルで扱う

`out/` の文字起こしは、元の音声と同じ機密レベルです。共有フォルダに置く前に、中身を確認してください。不要になったら削除します。

### 文字起こしは、必ず人が確認してから使う

数字、固有名詞、否定表現（「〜ではない」）の取り違えは、業務判断に直結します。確認なしで議事録として配らないでください。無音での誤認識（幻覚）対策として、`local` では無音除去（VAD）が既定で有効です。専門用語の誤認識には `--prompt` でヒントを渡し、結果を比べてください。

### 変わりやすい情報は、公開時に確認する

OpenAI の音声モデル名、Azure の提供状況・リージョン・`api-version`、各サービスの料金は、変わります。スクリプトの `whisper-1` と `2024-02-01` は、執筆時点の公式の例に基づく値です。**使う前に、公式ドキュメントで最新を確認してください。**

---

## カスタマイズのヒント

| やりたいこと | 方向性 |
|---|---|
| 日本語以外の音声 | 2 か所の `language="ja"` を、引数で渡す形に変える |
| 字幕（SRT）出力 | local の `segments` の開始・終了秒から、SRT 形式に整形する |
| 話者分離 | 本スクリプトの範囲外。対応するモデル・サービスは、公式で確認する |
| ログ | `print` を `logging` に置き換え、処理時間とファイル名を記録する |
| 失敗時の再開 | ファイルごとに結果を保存し、既に結果があるファイルをスキップする |

---

## よくあるエラー

| 表示 | 原因と対処 |
|---|---|
| `OPENAI_API_KEY がありません` | `.env` が無い、または名前の誤り。ファイル名の先頭の `.` も確認 |
| `.env に未設定: AZURE_...` | 表示された項目を `.env` に追加 |
| Azure で `DeploymentNotFound` | `AZURE_OPENAI_DEPLOYMENT` に、モデル名ではなくデプロイ名を入れる |
| `ffmpeg がありません` | `winget install Gyan.FFmpeg` を実行し、VSCode を再起動 |
| `AuthenticationError` | キーの誤り・失効。キーを再確認 |
| CUDA / cublas / cudnn の DLL エラー | GPU で動かすには、CUDA 12 用の cuBLAS と cuDNN 9 が必要。難しければ `--device cpu` か、Colab を使う |
| 初回が極端に遅い（local） | モデルのダウンロード中。2 回目以降は速くなる |

---

## まとめ

- **1ファイルで、local / OpenAI / Azure を切り替えられる**
- 25MB 超は、**自動分割**。分割ファイルは**自動削除**
- Azure の `model=` は**デプロイ名**
- `segments` は**遅延評価**。取り出して初めて処理が進む
- 方式は、**音声の機密度**で選ぶ。結果は**人が確認**してから使う
- **未検証の範囲**（実送信・モデル実行・Windows 実機・GPU）は、使う前に自分の環境で確認する

## 参考

- [OpenAI Speech to text ガイド](https://platform.openai.com/docs/guides/speech-to-text)
- [Azure OpenAI Whisper クイックスタート](https://learn.microsoft.com/ja-jp/azure/ai-services/openai/whisper-quickstart)
- [faster-whisper（GitHub）](https://github.com/SYSTRAN/faster-whisper)
