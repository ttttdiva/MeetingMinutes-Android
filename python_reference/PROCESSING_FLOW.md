# Python版 処理フローリファレンス

## 全体の処理フロー

```
1. 音声入力
   ├── 録音 (AudioRecorder)
   └── ファイル選択 (drag_drop_process.py)
      ↓
2. 音声ファイル保存
   └── outputs/ ディレクトリに保存
      ↓
3. 文字起こし処理
   ├── Whisper API (OpenAI)
   ├── Gemini API (Google)
   └── フォールバック機能
      ↓
4. 文字起こし結果保存
   └── transcript_YYYYMMDD_HHMMSS.txt
      ↓
5. 議事録生成処理
   ├── Gemini API
   ├── OpenAI GPT API
   └── フォールバック機能
      ↓
6. 議事録保存
   └── summary_YYYYMMDD_HHMMSS.md (Markdown形式)
```

## 主要クラスと責務

### MeetingMinutesSystem (main.py)
- 全体の処理フローを制御
- 設定ファイル(config.yaml)の読み込み
- 各プロセッサーの初期化と実行

### AudioRecorder (src/record.py)
- 音声録音機能
- FFmpegを使用した音声ファイル処理
- 録音設定（サンプルレート、チャンネル数など）

### ProcessorFactory (src/processor_factory.py)
- 各種プロセッサーの生成
- モデル選択の制御

### 文字起こしプロセッサー
- **WhisperTranscriber**: OpenAI Whisper APIを使用
- **GeminiTranscriber**: Google Gemini APIを使用（話者分離対応）

### 要約プロセッサー
- **GeminiSummarizer**: Gemini APIで議事録生成
- **GPTSummarizer**: OpenAI GPT APIで議事録生成

## API呼び出しパターン

### Whisper API (文字起こし)
```python
# APIキー設定
client = OpenAI(api_key=api_key)

# 文字起こし実行
with open(audio_file_path, "rb") as audio_file:
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="ja",
        response_format="text"
    )
```

### Gemini API (文字起こし・要約)
```python
# APIキー設定
genai.configure(api_key=api_key)

# 音声ファイルアップロード（文字起こし用）
audio_file = genai.upload_file(path=audio_file_path)

# 文字起こし実行
model = genai.GenerativeModel(model_name)
response = model.generate_content(
    [prompt, audio_file],
    generation_config=generation_config
)

# 議事録生成
response = model.generate_content(
    prompt + transcript,
    generation_config=generation_config
)
```

### OpenAI GPT API (要約)
```python
# APIキー設定
client = OpenAI(api_key=api_key)

# 議事録生成
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": transcript}
    ],
    temperature=0.3,
    max_tokens=4000
)
```

## エラーハンドリング

### フォールバック機能
1. プライマリモデルで処理実行
2. エラー発生時、フォールバックモデルリストを順番に試行
3. リモートフォールバック（SSH経由）も対応
4. 全て失敗した場合はエラーメッセージを表示

### 主なエラーパターン
- API認証エラー → APIキー確認
- レート制限エラー → リトライまたはフォールバック
- ファイルサイズ制限 → 音声ファイル分割処理
- ネットワークエラー → リトライ処理

## 設定管理

### 環境変数 (.env)
```
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
HF_TOKEN=your_huggingface_token  # 話者分離用
```

### 設定ファイル (config.yaml)
- 使用モデルの選択
- フォールバック設定
- 処理モード（local/remote）
- 各種パラメータ

## 出力ファイル形式

### 文字起こしファイル (transcript_*.txt)
- プレーンテキスト形式
- 話者分離がある場合は話者ラベル付き

### 議事録ファイル (summary_*.md)
- Markdown形式
- セクション構成：
  - 会議概要
  - 主な議題
  - 決定事項
  - アクションアイテム
  - 次回予定