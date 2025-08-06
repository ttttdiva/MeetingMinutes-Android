# API呼び出しパターン詳細

## Kotlin/Android実装用のAPI呼び出しサンプル

### 1. OpenAI Whisper API (文字起こし)

#### Python実装
```python
from openai import OpenAI

client = OpenAI(api_key=api_key)
with open(audio_file_path, "rb") as audio_file:
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="ja",
        response_format="text"
    )
transcript = response
```

#### Kotlin実装イメージ
```kotlin
// Retrofit + OkHttpを使用
interface WhisperApi {
    @Multipart
    @POST("audio/transcriptions")
    suspend fun transcribe(
        @Part file: MultipartBody.Part,
        @Part("model") model: RequestBody,
        @Part("language") language: RequestBody,
        @Part("response_format") format: RequestBody
    ): TranscriptResponse
}

// 使用例
val file = File(audioFilePath)
val requestFile = file.asRequestBody("audio/*".toMediaType())
val body = MultipartBody.Part.createFormData("file", file.name, requestFile)
val response = api.transcribe(
    file = body,
    model = "whisper-1".toRequestBody(),
    language = "ja".toRequestBody(),
    format = "text".toRequestBody()
)
```

### 2. Google Gemini API

#### Python実装（文字起こし）
```python
import google.generativeai as genai

genai.configure(api_key=api_key)
audio_file = genai.upload_file(path=audio_file_path)
model = genai.GenerativeModel("gemini-1.5-flash")

prompt = "この音声を文字起こししてください。"
response = model.generate_content([prompt, audio_file])
transcript = response.text
```

#### Kotlin実装イメージ
```kotlin
// Google AI SDK for Android使用
val generativeModel = GenerativeModel(
    modelName = "gemini-1.5-flash",
    apiKey = BuildConfig.GEMINI_API_KEY
)

// 音声ファイルをBase64エンコード
val audioBytes = File(audioFilePath).readBytes()
val audioBase64 = Base64.encodeToString(audioBytes, Base64.DEFAULT)

// プロンプトと音声データを送信
val prompt = "この音声を文字起こししてください。"
val response = generativeModel.generateContent(
    content {
        text(prompt)
        blob("audio/mp3", audioBytes)
    }
)
val transcript = response.text
```

### 3. 議事録生成 (Gemini)

#### Python実装
```python
prompt = """
以下の会議内容から議事録を作成してください：

# 出力形式
## 会議概要
## 主な議題
## 決定事項
## アクションアイテム
## 次回予定

# 会議内容
"""

response = model.generate_content(
    prompt + transcript,
    generation_config=genai.GenerationConfig(
        temperature=0.3,
        max_output_tokens=4000
    )
)
summary = response.text
```

#### Kotlin実装イメージ
```kotlin
val summaryPrompt = """
    |以下の会議内容から議事録を作成してください：
    |
    |# 出力形式
    |## 会議概要
    |## 主な議題
    |## 決定事項
    |## アクションアイテム
    |## 次回予定
    |
    |# 会議内容
    |$transcript
""".trimMargin()

val generationConfig = generationConfig {
    temperature = 0.3f
    maxOutputTokens = 4000
}

val response = generativeModel.generateContent(
    summaryPrompt,
    generationConfig = generationConfig
)
val summary = response.text
```

### 4. OpenAI GPT API (議事録生成)

#### Python実装
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "あなたは会議の議事録を作成する専門家です。"
        },
        {
            "role": "user",
            "content": f"以下の会議内容から議事録を作成してください：\n{transcript}"
        }
    ],
    temperature=0.3,
    max_tokens=4000
)
summary = response.choices[0].message.content
```

#### Kotlin実装イメージ
```kotlin
data class ChatRequest(
    val model: String,
    val messages: List<Message>,
    val temperature: Float,
    val max_tokens: Int
)

data class Message(
    val role: String,
    val content: String
)

// API呼び出し
val request = ChatRequest(
    model = "gpt-4o-mini",
    messages = listOf(
        Message(
            role = "system",
            content = "あなたは会議の議事録を作成する専門家です。"
        ),
        Message(
            role = "user",
            content = "以下の会議内容から議事録を作成してください：\n$transcript"
        )
    ),
    temperature = 0.3f,
    max_tokens = 4000
)

val response = openAiApi.createChatCompletion(request)
val summary = response.choices.first().message.content
```

## エラーハンドリング

### 共通エラーパターン

1. **認証エラー (401)**
   - APIキーの確認
   - ヘッダー設定の確認

2. **レート制限 (429)**
   - リトライ処理の実装
   - exponential backoff

3. **ファイルサイズ制限**
   - Whisper API: 25MB制限
   - 音声ファイルの圧縮/分割

4. **ネットワークエラー**
   - タイムアウト設定
   - リトライ処理

### Kotlinでのエラーハンドリング例
```kotlin
sealed class ApiResult<T> {
    data class Success<T>(val data: T) : ApiResult<T>()
    data class Error<T>(val exception: Exception) : ApiResult<T>()
}

suspend fun transcribeWithRetry(
    audioFile: File,
    maxRetries: Int = 3
): ApiResult<String> {
    var retryCount = 0
    var lastException: Exception? = null
    
    while (retryCount < maxRetries) {
        try {
            val result = transcribeAudio(audioFile)
            return ApiResult.Success(result)
        } catch (e: HttpException) {
            when (e.code()) {
                429 -> {
                    // レート制限の場合は待機
                    delay(2000L * (retryCount + 1))
                }
                401 -> {
                    // 認証エラーは即座に失敗
                    return ApiResult.Error(e)
                }
            }
            lastException = e
        } catch (e: Exception) {
            lastException = e
        }
        retryCount++
    }
    
    return ApiResult.Error(lastException ?: Exception("Unknown error"))
}
```

## 必要な依存関係

### Gradle依存関係
```gradle
dependencies {
    // Networking
    implementation 'com.squareup.retrofit2:retrofit:2.9.0'
    implementation 'com.squareup.retrofit2:converter-gson:2.9.0'
    implementation 'com.squareup.okhttp3:okhttp:4.11.0'
    implementation 'com.squareup.okhttp3:logging-interceptor:4.11.0'
    
    // Google Gemini
    implementation 'com.google.ai.client.generativeai:generativeai:0.1.2'
    
    // Coroutines
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
    
    // JSON parsing
    implementation 'com.google.code.gson:gson:2.10.1'
}
```

## APIキー管理

### BuildConfig使用（推奨）
```gradle
// build.gradle (app)
android {
    defaultConfig {
        buildConfigField "String", "OPENAI_API_KEY", "\"${OPENAI_API_KEY}\""
        buildConfigField "String", "GEMINI_API_KEY", "\"${GEMINI_API_KEY}\""
    }
}
```

### local.properties
```properties
OPENAI_API_KEY=your_api_key_here
GEMINI_API_KEY=your_api_key_here
```

### SharedPreferencesで暗号化保存
```kotlin
class ApiKeyManager(context: Context) {
    private val encryptedPrefs = EncryptedSharedPreferences.create(
        context,
        "api_keys",
        MasterKeys.getOrCreate(MasterKeys.AES256_GCM_SPEC),
        EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
        EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
    )
    
    fun saveApiKey(service: String, key: String) {
        encryptedPrefs.edit().putString(service, key).apply()
    }
    
    fun getApiKey(service: String): String? {
        return encryptedPrefs.getString(service, null)
    }
}
```