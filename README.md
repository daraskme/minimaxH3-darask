# H3 Studio

MiniMax H3 の映像と音声を、独立したローカルエンジンで生成する Windows 向け GUI です。ComfyUI の起動、API、Python 環境、カスタムノードには依存しません。ブラウザ UI と生成 API は `127.0.0.1` のみに公開し、生成物は日付別の `outputs/YYYY-MM-DD` に保存します。

対象環境は 128 GB RAM / RTX PRO 6000 Blackwell 96 GB です。公式 Diffusers の `MiniMaxH3ModularPipeline`、TorchAO INT8 weight-only v2、コンポーネント単位の自動 CPU オフロード（VRAM 12 GB を作業領域として確保）、SageAttention を使います。低メモリモードだけがブロック単位ストリーミングを使います。

## セットアップ

PowerShell で次を実行します。

```powershell
.\bootstrap.ps1
.\.venv\Scripts\python.exe .\scripts\download_model.py
.\launch.ps1
```

`bootstrap.ps1` は Python 依存関係に加え、SHA256 を固定した Real-ESRGAN x2plus と RIFE 4.26 の重み（合計約 90 MB）を公式配布元から導入します。H3 モデルは約 144 GB（Ref2VA transformer を除く T2VA / FL2VA と共有コンポーネント）です。H3 のダウンロードは明示的に実行し、アプリ起動時には行いません。`start.cmd` でも起動できます。

DLSS は通常セットアップとは分離しています。公式 NVIDIA DLSS SDK の Super Resolution は `scripts/build_dlss_exporter.ps1` でローカルビルドし、`scripts/validate_dlss_exporter.py` で短い動画を書き出せた場合だけ有効になります。実験的な DLSS 5 Neural Rendering はコミュニティ改変の未署名ランタイムを使うため、次の明示操作で固定ZIPの全ファイルを検証し、隔離 worker のプリフライトと実ファイル書き出しに成功した場合だけ表示します。

```powershell
.\.venv\Scripts\python.exe .\scripts\stage_dlss5_runtime.py
.\.venv\Scripts\python.exe .\scripts\validate_dlss5_nr_runtime.py --allow-unsigned-runtime --source .\短いCFR動画.mp4
```

モデルの保存場所や LoRA フォルダを変える場合は `config.example.json` を `config.json` にコピーして編集します。初期値はモデルが `./models`、LoRA が `./models/loras` です。生成エンジン、LoRA、後処理はいずれも ComfyUI を起動せずに動作します。

## 生成仕様

- 標準は 960×544、24 fps、124 frames（約 5.17 秒）です。幅と高さは 32 の倍数、フレームは `17k+5` に切り上げます。最大は 345 frames（14.375 秒）です。Real-ESRGAN x2 の出力は実寸 1920×1088 です。
- UI の Steps は実際の Transformer 評価回数です。H3 のスケジューラは終端 0 を含むため、内部には `Steps + 1` 個の sigma 点を渡します。
- 最初／最後の画像を個別に指定できます。最後の画像だけを指定する生成も公式 FL2VA が対応します。
- 複数 LoRA は画面の上から順に、個別の重みで適用します。無効化した項目は履歴へ残りますがモデルには適用しません。読み込み後に PEFT adapter の登録と順序を確認し、一致しない LoRA はエラーにします。
- モデルは画面の「モデルを読み込む」で明示的に準備します。LoRA の追加、順序、重み、有効状態は変更後にGPUキューへ自動反映され、生成中のモデルは途中で変更しません。モデル未読み込み時は最新のLoRA構成を保留し、明示読み込み時に適用します。
- 高速プリセットは検証済みの 8-step / 4-step Turbo LoRA と固定ペアです。8-step は video/audio shift 12/3、4-step 768p は 6/3 を使います。異なる高速 LoRA、Ref2VA 用、複数高速 LoRAの重ね掛けは拒否します。
- 生成設定には24 FPSの推奨キャンバスと、124 / 243 / 345フレーム（約5.17 / 10.13 / 14.38秒）を用意しています。Latent 2-passは最初の実機出力が画質不合格だったため、現在API/UIで利用停止中です。24ch正規化の修正は保存済みですが、完全な実機再検証に合格するまで有効化しません。
- VC-Attention は無効です。参照された実装を監査した結果、論文のブロック平均復元と同等と確認できず、パディングのマスクにも未解決点があるためです。

## アップスケールとフレーム補間

- Real-ESRGAN x2plus を Spandrel 0.4.2 から直接実行し、RGB フレームをストリーミング処理します。960×544 の入力は 1920×1088 になります。大きな入力にはオーバーラップ付きタイル処理を使います。
- RIFE 4.26 は監査済みの公式 IFNet 推論グラフを FP32 で直接実行します。2×／3×／4×を選べ、元フレームを整数間隔に保ち、末尾を保持して長さを変えません。シーン切替を検出した区間では架空の中間像を作らず直前フレームを保持します。
- AI を使わない Lanczos 2×／3×／4×も明示的な従来方式として選べます。VFR 入力は Lanczos だけが平均レートの CFR へ正規化し、AI 方式は明確に拒否します。
- 元の音声は再エンコードせず保持します。派生 MP4 には元ファイルの SHA256、生成設定、派生履歴、実方式、正確な入出力 FPS とフレーム数を保存します。入力と派生ファイルは上書きしません。
- NVIDIA公式SDKの DLSS Super Resolution は2倍の実ファイル出力です。実際のNGX評価回数、フレーム数、出力寸法、ドライバーを受領書で照合し、音声をストリームコピーします。
- DLSS 5 Neural Rendering（実験）は超解像ではなく、元解像度のまま feature 18 を適用します。固定した v0.24.0 worker のバイナリプロトコル受領書で feature 18 の作成・評価・全フレームを検証します。未署名のコミュニティ改変ランタイムであり、NVIDIA公式SDK版とはUI・メタデータ上で区別します。
- どちらもドライバー610.47以上と、このGPU・ドライバー上の実ファイルスモークが必要です。アプリがドライバーを更新したり、従来拡大をDLSSと表示したりすることはありません。

## 出力と履歴

MP4 には映像と生成音声を格納します。生成後にストリームコピーで JSON を `comment` metadata に埋め込み、同じ内容の `.json` を隣へ保存します。プロンプト、確定 seed、順序付き LoRA、モデル、要求値と解決値、実際の Attention / 精度 / メモリ方式、フロー shift、評価回数を記録します。アップスケール／補間では元の埋め込み JSON も命令として扱わず来歴データとして引き継ぎます。メタデータ処理だけ失敗した場合は高コストな MP4 を削除せず、履歴から後処理を再試行できます。

履歴は `data/studio.sqlite3` に保存します。キューは一度に 1 本を生成し、選択したジョブだけをキャンセルします。アプリ再起動で実行中だったジョブは「中断」として失敗に更新し、別プロセスや別ジョブを停止しません。

## 確認

```powershell
.\.venv\Scripts\python.exe -m pytest
```

現在の検証状態と次回作業の入口は [docs/HANDOFF.md](docs/HANDOFF.md) にあります。実装と受け入れ条件は [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)、DLSSの固定契約は [docs/DLSS.md](docs/DLSS.md)、Astra の監査は [docs/ASTRA_CODE_REVIEW.md](docs/ASTRA_CODE_REVIEW.md) にあります。
