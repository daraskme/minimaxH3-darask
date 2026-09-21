# H3 Studio 実装計画と受け入れ条件

## 構成

FastAPI は静的 UI、入力検証、アップロード、履歴、出力配信だけを担当する。`JobRunner` が生成・アップスケール・補間を共通の GPU キューで 1 本ずつ所有する。生成設定は `StandaloneH3Engine`、派生処理は `VideoPostprocessEngine` に渡す。H3 エンジンはローカルの公式 Diffusers H3 リポジトリを直接読み、ComfyUI のコード、サーバー、ノード、モデル管理を呼ばない。

96 GB GPU の標準プロファイルは Transformer と Qwen3-VL を TorchAO INT8 にし、Diffusers `ComponentsManager` がステージごとにモデル全体を GPU へ置く。12 GB の余白を保持し、Transformer の各ブロックを毎 step PCIe 転送しない。`low_memory` は RAM / VRAM 制約時の明示的なフォールバックとして group offload を使う。

生成は T2VA と FL2VA を提供する。公式モデルの guidance-distilled 契約に従い、negative prompt / CFG / 任意 sampler を装って公開しない。進捗とキャンセルは Transformer の forward pre-hook で評価ごとに確認し、例外時は hook、offload manager、部分ロード済み pipeline を破棄する。

## 高速化

1. SageAttention 2.2 の実カーネルを明示選択し、利用不可なら SDPA へ黙って偽装せず失敗させる。
2. 8-step Turbo v1.0 は shift 12/3、4-step 768p v1.0 は shift 6/3 と固定する。
3. INT8 weight-only v2 では入力投影、time embedding、token refiner、出力 head など公式除外リストを BF16/FP32 のまま保持する。
4. 高速 LoRA の互換表はファイル名、workflow、NFE、shift を一体で管理する。スタイル LoRA は追加できるが、高速 LoRA は 1 個だけにする。
5. VC-Attention は検証未完了として選択肢を無効にする。

## 派生処理

1. Real-ESRGAN x2plus は検証済み重みを Spandrel でロードし、FP16 CUDA、フレーム単位、一定面積を超える場合はオーバーラップ付きタイルで処理する。
2. RIFE 4.26 は公式 IFNet と warplayer を同梱し、ダウンロードする ZIP からは SHA256 検証した `flownet.pkl` だけを展開する。`teacher.*` と `caltime.*` だけを除去した後 `strict=True` でロードする。
3. 補間は元の N フレームを k 間隔に配置し、末尾を k-1 フレーム保持して常に N×k フレームとする。FPS は有理数のまま k 倍し、240 fps を上限にする。
4. 音声は元 MP4 から stream copy する。入力 SHA256 と固定 origin、直近 8 件の lineage、処理方法、モデル SHA256、フレーム/FPSを MP4 と sidecar JSON に保存する。
5. VFR は RIFE / Real-ESRGAN では拒否する。Lanczos は `fps` filter で CFR 化することを metadata に記録する。出力は各辺 8192、4000 万画素以下に制限する。
6. SeedVR2 3B は固定したモデルだけを別エンジンで実行し、時系列窓のoverlapを合成して元の有理数FPSと音声を保持する。
7. NVIDIA公式SDKのDLSS SRは2倍のNGX出力として扱う。コミュニティ版DLSS 5 Neural Renderingは元解像度の実験方式として分離し、固定runtime、protocol 6受領書、現在のドライバーでの実ファイルスモークが揃うまで利用不可にする。詳細は [DLSS.md](DLSS.md) に記す。

## 受け入れ条件

- [x] 幅／高さ／フレーム、Turbo ペア、LoRA 順序、パストラバーサルを単体テストする。
- [x] 実 MP4 の日本語複数行 metadata が完全に復元され、映像と音声の両 stream が残ることをテストする。
- [x] queued / running の選択キャンセルと終端状態の競合を直列化する。
- [x] Host と mutation Origin をローカルに制限する。
- [x] モデル shard が全部揃うまで inventory を ready にしない。
- [x] Real-ESRGAN x2 の実 GPU 処理で 24 frames / 24 fps / 音声付き入力が、同じ 24 frames / 24 fps、384×256、1.0 秒、音声付きになることを確認する。
- [x] RIFE 2×／3×／4×の実 GPU 処理で 48／72／96 frames、48／72／96 fps、1.0 秒、音声保持になることを確認する。
- [x] CFR/VFR 判定、MP4 を装った Matroska の拒否、FFmpeg の取消応答と大量 stderr drain、埋め込み来歴の round-trip を検証する。
- [x] SeedVR2 3B FP16 の実GPU処理で5 framesの小型入力と、24 frames / 24000/1001 fps / 音声付き統合入力の寸法・FPS・音声保持を確認する。
- [x] 公式DLSS SRで24回のNGX評価、2倍寸法、24 frames / 24 fps / 音声保持を確認する。
- [x] DLSS 5 Neural Renderingの固定runtimeでpreflightと24 framesの実書き出しを行い、feature 18、verified frames、元解像度、CFR、音声packet一致を確認する。
- [x] 公式フルモデルで T2VA 124 frames を生成し、native audio、Sage dispatch、埋め込み metadata を確認する。
- [x] Turbo8 の第1パス8 evaluations、shift 12/3、LoRA 312 modules登録を履歴と metadata で確認する。
- [x] Latent 2-passの24ch正規化を修正し、512×512と標準960×544の124-frame実生成で先頭・中間・末尾の画質、音声、FPS、metadataを確認する。
- [x] スタイル LoRA 2 個をTurbo8と同時に異なる重みで適用し、adapter の登録数と順序を確認する。映像への個別効果評価は別途行う。
- [ ] 最初／最後フレーム、モデル切替、キャンセル、後処理再試行をブラウザから確認する。

実機検証は RTX PRO 6000 / 128 GB RAM で行った。最初のLatent 2-pass出力は24ch正規化の欠落によりピンク格子となった。修正後、256→512と標準480×288→960×544の両方で、124 frames／24 fps、Turbo8の第1パス8 evaluations、学習済み24ch潜在拡大、高解像度4 evaluations（strength 0.18）を実行した。先頭・中間・末尾フレームで同じ被写体と場面を確認し、格子は消失した。両MP4は5.1667秒、音声付きで、埋め込みJSONとsidecarも一致した。ロード時の最大host RSSは73.37 GB、最小system freeは39.45 GB、GPU peakは68,935 MiBだった。さらにTurbo8 1.0、style LoRA 0.7／0.4をこの順で登録し、対象module数312／312／208とruntime順序を確認した。
