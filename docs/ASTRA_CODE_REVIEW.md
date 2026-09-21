# Astra adversarial implementation review

Review started 2026-09-21 JST. Scope: standalone application sources and installed Diffusers 0.40 / TorchAO 0.18 / PEFT 0.21 interfaces. Only review documents were edited by Astra; implementation changes belong to Sol. This is a live review, not a completed GPU-generation approval.

## Confirmed issues sent to Sol

| Priority | Location | Reproduction / consequence | Required resolution |
|---|---|---|---|
| P1 | `main.py:replay_job` | Original request has `seed=None` and resolved seed 12345; replay forwards `None`, drawing a new seed. | Replay the resolved seed; restore it in the UI. Preserve seed zero and avoid JavaScript loss of 63-bit precision. |
| P1 | `runner.py:_run` completion | Inject cancellation immediately before the completed store write: cancel returns `cancelling`, then final state becomes `completed`. | Serialize the final cancellation decision and completed write under the same lock as cancellation. |
| P1 | `main.py` local HTTP boundary | TestClient POST to `/api/outputs/open` with foreign Origin and Host returns 200 and invokes the mocked launcher. | Validate local Host and reject foreign-origin mutations, including bodyless requests and uploads. Local binding alone is insufficient against cross-origin requests / DNS rebinding. |
| P1 | `inventory.py:_repository_status` | A directory containing only `modular_model_index.json` with `{}` is reported ready. | Validate H3 class and every required configuration/tokenizer/processor/model weight or shard index. Missing indices must mean incomplete, not ready. |
| P1 | `schemas.py:combinations` / `engine.py` | Filename substring matching accepts a Ref2VA Turbo4 adapter against FL2VA, then applies an incompatible global shift 6. Multiple acceleration LoRAs can be stacked. | Explicit compatible recipe mapping by artifact/partition/steps/shifts; reject multiple acceleration adapters and wrong partition. |
| P1 | `runner.py:_run` error recovery | Any error during metadata remux/sidecar write deletes the already encoded GPU-generated video. | Preserve a valid clip as a failed-finalization artifact with an error and recovery path; do not mark it completed until required metadata succeeds. |

Two earlier issues have been fixed in source: claiming a queued job is now atomic with the running state, and LoRA root resolution no longer appends an extra `loras` folder. These changes still need the maintained regression suite.

## Independent runtime checks already passed

- The new FFmetadata temporary-file implementation avoids Windows' command-line length limit. Encoded a small real MP4 with audio; embedded a 19,200-character Japanese/newline/emoji prompt containing backslashes, equals, semicolons and hashes plus a 63-bit seed and two adapter settings. The entire JSON `comment` round-tripped exactly. One video stream and one audio stream survived the remux.
- The authoritative JSON comment retains original line endings. The convenience description field did not exactly match the separately normalized original string; this does not lose prompt data because the JSON is authoritative.
- Installed loader source confirms `num_inference_steps=N+1` yields N evaluations, `set_shift` is a real API and H3 supports `output_type="pil"`. No fabricated conventional pipeline callbacks are used.
- Header/meta-tensor checks for selected Turbo artifacts and actual small INT8 GPU multiple-adapter forward are recorded in `ASTRA_TURBO_VALIDATION.md`.

## Additional implementation inspection gates

- Full model-load failure and cancellation must drop partially allocated components, offload hooks and manager registrations; do not retain failed GPU state for the next job. Callbacks must be detached in finally blocks.
- Selectable optimization labels must correspond to actual applied kernels and a compatible recipe. Do not equate module discovery with validated kernel execution. Parent has separately exercised the installed Sage GPU dispatch.
- The 96 GB profile should use resident/staged INT8 components with GPU headroom; continuously streaming every layer is a lower-memory fallback, not automatically the optimized workstation default.
- Before marking ready, validate required runtime imports and complete native checkpoint assets. `load_components` catches individual component errors and logs warnings, so explicitly assert expected loaded components before publishing the cached pipeline.
- Full native H3 generation is still unverified during the large checkpoint download. No amount of mock/API/unit success substitutes for an actual video+audio generation, model/LoRA selection and metadata recovery.

## Fix verification, 2026-09-22

The confirmed application issues above are resolved in the current source and were rechecked independently:

- Replay of an originally random request now forwards its resolved seed 12345. Seeds are constrained to JavaScript's exact integer range, and the UI uses nullish fallback so seed zero survives restoration.
- A genuine second cancellation thread started immediately before the terminal store write blocks on the completion lock. When the completed write wins, cancellation subsequently returns `completed`; it no longer acknowledges cancellation and then overwrites it.
- Foreign Origin mutation returns 403; foreign Host read returns 400. The mocked output-folder launcher was not called.
- Simulated MP4 metadata remux failure leaves status `failed` while preserving both the generated clip and its JSON recovery record. A retry-finalize route is present.
- Inventory now validates model class, required components/configuration, tokenizer vocabularies, standalone weights or shard indices, nonempty shard maps, and referenced shard existence. An incomplete repository is not advertised as ready. No automatic remote model lookup occurs during generation.
- Explicit Turbo artifact recipes reject wrong-partition / unknown / stacked acceleration adapters and preserve independently weighted style adapters. Turbo8 uses 12/3 shifts; the known Turbo4 768p artifact uses 6/3.
- Failed/cancelled load or inference invalidates and unloads the cached pipeline. Loaded components are asserted explicitly. The 96 GB default is ComponentsManager automatic CPU offload with a 12 GB VRAM reserve; layer streaming is an explicit fallback.
- **Local component-root correction:** the official manifest retains Hub repository identifiers in ComponentSpecs. The engine now passes `pretrained_model_name_or_path=model_id` to `load_components`, preventing the remaining components from resolving to the wrong cache. A bounded independent runtime test loaded only the two weightless scheduler ComponentSpecs with this override: both resolved to the local project model directory, and five grid points produced four evaluations with the correct video/audio sigmas. No full-model allocation was needed for this test.

The maintained test suite passed (11 passed, one system-ffmpeg test skipped at the time of the run). The skipped test was also invoked independently using the bundled imageio ffmpeg and passed, including exact Japanese prompt description and audio preservation. The longer 19,200-character metadata test above also passed.

## Final frontend source follow-up

After the earlier fixes, a separate read-only audit of the completed frontend found two state issues, sent to Sol for correction:

- Refreshing inventory or restoring a history item with a missing LoRA selects the first available adapter silently. Preserve the requested missing adapter visibly and prevent generation while it remains enabled.
- Generation is allowed during an input-frame upload, when the frame token is still null. A late upload response can also restore a cleared/replaced frame. Track pending uploads, prevent generation until they settle, and discard stale responses after clear/restore/replacement.

Minor accessibility follow-up: associate the dynamic LoRA selector with an accessible label and provide a visible focus treatment for the visually hidden file inputs. Keyboard LoRA ordering, semantic field labels, live status regions, and reduced-motion styling are otherwise present in source.

Runtime diagnostics currently verify CUDA and Diffusers but should also check the required TorchAO / PEFT / Transformers classes before reporting readiness. This can use bounded imports without constructing model weights.

## Additional root-agent verification evidence

The root agent independently repeated MP4 verification after the LF-writing correction: a 15,000-character Japanese/emoji/newline prompt with `#;=` and backslash round-tripped exactly through both embedded JSON and description. The JSON sidecar matched, and 24 video frames plus the AAC audio stream survived.

The root agent also performed browser QA at desktop width 1440 and a narrow layout initially around 680 pixels. Primary layout, collapsed advanced/runtime settings, Turbo8's exact adapter auto-selection, two LoRA weights (1.0 / 0.6), and generation disabled for an incomplete model were checked. Astra did not reuse or control that browser tab; the independent frontend follow-up above was source-only.

The maintained suite subsequently reached 13 passed, including bundled-ffmpeg coverage, as reported by Sol. These reports are distinguished from Astra's earlier independent test runs.

## Current decision

The final generation-frontend issues are now fixed in source: missing LoRAs are preserved as explicit missing options and block submission when enabled; a restored missing model remains explicit; upload operations carry versions and AbortControllers, pending uploads block submission, and stale responses are discarded. LoRA controls update the generation gate, and an early `state.submitted` guard prevents duplicate keyboard submission. Dynamic LoRA selectors have accessible labels and file-input focus is visible.

Runtime diagnostics now import PEFT, TorchAO, Transformers and the required H3/Qwen/offload classes, and validate `Int8WeightOnlyConfig(version=2)` without constructing weights. The earlier readiness follow-up is resolved at source level.

**This is not an end-to-end engine approval:** complete native model loading, a real video with audio, practical model/LoRA switching, and the claimed acceleration's timing still require their runtime evidence after the model download. Do not claim completion or measured performance before those gates pass.

The later user-requested upscaling/interpolation expansion is reviewed separately in `ASTRA_POSTPROCESS_REVIEW.md`. Its bounded Real-ESRGAN/RIFE implementation now has a separate affirmative review with actual application GPU/media evidence, exact provenance verification, and 17 independently passing tests. That approval does not substitute for the outstanding full H3 generation gate.

The root agent subsequently closed the fractional-FPS and full AI provenance/audio checks: exact 24000/1001→48000/1001 FPS and 1001/1000-second duration; original metadata and prompt preserved through Real-ESRGAN→RIFE; AAC packet-byte hashes identical across the chain. Details and evidence ownership are recorded in the post-processing review. Full native H3 generation remains the major outstanding execution verification.

Final post-processing publication/cancellation checks also passed after restart: an actual RIFE output recorded its validation probe; cancelling Real-ESRGAN during GPU processing published no output; the next RIFE replay completed successfully (`30f653b3`). The post-processing backend review is closed and approved. Remaining final UI viewing and full H3 execution are separate from this completed backend review.

## Final history playback/resource follow-up

The root agent's final UI check found that polling reinserted unchanged history cards, disturbing their media elements. The page also crashed with 13 live video elements; the precise crash cause was not established, and a standalone MP4 played to completion. Do not claim that decoder count was conclusively the crash cause.

Sol changed history to one selected video player and lazy JPEG thumbnails for other clips. Astra's read-only source audit confirms that unchanged cards remain at their current DOM position, the selected preview stays first, and removed/replaced video elements are paused, have `src` removed, and are reset before disposal. The UI consumes the backend's `thumbnail_url` directly and includes it in the card signature. Failed thumbnails fall back to the ordinary placeholder.

The thumbnail route resolves only a stored output path under the configured outputs root, requires MP4 and forces the MOV decoder with restricted protocols. It creates a bounded first-frame JPEG on CPU, publishes atomically, caches by source revision, and reuses the existing JPEG. Stable URLs now require browser revalidation rather than incorrectly claiming a year of immutability. Cache target, temporary files and stale-revision scans use the resolved cache root.

Astra independently passed the thumbnail test and additional real-file checks: relative cache directories work, cache hits retain their original modification time, changing the video revision creates a new JPEG and removes the old revision, and an output-path traversal attempt is rejected with HTTP 404. Sol reported the maintained suite at **18 passed**; Astra's targeted thumbnail and adversarial cache checks are distinguished from that full-suite report.

The root agent's final browser QA passed after restarting the thumbnail-enabled server. A fresh in-app browser tab displayed 13 history records with one history video and 11 loaded JPEG thumbnails. A real eight-second MP4 played through the 1.8-second polling updates to `currentTime=8`, `duration=8`, `ended=true`, `readyState=4`, with no media error. Switching to a different job retained exactly one history video and selected the correct new source. Upscaling independently showed a ready status while H3 model download remained pending. This closes the playback-across-polling gate; Astra did not open or control that browser tab.

The root agent also completed the 390×844 responsive check: document client width and scroll width both measured 375 pixels (the remaining 15 pixels were the scrollbar), controls fit without horizontal overflow, and no new console errors appeared. The viewport was restored and the deliverable browser tab retained. This closes that responsive-view gate.
