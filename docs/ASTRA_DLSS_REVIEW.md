# Astra DLSS runtime and file-export plan review

Reviewed 2026-09-22 for the user's explicit request to try DLSS on RTX PRO 6000 Blackwell, with driver/runtime setup and real file export. Astra's work here is source/provenance/plan review; no downloaded executable or unsigned runtime was run by Astra. Existing Real-ESRGAN/RIFE approval does not extend automatically to this new engine.

## Exact upstream and artifact identity

The actual current release is the **pre-release** tag `dlss5-video-player-v0.24.0`, not `v0.24.0`. Direct GitHub API checks of `/releases/latest` and the latter tag returned 404 because the first excludes pre-releases and the second was the wrong name. They did not prove the repository unavailable. The correct tagged API request succeeded and supplied these exact assets. [Release](https://github.com/2600th/dlss5-video-player/releases/tag/dlss5-video-player-v0.24.0)

- Commit: `ffb01f5d015ee1a0544a6a46452cfe4b334553a3` (annotated tag object `9c966276bb0bf2dbc4028e50533e5141b3d1775d`).
- Complete ZIP: `dlss5-video-player-v0.24.0-win64.zip`, 327,555,972 bytes, SHA-256 `66947ca33b84459d13b3002cfea153f295c1d0543260713eced6b9766a136e78`.
- Core-only ZIP: `DLSSVideoPlayer-v0.24.0-core-win64.zip`, 36,678,552 bytes, SHA-256 `19a5fd715e3acda98ca089f8a74c5a81a666bcef7e149aa3eeed73021dd10dc1`.

Exact-tag C++ and runtime-lock files were downloaded as text into `.cache/research/dlss5-v0.24.0`. Cached web views of `main` were inconsistent: one showed a version-3 protocol while newer worker source used additional messages. The exact release header confirms **protocol 6**. Mixing cached main snippets with the release binary is a concrete integration bug; pin the matching artifacts and source together.

## What can actually be exported

The community project's neural helper uses a native-resolution DLAA carrier. Its validated neural cache/export remains at source resolution. The separate playback SR toggle and presentation adjustments do not affect that exported result. Output is 8-bit, so changing a container cannot restore original HDR precision. These limits are stated in the [technical overview](https://github.com/2600th/dlss5-video-player/blob/dlss5-video-player-v0.24.0/TECHNICAL_OVERVIEW.md).

| Mode | Correct integration claim |
| --- | --- |
| Community DLSS neural rendering | Experimental same-resolution video transformation through the external worker, once actual feature-18 execution and export are verified. |
| DLSS Super Resolution | A separate SR operation. The existing player's display toggle is not a file-export implementation. |
| DLAA carrier | Native NGX carrier used by the neural interception path; successful carrier evaluation alone does not prove neural rendering occurred. |

Therefore the first working worker adapter must not be represented as a 2× upscaler. If higher resolution is wanted, chain a separately named, verified upscaler after neural rendering. A genuine SR file exporter requires its own evaluated-output capture path and tests; multiplying the saved video with ffmpeg is not DLSS SR.

## Hardware, runtime provenance and terms

The upstream tests require Windows x64, D3D12 NVIDIA RTX and driver 610.47 or newer. Its documentation discusses RTX-branded workstation hardware but does not make the GPU name sufficient evidence; actual feature-18 capability and render receipts decide support. The current local driver reported earlier was 597.06. Driver staging/installation and any reboot coordination belong to the root agent's separate system-change work. [Runtime setup](https://github.com/2600th/dlss5-video-player/blob/dlss5-video-player-v0.24.0/docs/DLSS5_SETUP.md)

The project is not an NVIDIA product or official DLSS 5 SDK integration. Its neural stack includes a modified runtime derived from a leaked binary, with NVIDIA's signature removed; ReShade/RenoDX are also unsigned. The source is MIT, but that does not relicense NVIDIA components. Upstream states that redistribution of the combined experimental binary set is unresolved. For this local integration, preserve the notices and provenance; do not bundle these binaries into the repository or publish them under the project's source license. These are upstream disclosures, not a new license grant. [Third-party notices](https://github.com/2600th/dlss5-video-player/blob/dlss5-video-player-v0.24.0/THIRD_PARTY.md)

The exact-tag [runtime lock](https://github.com/2600th/dlss5-video-player/blob/dlss5-video-player-v0.24.0/packaging/runtime-lock.json) pins every component. For example, `nvngx_dlssnr.dll` is 165,830,144 bytes, SHA-256 `6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927`, numeric version 310.8.2.0, explicitly `NotSigned`. Signed `nvngx_dlss.dll` is a different SR component; its valid NVIDIA signature must not be generalized to the neural DLL. A matching archive/file hash proves identity, not safety or official endorsement. The [security document](https://github.com/2600th/dlss5-video-player/blob/dlss5-video-player-v0.24.0/SECURITY.md) likewise distinguishes these properties.

## Concrete worker contract

This is a binary Win32 inherited-handle protocol, not a conventional `--input --output` executable returning JSON on stdout. The [exact-tag launcher](https://github.com/2600th/dlss5-video-player/blob/dlss5-video-player-v0.24.0/src/NeuralWorker.cpp) defines:

- Preflight: `--neural-preflight --metadata-handle HANDLE`.
- Single render: `--neural-worker`, metadata handle, source path, staging path, width, height, FPS, duration in 100 ns units, numeric job id, range start/end, preroll frame count, frame retry limit and canonical guides `mv=1,depth=1`.
- The corresponding option names are `--source`, `--staging`, `--width`, `--height`, `--fps`, `--duration-100ns`, `--job-id`, `--range-start-100ns`, `--range-end-100ns`, `--preroll-frames`, `--frame-retry-limit`, `--guides`.
- A configuration-repair exit may require one clean relaunch with `--configuration-restarted`; do not loop indefinitely.

The [release protocol header](https://github.com/2600th/dlss5-video-player/blob/dlss5-video-player-v0.24.0/src/NeuralWorkerProtocol.h) uses magic `0x3152574E`, version 6 and a 12-byte packed header. Payload limit is 64 KiB; Progress is 52 bytes, Result 152 bytes plus bounded UTF-16 detail, Preflight 8 bytes plus UTF-8 JSON, Timeline 80 bytes, Memory 16 bytes. The adapter must reject wrong versions, truncated/oversized messages, invalid enums/booleans, nonfinite metrics and mismatched job identities. A successful result requires nonzero frames/evaluations, adequate verified neural frames, feature-18 armed before capture, valid interception/evaluation evidence, no later failure and no cancellation. An exit code of zero can accompany a failure receipt: process exit alone must never publish a successful DLSS result.

The [preflight contract](https://github.com/2600th/dlss5-video-player/blob/dlss5-video-player-v0.24.0/src/NeuralPreflight.h) distinguishes feature-18 creation from the SR carrier's creation result. It also records GPU/driver, runtime identities and model-store identity. Preserve those distinctions in diagnostics and metadata rather than turning any successful NGX call into a green status.

## Integration and adversarial acceptance gates

The following are proposed application requirements, rather than claims that the integration already works:

1. Keep the external runtime in its own versioned directory with the complete locked layout. Keep its proxy DLL out of H3 Studio's root and Python environment. Verify archive identity, contained extraction, component hashes and expected signature states before capability probing. Do not lower Defender or signature policies to make a probe succeed.
2. Use the existing shared GPU queue and unload the H3 pipeline before the worker starts. Prefer a single-shot hidden worker initially so process exit releases the D3D12/NGX workset before the next H3/RIFE job. A retained worker needs explicit memory ownership and is unnecessary for first correctness.
3. Use a narrow inherited handle list, an actual metadata pipe reader, no command shell, job-specific staging, bounded logs and timeouts. Kill and reap the worker plus owned descendants on cancellation/crash; close handles before deleting staging on Windows. Avoid leaving a decoder/encoder child holding GPU resources after the parent exits.
4. Keep capability states separate: files installed, driver acceptable, runtime identity verified, preflight passed, export smoke passed. Driver/model name or file presence alone cannot enable a verified engine. Cache probe results by driver, runtime hash and GPU identity; invalidate them after any of those changes.
5. Start with complete local CFR SDR clips. Preserve source resolution, exact frame count/timeline and original audio; reject unsupported HDR/VFR inputs explicitly or use a separately disclosed normalization path. Save the original settings, source hash, actual runtime/driver, effective neural settings and receipt alongside the existing provenance chain.
6. Require genuine feature-18 receipt evidence and a independently decoded output before remuxing audio, embedding metadata and atomic publication. Reuse the existing output geometry/FPS/duration/frame-count/audio checks. Pass-through, carrier-only success, partial output or an old receipt must fail closed.
7. Exercise invalid/missing runtime, old driver, protocol mismatch, failed receipt with exit zero, truncated pipe, cancellation and subsequent next-job recovery before approval. On the RTX PRO 6000, run a short real video and compare original versus rendered pixels while confirming the receipt: pixel differences by themselves can also come from lossy encoding.

## Decision and current limits

The isolated-worker architecture is implementable against the pinned contract. Same-resolution experimental neural export is the supported initial scope of that adapter. The existing upstream playback SR feature is not yet evidence of SR file export.

No unsigned runtime has been executed during this review and no local DLSS capability/export success is claimed. Runtime and driver setup, actual feature-18 probing and real media output remain execution gates for Sol/root. The user has subsequently also requested latent two-pass, SeedVR2 and VC Attention work; those have separate engine contracts and must not inherit a DLSS approval or use a reference approximation under the real engine's name.

## Implementation follow-up: signed SR route

Sol has selected a separate locally built SR exporter using [NVIDIA/DLSS v310.9.1](https://github.com/NVIDIA/DLSS/tree/v310.9.1), revision `374959484e79a640feaba44c93ac8cfb0a03f5b5`. Sol reports the official x64 release `nvngx_dlss.dll` has a valid NVIDIA signature and SHA-256 `3975567b8943c53acce397f2b72380092f84f162d00b0d2c7d08a1025c563983`. That SDK contains SR/DLAA, RR and FG runtimes; it does not supply the community neural runtime. This route can provide actual DLSS SR export if the new native capture implementation passes real media tests; it does not complete DLSS 5 neural rendering.

Early source review of `h3studio/dlss.py` was sent to Sol before the implementation was declared coherent: readiness must not turn green from DLL/EXE presence alone; inherited pipes may split a protocol header across reads; receipt decoding must reject invalid boolean/enum/reserved values, nonfinite metrics and malformed detail strings. These remain review findings pending the final corrected source and tests, rather than a final verdict on unfinished code.
