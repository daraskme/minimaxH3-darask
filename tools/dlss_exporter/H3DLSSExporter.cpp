// Standalone NVIDIA DLSS Super Resolution file exporter for H3 Studio.
//
// This target is built against the exact audited dlss5-video-player source
// revision and NVIDIA's official DLSS SDK.  It deliberately does not load the
// separate, unsigned Neural Rendering runtime used by that player's
// experimental DLSS 5 path.

#include "D3D12Renderer.h"
#include "GuideControls.h"
#include "MediaPipeline.h"
#include "TemporalGuides.h"
#include "VideoDecoder.h"

#include <windows.h>
#include <mfapi.h>

#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <string_view>

namespace {

constexpr std::string_view kContract = "h3studio-dlss-sr-v1";

std::string NarrowAscii(std::wstring_view value)
{
    std::string result;
    result.reserve(value.size());
    for (const wchar_t character : value) {
        if (character < 0x20 || character > 0x7e) return {};
        result.push_back(static_cast<char>(character));
    }
    return result;
}

std::map<std::wstring, std::wstring> ParseArguments(int argc, wchar_t** argv)
{
    std::map<std::wstring, std::wstring> values;
    if (argc < 2 || std::wstring_view(argv[1]) != L"--h3-dlss-sr") return {};
    if ((argc - 2) % 2 != 0) return {};
    for (int index = 2; index < argc; index += 2) {
        std::wstring key(argv[index]);
        if (!key.starts_with(L"--") || values.contains(key)) return {};
        values.emplace(std::move(key), argv[index + 1]);
    }
    return values;
}

bool ParseUnsigned(const std::wstring& text, uint32_t& value)
{
    const std::string narrow = NarrowAscii(text);
    if (narrow.empty()) return false;
    uint64_t parsed{};
    const auto result = std::from_chars(narrow.data(), narrow.data() + narrow.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != narrow.data() + narrow.size() || parsed > UINT32_MAX)
        return false;
    value = static_cast<uint32_t>(parsed);
    return true;
}

bool ParseDouble(const std::wstring& text, double& value)
{
    const std::string narrow = NarrowAscii(text);
    if (narrow.empty()) return false;
    const auto result = std::from_chars(narrow.data(), narrow.data() + narrow.size(), value);
    return result.ec == std::errc{} && result.ptr == narrow.data() + narrow.size() &&
           std::isfinite(value) && value > 0.0;
}

std::string JsonEscape(std::string_view value)
{
    std::string result;
    for (const unsigned char character : value) {
        switch (character) {
        case '"': result += "\\\""; break;
        case '\\': result += "\\\\"; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default:
            if (character >= 0x20) result.push_back(static_cast<char>(character));
            break;
        }
    }
    return result;
}

bool WriteReceipt(const std::filesystem::path& path, bool ok, std::string_view detail,
                  uint32_t width, uint32_t height, double fps,
                  uint64_t frames, uint64_t evaluations)
{
    const auto temporary = path.wstring() + L".tmp";
    std::ofstream output(std::filesystem::path(temporary), std::ios::binary | std::ios::trunc);
    if (!output) return false;
    output << "{\"contract\":\"" << kContract << "\",\"ok\":" << (ok ? "true" : "false")
           << ",\"detail\":\"" << JsonEscape(detail) << "\",\"width\":" << width
           << ",\"height\":" << height << ",\"fps\":" << fps << ",\"frames\":" << frames
           << ",\"dlss_evaluations\":" << evaluations << "}";
    output.close();
    if (!output) return false;
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        return false;
    }
    return true;
}

int Fail(const std::filesystem::path& receipt, std::string_view detail,
         uint32_t width = 0, uint32_t height = 0, double fps = 0.0,
         uint64_t frames = 0, uint64_t evaluations = 0)
{
    WriteReceipt(receipt, false, detail, width, height, fps, frames, evaluations);
    std::cerr << "H3DLSS_ERROR " << detail << '\n';
    return 1;
}

class ComMfScope {
public:
    bool Start()
    {
        com_ = SUCCEEDED(CoInitializeEx(nullptr, COINIT_MULTITHREADED));
        mf_ = SUCCEEDED(MFStartup(MF_VERSION));
        return com_ && mf_;
    }
    ~ComMfScope()
    {
        if (mf_) MFShutdown();
        if (com_) CoUninitialize();
    }
private:
    bool com_{};
    bool mf_{};
};

} // namespace

int wmain(int argc, wchar_t** argv)
{
    const auto arguments = ParseArguments(argc, argv);
    const auto sourceIt = arguments.find(L"--source");
    const auto outputIt = arguments.find(L"--output");
    const auto receiptIt = arguments.find(L"--receipt");
    const auto helperIt = arguments.find(L"--ffmpeg-dir");
    const auto widthIt = arguments.find(L"--target-width");
    const auto heightIt = arguments.find(L"--target-height");
    const auto fpsIt = arguments.find(L"--expected-fps");
    const auto framesIt = arguments.find(L"--expected-frames");
    if (arguments.size() != 8 || sourceIt == arguments.end() || outputIt == arguments.end() ||
        receiptIt == arguments.end() || helperIt == arguments.end() || widthIt == arguments.end() ||
        heightIt == arguments.end() || fpsIt == arguments.end() || framesIt == arguments.end()) {
        std::cerr << "Usage: H3DLSSExporter --h3-dlss-sr --source FILE --output FILE --receipt FILE "
                     "--ffmpeg-dir DIR --target-width W --target-height H --expected-fps FPS --expected-frames N\n";
        return 2;
    }

    const std::filesystem::path source(sourceIt->second);
    const std::filesystem::path output(outputIt->second);
    const std::filesystem::path receipt(receiptIt->second);
    uint32_t targetWidth{}, targetHeight{}, expectedFrames{};
    double expectedFps{};
    if (!ParseUnsigned(widthIt->second, targetWidth) || !ParseUnsigned(heightIt->second, targetHeight) ||
        !ParseUnsigned(framesIt->second, expectedFrames) || !ParseDouble(fpsIt->second, expectedFps) ||
        targetWidth < 2 || targetHeight < 2 || targetWidth % 2 || targetHeight % 2 || !expectedFrames) {
        return Fail(receipt, "invalid numeric arguments");
    }
    if (!std::filesystem::is_regular_file(source) || std::filesystem::exists(output) ||
        std::filesystem::exists(receipt) || !std::filesystem::is_directory(output.parent_path()) ||
        !std::filesystem::is_directory(receipt.parent_path())) {
        return Fail(receipt, "invalid source or output paths");
    }

    ComMfScope platform;
    if (!platform.Start()) return Fail(receipt, "COM or Media Foundation initialization failed");

    VideoDecoder decoder;
    if (!decoder.Open(source.wstring(), MediaSourceKind::LocalFile))
        return Fail(receipt, "source decode initialization failed");
    // H3 Studio has already decoded every source PTS and rejected VFR before
    // launching this isolated worker. Media Foundation exposes only one rate,
    // so ConstantFrameRate() cannot independently corroborate it here. The
    // worker still checks the declared rate and exact decoded frame count.
    if (!decoder.FrameRateKnown() || std::abs(decoder.FrameRate() - expectedFps) > 0.005) {
        return Fail(receipt, "source frame rate is missing or differs from the validated request");
    }
    if (uint64_t{targetWidth} * targetHeight > uint64_t{7680} * 4320 ||
        targetWidth <= decoder.Width() || targetHeight <= decoder.Height()) {
        return Fail(receipt, "unsupported DLSS SR output geometry");
    }

    HWND window = CreateWindowExW(0, L"STATIC", L"H3 Studio DLSS SR", WS_POPUP,
                                  0, 0, 16, 16, nullptr, nullptr, GetModuleHandleW(nullptr), nullptr);
    if (!window) return Fail(receipt, "hidden D3D12 window creation failed");

    int exitCode = 1;
    {
        auto renderer = MakeD3D12Renderer();
        renderer->SetCaptureFormat(CaptureFormat::Nv12);
        const auto [gridWidth, gridHeight] =
            TemporalGuideGenerator::AnalysisGrid(decoder.Width(), decoder.Height(), decoder.FrameRate());
        if (!renderer->Initialize(window, decoder.Width(), decoder.Height(), targetWidth, targetHeight,
                                  gridWidth, gridHeight, NVSDK_NGX_PerfQuality_Value_MaxQuality, true) ||
            !renderer->DLSSAvailable()) {
            exitCode = Fail(receipt, "official NVIDIA DLSS SR initialization was rejected");
        } else if (renderer->OutputW() != targetWidth || renderer->OutputH() != targetHeight) {
            exitCode = Fail(receipt, "DLSS runtime reduced the requested output geometry",
                            renderer->OutputW(), renderer->OutputH(), decoder.FrameRate());
        } else {
            RawVideoEncoder encoder(helperIt->second);
            const EncoderPixelFormat pixelFormat = renderer->ActiveCaptureFormat() == CaptureFormat::Nv12
                ? EncoderPixelFormat::Nv12 : EncoderPixelFormat::Bgra;
            const EncoderSpec spec{targetWidth, targetHeight, decoder.FrameRate(),
                                   EncoderKind::H264Software, pixelFormat, 5};
            const EncodeError started = encoder.Start(spec, output);
            if (started != EncodeError::None) {
                exitCode = Fail(receipt, "H.264 output initialization failed", targetWidth, targetHeight,
                                decoder.FrameRate());
            } else {
                TemporalGuideGenerator guides;
                VideoFrame frame;
                CapturedVideoFrame captured;
                uint64_t frameCount{};
                bool ok = true;
                const float frameMilliseconds = static_cast<float>(1000.0 / decoder.FrameRate());
                while (decoder.ReadNext(frame)) {
                    GuideFrame guide;
                    const FrameIdentity identity = IdentityOf(
                        frame, guides.HistoryGeneration(), 1,
                        frameCount == 0 ? HistoryReset::FirstFrame : HistoryReset::None);
                    ok = guides.Generate(frame.bgra.data(), decoder.Width(), decoder.Height(), decoder.Width(),
                                         decoder.Height(), decoder.FrameRate(), identity, guide, frame.layout) &&
                         renderer->RenderFrameForCache(frame.bgra.data(), frame.bgra.size(), identity,
                                                       guide, frameMilliseconds, captured) &&
                         renderer->LastFrameUsedDLSS() && captured.width == targetWidth &&
                         captured.height == targetHeight &&
                         encoder.WriteFrame(captured.pixels) == EncodeError::None;
                    if (!ok) break;
                    ++frameCount;
                    std::cerr << "H3DLSS_PROGRESS " << frameCount << ' ' << expectedFrames << '\n';
                }
                const uint64_t evaluations = renderer->DLSSEvaluations();
                const EncodeError finished = ok ? encoder.Finish() : EncodeError::FinishFailed;
                if (!ok || finished != EncodeError::None || frameCount != expectedFrames ||
                    evaluations != frameCount || !std::filesystem::is_regular_file(output) ||
                    std::filesystem::file_size(output) == 0) {
                    encoder.Cancel();
                    std::error_code ignored;
                    std::filesystem::remove(output, ignored);
                    exitCode = Fail(receipt, "DLSS SR did not produce one evaluated frame per source frame",
                                    targetWidth, targetHeight, decoder.FrameRate(), frameCount, evaluations);
                } else if (!WriteReceipt(receipt, true, "", targetWidth, targetHeight, decoder.FrameRate(),
                                         frameCount, evaluations)) {
                    std::error_code ignored;
                    std::filesystem::remove(output, ignored);
                    exitCode = 1;
                } else {
                    exitCode = 0;
                }
            }
        }
        renderer.reset();
    }
    DestroyWindow(window);
    return exitCode;
}
