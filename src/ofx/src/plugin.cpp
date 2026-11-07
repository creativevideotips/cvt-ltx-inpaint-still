// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.
//
// LTX Inpaint Still: an OFX effect for Fusion that makes clean plate stills with the LTX
// in/outpainting IC-LoRA. Based on the CVT port of LTX Alpha Gen (see CVT_PORT.md).
//
// Process reads one frame (the Reference Frame) of the picture and its mask, and starts a job in
// its own process (worker/still_job.py), which shows the progress window. When the job ends, the
// node holds the chosen plate on every frame, so it can go straight into a tracker.
//
// Facts about the Fusion host this relies on, tested with Alpha Gen: images are trimmed to their
// Domain of Definition, so everything is placed on the frame's canvas; background setValue works;
// label params show their name, so read-only text is a disabled single-line field.
#include <dlfcn.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "cache_folder.h"
#include "half.h"
#include "log.h"
#include "marker.h"
#include "ofxsImageEffect.h"
#include "plate_io.h"
#include "text_wrap.h"
#include "platform.h"
#include "process.h"
#include "transport.h"
#include "user_config.h"

#ifndef LTXIS_VERSION
#define LTXIS_VERSION "0.0"
#endif
#ifndef LTXIS_VERSION_MAJOR
#define LTXIS_VERSION_MAJOR 1
#define LTXIS_VERSION_MINOR 0
#endif

namespace {

using namespace ltxag;

constexpr const char* kPluginId = "com.creativevideotips.ltx.inpaintstill";
constexpr const char* kPluginLabel = "LTX Inpaint Still";
constexpr const char* kPluginGroup = "Creative Video Tips";
constexpr const char* kMaskClip = "Mask";
constexpr const char* kWebsiteUrl = "https://creativevideotips.com";
constexpr const char* kModelPageUrl = "https://huggingface.co/Lightricks/LTX-2.5";
constexpr const char* kLoraPageUrl = "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting";
constexpr const char* kTokenPageUrl = "https://huggingface.co/settings/tokens";
constexpr const char* kPShowSetup = "showSetup";
constexpr const char* kPViewLicence = "viewLicence";
constexpr const char* kPAccept = "acceptLicence";
constexpr const char* kPModelPage = "openModelPage";
constexpr const char* kPLoraPage = "openLoraPage";
constexpr const char* kPTokenPage = "openTokenPage";
constexpr const char* kPToken = "hfToken";
constexpr const char* kPDownload = "download";
constexpr const char* kPExistingDir = "existingModels";
constexpr const char* kPLinkExisting = "linkExisting";
constexpr const char* kPChooseStills = "chooseStillsFolder";
constexpr const char* kPChooseExisting = "chooseExistingModels";
constexpr const char* kPChooseModel = "chooseModelFolder";
constexpr const char* kPChooseCache = "chooseCacheFolder";
constexpr const char* kPSetupStatus = "setupStatus";
constexpr const char* kPCollectLogs = "collectLogs";
constexpr double kMemoryNeedGB = 42.0;  // measured: 41 to 49 GB reserved at HD on an M5 Max
constexpr double kMemoryNeedLinearGB = 52.0;  // ACEScg or Linear sRGB: about 56 GB reserved, with the VAE at float32
constexpr int kMaxVariations = 8;
constexpr int kMaxSeed = 1000000;

const char* const kMethodNames[] = {"Detail (one pass)", "Structure (two pass)"};
const char* const kMethodIds[] = {"detail", "structure"};
const char* const kViewNames[] = {"LTX AI Still", "Original"};
const char* const kScaleNames[] = {"1x", "2x", "4x"};
const int kScales[] = {1, 2, 4};

constexpr const char* kPProcess = "process";
constexpr const char* kPInputs = "inputsInfo";
constexpr const char* kPStatus = "status";
constexpr const char* kPReference = "referenceFrame";
constexpr const char* kPUseCurrent = "useCurrentFrame";
constexpr const char* kPMethod = "method";
constexpr const char* kPVariations = "variations";
constexpr const char* kPSeed = "seed";
constexpr const char* kPPrompt = "prompt";
constexpr const char* kPPersonCheck = "personCheck";
constexpr const char* kPCopyLlm = "copyForLlm";
constexpr const char* kPColour = "inputColour";
constexpr const char* kColourIds[] = {"as_is", "acescg", "srgb_linear"};
constexpr const char* kColourNames[] = {"As Is (Rec.709, faster)", "ACEScg", "Linear sRGB"};
constexpr size_t kPromptColumns = 48;  // about what the Inspector's box shows at its usual width
constexpr const char* kPWorkScale = "workScale";
constexpr const char* kPGrow = "maskGrow";
constexpr const char* kPBlend = "blend";
constexpr const char* kPShow = "showVariation";
constexpr const char* kPView = "viewMode";
constexpr const char* kPFolder = "plateFolder";
constexpr const char* kPOpenFolder = "openStillsFolder";
constexpr const char* kPRelease = "releaseMemory";
constexpr const char* kPDwab = "dwabLevel";
constexpr const char* kPName = "plateName";
constexpr const char* kPModelDir = "modelDir";
constexpr const char* kPCacheDir = "cacheDir";
constexpr const char* kPVersion = "versionInfo";
constexpr const char* kPCredits = "credits";
constexpr const char* kPWebsite = "openWebsite";
constexpr const char* kPNodeId = "nodeId";
constexpr const char* kPReferenceSet = "referenceSet";
constexpr const char* kPResult = "resultState";
constexpr const char* kPTick = "refreshTick";

std::string resourcesDir() {
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void*>(&resourcesDir), &info) || !info.dli_fname) return "";
  std::string p = info.dli_fname;  // .../Contents/MacOS/LTXInpaintStill.ofx
  p = p.substr(0, p.rfind('/'));
  p = p.substr(0, p.rfind('/'));
  return p + "/Resources";
}

std::string homeDir() {
  const char* h = std::getenv("HOME");
  return h ? h : "/tmp";
}

std::string cacheRoot() { return homeDir() + "/.cache/ltx-inpaint-still"; }

std::string cacheFolderSetting();  // below, once the settings can be read

std::string randomId() {
  std::random_device rd;
  char b[9];
  std::snprintf(b, sizeof b, "%08x", rd());
  return b;
}

std::string jsonStr(const std::string& s) {
  std::string o = "\"";
  for (const char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    if (static_cast<unsigned char>(c) < 0x20) {  // a typed prompt can hold these, and raw they aren't valid JSON
      char escaped[8];
      std::snprintf(escaped, sizeof escaped, "\\u%04x", static_cast<unsigned>(c));
      o += escaped;
      continue;
    }
    o += c;
  }
  return o + "\"";
}

// Value of "key" in a status file written by Python's json.dump.
std::string jsonField(const std::string& j, const std::string& key) {
  const std::string k = "\"" + key + "\": ";
  const auto p = j.find(k);
  if (p == std::string::npos) return "";
  const size_t b = p + k.size();
  if (b < j.size() && j[b] == '"') return j.substr(b + 1, j.find('"', b + 1) - b - 1);
  const auto e = j.find_first_of(",}", b);
  return j.substr(b, e - b);
}

std::string readFile(const std::string& path) {
  std::ifstream in(path);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// A setting from this effect's config, else Alpha Gen's, whose models and runtime it shares.
std::string sharedSetting(const std::string& key) {
  const UserConfig own = loadUserConfig();
  if (own.count(key)) return own.at(key);
  std::ifstream in(homeDir() + "/.config/ltx-alphagen/config.conf");
  std::string line;
  while (std::getline(in, line))
    if (line.rfind(key + "=", 0) == 0) return line.substr(key.size() + 1);
  return "";
}

bool appleToolsInstalled() {
  if (std::getenv("LTXAG_TEST_NO_APPLE_TOOLS")) return false;
  std::vector<std::string> roots = {"/Library/Developer/CommandLineTools",
                                    "/Applications/Xcode.app/Contents/Developer"};
  for (const char* selected : {"/var/select/developer_dir", "/var/db/xcode_select_link"}) {
    char link[1024];
    const ssize_t n = readlink(selected, link, sizeof(link));
    if (n > 0) roots.emplace_back(link, size_t(n));
  }
  for (const auto& root : roots)
    if (access((root + "/usr/bin/python3").c_str(), X_OK) == 0) return true;
  return false;
}

std::string timeStr(const char* fmt) {
  std::time_t t = std::time(nullptr);
  std::tm tm{};
  localtime_r(&t, &tm);
  char b[32];
  std::strftime(b, sizeof b, fmt, &tm);
  return b;
}

std::string trimSlashes(std::string p) {
  while (p.size() > 1 && p.back() == '/') p.pop_back();
  return p;
}

uint64_t fnv(const std::string& s) {
  uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : s) h = (h ^ c) * 1099511628211ULL;
  return h;
}

std::string licencePath() { return resourcesDir() + "/licence.txt"; }

std::string licenceHash() {
  char b[17];
  std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(fnv(readFile(licencePath()))));
  return b;
}

// Per user. Someone with LTX Alpha Gen already has the models and the runtime: it uses those.
std::string modelFolderSetting() {
  const std::string shared = sharedSetting("model_dir");
  // A folder that was set up and has since been deleted is left over in the settings, not a choice. A
  // folder on a drive that's away may come back, and one typed in but not made yet is made by Setup.
  struct stat st {};
  const bool leftOver = !shared.empty() && !sharedSetting("models_ready").empty() && stat(shared.c_str(), &st) != 0 &&
                        !volumeIsMissing(shared);
  return shared.empty() || leftOver ? homeDir() + "/LTX-2.5" : shared;
}

struct SetupState {
  bool licenceAccepted = false;
  bool modelsReady = false;
  bool ready() const { return licenceAccepted && modelsReady; }
};

// The licence is the same one Alpha Gen shows, so accepting it there counts here. Ready needs the runtime
// and this effect's own LoRA, which someone coming from Alpha Gen doesn't have yet.
SetupState setupState() {
  SetupState s;
  s.licenceAccepted = !sharedSetting("licence_accepted").empty() && sharedSetting("licence_hash") == licenceHash();
  const std::string dir = modelFolderSetting();
  const std::string python = sharedSetting("ltx_python");
  struct stat st {};
  s.modelsReady = !python.empty() && stat(python.c_str(), &st) == 0;
  for (const char* file : {"/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
                           "/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
                           "/vae/ltx-2.5-video-vae-bf16.safetensors",
                           "/loras/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"})
    s.modelsReady = s.modelsReady && stat((dir + file).c_str(), &st) == 0;
  return s;
}

// Per user, like the model folder.
std::string cacheFolderSetting() {
  const UserConfig own = loadUserConfig();
  const auto found = own.find("still_cache_dir");
  if (found != own.end() && !found->second.empty()) return found->second;
  return defaultCacheFolder(sharedSetting("cache_dir"), cacheRoot());
}

bool containsNoCase(std::string s, const char* what) {
  for (auto& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
  return s.find(what) != std::string::npos;
}

bool hostIsResolve() {
  const auto* d = OFX::getImageEffectHostDescription();
  return d && (containsNoCase(d->hostName, "resolve") || containsNoCase(d->hostLabel, "resolve"));
}

int componentsOf(const OFX::Image& img) {
  switch (img.getPixelComponents()) {
    case OFX::ePixelComponentRGBA: return 4;
    case OFX::ePixelComponentRGB: return 3;
    case OFX::ePixelComponentAlpha: return 1;
    default: return 0;
  }
}

struct Canvas {
  int x = 0, y = 0, w = 0, h = 0;  // OFX pixel coordinates, bottom-left origin
  bool valid() const { return w > 0 && h > 0; }
};

std::string formatCanvas(const Canvas& c) {
  return std::to_string(c.x) + "," + std::to_string(c.y) + "," + std::to_string(c.w) + "," + std::to_string(c.h);
}

// An image on the canvas: c.w * c.h * 4 floats, rows top-down. A one-channel image goes in every
// channel. Alpha is 1 where an image has none, and everything is 0 outside the image.
void placeOnCanvas(const OFX::Image& img, const Canvas& c, std::vector<float>& buf) {
  buf.assign(size_t(c.w) * c.h * 4, 0.0f);
  const OfxRectI b = img.getBounds();
  const int comps = componentsOf(img);
  const int x0 = std::max(b.x1, c.x), x1 = std::min(b.x2, c.x + c.w);
  if (x0 >= x1 || comps == 0) return;
  for (int y = std::max(b.y1, c.y); y < std::min(b.y2, c.y + c.h); ++y) {
    const auto* p = static_cast<const float*>(img.getPixelAddress(x0, y));
    float* q = buf.data() + (size_t(c.y + c.h - 1 - y) * c.w + (x0 - c.x)) * 4;
    for (int x = x0; x < x1; ++x, p += comps, q += 4) {
      if (comps == 1) {
        q[0] = q[1] = q[2] = q[3] = p[0];
      } else {
        q[0] = p[0], q[1] = p[1], q[2] = p[2];
        q[3] = comps == 4 ? p[3] : 1.0f;
      }
    }
  }
}

// The mask from an image on the canvas: its alpha, or its red channel if the alpha says nothing
// (a mask drawn as a white-on-black picture has solid alpha). Empty means no usable mask.
std::vector<float> maskFrom(const std::vector<float>& rgba, bool allowRed) {
  const size_t n = rgba.size() / 4;
  std::vector<float> mask(n);
  float low = 1e9f, high = -1e9f;
  for (size_t i = 0; i < n; ++i) {
    const float a = rgba[i * 4 + 3];
    mask[i] = a;
    low = std::min(low, a), high = std::max(high, a);
  }
  if (high - low > 1e-3f) return mask;
  if (!allowRed) return {};
  low = 1e9f, high = -1e9f;
  for (size_t i = 0; i < n; ++i) {
    const float r = rgba[i * 4];
    mask[i] = r;
    low = std::min(low, r), high = std::max(high, r);
  }
  return high - low > 1e-3f ? mask : std::vector<float>{};
}

bool writeStaged(const std::string& path, int w, int h, uint32_t channels, const std::vector<uint16_t>& half,
                 std::string& err) {
  std::ofstream f(path, std::ios::binary);
  const FrameFileHeader header{{'L', 'T', 'X', 'A'}, kTransportVersion, uint32_t(w), uint32_t(h), 1u, channels};
  f.write(reinterpret_cast<const char*>(&header), sizeof header);
  f.write(reinterpret_cast<const char*>(half.data()), std::streamsize(half.size() * 2));
  f.close();
  if (!f) {
    err = "could not write " + path;
    return false;
  }
  chmod(path.c_str(), 0600);  // Resolve runs with umask 000
  return true;
}

// What the last finished job made, saved with the comp: frame|method|seeds|canvas|path,path,...
struct Result {
  int frame = 0;
  std::string method;
  std::vector<int> seeds;
  Canvas canvas;
  std::vector<std::string> paths;
  bool valid() const { return !paths.empty() && paths.size() == seeds.size(); }
};

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> parts;
  std::string part;
  std::stringstream ss(s);
  while (std::getline(ss, part, sep)) parts.push_back(part);
  return parts;
}

Result parseResult(const std::string& s) {
  Result r;
  const auto parts = split(s, '|');
  if (parts.size() != 5) return r;
  try {
    r.frame = std::stoi(parts[0]);
    r.method = parts[1];
    for (const auto& seed : split(parts[2], ',')) r.seeds.push_back(std::stoi(seed));
    if (std::sscanf(parts[3].c_str(), "%d,%d,%d,%d", &r.canvas.x, &r.canvas.y, &r.canvas.w, &r.canvas.h) != 4)
      return Result{};
    r.paths = split(parts[4], ',');
  } catch (const std::exception&) {
    return Result{};
  }
  return r;
}

std::atomic<unsigned> gJobSeq{0};

class InpaintStillEffect : public OFX::ImageEffect {
 public:
  explicit InpaintStillEffect(OfxImageEffectHandle handle) : ImageEffect(handle) {
    src_ = fetchClip(kOfxImageEffectSimpleSourceClipName);
    mask_ = fetchClip(kMaskClip);
    dst_ = fetchClip(kOfxImageEffectOutputClipName);
    status_ = fetchStringParam(kPStatus);
    reference_ = fetchIntParam(kPReference);
    method_ = fetchChoiceParam(kPMethod);
    variations_ = fetchIntParam(kPVariations);
    seed_ = fetchIntParam(kPSeed);
    prompt_ = fetchStringParam(kPPrompt);
    colour_ = fetchChoiceParam(kPColour);
    personCheck_ = fetchBooleanParam(kPPersonCheck);
    workScale_ = fetchChoiceParam(kPWorkScale);
    grow_ = fetchIntParam(kPGrow);
    blend_ = fetchIntParam(kPBlend);
    show_ = fetchIntParam(kPShow);
    view_ = fetchChoiceParam(kPView);
    folder_ = fetchStringParam(kPFolder);
    dwab_ = fetchIntParam(kPDwab);
    name_ = fetchStringParam(kPName);
    modelDir_ = fetchStringParam(kPModelDir);
    cacheDir_ = fetchStringParam(kPCacheDir);
    showSetup_ = fetchBooleanParam(kPShowSetup);
    accept_ = fetchBooleanParam(kPAccept);
    token_ = fetchStringParam(kPToken);
    existingDir_ = fetchStringParam(kPExistingDir);
    linkExisting_ = fetchBooleanParam(kPLinkExisting);
    setupStatus_ = fetchStringParam(kPSetupStatus);
    nodeIdParam_ = fetchStringParam(kPNodeId);
    referenceSet_ = fetchBooleanParam(kPReferenceSet);
    result_ = fetchStringParam(kPResult);
    tick_ = fetchIntParam(kPTick);
    try {
      nodeIdParam_->getValue(nodeId_);
      if (nodeId_.empty()) {
        nodeId_ = randomId();  // saved with the comp; copies of a node share it
        nodeIdParam_->setValue(nodeId_);
      }
      LogScope scope(nodeId_);
      LOGI("createInstance %p", static_cast<void*>(this));
      modelDir_->setValue(modelFolderSetting());
      cacheDir_->setValue(cacheFolderSetting());
      accept_->setValue(setupState().licenceAccepted);
      refreshSetupUi();
    } catch (const std::exception& e) {
      LOGE("createInstance FAILED: %s", e.what());
    }
  }

  ~InpaintStillEffect() override {
    stopFlag_ = true;
    if (watcher_.joinable()) watcher_.join();
    stopSetupWatcher();
    stopChoosingFolder();
  }

  void getClipPreferences(OFX::ClipPreferencesSetter& prefs) override {
    prefs.setClipComponents(*dst_, OFX::ePixelComponentRGBA);
    prefs.setOutputPremultiplication(OFX::eImageOpaque);
  }

  void changedParam(const OFX::InstanceChangedArgs& args, const std::string& name) override {
    if (args.reason != OFX::eChangeUserEdit || inChangedParam_) return;
    struct Guard {
      bool& b;
      explicit Guard(bool& b) : b(b) { b = true; }
      ~Guard() { b = false; }
    } guard(inChangedParam_);
    // The saved id is the one a script finds the node by, and Fusion can make a second instance that
    // changes it while this one lives on. So it is read fresh, never trusted from createInstance.
    try {
      std::string saved;
      nodeIdParam_->getValue(saved);
      if (!saved.empty()) nodeId_ = saved;
    } catch (const std::exception&) {
    }
    LogScope scope(nodeId_);
    LOGI("changedParam %s time=%g", name.c_str(), args.time);
    try {
      if (name == kPProcess) {
        process(args.time);
      } else if (name == kPUseCurrent) {
        reference_->setValue(int(std::lround(args.time)));
        referenceSet_->setValue(true);
      } else if (name == kPReference) {
        // Fusion also reports this as a user edit when a node is pasted, with the field still at its
        // default of 0, so 0 doesn't count as choosing a frame.
        int value = 0;
        reference_->getValue(value);
        if (value != 0) referenceSet_->setValue(true);
      } else if (name == kPPrompt) {
        // OpenFX has no word-wrap setting, so the text itself gets line breaks to fit the box.
        std::string text;
        prompt_->getValue(text);
        const std::string wrapped = wrapText(text, kPromptColumns);
        if (wrapped != text) prompt_->setValue(wrapped);
      } else if (name == kPChooseStills) {
        chooseFolder(folder_, "Choose the folder for stills", "");
      } else if (name == kPChooseExisting) {
        chooseFolder(existingDir_, "Choose a folder that has the model files", "");
      } else if (name == kPChooseModel) {
        chooseFolder(modelDir_, "Choose the folder for the LTX-2.5 models", "model_dir");
      } else if (name == kPChooseCache) {
        chooseFolder(cacheDir_, "Choose the cache folder", "still_cache_dir");
      } else if (name == kPCopyLlm) {
        copyForLlm(args.time);
      } else if (name == kPOpenFolder) {
        openStillsFolder();
      } else if (name == kPRelease) {
        releaseMemory();
      } else if (name == kPWebsite) {
        std::string err;
        launchDetached({"/usr/bin/open", kWebsiteUrl}, cleanEnvironment(), err);
      } else if (name == kPCacheDir) {
        std::string dir;
        cacheDir_->getValue(dir);
        UserConfig cfg = loadUserConfig();
        cfg["still_cache_dir"] = dir;
        saveUserConfig(cfg);
      } else if (name == kPShowSetup) {
        refreshSetupUi();
      } else if (name == kPViewLicence) {
        std::string err;
        if (!showTextFile("LTX-2.x Licence", licencePath(), cleanEnvironment(), err)) showError("Could not show the licence: " + err);
      } else if (name == kPAccept) {
        bool accepted = false;
        accept_->getValue(accepted);
        UserConfig cfg = loadUserConfig();
        cfg["licence_accepted"] = accepted ? timeStr("%Y-%m-%d") : "";
        cfg["licence_hash"] = accepted ? licenceHash() : "";
        saveUserConfig(cfg);
        LOGI("licence %s", accepted ? "accepted" : "withdrawn");
        refreshSetupUi();
      } else if (name == kPModelPage) {
        openPage(kModelPageUrl);
      } else if (name == kPLoraPage) {
        openPage(kLoraPageUrl);
      } else if (name == kPTokenPage) {
        openPage(kTokenPageUrl);
      } else if (name == kPDownload) {
        download();
      } else if (name == kPCollectLogs) {
        collectLogs();
      } else if (name == kPModelDir) {
        std::string dir;
        modelDir_->getValue(dir);
        UserConfig cfg = loadUserConfig();
        cfg["model_dir"] = dir;
        saveUserConfig(cfg);
        refreshSetupUi();
      }
    } catch (const std::exception& e) {
      LOGE("changedParam %s FAILED: %s", name.c_str(), e.what());
      if (name == kPProcess) showError(std::string("Process failed (") + e.what() + "): see the log.");
      throw;
    }
  }

  void render(const OFX::RenderArguments& args) override {
    LogScope scope(nodeId_);
    std::unique_ptr<OFX::Image> out(dst_->fetchImage(args.time));
    if (!out) OFX::throwSuiteStatusException(kOfxStatFailed);
    if (onEditOrColorPage()) {  // the effect cannot be hidden there
      drawCaption(*out, args, "USE FUSION PAGE");
      return;
    }
    std::string state;
    result_->getValueAtTime(args.time, state);
    const Result result = parseResult(state);
    if (result.valid() && showingOriginal(args.time)) {
      drawSource(*out, args, double(result.frame));  // the frame the stills were made from
      return;
    }
    if (result.valid()) {
      const auto plate = currentPlate(result, args.time);
      if (plate && plate->valid()) {
        drawPlate(*out, args, *plate, result.canvas);
        return;
      }
      drawCaption(*out, args, "STILL MISSING");
      return;
    }
    if (!setupState().ready()) {  // as in Alpha Gen: says what to do before anyone presses Process
      drawCaption(*out, args, "SETUP REQUIRED");
      return;
    }
    drawSource(*out, args, args.time);  // not processed: the picture passes through
  }

  // The host must be told when a render reads the picture at another frame than its own.
  void getFramesNeeded(const OFX::FramesNeededArguments& args, OFX::FramesNeededSetter& frames) override {
    std::string state;
    result_->getValueAtTime(args.time, state);
    const Result result = parseResult(state);
    double t = args.time;
    if (result.valid() && showingOriginal(args.time)) t = double(result.frame);
    OfxRangeD range;
    range.min = range.max = t;
    frames.setFramesNeeded(*src_, range);
    range.min = range.max = args.time;
    frames.setFramesNeeded(*mask_, range);
  }

 private:
  bool showingOriginal(double time) {
    int view = 0;
    view_->getValueAtTime(time, view);
    return view == 1;
  }

  // Ends the resident worker now rather than after its idle time, so another tool such as LTX Alpha
  // Gen has the memory. The worker can be shared with other nodes and comps, so it goes only when idle.
  void releaseMemory() {
    const std::string socket = cacheRoot() + "/run/worker.sock";
    const std::string info = readFile(socket + ".json");
    const std::string pidText = jsonField(info, "pid");
    const pid_t pid = pid_t(std::atoi(pidText.c_str()));
    if (pid <= 0 || kill(pid, 0) != 0 || !containsNoCase(processName(unsigned(pid)), "python")) {
      setText(status_, "The model is not loaded: nothing to release.");
      return;
    }
    const std::string state = jsonField(info, "state");
    if (state != "ready") {
      setText(status_, "The model is in use by a job: wait for it to finish, or press Cancel in its window.");
      return;
    }
    const double held = std::atof(jsonField(info, "vram_gb").c_str());
    kill(pid, SIGTERM);
    for (int i = 0; i < 30 && kill(pid, 0) == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (kill(pid, 0) == 0) kill(pid, SIGKILL);
    unlink((socket + ".json").c_str());
    unlink(socket.c_str());
    char text[120];
    std::snprintf(text, sizeof text, "Released the model's memory (about %.0f GB). The next Process loads it again.", held);
    LOGI("%s", text);
    setText(status_, text);
  }

  // Makes the folder stills will go in and shows it, so the location can be checked before a job. In
  // its own process: it asks the host where a path map points, which the host can't answer while this
  // button press is running. Without the runtime, the folder of the last stills made.
  void openStillsFolder() {
    std::string folder, err;
    folder_->getValue(folder);
    const std::string python = sharedSetting("ltx_python");
    struct stat st {};
    if (!python.empty() && stat(python.c_str(), &st) == 0) {
      if (!launchDetached({python, "-I", "-B", resourcesDir() + "/worker/still_open_folder.py",
                           hostIsResolve() ? "resolve" : "fusion", nodeId_, folder},
                          cleanEnvironment(), err))
        showError("Could not open the stills folder: " + err);
      return;
    }
    std::string state;
    result_->getValue(state);
    const Result result = parseResult(state);
    if (!result.valid()) {
      setText(status_, "No stills yet, and the LTX runtime was not found.");
      return;
    }
    const std::string& path = result.paths.front();
    const std::string last = path.substr(0, path.rfind('/'));
    if (!launchDetached({"/usr/bin/open", last}, cleanEnvironment(), err)) showError("Could not open " + last + ": " + err);
  }

  bool onEditOrColorPage() {
    if (!hostIsResolve()) return false;
    try {
      const OfxRangeD r = dst_->getFrameRange();
      return r.max >= r.min;
    } catch (const std::exception&) {
      return false;
    }
  }

  // The plate to show, read once and kept while the choice stays the same. The output is the same
  // on every frame, so it can go straight into a tracker.
  std::shared_ptr<ltxis::Plate> currentPlate(const Result& result, double time) {
    int which = 1;
    show_->getValueAtTime(time, which);
    const size_t i = size_t(std::min(std::max(which, 1), int(result.paths.size())) - 1);
    const std::string& path = result.paths[i];
    std::lock_guard<std::mutex> lock(plateMutex_);
    if (path != platePath_) {
      auto plate = std::make_shared<ltxis::Plate>();
      std::string err;
      if (!ltxis::readPlate(path, *plate, err)) {
        LOGW("render: %s", err.c_str());
        plate.reset();
      }
      plate_ = plate;
      platePath_ = path;
    }
    return plate_;
  }

  void drawPlate(OFX::Image& out, const OFX::RenderArguments& args, const ltxis::Plate& plate, const Canvas& canvas) {
    const OfxRectI b = out.getBounds();
    const OfxRectI& win = args.renderWindow;
    const int comps = componentsOf(out);
    for (int y = std::max(win.y1, b.y1); y < std::min(win.y2, b.y2); ++y) {
      auto* px = static_cast<float*>(out.getPixelAddress(std::max(win.x1, b.x1), y));
      const int row = canvas.h - 1 - (y - canvas.y);
      for (int x = std::max(win.x1, b.x1); x < std::min(win.x2, b.x2); ++x, px += comps) {
        const int col = x - canvas.x;
        const bool inside = row >= 0 && row < plate.height && col >= 0 && col < plate.width;
        const float* p = inside ? plate.rgb.data() + (size_t(row) * plate.width + col) * 3 : nullptr;
        for (int c = 0; c < std::min(comps, 3); ++c) px[c] = p ? p[c] : 0.0f;
        if (comps == 4) px[3] = inside ? plate.alpha[size_t(row) * plate.width + col] : 0.0f;
      }
    }
  }

  void drawSource(OFX::Image& out, const OFX::RenderArguments& args, double time) {
    std::unique_ptr<OFX::Image> src(src_->isConnected() ? src_->fetchImage(time) : nullptr);
    const OfxRectI b = out.getBounds();
    const OfxRectI& win = args.renderWindow;
    const int comps = componentsOf(out);
    const int srcComps = src ? componentsOf(*src) : 0;
    const OfxRectI sb = src ? src->getBounds() : OfxRectI{0, 0, 0, 0};
    for (int y = std::max(win.y1, b.y1); y < std::min(win.y2, b.y2); ++y) {
      auto* px = static_cast<float*>(out.getPixelAddress(std::max(win.x1, b.x1), y));
      for (int x = std::max(win.x1, b.x1); x < std::min(win.x2, b.x2); ++x, px += comps) {
        const bool inside = srcComps >= 3 && x >= sb.x1 && x < sb.x2 && y >= sb.y1 && y < sb.y2;
        const auto* p = inside ? static_cast<const float*>(src->getPixelAddress(x, y)) : nullptr;
        for (int c = 0; c < std::min(comps, 3); ++c) px[c] = p ? p[c] : 0.0f;
        if (comps == 4) px[3] = p ? (srcComps == 4 ? p[3] : 1.0f) : 0.0f;
      }
    }
  }

  void drawCaption(OFX::Image& out, const OFX::RenderArguments& args, const std::string& caption) {
    const OfxRectI b = out.getBounds();
    const int w = b.x2 - b.x1, h = b.y2 - b.y1;
    std::vector<float> text;
    text.assign(size_t(w) * h, 0.0f);
    drawMarker(text, w, h, caption);
    const OfxRectI& win = args.renderWindow;
    const int comps = componentsOf(out);
    for (int y = std::max(win.y1, b.y1); y < std::min(win.y2, b.y2); ++y) {
      auto* px = static_cast<float*>(out.getPixelAddress(std::max(win.x1, b.x1), y));
      const int row = h - 1 - (y - b.y1);
      for (int x = std::max(win.x1, b.x1); x < std::min(win.x2, b.x2); ++x, px += comps) {
        const float v = text[size_t(row) * w + (x - b.x1)];
        for (int c = 0; c < std::min(comps, 3); ++c) px[c] = v;
        if (comps == 4) px[3] = 1.0f;
      }
    }
  }

  void setText(OFX::StringParam* p, const std::string& v) {
    try {
      std::string cur;
      p->getValue(cur);
      if (cur != v) p->setValue(v);
    } catch (const std::exception& e) {
      LOGW("setValue FAILED: %s", e.what());
    }
  }

  void showError(const std::string& msg) {
    LOGW("shown to the user: %s", msg.c_str());
    setText(status_, msg);
    // A window of its own: the Status field alone was missed, and the host's sendMessage showed
    // nothing. The text goes as an argument, so a path in it can't be read as a script.
    std::string err;
    if (!launchDetached({"/usr/bin/osascript", "-e", "on run argv", "-e",
                         "display dialog (item 1 of argv) with title \"LTX Inpaint Still\" buttons {\"OK\"} "
                         "default button \"OK\" with icon caution",
                         "-e", "end run", msg},
                        cleanEnvironment(), err))
      LOGW("could not show the message: %s", err.c_str());
  }

  // Memory the worker can use: what is free, plus what a resident worker of ours already holds.
  double workerHeldGB(const std::string& socket) {
    const std::string pid = jsonField(readFile(socket + ".json"), "pid");
    if (pid.empty() || kill(std::atoi(pid.c_str()), 0) != 0) return 0.0;
    return std::atof(jsonField(readFile(socket + ".json"), "vram_gb").c_str());
  }

  // The picture and mask at a frame, as Process and Copy for LLM use them. False: the user has been told why.
  bool readFrame(int frame, Canvas& canvas, std::vector<float>& picture, std::vector<float>& maskValues) {
    const auto fail = [this](const std::string& msg) {
      showError(msg);
      return false;
    };
    const double t = double(frame);
    const OfxRectD rod = src_->getRegionOfDefinition(t);
    canvas.x = int(std::floor(rod.x1)), canvas.y = int(std::floor(rod.y1));
    canvas.w = int(std::ceil(rod.x2)) - canvas.x, canvas.h = int(std::ceil(rod.y2)) - canvas.y;
    if (!canvas.valid()) return fail("The picture is empty at frame " + std::to_string(frame) + ".");

    // The host refuses a frame outside the picture by throwing, which would only say kOfxStatFailed.
    auto fetch = [&](OFX::Clip* clip) -> OFX::Image* {
      try {
        return clip->fetchImage(t);
      } catch (const std::exception& e) {
        LOGW("fetchImage frame %d: %s", frame, e.what());
        return nullptr;
      }
    };
    const std::string noFrame = "Fusion could not provide frame " + std::to_string(frame) +
                                ". Is Reference Frame inside the comp? Use Current Frame sets it to the frame on screen.";
    std::vector<float> maskRgba;
    {
      std::unique_ptr<OFX::Image> img(fetch(src_));
      if (!img) return fail(noFrame);
      if (img->getPixelDepth() != OFX::eBitDepthFloat) return fail("Only 32-bit float images are supported.");
      placeOnCanvas(*img, canvas, picture);
    }
    if (mask_->isConnected()) {
      // A larger mask would land on the wrong area without a word. A smaller one can be right: a mask's
      // area can be trimmed to what it covers.
      const OfxRectD maskRod = mask_->getRegionOfDefinition(t);
      const int maskW = int(std::ceil(maskRod.x2)) - int(std::floor(maskRod.x1));
      const int maskH = int(std::ceil(maskRod.y2)) - int(std::floor(maskRod.y1));
      if (maskW > canvas.w || maskH > canvas.h) {
        return fail("The mask is " + std::to_string(maskW) + "x" + std::to_string(maskH) + " but the picture is " +
                         std::to_string(canvas.w) + "x" + std::to_string(canvas.h) +
                         ". Make them the same size, or the fill lands on the wrong area.");
      }
      std::unique_ptr<OFX::Image> img(fetch(mask_));
      if (!img) return fail(noFrame);
      placeOnCanvas(*img, canvas, maskRgba);
      maskValues = maskFrom(maskRgba, true);
      if (maskValues.empty()) return fail("The green Mask input is all one value at this frame: white marks what to fill.");
    } else {
      maskValues = maskFrom(picture, false);
      if (maskValues.empty()) return fail("No mask: connect the green Mask input, or give the picture an alpha.");
    }

    return true;
  }

  bool stageFrame(const Canvas& canvas, const std::vector<float>& picture, const std::vector<float>& maskValues,
                  const std::string& picturePath, const std::string& maskPath, std::string& err) {
    const size_t n = size_t(canvas.w) * canvas.h;
    std::vector<float> rgb(n * 3);
    for (size_t i = 0; i < n; ++i)
      rgb[i * 3] = picture[i * 4], rgb[i * 3 + 1] = picture[i * 4 + 1], rgb[i * 3 + 2] = picture[i * 4 + 2];
    std::vector<uint16_t> half(n * 3);
    floatToHalf(rgb.data(), half.data(), n * 3);
    if (!writeStaged(picturePath, canvas.w, canvas.h, 3, half, err)) return false;
    half.resize(n);
    floatToHalf(maskValues.data(), half.data(), n);
    if (!writeStaged(maskPath, canvas.w, canvas.h, 1, half, err)) {
      unlink(picturePath.c_str());
      return false;
    }
    return true;
  }

  // The frame a job would use, without choosing it: a stored Reference Frame, or else the one on screen.
  int frameToUse(double time, bool& chosen) {
    referenceSet_->getValue(chosen);
    int frame = int(std::lround(time));
    if (chosen) {
      // A node pasted by an earlier build can say frame 0 was chosen when it wasn't. Comps usually start
      // at 1001, so a 0 the picture doesn't have means the frame on screen.
      int stored = 0;
      reference_->getValue(stored);
      const OfxRangeD range = src_->getFrameRange();
      if (stored == 0 && (range.min > 0 || range.max < 0)) chosen = false;
    }
    if (chosen) reference_->getValue(frame);
    return frame;
  }

  // The Mac's own folder picker. Fusion's browse control for a folder field only picks files. It runs on its
  // own thread so the host keeps drawing while the window is open, and Cancel changes nothing.
  void chooseFolder(OFX::StringParam* field, const char* prompt, const char* configKey) {
    if (browsing_.exchange(true)) return;
    if (browseThread_.joinable()) browseThread_.join();
    std::string start;
    field->getValue(start);
    struct stat st {};
    if (start.empty() || start[0] != '/' || stat(start.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) start = homeDir();
    const std::string node = nodeId_, title = prompt, key = configKey;
    browseThread_ = std::thread([this, field, node, title, start, key] {
      LogScope scope(node);
      auto proc = std::make_shared<WorkerProcess>();
      {
        std::lock_guard<std::mutex> lock(browseMutex_);
        browseProc_ = proc;
      }
      std::string err;
      std::vector<std::string> lines;
      if (proc->start({"/usr/bin/osascript", "-e", "on run argv", "-e", "activate", "-e",
                       "return POSIX path of (choose folder with prompt (item 1 of argv) default location "
                       "(POSIX file (item 2 of argv) as alias))",
                       "-e", "end run", title, start},
                      cleanEnvironment(), err)) {
        while (proc->readLines(200, lines)) {
        }
        proc->wait();
      } else {
        LOGW("choose folder: %s", err.c_str());
      }
      {
        std::lock_guard<std::mutex> lock(browseMutex_);
        browseProc_.reset();
      }
      std::string chosen = lines.empty() ? "" : lines.front();
      while (chosen.size() > 1 && chosen.back() == '/') chosen.pop_back();
      if (!chosen.empty() && chosen[0] == '/') {
        try {
          field->setValue(chosen);
          if (!key.empty()) {  // a change made here isn't a user edit, so it is saved here
            UserConfig cfg = loadUserConfig();
            cfg[key] = chosen;
            saveUserConfig(cfg);
          }
          refreshSetupUi();
        } catch (const std::exception& e) {
          LOGE("choose folder: %s", e.what());
        }
      }
      browsing_ = false;
    });
  }

  void stopChoosingFolder() {
    {
      std::lock_guard<std::mutex> lock(browseMutex_);
      if (browseProc_) browseProc_->kill();
    }
    if (browseThread_.joinable()) browseThread_.join();
  }

  void openPage(const char* url) {
    std::string err;
    if (!launchDetached({"/usr/bin/open", url}, cleanEnvironment(), err)) showError("Could not open " + std::string(url) + ": " + err);
  }

  // Once Setup is complete, the controls only needed for the download are hidden, and Show Setup Options
  // brings them back.
  void refreshSetupUi() {
    const SetupState st = setupState();
    bool show = false;
    showSetup_->getValue(show);
    const bool hide = st.ready() && !show;
    for (const char* p : {kPViewLicence, kPAccept, kPModelPage, kPLoraPage, kPTokenPage, kPToken, kPDownload, kPExistingDir, kPLinkExisting, kPChooseExisting})
      getParam(p)->setIsSecret(hide);
    showSetup_->setIsSecret(!st.ready());
    setText(setupStatus_, std::string("Licence: ") + (st.licenceAccepted ? "OK" : "NO") + " | Models: " + (st.modelsReady ? "OK" : "NO"));
  }

  void download() {
    std::string token, dir, existing;
    token_->getValue(token);
    modelDir_->getValue(dir);
    existingDir_->getValue(existing);
    dir = trimSlashes(dir);
    existing = trimSlashes(existing);
    if (!setupState().licenceAccepted) return showError("Tick I Accept for the LTX-2 Community License before downloading.");
    if (!appleToolsInstalled())
      return showError("Apple's Command Line Tools are missing. In Terminal, run xcode-select --install, then press Download again.");
    std::vector<std::string> argv = {"/usr/bin/python3", "-I", "-B", resourcesDir() + "/helper/setup_helper.py", "--model-dir", dir,
                                     "--cache-root", cacheRoot()};
    if (!existing.empty()) {
      argv.insert(argv.end(), {"--existing", existing});
      bool link = false;
      linkExisting_->getValue(link);
      if (link) argv.push_back("--link");
    }
    std::vector<std::string> extras = {"LTXAG_LOG_LEVEL=info"};
    const bool hasToken = !token.empty();
    if (hasToken) extras.push_back("HF_TOKEN=" + token);  // in its environment, never on a command line
    std::string err;
    const bool ok = launchDetached(argv, cleanEnvironment(extras), err);
    token.assign(token.size(), '\0');
    extras.back().assign(extras.back().size(), '\0');
    token_->setValue("");  // used once, then gone
    if (!ok) return showError("Download failed to start: " + err);
    LOGI("download: helper started, %s token; token field cleared", hasToken ? "with a" : "without a");
    setText(status_, hasToken ? "Setup is running in its own window. The token was used and not stored." : "Setup is running in its own window.");
    startSetupWatcher();
  }

  void startSetupWatcher() {
    stopSetupWatcher();
    setupStop_ = false;
    const std::string node = nodeId_;
    setupWatcher_ = std::thread([this, node] {
      LogScope scope(node);
      const auto t0 = std::chrono::steady_clock::now();
      while (!setupStop_ && std::chrono::steady_clock::now() - t0 < std::chrono::hours(4)) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        if (!setupState().ready()) continue;
        try {
          refreshSetupUi();
          setText(status_, "Setup is complete.");
        } catch (const std::exception& e) {
          LOGE("setup watcher: %s", e.what());
        }
        return;
      }
    });
  }

  void stopSetupWatcher() {
    setupStop_ = true;
    if (setupWatcher_.joinable()) setupWatcher_.join();
  }

  // A support bundle: logs and job status, with no pictures and no token, in the home folder.
  void collectLogs() {
    const std::string bundle = homeDir() + "/ltx-inpaint-still-support-" + timeStr("%Y%m%d-%H%M%S") + ".tar.gz";
    std::string err;
    std::vector<std::string> argv = {"/usr/bin/tar", "-czf", bundle, "-C", cacheRoot(), "logs"};
    struct stat st {};
    if (stat((cacheRoot() + "/jobs").c_str(), &st) == 0) argv.push_back("jobs");
    if (launchDetached(argv, cleanEnvironment(), err))
      setText(status_, "Support bundle: " + bundle + ". Review it before sending: it has your logs and folder paths, but no pictures and no token.");
    else
      showError("Could not write the support bundle: " + err);
  }

  std::string colourId() {
    int index = 0;
    colour_->getValue(index);
    return kColourIds[std::min(std::max(index, 0), 2)];
  }

  // Where stills in progress, encoded prompts and Copy for LLM pictures go right now.
  std::string workFolder(std::string& note) {
    std::string chosen;
    cacheDir_->getValue(chosen);
    return chooseCacheFolder(chosen, cacheRoot(), note);
  }

  void copyForLlm(double time) {
    if (onEditOrColorPage()) return showError("LTX Inpaint Still works on the Fusion page only.");
    if (!src_->isConnected()) return showError("Connect a picture to the yellow Image input.");
    const std::string python = sharedSetting("ltx_python");
    struct stat st {};
    if (python.empty() || stat(python.c_str(), &st) != 0)
      return showError("The LTX runtime was not found: press Download in the Setup section.");
    bool chosen = false;
    const int frame = frameToUse(time, chosen);
    Canvas canvas;
    std::vector<float> picture, maskValues;
    if (!readFrame(frame, canvas, picture, maskValues)) return;
    const std::string id = std::to_string(getpid()) + "-" + std::to_string(gJobSeq++);
    const std::string picturePath = stagingDir() + "/ltxis_llm_" + id + "_picture.bin";
    const std::string maskPath = stagingDir() + "/ltxis_llm_" + id + "_mask.bin";
    std::string err, note;
    const std::string work = workFolder(note);
    if (!stageFrame(canvas, picture, maskValues, picturePath, maskPath, err)) return showError(err);
    if (!launchDetached({python, "-I", "-B", resourcesDir() + "/worker/still_llm.py", picturePath, maskPath,
                         std::to_string(frame), work, colourId()},
                        cleanEnvironment({"LTXAG_LOG_LEVEL=info"}), err)) {
      unlink(picturePath.c_str());
      unlink(maskPath.c_str());
      return showError("Could not start Copy for LLM: " + err);
    }
    setText(status_, "Copied for LLM from frame " + std::to_string(frame) + ": see the message." +
                         (note.empty() ? "" : " " + note));
  }

  void process(double time) {
    {
      std::lock_guard<std::mutex> lock(jobMutex_);
      if (jobRunning_) {
        setText(status_, "Already processing: use Cancel in the progress window first.");
        return;
      }
    }
    if (onEditOrColorPage()) return showError("LTX Inpaint Still works on the Fusion page only.");
    if (!src_->isConnected()) return showError("Connect a picture to the yellow Image input.");
    const SetupState setup = setupState();
    if (!setup.ready())
      return showError(setup.licenceAccepted ? "The models aren't installed yet: paste your Hugging Face token in the Setup section and press Download."
                                             : "Open the Setup section and tick I Accept for the licence, then paste your Hugging Face token and press Download.");
    std::string modelDir;
    modelDir_->getValue(modelDir);
    while (modelDir.size() > 1 && modelDir.back() == '/') modelDir.pop_back();
    const std::string python = sharedSetting("ltx_python");
    struct stat st {};
    if (python.empty() || stat(python.c_str(), &st) != 0)
      return showError("The LTX runtime was not found: press Download in the Setup section.");
    const std::string lora = modelDir + "/loras/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors";
    if (stat(lora.c_str(), &st) != 0) return showError("The inpainting LoRA is missing: " + lora);
    const std::string socket = cacheRoot() + "/run/worker.sock";
    const VramInfo mem = queryVram();
    const double available = mem.freeGB + workerHeldGB(socket);
    const bool linearColour = colourId() != kColourIds[0];
    const double memoryNeed = linearColour ? kMemoryNeedLinearGB : kMemoryNeedGB;
    if (mem.ok && available < memoryNeed) {
      char b[160];
      std::snprintf(b, sizeof b, "Not enough free memory: %.0f GB, about %.0f GB needed. Is LTX Alpha Gen still running?",
                    available, memoryNeed);
      return showError(b);
    }

    bool chosen = false;
    int frame = frameToUse(time, chosen);
    if (!chosen) {  // first Process: the frame on screen
      reference_->setValue(frame);
      referenceSet_->setValue(true);
    }

    Canvas canvas;
    std::vector<float> picture, maskValues;
    if (!readFrame(frame, canvas, picture, maskValues)) return;

    const std::string jobId = std::to_string(getpid()) + "-" + std::to_string(gJobSeq++) + "-" +
                              std::to_string(std::time(nullptr));
    const std::string staging = stagingDir();
    const std::string picturePath = staging + "/ltxis_job_" + jobId + "_picture.bin";
    const std::string maskPath = staging + "/ltxis_job_" + jobId + "_mask.bin";
    const std::string manifest = staging + "/ltxis_job_" + jobId + ".json";
    std::string err;
    if (!stageFrame(canvas, picture, maskValues, picturePath, maskPath, err)) return showError(err);

    int methodIndex = 0, count = 1, seed = 1, scaleIndex = 0, grow = 16, blend = 1, dwabLevel = 45;
    dwab_->getValue(dwabLevel);
    method_->getValue(methodIndex);
    variations_->getValue(count);
    seed_->getValue(seed);
    workScale_->getValue(scaleIndex);
    grow_->getValue(grow);
    blend_->getValue(blend);
    std::string folder, name, promptText;
    bool personCheck = true;
    prompt_->getValue(promptText);
    personCheck_->getValue(personCheck);
    folder_->getValue(folder);
    name_->getValue(name);
    const std::string method = kMethodIds[std::min(std::max(methodIndex, 0), 1)];
    std::string cacheNote;
    const std::string work = workFolder(cacheNote);
    const std::string plateDir = work + "/plates/" + nodeId_;
    // The same empty-prompt embedding as Alpha Gen, if it is there; the worker builds it if not.
    std::string embeddings = homeDir() + "/.cache/ltx-alphagen/mattes/prompt-empty.safetensors";
    if (stat(embeddings.c_str(), &st) != 0) embeddings = cacheRoot() + "/prompt-empty.safetensors";

    std::ostringstream js;
    js << "{\"job_id\":" << jsonStr(jobId) << ",\"node_id\":" << jsonStr(nodeId_) << ",\"picture\":"
       << jsonStr(picturePath) << ",\"mask\":" << jsonStr(maskPath) << ",\"method\":" << jsonStr(method)
       << ",\"seeds\":[";
    std::string seeds;
    for (int i = 0; i < count; ++i) {
      js << (i ? "," : "") << seed + i;
      seeds += (i ? "," : "") + std::to_string(seed + i);
    }
    js << "],\"work_scale\":" << kScales[std::min(std::max(scaleIndex, 0), 2)] << ",\"grow\":" << grow
       << ",\"blend\":" << blend << ",\"frame\":" << frame << ",\"name\":" << jsonStr(name)  // empty: the comp's name
       << ",\"out_dir\":" << jsonStr(plateDir) << ",\"plate_folder\":" << jsonStr(folder)
       << ",\"prompt\":" << jsonStr(promptText) << ",\"person_check\":" << (personCheck ? "true" : "false")
       << ",\"input_colour\":" << jsonStr(colourId()) << ",\"work_root\":" << jsonStr(work) << ",\"cache_note\":" << jsonStr(cacheNote)
       << ",\"dwab_level\":" << dwabLevel << ",\"cache_root\":"
       << jsonStr(cacheRoot()) << ",\"worker\":" << jsonStr(resourcesDir() + "/worker/still_worker.py")
       << ",\"worker_python\":" << jsonStr(python) << ",\"worker_socket\":" << jsonStr(socket)
       << ",\"model_dir\":" << jsonStr(modelDir) << ",\"embeddings\":" << jsonStr(embeddings)
       << ",\"precision\":\"" << (sharedSetting("transformer_fp16") == "1" ? "fp16" : "bf16") << "\""
       << ",\"host_pid\":" << getpid() << ",\"host\":\"" << (hostIsResolve() ? "resolve" : "fusion") << "\""
       << ",\"title\":" << jsonStr(std::string(kPluginLabel) + " - frame " + std::to_string(frame)) << "}";
    {
      std::ofstream mf(manifest);
      mf << js.str();
    }
    chmod(manifest.c_str(), 0600);
    if (!launchDetached({python, "-I", "-B", resourcesDir() + "/worker/still_job.py", manifest},
                        cleanEnvironment({"LTXAG_LOG_LEVEL=info"}), err)) {
      unlink(picturePath.c_str());
      unlink(maskPath.c_str());
      unlink(manifest.c_str());
      return showError("Could not start processing: " + err);
    }
    const std::string statusPath = cacheRoot() + "/jobs/" + jobId + ".json";
    LOGI("process: job %s frame %d %s seeds %s canvas %s", jobId.c_str(), frame, method.c_str(), seeds.c_str(),
         formatCanvas(canvas).c_str());
    setText(status_, "Processing frame " + std::to_string(frame));
    const std::string head = std::to_string(frame) + "|" + method + "|" + seeds + "|" + formatCanvas(canvas) + "|";
    {
      std::lock_guard<std::mutex> lock(jobMutex_);
      jobRunning_ = true;
    }
    stopFlag_ = true;
    if (watcher_.joinable()) watcher_.join();
    stopFlag_ = false;
    const int nextSeed = seed + count;
    watcher_ = std::thread([this, statusPath, head, nextSeed] { watch(statusPath, head, nextSeed); });
  }

  // Follows the job's status file, and when the job ends stores the result and redraws the viewer.
  void watch(const std::string& statusPath, const std::string& head, int nextSeed) {
    LogScope scope(nodeId_);
    const auto started = std::chrono::steady_clock::now();
    while (!stopFlag_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      const std::string status = readFile(statusPath);
      const std::string state = jsonField(status, "state");
      if (state == "running") {
        const std::string text = jsonField(status, "message");
        if (!text.empty()) setText(status_, text);
        continue;
      }
      if (state.empty()) {  // no status yet; a job that never wrote one has died
        if (std::chrono::steady_clock::now() - started > std::chrono::minutes(2)) {
          setText(status_, "The job did not start: see the log.");
          break;
        }
        continue;
      }
      if (state == "done") {
        const std::string paths = jsonField(status, "paths");
        result_->setValue(head + paths);
        show_->setValue(1);
        view_->setValue(0);  // show what was just made
        // Past every seed used, including any the person check threw out, so running again never repeats them.
        const std::string used = jsonField(status, "next_seed");
        seed_->setValue(std::min(used.empty() ? nextSeed : std::atoi(used.c_str()), kMaxSeed));
        {
          std::lock_guard<std::mutex> lock(plateMutex_);
          platePath_.clear();
        }
        setText(status_, jsonField(status, "message"));
      } else {
        setText(status_, state == "cancelled" ? "Cancelled." : "Failed: " + jsonField(status, "message"));
      }
      try {
        int tick = 0;
        tick_->getValue(tick);
        tick_->setValue(tick + 1);  // a change here redraws the viewer
      } catch (const std::exception& e) {
        LOGW("redraw FAILED: %s", e.what());
      }
      break;
    }
    std::lock_guard<std::mutex> lock(jobMutex_);
    jobRunning_ = false;
  }

  OFX::Clip* src_ = nullptr;
  OFX::Clip* mask_ = nullptr;
  OFX::Clip* dst_ = nullptr;
  OFX::StringParam* status_ = nullptr;
  OFX::IntParam* reference_ = nullptr;
  OFX::ChoiceParam* method_ = nullptr;
  OFX::IntParam* variations_ = nullptr;
  OFX::IntParam* seed_ = nullptr;
  OFX::StringParam* prompt_ = nullptr;
  OFX::ChoiceParam* colour_ = nullptr;
  OFX::StringParam* cacheDir_ = nullptr;
  OFX::BooleanParam* showSetup_ = nullptr;
  OFX::BooleanParam* accept_ = nullptr;
  OFX::StringParam* token_ = nullptr;
  OFX::StringParam* existingDir_ = nullptr;
  OFX::BooleanParam* linkExisting_ = nullptr;
  OFX::StringParam* setupStatus_ = nullptr;
  std::thread setupWatcher_;
  std::thread browseThread_;
  std::atomic<bool> browsing_{false};
  std::mutex browseMutex_;
  std::shared_ptr<WorkerProcess> browseProc_;
  std::atomic<bool> setupStop_{false};
  OFX::BooleanParam* personCheck_ = nullptr;
  OFX::ChoiceParam* workScale_ = nullptr;
  OFX::IntParam* grow_ = nullptr;
  OFX::IntParam* blend_ = nullptr;
  OFX::IntParam* show_ = nullptr;
  OFX::ChoiceParam* view_ = nullptr;
  OFX::StringParam* folder_ = nullptr;
  OFX::IntParam* dwab_ = nullptr;
  OFX::StringParam* name_ = nullptr;
  OFX::StringParam* modelDir_ = nullptr;
  OFX::StringParam* nodeIdParam_ = nullptr;
  OFX::BooleanParam* referenceSet_ = nullptr;
  OFX::StringParam* result_ = nullptr;
  OFX::IntParam* tick_ = nullptr;
  std::string nodeId_;
  bool inChangedParam_ = false;
  std::mutex jobMutex_;
  bool jobRunning_ = false;
  std::atomic<bool> stopFlag_{false};
  std::thread watcher_;
  std::mutex plateMutex_;
  std::string platePath_;
  std::shared_ptr<ltxis::Plate> plate_;
};

class InpaintStillFactory : public OFX::PluginFactoryHelper<InpaintStillFactory> {
 public:
  InpaintStillFactory()
      : PluginFactoryHelper<InpaintStillFactory>(kPluginId, LTXIS_VERSION_MAJOR, LTXIS_VERSION_MINOR) {}

  void describe(OFX::ImageEffectDescriptor& desc) override {
    logConfigure();
    rotateLogs();
    LOGI("==== plugin load, version %s ====", LTXIS_VERSION);
    desc.setLabels(kPluginLabel, kPluginLabel, kPluginLabel);
    desc.setPluginGrouping(kPluginGroup);
    desc.addSupportedContext(OFX::eContextFilter);
    desc.addSupportedContext(OFX::eContextGeneral);
    desc.addSupportedBitDepth(OFX::eBitDepthFloat);
    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setSupportsMultiResolution(false);
    desc.setSupportsTiles(false);
    desc.setTemporalClipAccess(true);
    desc.setRenderTwiceAlways(false);
    desc.setSupportsMultipleClipPARs(false);
    desc.setRenderThreadSafety(OFX::eRenderInstanceSafe);
  }

  void describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum ctx) override {
    LOGI("describeInContext %d", int(ctx));
    OFX::ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->setLabels("Image", "Image", "Image");
    src->addSupportedComponent(OFX::ePixelComponentRGBA);
    src->addSupportedComponent(OFX::ePixelComponentRGB);
    src->setTemporalClipAccess(true);
    src->setSupportsTiles(false);
    OFX::ClipDescriptor* mask = desc.defineClip(kMaskClip);
    mask->setLabels("Mask", "Mask (white = fill)", "Mask (white = fill)");
    mask->addSupportedComponent(OFX::ePixelComponentRGBA);
    mask->addSupportedComponent(OFX::ePixelComponentRGB);
    mask->addSupportedComponent(OFX::ePixelComponentAlpha);
    mask->setOptional(true);
    mask->setTemporalClipAccess(true);
    mask->setSupportsTiles(false);
    OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->setSupportsTiles(false);

    OFX::PageParamDescriptor* page = desc.definePageParam("Inpaint Still");
    const auto group = [&](const char* id, const char* text, bool open) {
      auto* g = desc.defineGroupParam(id);
      g->setLabels(text, text, text);
      g->setOpen(open);
      page->addChild(*g);
      return g;
    };
    const auto labelled = [](auto* p, const char* text, const char* hint, OFX::GroupParamDescriptor* g) {
      p->setLabels(text, text, text);
      p->setHint(hint);
      p->setParent(*g);
      return p;
    };
    const auto integer = [&](const char* id, const char* text, const char* hint, OFX::GroupParamDescriptor* g, int def,
                             int low, int high) {
      auto* p = labelled(desc.defineIntParam(id), text, hint, g);
      p->setDefault(def);
      p->setRange(low, high);
      p->setDisplayRange(low, high);
      p->setAnimates(false);
      return p;
    };
    const auto readOnly = [&](const char* id, const char* text, const char* def, OFX::GroupParamDescriptor* g) {
      auto* p = labelled(desc.defineStringParam(id), text, "", g);
      p->setStringType(OFX::eStringTypeSingleLine);
      p->setEnabled(false);
      p->setDefault(def);
      p->setIsPersistant(false);
      p->setEvaluateOnChange(false);
      p->setAnimates(false);
      return p;
    };
    const auto hidden = [](auto* p) {
      p->setIsSecret(true);
      p->setAnimates(false);
      return p;
    };

    auto* main = group("grpMain", "Inpaint Still", true);
    labelled(desc.definePushButtonParam(kPProcess), "Process",
             "Make the stills from the Reference Frame, in the background. Progress and Cancel appear in a separate "
             "window.", main);
    readOnly(kPStatus, "Status", "Not processed", main);
    auto* reference = integer(kPReference, "Reference Frame",
                              "The frame the still is made from, and your match-move reference. Before the first "
                              "Process, the frame on screen is used.", main, 0, -100000, 10000000);
    reference->setDisplayRange(0, 2000);
    reference->setEvaluateOnChange(false);
    labelled(desc.definePushButtonParam(kPUseCurrent), "Use Current Frame", "Set Reference Frame to the frame on screen.",
             main);
    auto* colourParam = labelled(desc.defineChoiceParam(kPColour), "Input Colour",
                                 "What the picture coming in is. ACEScg or Linear sRGB: the model works in ACEScct, and "
                                 "the stills come back in the same space, with values above 1.0 kept. As Is: the "
                                 "picture goes to the model as it is, so use it for Rec.709, and convert log footage "
                                 "with a CST first.", main);
    for (const char* name : kColourNames) colourParam->appendOption(name);
    colourParam->setDefault(0);
    colourParam->setAnimates(false);
    colourParam->setEvaluateOnChange(false);
    auto* promptParam = labelled(desc.defineStringParam(kPPrompt), "Prompt",
                                 "Optional. Describe the whole scene as it should look, not the change, and only what "
                                 "should be there: a wooden staircase in a bright home, not remove the man. On a big "
                                 "hole this helps keep invented people and objects out. A new prompt takes a little "
                                 "longer the first time.", main);
    promptParam->setStringType(OFX::eStringTypeMultiLine);
    promptParam->setDefault("");
    promptParam->setAnimates(false);
    promptParam->setEvaluateOnChange(false);
    labelled(desc.definePushButtonParam(kPCopyLlm), "Copy for LLM",
             "To get a suggested prompt: copies instructions to the clipboard and shows this frame in Finder, with the "
             "area to fill in grey. Paste both into any LLM, then put its sentence in Prompt.", main);
    auto* method = labelled(desc.defineChoiceParam(kPMethod), "Method",
                            "Detail (one pass): the most detail. Structure (two pass): softer, but safer on very big "
                            "holes, where Detail can invent objects.", main);
    for (const char* name : kMethodNames) method->appendOption(name);
    method->setDefault(0);
    method->setAnimates(false);
    method->setEvaluateOnChange(false);
    integer(kPVariations, "Variations", "How many stills to make, each from its own seed.", main, 3, 1, kMaxVariations)
        ->setEvaluateOnChange(false);
    integer(kPSeed, "Seed", "The first variation's seed; the others count up from it. It moves on by itself after each "
            "job, so running again makes new stills. Type a seed to go back to one.", main, 1, 0, kMaxSeed)
        ->setEvaluateOnChange(false);
    auto* personCheck = labelled(desc.defineBooleanParam(kPPersonCheck), "Discard Stills with People",
                                 "Checks each finished still, and if a person appears in the filled area, throws that "
                                 "still away and tries the next seed. It can make the job take longer. It can miss a "
                                 "partial head, so still look at your stills. To help keep people from being generated "
                                 "at all, describe the scene in Prompt.",
                                 main);
    personCheck->setDefault(true);
    personCheck->setAnimates(false);
    personCheck->setEvaluateOnChange(false);
    auto* scale = labelled(desc.defineChoiceParam(kPWorkScale), "Work Scale",
                           "Enlarge the picture for the model, then scale the fill back. Small crops need this: a "
                           "130-pixel crop only worked at 4x.", main);
    for (const char* name : kScaleNames) scale->appendOption(name);
    scale->setDefault(0);
    scale->setAnimates(false);
    scale->setEvaluateOnChange(false);
    integer(kPGrow, "Mask Grow", "Pixels to grow the mask before filling. Lower for small objects.", main, 16, 0, 200)
        ->setEvaluateOnChange(false);
    integer(kPBlend, "Blend", "How far the fill's brightness blends into its surroundings.", main, 1, 0, 15)
        ->setEvaluateOnChange(false);
    // What to make is above; what happens with the results is in its own section.
    auto* results = group("grpResults", "Results", true);
    integer(kPShow, "Show Variation", "Which still the node outputs.", results, 1, 1, kMaxVariations);
    auto* view = labelled(desc.defineChoiceParam(kPView), "View",
                          "LTX AI Still: the chosen variation. Original: the picture at the frame the stills were made "
                          "from, held on every frame, to compare with the still.", results);
    for (const char* option : kViewNames) view->appendOption(option);
    view->setDefault(0);
    view->setAnimates(false);
    // Resolve's Fusion page has no comp file; ProjectMedia: is the project's media location.
    auto* folder = labelled(desc.defineStringParam(kPFolder), "Stills Folder",
                            "Where stills are saved. Comp: is the comp's folder in Fusion Studio. ProjectMedia: is the "
                            "project's media location in Resolve. A full path works too.", results);
    folder->setStringType(OFX::eStringTypeSingleLine);
    folder->setDefault(hostIsResolve() ? "ProjectMedia:/VFX/ltx_inpaint_stills/" : "Comp:/ltx_inpaint_stills/");
    folder->setAnimates(false);
    folder->setEvaluateOnChange(false);
    labelled(desc.definePushButtonParam(kPChooseStills), "Choose Stills Folder",
             "Pick a folder in Finder. This sets a full path, in place of Comp: or ProjectMedia:.", results);
    labelled(desc.definePushButtonParam(kPOpenFolder), "Open Stills Folder",
             "Makes the folder if needed and shows it in Finder, so you can check where stills will go before "
             "running a job, or bring them in by hand.", results);
    integer(kPDwab, "DWAB Level", "Compression of the saved EXRs. Higher is smaller, lower is closer to the original.",
            results, 45, 1, 200)
        ->setEvaluateOnChange(false);
    auto* name = labelled(desc.defineStringParam(kPName), "Prefix",
                          "Optional. Goes before the clip's name on the timeline (the comp's name in Fusion Studio): sign "
                          "gives sign_sh010_LTXStill_f01032_A.exr. Useful when removing several objects on one frame.",
                          results);
    name->setStringType(OFX::eStringTypeSingleLine);  // without this, Fusion draws a tall box
    name->setDefault("");
    name->setAnimates(false);
    name->setEvaluateOnChange(false);
    labelled(desc.definePushButtonParam(kPRelease), "Release Memory",
             "Unload the model now, for example before running LTX Alpha Gen. It otherwise unloads by itself about 5 "
             "minutes after a job. The next Process takes about 15 seconds longer.", results);

    // Open until Setup is complete, folded after that.
    auto* setup = group("grpSetup", "Setup", !setupState().ready());
    const auto perUser = [](auto* p) {
      p->setIsPersistant(false);  // never saved with a comp
      p->setAnimates(false);
      p->setEvaluateOnChange(false);
      return p;
    };
    readOnly(kPSetupStatus, "Setup", "", setup);
    auto* showSetup = perUser(labelled(desc.defineBooleanParam(kPShowSetup), "Show Setup Options",
                                       "Show the licence, token and download controls again: to download the models "
                                       "again, use model files from another folder, or read the licence.", setup));
    showSetup->setDefault(false);
    labelled(desc.definePushButtonParam(kPViewLicence), "View Licence", "Show the LTX-2.x Community License terms.", setup);
    perUser(labelled(desc.defineBooleanParam(kPAccept), "I Accept",
                     "I accept the LTX-2.x Community License. Free below USD 10M annual revenue; above that a paid "
                     "Lightricks licence is required.", setup))->setDefault(false);
    labelled(desc.definePushButtonParam(kPModelPage), "LTX-2.5 Model Page",
             "Open the LTX-2.5 page on Hugging Face in your browser. Log in and accept the terms there before downloading.", setup);
    labelled(desc.definePushButtonParam(kPLoraPage), "In/Outpainting LoRA Page",
             "Open the LTX in/outpainting LoRA page on Hugging Face in your browser. Log in and accept the terms there "
             "before downloading.", setup);
    labelled(desc.definePushButtonParam(kPTokenPage), "Token Page",
             "Open your access tokens page on Hugging Face in your browser. Create a Read token there and paste it below.", setup);
    auto* token = perUser(labelled(desc.defineStringParam(kPToken), "Hugging Face Token",
                                   "Your Hugging Face access token (hf_...). Used for this download only, never stored.", setup));
    token->setStringType(OFX::eStringTypeSingleLine);
    token->setDefault("");
    labelled(desc.definePushButtonParam(kPDownload), "Download",
             "Install the LTX runtime and the models into the Model Folder: keeps files already there, reuses verified files "
             "from Existing Models, and downloads the rest with your token.", setup);
    auto* existing = perUser(labelled(desc.defineStringParam(kPExistingDir), "Existing Models",
                                      "Optional: a folder that may already contain the model files, such as a ComfyUI "
                                      "models folder. Matching files are verified and reused instead of downloaded.", setup));
    existing->setStringType(OFX::eStringTypeSingleLine);
    existing->setDefault("");
    labelled(desc.definePushButtonParam(kPChooseExisting), "Choose Existing Models", "Pick the folder in Finder.", setup);
    perUser(labelled(desc.defineBooleanParam(kPLinkExisting), "Link Existing Models",
                     "Link to the files in Existing Models instead of copying them: saves up to 73 GB, but that folder must "
                     "stay where it is and stay available. Off: they are copied into the Model Folder.", setup))->setDefault(false);
    readOnly(kPInputs, "Inputs", "Yellow: image. Green: mask.", setup);
    auto* model = labelled(desc.defineStringParam(kPModelDir), "Model Folder",
                           "Where the LTX-2.5 models and runtime are kept: about 80 GB, or about 125 GB on M1 and M2 Macs, "
                           "where Setup adds a faster copy of the main model.", setup);
    model->setStringType(OFX::eStringTypeSingleLine);
    model->setDefault(homeDir() + "/LTX-2.5");
    model->setIsPersistant(false);
    model->setAnimates(false);
    model->setEvaluateOnChange(false);
    labelled(desc.definePushButtonParam(kPChooseModel), "Choose Model Folder", "Pick the folder in Finder.", setup);
    auto* cache = labelled(desc.defineStringParam(kPCacheDir), "Cache Folder",
                           "Where stills in progress, encoded prompts and Copy for LLM pictures are kept. Saved stills "
                           "still go to Stills Folder. If this folder isn't available, the home folder's cache is used "
                           "and Status says so.", setup);
    cache->setStringType(OFX::eStringTypeSingleLine);
    cache->setDefault(cacheRoot());
    cache->setIsPersistant(false);
    cache->setAnimates(false);
    cache->setEvaluateOnChange(false);
    labelled(desc.definePushButtonParam(kPChooseCache), "Choose Cache Folder", "Pick the folder in Finder.", setup);
    readOnly(kPVersion, "Version", LTXIS_VERSION, setup);
    readOnly(kPCredits, "Credits", "Created by Chadwick, Creative Video Tips. Open source, MPL 2.0.",
             setup);
    labelled(desc.definePushButtonParam(kPWebsite), "Creative Video Tips", "Open creativevideotips.com.", setup);

    hidden(desc.defineStringParam(kPNodeId))->setDefault("");
    hidden(desc.defineBooleanParam(kPReferenceSet))->setDefault(false);
    auto* result = hidden(desc.defineStringParam(kPResult));
    result->setDefault("");
    hidden(desc.defineIntParam(kPTick))->setDefault(0);
  }

  OFX::ImageEffect* createInstance(OfxImageEffectHandle handle, OFX::ContextEnum) override {
    return new InpaintStillEffect(handle);
  }
};

}  // namespace

namespace OFX {
namespace Plugin {
void getPluginIDs(OFX::PluginFactoryArray& ids) {
  static InpaintStillFactory factory;
  ids.push_back(&factory);
}
}  // namespace Plugin
}  // namespace OFX
