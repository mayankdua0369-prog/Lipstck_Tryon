"use client";

import { useEffect, useRef, useState, type ChangeEvent } from "react";

import {
  fetchShades,
  submitTryOn,
  type Recommendation,
  type ShadeCatalog,
  type ToneProfile,
} from "../lib/api";

const DEFAULT_FAMILY = "Pinks";
const DEFAULT_SUBCATEGORY = "Soft Pinks";
const DEFAULT_SHADE = "Blush Pink";
const SNAPSHOT_MAX_WIDTH = 960;
const REALTIME_MAX_WIDTH = 256;
const REALTIME_DELAY_MS = 120;

function dataUrlFromBase64(base64: string) {
  return `data:image/png;base64,${base64}`;
}

export default function HomePage() {
  const [shades, setShades] = useState<ShadeCatalog>({});
  const [selectedFamily, setSelectedFamily] = useState(DEFAULT_FAMILY);
  const [selectedSubcategory, setSelectedSubcategory] =
    useState(DEFAULT_SUBCATEGORY);
  const [selectedShade, setSelectedShade] = useState(DEFAULT_SHADE);
  const [useCustomColor, setUseCustomColor] = useState(false);
  const [customHex, setCustomHex] = useState("#C96A78");
  const [opacity, setOpacity] = useState(0.72);
  const [finish, setFinish] = useState("Matte");
  const [loading, setLoading] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [realtimeEnabled, setRealtimeEnabled] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [tunedHex, setTunedHex] = useState<string | null>(null);
  const [toneProfile, setToneProfile] = useState<ToneProfile | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [activeMode, setActiveMode] = useState<"live" | "upload">("live");
  const [shadeQuery, setShadeQuery] = useState("");

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processingRef = useRef(false);
  const realtimeTimeoutRef = useRef<number | null>(null);

  useEffect(() => {
    async function loadShades() {
      try {
        const catalog = await fetchShades();
        setShades(catalog);
      } catch {
        setError("Unable to load shades from the backend.");
      }
    }

    loadShades();
  }, []);

  useEffect(() => {
    if (!shades[selectedFamily]) {
      return;
    }
    const subcategories = Object.keys(shades[selectedFamily]);
    if (!subcategories.includes(selectedSubcategory)) {
      const nextSubcategory =
        selectedFamily === DEFAULT_FAMILY && subcategories.includes(DEFAULT_SUBCATEGORY)
          ? DEFAULT_SUBCATEGORY
          : subcategories[0];
      setSelectedSubcategory(nextSubcategory);
      return;
    }

    const options = shades[selectedFamily][selectedSubcategory] ?? [];
    const names = options.map((item) => item.name);
    if (!names.includes(selectedShade)) {
      setSelectedShade(
        selectedFamily === DEFAULT_FAMILY &&
          selectedSubcategory === DEFAULT_SUBCATEGORY &&
          names.includes(DEFAULT_SHADE)
          ? DEFAULT_SHADE
          : names[0],
      );
    }
  }, [selectedFamily, selectedSubcategory, selectedShade, shades]);

  useEffect(() => {
    return () => {
      clearRealtimeLoop();
      stopCamera();
    };
  }, []);

  useEffect(() => {
    if (!realtimeEnabled || !cameraActive) {
      clearRealtimeLoop();
      return;
    }

    queueRealtimeFrame();
    return () => {
      clearRealtimeLoop();
    };
  }, [realtimeEnabled, cameraActive, selectedShade, useCustomColor, customHex, opacity, finish]);

  const families = Object.keys(shades);
  const subcategories = Object.keys(shades[selectedFamily] ?? {});
  const shadeOptions = shades[selectedFamily]?.[selectedSubcategory] ?? [];
  const filteredShadeOptions = shadeOptions.filter((shade) =>
    shade.name.toLowerCase().includes(shadeQuery.toLowerCase()),
  );
  const pinkShadeCount = Object.values(shades.Pinks ?? {}).reduce(
    (total, items) => total + items.length,
    0,
  );

  async function processFile(file: File, originalPreview?: string, isRealtime = false) {
    if (!isRealtime) {
      setLoading(true);
    }
    setError(null);
    if (!isRealtime) {
      setMessage(null);
    }

    try {
      const response = await submitTryOn({
        file,
        shadeName: useCustomColor ? undefined : selectedShade,
        customHex: useCustomColor ? customHex : undefined,
        opacity,
        finish,
      });

      if (!response.detected || !response.image_base64) {
        if (!isRealtime) {
          setMessage(response.message ?? "No face detected.");
          setResultImage(null);
          setRecommendations([]);
          setToneProfile(null);
          setTunedHex(null);
        }
        return;
      }

      if (originalPreview && !isRealtime) {
        setOriginalImage(originalPreview);
      }
      setResultImage(dataUrlFromBase64(response.image_base64));
      setRecommendations(response.recommendations ?? []);
      setToneProfile(response.tone_profile ?? null);
      setTunedHex(response.tuned_hex ?? null);
    } catch {
      if (!isRealtime) {
        setError("The try-on request failed. Check that the FastAPI backend is running.");
      }
    } finally {
      if (!isRealtime) {
        setLoading(false);
      }
      processingRef.current = false;
    }
  }

  async function onUploadChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    const previewUrl = URL.createObjectURL(file);
    setOriginalImage(previewUrl);
    setActiveMode("upload");
    await processFile(file, previewUrl, false);
  }

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 960 },
        },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setActiveMode("live");
      setCameraActive(true);
      setError(null);
    } catch {
      setError("Camera access was denied or is unavailable on this device.");
    }
  }

  function stopCamera() {
    clearRealtimeLoop();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraActive(false);
    setRealtimeEnabled(false);
  }

  function clearRealtimeLoop() {
    if (realtimeTimeoutRef.current !== null) {
      window.clearTimeout(realtimeTimeoutRef.current);
      realtimeTimeoutRef.current = null;
    }
  }

  function queueRealtimeFrame() {
    clearRealtimeLoop();
    realtimeTimeoutRef.current = window.setTimeout(() => {
      void processCurrentCameraFrame(true);
    }, REALTIME_DELAY_MS);
  }

  async function processCurrentCameraFrame(isRealtime: boolean) {
    if (!videoRef.current || !canvasRef.current || processingRef.current) {
      if (isRealtime && realtimeEnabled) {
        queueRealtimeFrame();
      }
      return;
    }
    if (videoRef.current.videoWidth === 0 || videoRef.current.videoHeight === 0) {
      if (isRealtime && realtimeEnabled) {
        queueRealtimeFrame();
      }
      return;
    }

    processingRef.current = true;
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const maxWidth = isRealtime ? REALTIME_MAX_WIDTH : SNAPSHOT_MAX_WIDTH;
    const scale = Math.min(1, maxWidth / video.videoWidth);
    const targetWidth = Math.max(1, Math.round(video.videoWidth * scale));
    const targetHeight = Math.max(1, Math.round(video.videoHeight * scale));

    canvas.width = targetWidth;
    canvas.height = targetHeight;

    const context = canvas.getContext("2d");
    if (!context) {
      processingRef.current = false;
      if (isRealtime && realtimeEnabled) {
        queueRealtimeFrame();
      }
      return;
    }

    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "medium";
    context.drawImage(video, 0, 0, targetWidth, targetHeight);
    const originalPreview = canvas.toDataURL(
      "image/jpeg",
      isRealtime ? 0.72 : 0.9,
    );
    if (!isRealtime) {
      setOriginalImage(originalPreview);
    }

    const blob = await new Promise<Blob | null>((resolve) => {
      canvas.toBlob(resolve, "image/jpeg", isRealtime ? 0.72 : 0.9);
    });

    if (!blob) {
      processingRef.current = false;
      if (isRealtime && realtimeEnabled) {
        queueRealtimeFrame();
      }
      return;
    }

    const file = new File([blob], "camera-frame.jpg", { type: "image/jpeg" });
    await processFile(file, originalPreview, isRealtime);
    if (isRealtime && realtimeEnabled) {
      queueRealtimeFrame();
    }
  }

  return (
    <main className="page-shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Live Lip Color Studio</p>
          <h1>LipTone Studio</h1>
          <p className="hero-copy">
            Try lipstick shades in a cleaner, live-first studio made for phone use,
            quick shade switching, and a much bigger pink collection.
          </p>
        </div>
        <div className="hero-stats">
          <div className="stat-card">
            <span>Live first</span>
            <strong>Camera opens before uploads</strong>
          </div>
          <div className="stat-card">
            <span>Pink library</span>
            <strong>{pinkShadeCount}+ pink shades</strong>
          </div>
          <div className="stat-card">
            <span>Collections</span>
            <strong>Reds, nudes, browns, corals, plums</strong>
          </div>
        </div>
      </section>

      <section className="studio-toolbar">
        <div className="mode-switch" role="tablist" aria-label="Try-on mode">
          <button
            type="button"
            className={`mode-chip ${activeMode === "live" ? "active" : ""}`}
            onClick={() => setActiveMode("live")}
          >
            Live try-on
          </button>
          <button
            type="button"
            className={`mode-chip ${activeMode === "upload" ? "active" : ""}`}
            onClick={() => setActiveMode("upload")}
          >
            Upload photo
          </button>
        </div>
        <div className="toolbar-note">
          <strong>{selectedShade}</strong>
          <span>
            {finish} finish · {(opacity * 100).toFixed(0)}% coverage
          </span>
        </div>
      </section>

      <section className="app-grid">
        <section className="preview-panel live-first">
          <div className="result-header">
            <div>
              <p className="eyebrow">
                {activeMode === "live" ? "Live Session" : "Photo Session"}
              </p>
              <h2>
                {activeMode === "live" ? "Realtime Camera Try-On" : "Upload Photo Try-On"}
              </h2>
            </div>
            {loading ? <span className="status-chip">Processing...</span> : null}
          </div>

          {error ? <div className="alert error">{error}</div> : null}
          {message ? <div className="alert info">{message}</div> : null}

          {activeMode === "live" ? (
          <div className="live-stage">
            <div className="image-card live-card">
              <div className="live-card-head">
                <p>Camera</p>
                <div className="camera-actions">
                  {!cameraActive ? (
                    <button type="button" className="primary-button" onClick={startCamera}>
                      Start live camera
                    </button>
                  ) : (
                    <>
                      <button
                        type="button"
                        className="secondary-button"
                        onClick={() => void processCurrentCameraFrame(false)}
                      >
                        Capture HD
                      </button>
                      <button
                        type="button"
                        className="primary-button"
                        onClick={() => {
                          setRealtimeEnabled((current) => {
                            const next = !current;
                            if (!next) {
                              clearRealtimeLoop();
                            }
                            return next;
                          });
                        }}
                      >
                        {realtimeEnabled ? "Stop live stream" : "Start live stream"}
                      </button>
                      <button type="button" className="ghost-button" onClick={stopCamera}>
                        Stop
                      </button>
                    </>
                  )}
                </div>
              </div>
              <div className="camera-frame live-frame">
                <video ref={videoRef} muted playsInline />
                <canvas ref={canvasRef} hidden />
              </div>
              <p className="muted-copy live-note">
                Live mode prioritizes speed by sending smaller frames. Use `Capture HD` for a sharper still result.
              </p>
            </div>

            <div className="image-card">
              <p>Live result</p>
              {resultImage && activeMode === "live" ? (
                <img src={resultImage} alt="Live lipstick try-on result" />
              ) : (
                <div className="image-placeholder">Start the camera to preview the live result</div>
              )}
            </div>
          </div>
          ) : (
          <div className="image-grid upload-focus">
            <div className="image-card">
              <p>Original</p>
              {originalImage ? (
                <img src={originalImage} alt="Original upload or camera frame" />
              ) : (
                <div className="image-placeholder">Upload a portrait to begin</div>
              )}
            </div>
            <div className="image-card">
              <p>Try-on result</p>
              {resultImage && activeMode === "upload" ? (
                <img src={resultImage} alt="Lipstick try-on result" />
              ) : (
                <div className="image-placeholder">Your processed result appears here</div>
              )}
            </div>
          </div>
          )}

          <div className="insights-grid">
            <div className="panel-card">
              <h2>Detected Tone</h2>
              {toneProfile ? (
                <ul className="metric-list">
                  <li>
                    <span>Undertone</span>
                    <strong>{toneProfile.undertone}</strong>
                  </li>
                  <li>
                    <span>Depth</span>
                    <strong>{toneProfile.depth}</strong>
                  </li>
                  <li>
                    <span>Skin sample</span>
                    <strong>{toneProfile.skin_hex ?? "n/a"}</strong>
                  </li>
                  <li>
                    <span>Tuned lipstick</span>
                    <strong>{tunedHex ?? "n/a"}</strong>
                  </li>
                </ul>
              ) : (
                <p className="muted-copy">
                  Tone analysis appears after a successful try-on.
                </p>
              )}
            </div>

            <div className="panel-card">
              <h2>Recommended Pinks</h2>
              {recommendations.length > 0 ? (
                <div className="recommendation-list">
                  {recommendations.map((shade) => (
                    <button
                      key={`${shade.subcategory}-${shade.name}`}
                      type="button"
                      className="recommendation-card"
                      onClick={() => {
                        setUseCustomColor(false);
                        setSelectedFamily(shade.family);
                        setSelectedSubcategory(shade.subcategory);
                        setSelectedShade(shade.name);
                      }}
                    >
                      <span
                        className="shade-swatch"
                        style={{ backgroundColor: shade.hex }}
                      />
                      <div>
                        <strong>{shade.name}</strong>
                        <small>
                          {shade.subcategory} · {shade.undertone}
                        </small>
                      </div>
                    </button>
                  ))}
                </div>
              ) : (
                <p className="muted-copy">
                  The backend will suggest pink shades that better fit the detected tone.
                </p>
              )}
            </div>
          </div>
        </section>

        <aside className="control-panel">
          <div className="panel-card">
            <h2>Shade Controls</h2>
            <div className="collection-banner">
              <strong>{selectedFamily}</strong>
              <span>{selectedSubcategory}</span>
            </div>
            <label>
              <span>Family</span>
              <select
                value={selectedFamily}
                onChange={(event) => setSelectedFamily(event.target.value)}
              >
                {families.map((family) => (
                  <option key={family} value={family}>
                    {family}
                  </option>
                ))}
              </select>
            </label>

            <label>
              <span>Subcategory</span>
              <select
                value={selectedSubcategory}
                onChange={(event) => setSelectedSubcategory(event.target.value)}
              >
                {subcategories.map((subcategory) => (
                  <option key={subcategory} value={subcategory}>
                    {subcategory}
                  </option>
                ))}
              </select>
            </label>

            <p className="muted-copy">
              Pick from larger color collections. Pinks now include soft, cool, warm, nude,
              bright, mauve, rose, berry, pastel, and deep rose groups.
            </p>

            <label>
              <span>Search shade</span>
              <input
                type="text"
                placeholder="Find mink pink, rose, berry..."
                value={shadeQuery}
                onChange={(event) => setShadeQuery(event.target.value)}
              />
            </label>

            <div className="shade-grid">
              {filteredShadeOptions.map((shade) => (
                <button
                  key={shade.name}
                  type="button"
                  className={`shade-tile ${
                    !useCustomColor && selectedShade === shade.name ? "active" : ""
                  }`}
                  onClick={() => {
                    setUseCustomColor(false);
                    setSelectedShade(shade.name);
                  }}
                >
                  <span
                    className="shade-swatch"
                    style={{ backgroundColor: shade.hex }}
                  />
                  <span className="shade-label">{shade.name}</span>
                  <small>
                    {shade.undertone} · {shade.depth}
                  </small>
                </button>
              ))}
            </div>
            {filteredShadeOptions.length === 0 ? (
              <p className="muted-copy">No shades match this search in the selected subcategory.</p>
            ) : null}

            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={useCustomColor}
                onChange={(event) => setUseCustomColor(event.target.checked)}
              />
              <span>Use custom color</span>
            </label>

            {useCustomColor ? (
              <label>
                <span>Custom hex</span>
                <div className="color-picker-row">
                  <input
                    type="color"
                    value={customHex}
                    onChange={(event) => setCustomHex(event.target.value)}
                  />
                  <input
                    type="text"
                    value={customHex}
                    onChange={(event) => setCustomHex(event.target.value)}
                  />
                </div>
              </label>
            ) : null}

            <label>
              <span>Coverage: {opacity.toFixed(2)}</span>
              <input
                type="range"
                min="0.1"
                max="1"
                step="0.05"
                value={opacity}
                onChange={(event) => setOpacity(Number(event.target.value))}
              />
            </label>

            <label>
              <span>Finish</span>
              <div className="pill-row">
                {["Matte", "Satin", "Gloss"].map((item) => (
                  <button
                    key={item}
                    type="button"
                    className={`pill-button ${finish === item ? "active" : ""}`}
                    onClick={() => setFinish(item)}
                  >
                    {item}
                  </button>
                ))}
              </div>
            </label>
          </div>

          <div className="panel-card">
            <h2>Photo Upload</h2>
            <label className="upload-box">
              <input
                type="file"
                accept="image/*"
                onChange={onUploadChange}
                onClick={() => setActiveMode("upload")}
              />
              <span>Upload a selfie or portrait</span>
            </label>
            <p className="muted-copy">
              Upload mode gives a higher-quality still result. Live camera remains the primary experience above.
            </p>
          </div>
        </aside>
      </section>
    </main>
  );
}
