import { useEffect, useRef, useState } from "react";
import "./Deepfake.css";

const BACKEND = "http://localhost:5000";

// ── Call status constants ─────────────────────────────────────────────────────
const STATUS = {
	READY: "ready",
	CONNECTING: "connecting",
	ACTIVE: "active",
	ANALYSING: "analysing",
	ENDED: "ended",
	ERROR: "error",
};

// ── CallStatus sub-component ──────────────────────────────────────────────────
function CallStatus({ status, elapsedTime, errorMessage }) {
	const fmt = (s) =>
		`${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

	const dotMod = {
		[STATUS.READY]: "grey",
		[STATUS.CONNECTING]: "yellow",
		[STATUS.ACTIVE]: "green",
		[STATUS.ANALYSING]: "green",
		[STATUS.ENDED]: "grey",
		[STATUS.ERROR]: "red",
	}[status] ?? "grey";

	const label = {
		[STATUS.READY]: "Ready to start",
		[STATUS.CONNECTING]: "Connecting...",
		[STATUS.ACTIVE]: "Call active",
		[STATUS.ANALYSING]: "Analysing video...",
		[STATUS.ENDED]: "Call ended",
		[STATUS.ERROR]: errorMessage || "Connection error",
	}[status];

	return (
		<div className="call-header">
			<span className={`call-dot call-dot--${dotMod}`} />
			<span className="call-status-label">{label}</span>
			{status === STATUS.CONNECTING && <span className="header-spinner" />}
			<span className="call-timer">{fmt(elapsedTime)}</span>
		</div>
	);
}

// ── Main component ────────────────────────────────────────────────────────────
function Deepfake() {
	const remoteVideoRef = useRef(null);
	const localVideoRef = useRef(null);
	const fileInputRef = useRef(null);
	const canvasRef = useRef(null);
	const analyzeIntervalRef = useRef(null);
	const sessionIdRef = useRef(
		typeof crypto !== "undefined" && crypto.randomUUID
			? crypto.randomUUID()
			: Math.random().toString(36).slice(2)
	);
	const mediaStreamRef = useRef(null);
	const selectedCameraIdRef = useRef("");
	const inCallRef = useRef(false);
	const heatmapEnabledRef = useRef(false);
	const overlayCanvasRef = useRef(null);   // canvas drawn by rAF loop
	const heatmapImgRef = useRef(null);       // cached face-crop RGBA Image
	const heatmapBboxRef = useRef(null);      // cached bbox [x1,y1,x2,y2]
	const rafRef = useRef(null);              // requestAnimationFrame handle
	const heatmapLoopActiveRef = useRef(false); // cancel flag for fetch loop

	// ── Call state ─────────────────────────────────────────────────────────────────────────
	const [callStatus, setCallStatus] = useState(STATUS.READY);
	const [elapsedTime, setElapsedTime] = useState(0);
	const [errorMessage, setErrorMessage] = useState(null);
	const [audioEnabled, setAudioEnabled] = useState(true);
	const [videoEnabled, setVideoEnabled] = useState(true);
	const [uploadedVideoURL, setUploadedVideoURL] = useState(null);
	const [participantName] = useState("Alex Chen");
	const [detectionResult, setDetectionResult] = useState(null);
	const [stableReason, setStableReason] = useState(null);
	const stableReasonRef = useRef(null);
	const [backendError, setBackendError] = useState(null);
	const [cameras, setCameras] = useState([]);
	const [selectedCameraId, setSelectedCameraId] = useState("");
	const [showSettings, setShowSettings] = useState(false);
	const [cameraError, setCameraError] = useState(null);
	const [heatmapEnabled, setHeatmapEnabled] = useState(false);

	// Stable reason: only update when the explanation text actually changes so it
	// doesn't flicker on every frame even though detectionResult updates every frame.
	useEffect(() => {
		const incoming = detectionResult?.reason;
		if (incoming && incoming !== stableReasonRef.current) {
			stableReasonRef.current = incoming;
			setStableReason(incoming);
		}
	}, [detectionResult?.reason]);

	// Derived: call is live when ACTIVE or ANALYSING
	const inCall = callStatus === STATUS.ACTIVE || callStatus === STATUS.ANALYSING;

	// Sync heatmap ref (loop lifecycle is managed by the dedicated useEffect below)
	useEffect(() => {
		heatmapEnabledRef.current = heatmapEnabled;
	}, [heatmapEnabled]);

	// Keep inCallRef in sync for use inside async closures
	useEffect(() => {
		inCallRef.current = inCall;
	}, [inCall]);

	useEffect(() => {
		// Enumerate cameras on mount without requesting media access.
		// Labels may be empty until the user grants permission via Start Call.
		enumerateCameras();

		return () => {
			// cleanup on unmount
			if (analyzeIntervalRef.current) clearInterval(analyzeIntervalRef.current);
			if (mediaStreamRef.current) {
				mediaStreamRef.current.getTracks().forEach((t) => t.stop());
			}
			if (uploadedVideoURL) {
				URL.revokeObjectURL(uploadedVideoURL);
			}
		};
	}, []);

	// Revoke uploaded object URL when it changes or component unmounts
	useEffect(() => {
		return () => {
			if (uploadedVideoURL) URL.revokeObjectURL(uploadedVideoURL);
		};
	}, [uploadedVideoURL]);

	// ── Call timer ─────────────────────────────────────────────────────────────────────────
	useEffect(() => {
		if (!inCall) return;
		const timer = setInterval(() => setElapsedTime((t) => t + 1), 1000);
		return () => clearInterval(timer);
	}, [inCall]);

	// ── Analysis loop: starts/stops with inCall ────────────────────────────────────────
	useEffect(() => {
		if (analyzeIntervalRef.current) {
			clearInterval(analyzeIntervalRef.current);
			analyzeIntervalRef.current = null;
		}

		if (!inCall) {
			// Keep detectionResult alive so ENDED state can still show last results
			setBackendError(null);
			return;
		}

		// Reset rolling window for the new call session
		fetch(`${BACKEND}/deepfake/reset`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ session_id: sessionIdRef.current }),
		}).catch(() => {});
		setDetectionResult(null);
		setBackendError(null);

		let isMounted = true;
		let currentAbort = null;

		// Small delay so the camera stream has time to fully initialise
		const startTimer = setTimeout(() => {
			if (!isMounted) return;

			analyzeIntervalRef.current = setInterval(async () => {
				const videoEl = remoteVideoRef.current;
				const canvas = canvasRef.current;
				if (!videoEl || !canvas || videoEl.videoWidth === 0) return;
				if (heatmapEnabledRef.current) return; // heatmap has its own fetch + rAF loop

				if (currentAbort) currentAbort.abort();
				currentAbort = new AbortController();

				canvas.width = videoEl.videoWidth;
				canvas.height = videoEl.videoHeight;
				canvas.getContext("2d").drawImage(videoEl, 0, 0);
				const base64 = canvas.toDataURL("image/jpeg", 0.8).split(",")[1];

				const endpoint = heatmapEnabledRef.current
					? `${BACKEND}/deepfake/heatmap`
					: `${BACKEND}/deepfake/score`;

				try {
					const res = await fetch(endpoint, {
						method: "POST",
						headers: { "Content-Type": "application/json" },
						body: JSON.stringify({
							frame: base64,
							session_id: sessionIdRef.current,
						}),
						signal: currentAbort.signal,
					});
					if (res.ok) {
						const data = await res.json();
						if (isMounted) {
							setDetectionResult(data);
							// Transition ACTIVE → ANALYSING on first result
							setCallStatus((prev) =>
								prev === STATUS.ACTIVE ? STATUS.ANALYSING : prev
							);
							setBackendError(null);
						}
					} else {
						if (isMounted) {
							let msg = `Detection server error (HTTP ${res.status})`;
							try {
								const body = await res.json();
								if (body?.error) msg = body.error;
							} catch (_) { /* body was not JSON */ }
							setBackendError(msg);
						}
					}
				} catch (err) {
					if (err.name !== "AbortError" && isMounted)
						setBackendError(
							"Cannot connect to detection backend — is the Flask server running on port 5000?"
						);
				}
			}, 1000);
		}, 2000); // 2 s warm-up before first capture

		return () => {
			isMounted = false;
			clearTimeout(startTimer);
			if (currentAbort) currentAbort.abort();
			if (analyzeIntervalRef.current) {
				clearInterval(analyzeIntervalRef.current);
				analyzeIntervalRef.current = null;
			}
		};
	}, [inCall]);

	// ── Heatmap: rAF render loop + adaptive fetch loop ────────────────────────
	// rAF draws cached heatmap at latest bbox every frame (30 fps, smooth).
	// Fetch loop fires as fast as backend responds, updating bbox each time and
	// updating the heatmap image only on Grad-CAM frames (every HEATMAP_STRIDE).
	useEffect(() => {
		if (!heatmapEnabled || !inCall) {
			heatmapLoopActiveRef.current = false;
			if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
			const cvs = overlayCanvasRef.current;
			if (cvs) { cvs.getContext("2d").clearRect(0, 0, cvs.width, cvs.height); cvs.style.display = "none"; }
			heatmapImgRef.current = null;
			heatmapBboxRef.current = null;
			return;
		}

		// rAF render loop: 30 fps, draws cached heatmap at latest bbox ──────────
		const drawFrame = () => {
			const cvs = overlayCanvasRef.current;
			const videoEl = remoteVideoRef.current;
			if (!cvs || !videoEl || videoEl.videoWidth === 0) {
				rafRef.current = requestAnimationFrame(drawFrame);
				return;
			}
			const cssW = videoEl.clientWidth, cssH = videoEl.clientHeight;
			if (cvs.width !== cssW || cvs.height !== cssH) { cvs.width = cssW; cvs.height = cssH; }
			const ctx = cvs.getContext("2d");
			ctx.clearRect(0, 0, cssW, cssH);
			const img = heatmapImgRef.current;
			const bbox = heatmapBboxRef.current;
			if (img && bbox) {
					// bbox is now normalised [0-1] fractions of the video frame,
					// so it maps correctly regardless of the capture resolution sent.
					const [rx1, ry1, rx2, ry2] = bbox;
					const natW = videoEl.videoWidth, natH = videoEl.videoHeight;
					const scale = Math.max(cssW / natW, cssH / natH);
					const offX = (cssW - natW * scale) / 2;
					const offY = (cssH - natH * scale) / 2;
					cvs.style.display = "block";
					ctx.drawImage(img,
						rx1 * natW * scale + offX,
						ry1 * natH * scale + offY,
						(rx2 - rx1) * natW * scale,
						(ry2 - ry1) * natH * scale
					);
			}
			rafRef.current = requestAnimationFrame(drawFrame);
		};
		rafRef.current = requestAnimationFrame(drawFrame);

		// Adaptive fetch loop: fires next request right after previous completes ──
		heatmapLoopActiveRef.current = true;
		const MIN_DELAY_MS = 50;

		const fetchLoop = async () => {
			if (!heatmapLoopActiveRef.current) return;
			const videoEl = remoteVideoRef.current;
			const capCanvas = canvasRef.current;
			if (videoEl && capCanvas && videoEl.videoWidth > 0) {
				const t0 = Date.now();
				// Downscale to 640 px wide before sending — 4× fewer pixels for
				// MTCNN and Grad-CAM without affecting model accuracy.
				const MAX_CAP_W = 640;
				const capScale = Math.min(1, MAX_CAP_W / videoEl.videoWidth);
				const capW = Math.round(videoEl.videoWidth  * capScale);
				const capH = Math.round(videoEl.videoHeight * capScale);
				capCanvas.width  = capW;
				capCanvas.height = capH;
				capCanvas.getContext("2d").drawImage(videoEl, 0, 0, capW, capH);
				const base64 = capCanvas.toDataURL("image/jpeg", 0.7).split(",")[1];
				try {
					const res = await fetch(`${BACKEND}/deepfake/heatmap`, {
						method: "POST",
						headers: { "Content-Type": "application/json" },
						body: JSON.stringify({ frame: base64, session_id: sessionIdRef.current }),
					});
					if (res.ok && heatmapLoopActiveRef.current) {
						const data = await res.json();
						setDetectionResult(data);
						setCallStatus(prev => prev === STATUS.ACTIVE ? STATUS.ANALYSING : prev);
						setBackendError(null);
						if (data.bbox) heatmapBboxRef.current = data.bbox;
						if (data.heatmap) {
							const im = new Image();
							im.onload = () => { heatmapImgRef.current = im; };
							im.src = `data:image/png;base64,${data.heatmap}`;
						}
					}
				} catch (_) {
					if (heatmapLoopActiveRef.current)
						setBackendError("Cannot connect to detection backend — is the Flask server running on port 5000?");
				}
				const elapsed = Date.now() - t0;
				if (heatmapLoopActiveRef.current)
					setTimeout(fetchLoop, Math.max(0, MIN_DELAY_MS - elapsed));
			} else {
				if (heatmapLoopActiveRef.current) setTimeout(fetchLoop, 100);
			}
		};
		const warmup = setTimeout(fetchLoop, 2000);

		return () => {
			clearTimeout(warmup);
			heatmapLoopActiveRef.current = false;
			if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
			const cvs = overlayCanvasRef.current;
			if (cvs) { cvs.getContext("2d").clearRect(0, 0, cvs.width, cvs.height); cvs.style.display = "none"; }
			heatmapImgRef.current = null;
			heatmapBboxRef.current = null;
		};
	}, [heatmapEnabled, inCall]);

	// Re-enumerate cameras whenever devices are added or removed
	useEffect(() => {
		if (!navigator.mediaDevices?.addEventListener) return;
		const onDeviceChange = () => enumerateCameras();
		navigator.mediaDevices.addEventListener("devicechange", onDeviceChange);
		return () => navigator.mediaDevices.removeEventListener("devicechange", onDeviceChange);
	}, []);

	// ── Show full-frame heatmap JPEG in the overlay img element ────────────────
	function showHeatmap(base64jpeg) { /* no-op: replaced by rAF canvas loop */ }

	async function enumerateCameras() {
		try {
			const devices = await navigator.mediaDevices.enumerateDevices();
			// Include all videoinput devices — physical and virtual (OBS, Deep-Live-Cam, etc.)
			const videoInputs = devices.filter((d) => d.kind === "videoinput");
			setCameras(videoInputs);
			return videoInputs;
		} catch (err) {
			console.error("Could not enumerate devices", err);
			return [];
		}
	}

	async function switchCamera(deviceId) {
		// Update preference; if not in a call, just store for the next startCall
		selectedCameraIdRef.current = deviceId;
		setSelectedCameraId(deviceId);
		setCameraError(null);

		if (!inCallRef.current || !mediaStreamRef.current) return;

		try {
			const newStream = await navigator.mediaDevices.getUserMedia({
				video: { deviceId: { exact: deviceId }, width: 1280, height: 720 },
				audio: false,
			});
			const newVideoTrack = newStream.getVideoTracks()[0];

			// Stop and remove old video tracks
			const oldTracks = mediaStreamRef.current.getVideoTracks();
			oldTracks.forEach((t) => {
				t.stop();
				mediaStreamRef.current.removeTrack(t);
			});

			// Add the new track and attach disconnect watcher
			mediaStreamRef.current.addTrack(newVideoTrack);
			newVideoTrack.addEventListener("ended", handleCameraDisconnect);

			// Refresh srcObject so the browser picks up the new track immediately
			if (remoteVideoRef.current) {
				remoteVideoRef.current.srcObject = mediaStreamRef.current;
			}
		} catch (err) {
			console.error("Could not switch camera", err);
			setCameraError("Could not switch to the selected camera.");
		}
	}

	async function handleCameraDisconnect() {
		const available = await enumerateCameras();
		if (available.length > 0) {
			const fallbackId = available[0].deviceId;
			setCameraError("Selected camera disconnected. Switched to default camera.");
			selectedCameraIdRef.current = fallbackId;
			setSelectedCameraId(fallbackId);
			if (inCallRef.current) {
				await switchCamera(fallbackId);
			}
		} else {
			setCameraError("Camera disconnected and no other cameras are available.");
		}
	}

	async function startCall() {
		setCallStatus(STATUS.CONNECTING);
		setErrorMessage(null);
		setElapsedTime(0);
		setDetectionResult(null);

		try {
			const camConstraint = selectedCameraIdRef.current
				? { deviceId: { exact: selectedCameraIdRef.current }, width: 1280, height: 720 }
				: { width: 1280, height: 720 };
			const stream = await navigator.mediaDevices.getUserMedia({
				video: camConstraint,
				audio: true,
			});
			mediaStreamRef.current = stream;

			// Enumerate cameras now that we have permission (device labels are populated after getUserMedia)
			const cams = await enumerateCameras();

			// Identify the active camera and watch for disconnects
			const activeVideoTrack = stream.getVideoTracks()[0];
			if (activeVideoTrack) {
				const settings = activeVideoTrack.getSettings();
				const activeId = settings.deviceId ?? (cams[0]?.deviceId ?? "");
				selectedCameraIdRef.current = activeId;
				setSelectedCameraId(activeId);
				activeVideoTrack.addEventListener("ended", handleCameraDisconnect);
			}

			// show local user in the large video area; small preview is reserved for uploaded remote video
			if (remoteVideoRef.current) remoteVideoRef.current.srcObject = stream;
			setAudioEnabled(true);
			setVideoEnabled(true);
			setCallStatus(STATUS.ACTIVE);
		} catch (err) {
			console.error("Could not get user media", err);
			let msg = "Failed to connect.";
			if (err.name === "NotAllowedError") {
				msg = "Camera or microphone permission denied. Please allow access in your browser settings and try again.";
			} else if (err.name === "NotFoundError") {
				msg = "No camera or microphone found on this device.";
			} else if (err.name === "NotReadableError") {
				msg = "Camera or microphone is already in use by another application.";
			}
			setErrorMessage(msg);
			setCallStatus(STATUS.ERROR);
		}
	}

	function endCall() {
		if (mediaStreamRef.current) {
			mediaStreamRef.current.getTracks().forEach((t) => t.stop());
			mediaStreamRef.current = null;
		}
		if (remoteVideoRef.current) remoteVideoRef.current.srcObject = null;
		setShowSettings(false);
		setCallStatus(STATUS.ENDED);
	}

	function retryCall() {
		startCall();
	}

	function toggleAudio() {
		if (!mediaStreamRef.current) return;
		const audioTracks = mediaStreamRef.current.getAudioTracks();
		audioTracks.forEach((t) => (t.enabled = !t.enabled));
		setAudioEnabled((v) => !v);
	}

	function toggleVideo() {
		if (!mediaStreamRef.current) return;
		const videoTracks = mediaStreamRef.current.getVideoTracks();
		videoTracks.forEach((t) => (t.enabled = !t.enabled));
		setVideoEnabled((v) => !v);
	}

	function onLocalPreviewClick() {
		// open file picker to upload a video that will simulate remote participant
		if (fileInputRef.current) fileInputRef.current.click();
	}

	function onFileSelected(e) {
		const f = e.target.files && e.target.files[0];
		if (!f) return;
		// revoke previous URL
		if (uploadedVideoURL) URL.revokeObjectURL(uploadedVideoURL);
		const url = URL.createObjectURL(f);
		setUploadedVideoURL(url);
		// set the small preview to play the uploaded video
		if (localVideoRef.current) {
			localVideoRef.current.srcObject = null;
			localVideoRef.current.src = url;
			localVideoRef.current.muted = true;
			localVideoRef.current.loop = true;
			localVideoRef.current.play().catch(() => {});
		}
		// clear input so same file can be re-selected later
		e.target.value = "";
	}

	// ── Detection panel content ──────────────────────────────────────────────────
	// Display-only remap: model raw scores are tiny (real ≈ 0–1%, deepfake ≈ 3–10%).
	// Normalise so 3% raw (the verdict threshold) maps to ~55%, deepfake at 5–10%
	// lands at 71–100%, and real below 1% shows 22–32%.
	// Backend verdict/threshold is unchanged — this is display only.
	function remapProb(p) {
		return Math.min(Math.sqrt(Math.max(p, 0) / 0.10), 1.0);
	}

	function renderResults() {
		if (!detectionResult) return null;
		const rollingPct  = (remapProb(detectionResult.rolling.prob) * 100).toFixed(1);
		const framePct    = (remapProb(detectionResult.frame.prob)   * 100).toFixed(1);
		const rollingFrac = remapProb(detectionResult.rolling.prob)  * 100;
		const frameFrac   = remapProb(detectionResult.frame.prob)    * 100;
		return (
			<div className="detection-results">
				<div
					className={`verdict-badge ${detectionResult.rolling.verdict === "DEEPFAKE" ? "deepfake" : "real"}`}
				>
					{detectionResult.rolling.verdict}
				</div>

				<div className="ai-explanation">
					<span className="ai-explanation-title">AI Explanation</span>
					<p className="ai-explanation-text">
						{stableReason || "Analysing recent frames…"}
					</p>
				</div>

				<div className="confidence-label">
					<span>Deepfake probability</span>
					<span className="confidence-pct">
						{rollingPct}%
					</span>
				</div>
				<div className="confidence-track">
					<div
						className={`confidence-fill ${detectionResult.rolling.verdict === "DEEPFAKE" ? "deepfake" : "real"}`}
						style={{ width: `${rollingPct}%` }}
					/>
				</div>

				<div className="detail-grid">
					<span className="detail-key">Frames in window</span>
					<span className="detail-val">{detectionResult.rolling.frames_in_window}</span>

					<span className="detail-key">Face detected</span>
					<span className={`detail-val ${detectionResult.frame.face_found ? "face-yes" : "face-no"}`}>
						{detectionResult.frame.face_found ? "Yes" : "No — position face in frame"}
					</span>

					<span className="detail-key">Frame score</span>
					<span className="detail-val">
						{framePct}%
					</span>
				</div>

				<div className="heatmap-scale">
					<div className="heatmap-scale-bar">
						<div
							className="heatmap-scale-marker"
							style={{ left: `${Math.min(frameFrac, 100)}%` }}
						/>
					</div>
					<div className="heatmap-scale-labels">
						<span>Real</span>
						<span>Uncertain</span>
						<span>Suspicious</span>
						<span>Fake</span>
					</div>
				</div>

				{!detectionResult.frame.face_found && (
					<p className="face-warning">⚠ No face detected. Results may be less accurate.</p>
				)}
			</div>
		);
	}

	function renderDetectionContent() {
		if (callStatus === STATUS.READY || callStatus === STATUS.CONNECTING) {
			return <p className="detection-idle">Start a call to begin live analysis.</p>;
		}
		if (callStatus === STATUS.ERROR) {
			return <p className="detection-idle">{errorMessage || "Connection failed."}</p>;
		}
		if (callStatus === STATUS.ENDED) {
			return (
				<>
					<p className="detection-ended-msg">Call ended. View the final analysis below.</p>
					{renderResults()}
				</>
			);
		}
		if (backendError) {
			return <p className="detection-error">{backendError}</p>;
		}
		if (callStatus === STATUS.ACTIVE) {
			return (
				<div className="detection-loading">
					<span className="detection-spinner" />
					<p className="detection-idle analyzing-pulse">Live deepfake analysis in progress</p>
				</div>
			);
		}
		// ANALYSING
		return renderResults() ?? <p className="detection-idle analyzing-pulse">Analysing webcam feed…</p>;
	}

	return (
		<section className="deepfake-panel">
			<div className="video-area">
				<CallStatus
					status={callStatus}
					elapsedTime={elapsedTime}
					errorMessage={errorMessage}
				/>

				<video
					ref={remoteVideoRef}
					className="remote-video"
					playsInline
					autoPlay
					muted
				/>

			{/* Heatmap canvas — drawn at 30 fps by rAF loop, face-bbox sized overlay */}
			<canvas ref={overlayCanvasRef} className="heatmap-overlay-canvas" />

			{/* Heatmap ON badge — top-right corner indicator */}
			{heatmapEnabled && inCall && (
				<div className="heatmap-badge">🌡 Heatmap ON</div>
			)}

			<div
				className={"local-preview" + (uploadedVideoURL ? " has-video" : " upload-area")}
				onClick={() => {
					if (!uploadedVideoURL) onLocalPreviewClick();
				}}
				role="button"
				tabIndex={0}
			>
				<input
						ref={fileInputRef}
						type="file"
						accept="video/mp4,video/quicktime,video/x-msvideo,video/*"
						style={{ display: "none" }}
						onChange={onFileSelected}
					/>
					<video
						ref={localVideoRef}
						className="local-video"
						playsInline
						autoPlay
						muted
					/>
					{!uploadedVideoURL && (
						<div className="upload-hint">Click to upload remote participant video</div>
					)}
					<div className="participant-meta">
						<span className="status-dot" />
						<span className="participant-name">{participantName}</span>
					</div>
				</div>

				{cameraError && (
					<div className="camera-error-toast">
						{cameraError}
						<button className="toast-close" onClick={() => setCameraError(null)}>✕</button>
					</div>
				)}

				{callStatus === STATUS.ERROR && errorMessage && (
					<div className="error-alert">
						<span className="error-alert-icon">⚠</span>
						<span className="error-alert-msg">{errorMessage}</span>
						<button className="retry-btn" onClick={retryCall}>Retry</button>
					</div>
				)}

				{showSettings && (
					<div className="settings-panel">
						<div className="settings-row">
							<label className="settings-label" htmlFor="camera-select">Video Input</label>
							<select
								id="camera-select"
								className="settings-select"
								value={selectedCameraId}
								onChange={(e) => switchCamera(e.target.value)}
							>
								{cameras.length === 0 && (
									<option value="">No cameras detected</option>
								)}
								{cameras.map((cam) => (
									<option key={cam.deviceId} value={cam.deviceId}>
										{cam.label || `Camera ${cam.deviceId.slice(0, 8)}`}
									</option>
								))}
							</select>
						</div>
					</div>
				)}

				<div className="controls">
					<button onClick={toggleAudio} className="control-btn" disabled={!inCall}>
						{audioEnabled ? "Mute" : "Unmute"}
					</button>
					<button onClick={toggleVideo} className="control-btn" disabled={!inCall}>
						{videoEnabled ? "Stop Video" : "Start Video"}
					</button>
					<button
						onClick={() => setShowSettings((v) => !v)}
						className={`control-btn${showSettings ? " settings-active" : ""}`}
					>
						⚙ Settings
					</button>
					<button
						onClick={() => setHeatmapEnabled((v) => !v)}
						className={`control-btn${heatmapEnabled ? " heatmap-active" : ""}`}
						disabled={!inCall}
						title="Toggle Grad-CAM heatmap overlay"
					>
						🌡 Heatmap
					</button>
					{callStatus === STATUS.CONNECTING ? (
						<button className="control-btn primary" disabled>
							<span className="btn-spinner" /> Connecting...
						</button>
					) : inCall ? (
						<button onClick={endCall} className="control-btn danger">
							End Call
						</button>
					) : (
						<button onClick={startCall} className="control-btn primary">
							Start Call
						</button>
					)}
				</div>
			</div>

			<aside className="detection-panel">
				<div className="detection-card">
					<h3 className="detection-title">AI Deepfake Detector</h3>
					{renderDetectionContent()}
				</div>
			</aside>

			{/* Hidden canvas used for frame capture */}
			<canvas ref={canvasRef} style={{ display: "none" }} />
		</section>
	);
}

export default Deepfake;
