import { useEffect, useRef, useState } from "react";
import "./Deepfake.css";

const BACKEND = "http://localhost:5000";

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
	const [audioEnabled, setAudioEnabled] = useState(true);
	const [videoEnabled, setVideoEnabled] = useState(true);
	const [inCall, setInCall] = useState(false);
	const [uploadedVideoURL, setUploadedVideoURL] = useState(null);
	const [participantName] = useState("Alex Chen");
	const [detectionResult, setDetectionResult] = useState(null);
	const [isAnalyzing, setIsAnalyzing] = useState(false);
	const [backendError, setBackendError] = useState(null);
	const [cameras, setCameras] = useState([]);
	const [selectedCameraId, setSelectedCameraId] = useState("");
	const [showSettings, setShowSettings] = useState(false);
	const [cameraError, setCameraError] = useState(null);

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

	// Start / stop live webcam analysis whenever the call state changes
	useEffect(() => {
		if (analyzeIntervalRef.current) {
			clearInterval(analyzeIntervalRef.current);
			analyzeIntervalRef.current = null;
		}

		if (!inCall) {
			setDetectionResult(null);
			setIsAnalyzing(false);
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
		setIsAnalyzing(false);
		setBackendError(null);

		let isMounted = true;
		let currentAbort = null;

		// Small delay so the camera stream has time to fully initialise
		const startTimer = setTimeout(() => {
			if (!isMounted) return;

			analyzeIntervalRef.current = setInterval(async () => {
				const videoEl = remoteVideoRef.current;
				const canvas = canvasRef.current;
				// Only skip if there is genuinely no frame yet
				if (!videoEl || !canvas || videoEl.videoWidth === 0) return;

				// Abort previous in-flight request before sending the next
				if (currentAbort) currentAbort.abort();
				currentAbort = new AbortController();

				canvas.width = videoEl.videoWidth;
				canvas.height = videoEl.videoHeight;
				canvas.getContext("2d").drawImage(videoEl, 0, 0);
				const base64 = canvas.toDataURL("image/jpeg", 0.8).split(",")[1];

				try {
					const res = await fetch(`${BACKEND}/deepfake/score`, {
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
							setIsAnalyzing(true);
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
						setBackendError("Cannot connect to detection backend — is the Flask server running on port 5000?");
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

	// Re-enumerate cameras whenever devices are added or removed
	useEffect(() => {
		if (!navigator.mediaDevices?.addEventListener) return;
		const onDeviceChange = () => enumerateCameras();
		navigator.mediaDevices.addEventListener("devicechange", onDeviceChange);
		return () => navigator.mediaDevices.removeEventListener("devicechange", onDeviceChange);
	}, []);

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
			if (localVideoRef.current && !uploadedVideoURL) {
				// keep small preview blank/upload placeholder until user provides a video
				localVideoRef.current.srcObject = null;
			}
			inCallRef.current = true;
			setInCall(true);
			setAudioEnabled(true);
			setVideoEnabled(true);
		} catch (err) {
			console.error("Could not get user media", err);
			alert("Unable to access camera/microphone.");
		}
	}

	function endCall() {
		if (mediaStreamRef.current) {
			mediaStreamRef.current.getTracks().forEach((t) => t.stop());
			mediaStreamRef.current = null;
		}
		if (remoteVideoRef.current) remoteVideoRef.current.srcObject = null;
		if (localVideoRef.current) {
			// keep uploaded video if present; otherwise clear preview
			if (!uploadedVideoURL) localVideoRef.current.srcObject = null;
		}
		inCallRef.current = false;
		setInCall(false);
		setShowSettings(false);
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

	return (
		<section className="deepfake-panel">
			<div className="video-area">
				<div className="call-header">
					<span className="call-dot" /> You are in call
					<span className="call-timer">00:00</span>
				</div>

				<video
					ref={remoteVideoRef}
					className="remote-video"
					playsInline
					autoPlay
					muted
				/>

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
					<button onClick={toggleAudio} className="control-btn">
						{audioEnabled ? "Mute" : "Unmute"}
					</button>
					<button onClick={toggleVideo} className="control-btn">
						{videoEnabled ? "Stop Video" : "Start Video"}
					</button>
					<button className="control-btn">Screen Share</button>
					<button className="control-btn">Chat</button>
					<button className="control-btn">More</button>
					<button
						onClick={() => setShowSettings((v) => !v)}
						className={`control-btn${showSettings ? " settings-active" : ""}`}
					>
						⚙ Settings
					</button>
					{!inCall ? (
						<button onClick={startCall} className="control-btn primary">
							Start Call
						</button>
					) : (
						<button onClick={endCall} className="control-btn danger">
							End Call
						</button>
					)}
				</div>
			</div>

			<aside className="detection-panel">
				<div className="detection-card">
					<h3 className="detection-title">AI Deepfake Detector</h3>

					{!inCall ? (
					<p className="detection-idle">Start a call to begin live analysis.</p>
				) : backendError ? (
					<p className="detection-error">
					{backendError}
					</p>
				) : !isAnalyzing ? (
					<p className="detection-idle analyzing-pulse">Analysing webcam feed…</p>
				) : (
					detectionResult && (
							<div className="detection-results">
								<div
									className={`verdict-badge ${detectionResult.rolling.verdict === "DEEPFAKE" ? "deepfake" : "real"}`}
								>
									{detectionResult.rolling.verdict}
								</div>

							<div className="ai-explanation">
								<span className="ai-explanation-title">AI Explanation</span>
								<p className="ai-explanation-text">
									{detectionResult.reason || "Analysing recent frames…"}
								</p>
							</div>
								<div className="confidence-label">
									<span>Deepfake probability</span>
									<span className="confidence-pct">
										{(detectionResult.rolling.prob * 100).toFixed(1)}%
									</span>
								</div>
								<div className="confidence-track">
									<div
										className={`confidence-fill ${detectionResult.rolling.verdict === "DEEPFAKE" ? "deepfake" : "real"}`}
										style={{
											width: `${(detectionResult.rolling.prob * 100).toFixed(1)}%`,
										}}
									/>
								</div>

								<div className="detail-grid">
									<span className="detail-key">Frames in window</span>
									<span className="detail-val">
										{detectionResult.rolling.frames_in_window}
									</span>

									<span className="detail-key">Face detected</span>
									<span
										className={`detail-val ${detectionResult.frame.face_found ? "face-yes" : "face-no"}`}
									>
										{detectionResult.frame.face_found ? "Yes" : "No (fallback)"}
									</span>

									<span className="detail-key">Frame score</span>
									<span className="detail-val">
										{(detectionResult.frame.prob * 100).toFixed(1)}%
									</span>
								</div>

							</div>
						)
					)}
				</div>
			</aside>

			{/* Hidden canvas used for frame capture */}
			<canvas ref={canvasRef} style={{ display: "none" }} />
		</section>
	);
}

export default Deepfake;
