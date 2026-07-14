import { useEffect, useRef, useState } from "react";
import "./Deepfake.css";

function Deepfake() {
	const remoteVideoRef = useRef(null);
	const localVideoRef = useRef(null);
	const fileInputRef = useRef(null);
	const mediaStreamRef = useRef(null);
	const [audioEnabled, setAudioEnabled] = useState(true);
	const [videoEnabled, setVideoEnabled] = useState(true);
	const [inCall, setInCall] = useState(false);
	const [uploadedVideoURL, setUploadedVideoURL] = useState(null);
	const [participantName] = useState("Alex Chen");

	useEffect(() => {
		// auto-start preview/call so user sees local video immediately
		(async function autoStart() {
			try {
				await startCall();
			} catch (e) {
				// ignore errors, user can start manually
			}
		})();

		return () => {
			// cleanup on unmount
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

	async function startCall() {
		try {
			const stream = await navigator.mediaDevices.getUserMedia({
				video: { width: 1280, height: 720 },
				audio: true,
			});
			mediaStreamRef.current = stream;
			// show local user in the large video area; small preview is reserved for uploaded remote video
			if (remoteVideoRef.current) remoteVideoRef.current.srcObject = stream;
			if (localVideoRef.current && !uploadedVideoURL) {
				// keep small preview blank/upload placeholder until user provides a video
				localVideoRef.current.srcObject = null;
			}
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
		setInCall(false);
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
				<div className="detection-placeholder">
					<h3>AI Deepfake Detector</h3>
					<p className="muted">Right-side panel reserved for detection UI (coming later).</p>
				</div>
			</aside>
		</section>
	);
}

export default Deepfake;
