function Phishing({ url, setUrl, result, confidence, explanation, explanationAvailable, loading, error, onCheckURL }) {
	return (
		<section>
			<p className="subtle">Paste any link and scan it with your trained model.</p>

			<div className="detector-row">
				<input
					placeholder="https://example.com"
					value={url}
					onChange={(e) => setUrl(e.target.value)}
				/>

				<button type="button" onClick={onCheckURL} disabled={loading || !url.trim()}>
					{loading ? "Scanning..." : "Check URL"}
				</button>
			</div>

			{error && <p className="error-text">{error}</p>}

			{result && (
				<section className={`result-pill ${result.includes("Phishing") ? "danger" : "safe"}`}>
					<p>{result}</p>
					{confidence !== null && confidence !== undefined && (
						<p className="confidence-text">Confidence: {(confidence * 100).toFixed(0)}%</p>
					)}
				</section>
			)}

			{result && explanation && (
				<section className={`ai-explanation ${explanationAvailable ? "" : "unavailable"}`}>
					<p className="ai-explanation-label">
						{explanationAvailable ? "Why this result" : "AI explanation unavailable"}
					</p>
					<p>{explanation}</p>
				</section>
			)}
		</section>
	);
}

export default Phishing;