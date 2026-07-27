// TextExtractor.jsx
//
// Presents the "Text Extractor" tool tab (SMS/message). All values come from
// UserDashboard.jsx, and every user action (typing, clicking "Check Message") is reported via setMessage/onCheckMessage.
function TextExtractor({
	message,               // the raw text currently in the textarea
	setMessage,            // updates `message` as the user types
	result,                // "Scam Message" | "Legitimate Message" | "" (empty = no check run yet)
	confidence,            // model's confidence in `result`, as a 0-1 float, or null if unavailable
	explanation,           // Gemini-generated explanation text
	explanationAvailable,  // false if the explanation call failed/was skipped - still show `explanation`'s fallback message, just styled differently
	translated,            // true only if the backend detected the input was NOT English and translated it
	translatedText,        // the English translation (only when `translated` is true)
	detectedLanguage,      // e.g. "Chinese", "Malay" - the language Gemini detected, or null
	loading,               // true while the /detect-text request is in flight
	error,                 // network/request-level error message
	onCheckMessage,        // handler for the "Check Message" button - triggers the API call in the parent
}) {
	return (
		<section>
			<p className="subtle">Paste any suspicious SMS or message text (English or another language) and scan it with your trained model.</p>

			{/* Controlled textarea since SMS/message bodies can run several sentences long - a single-line input would clip
			    the message as the user types. */}
			<textarea
				placeholder="Paste the message here, e.g. 'WINNER!! You have been selected to claim a $900 prize...'"
				value={message}
				onChange={(e) => setMessage(e.target.value)}
				rows={5}
			/>

			<div className="text-detector-actions">
				{/* Disabled while a request is in flight and while the textarea is empty/whitespace-only. */}
				<button type="button" onClick={onCheckMessage} disabled={loading || !message.trim()}>
					{loading ? "Scanning..." : "Check Message"}
				</button>
			</div>

			{/* Only for hard failures. A message that gets classified but comes back
			    "Scam" is not an error and is a normal successful result, so it's
			    rendered separately below via the result-pill section. */}
			{error && <p className="error-text">{error}</p>}

			{result && (
				// .danger (red) for scam verdicts, .safe (green) otherwise.
				<section className={`result-pill ${result.includes("Scam") ? "danger" : "safe"}`}>
					<p>{result}</p>
					{confidence !== null && confidence !== undefined && (
						<p className="confidence-text">Confidence: {(confidence * 100).toFixed(0)}%</p>
					)}
				</section>
			)}

			{/* Renders only when the backend translates. For English input `translated` stays false and this whole block
			    is skipped.*/}
			{result && translated && translatedText && (
				<section className="translation-box">
					<p className="translation-label">
						Translated{detectedLanguage ? ` from ${detectedLanguage}` : ""} to English
					</p>
					<p>{translatedText}</p>
				</section>
			)}

			{result && explanation && (
				// .unavailable is a muted/grey styling variant used when Gemini's
				// explanation call failed - `explanation` still holds a
				// human-readable fallback string, so there is
				// something to display, just marked as "this
				// wasn't AI-generated".
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

export default TextExtractor;
