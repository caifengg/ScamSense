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
	onUploadScreenshot,    // handler(file) for the screenshot <input type="file"> - triggers /extract-text-from-image in the parent
	extracting,            // true while the /extract-text-from-image request is in flight
	extractError,          // error message from the screenshot extraction step only

	// ── Flag-as-wrong (for the currently-shown result only) ──────────────────
	checkId,               // id of the saved text_checks row for the current result, or null if it couldn't be saved
	flagged,               // true once this specific check has been flagged - disables the button
	flagging,              // true while the flag request is in flight
	flagError,             // error message from the flag step only
	onFlagWrong,           // handler for the "Flag as wrong" button

	// ── Recent-checks history ─────────────────────────────────────────────────
	history = [],           // array of {id, message, result, confidence, flagged, created_at}, most recent first
	historyLoading,         // true while /text-checks is being fetched
	historyError,           // network/request-level error for the history list only
	historyDeletingId,      // id of the entry currently being deleted, or null
	onDeleteHistoryEntry,   // handler(id) for a history entry's delete button
}) {
	return (
		<section>
			<p className="subtle">Paste any suspicious SMS or message text (English or another language), or upload a screenshot, and scan it with your trained model.</p>

			{/* Controlled textarea since SMS/message bodies can run several sentences long - a single-line input would clip
			    the message as the user types. Also the destination for text transcribed from an uploaded screenshot,
			    so the user can review/correct it before scanning. */}
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

				{/* Upload a screenshot instead of typing. Selecting a file immediately
				    calls onUploadScreenshot(file); the input is reset by the parent's
				    handler completing, so choosing the same file twice in a row still fires
				    a change event. Disabled while a request of either kind is in flight. */}
				<label className="screenshot-upload-label">
					{extracting ? "Reading screenshot..." : "Upload Screenshot"}
					<input
						type="file"
						accept="image/*"
						onChange={(e) => {
							const file = e.target.files && e.target.files[0];
							if (file) onUploadScreenshot(file);
							e.target.value = "";
						}}
						disabled={loading || extracting}
						hidden
					/>
				</label>
			</div>

			{/* Screenshot transcription failures (unreadable image, API error) are
			    kept separate from `error` (the /detect-text network error) so the
			    two don't stomp on each other's messages. */}
			{extractError && <p className="error-text">{extractError}</p>}

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

					{/* Only shown when the check was actually saved (checkId != null) -
					    e.g. hidden entirely for a logged-out session. Once flagged==true
					    the button is disabled and relabelled rather than removed, so the
					    user gets confirmation their flag registered. */}
					{checkId && (
						<div className="flag-wrong-row">
							<button
								type="button"
								className="flag-wrong-button"
								onClick={onFlagWrong}
								disabled={flagging || flagged}
							>
								{flagged ? "Flagged as wrong" : flagging ? "Flagging..." : "This looks wrong"}
							</button>
							{flagError && <p className="error-text">{flagError}</p>}
						</div>
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

			{/* Recent-checks history - only meaningful for a logged-in user, since
			    the parent only fetches it when a userId exists. `history` stays an
			    empty array otherwise, so this section simply doesn't render anything
			    below the heading. */}
			<section className="text-history">
				<p className="text-history-label">Recent Checks</p>

				{historyLoading && <p className="subtle">Loading history...</p>}
				{historyError && <p className="error-text">{historyError}</p>}

				{!historyLoading && !historyError && history.length === 0 && (
					<p className="subtle">No checks saved yet.</p>
				)}

				{history.length > 0 && (
					<ul className="text-history-list">
						{history.map((entry) => (
							<li key={entry.id} className={`text-history-item ${entry.result.includes("Scam") ? "danger" : "safe"}`}>
								<div className="text-history-item-main">
									<p className="text-history-message">{entry.message}</p>
									<p className="text-history-meta">
										{entry.result}
										{entry.confidence !== null && entry.confidence !== undefined &&
											` · ${(entry.confidence * 100).toFixed(0)}% confidence`}
										{entry.flagged && " · Flagged as wrong"}
									</p>
								</div>
								<button
									type="button"
									className="text-history-delete"
									onClick={() => onDeleteHistoryEntry(entry.id)}
									disabled={historyDeletingId === entry.id}
								>
									{historyDeletingId === entry.id ? "Deleting..." : "Delete"}
								</button>
							</li>
						))}
					</ul>
				)}
			</section>
		</section>
	);
}

export default TextExtractor;
