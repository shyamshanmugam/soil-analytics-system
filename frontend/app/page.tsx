"use client";

import { FormEvent, useState } from "react";

const fields = [
  ["nitrogen", "Nitrogen (mg/kg)"],
  ["phosphorus", "Phosphorus (mg/kg)"],
  ["potassium", "Potassium (mg/kg)"],
  ["ph", "pH"],
  ["organic_matter", "Organic matter (%)"],
  ["moisture", "Moisture (%)"],
] as const;

type Result = {
  health_score: number;
  deficiencies: string[];
  degradation_risks: string[];
  recommendations: { title: string; guidance: string }[];
  crop_suitability: { crop: string; suitable: boolean }[];
  image_classification: { status: string; explanation: string } | null;
};

export default function Home() {
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setResult(null);
    const form = new FormData(event.currentTarget);
    const body = Object.fromEntries(fields.map(([key]) => [key, Number(form.get(key))]));
    const image = form.get("image");
    const hasImage = image instanceof File && image.size > 0;
    const requestBody = hasImage ? form : JSON.stringify(body);
    try {
      setBusy(true);
      const response = await fetch(
        hasImage ? "http://127.0.0.1:8000/api/v1/analyze-image" : "http://127.0.0.1:8000/api/v1/analyze",
        {
        method: "POST",
        ...(hasImage ? {} : { headers: { "Content-Type": "application/json" } }),
        body: requestBody,
      });
      if (!response.ok) throw new Error("The soil analysis could not be completed.");
      setResult(await response.json());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unexpected error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <section className="hero">
        <p className="eyebrow">SOIL ANALYTICS</p>
        <h1>Understand your soil.</h1>
        <p>Enter laboratory values to receive a transparent health baseline and improvement guidance.</p>
      </section>
      <form onSubmit={submit} className="card">
        <h2>Laboratory soil test</h2>
        <div className="grid">
          {fields.map(([key, label]) => (
            <label key={key}>{label}<input required name={key} type="number" step="any" min="0" /></label>
          ))}
        </div>
        <label className="upload">Soil image (optional)
          <input name="image" type="file" accept="image/jpeg,image/png" capture="environment" />
        </label>
        <button disabled={busy} type="submit">{busy ? "Analyzing..." : "Analyze soil"}</button>
        {error && <p className="error">{error}</p>}
      </form>
      {result && <section className="card result">
        <h2>Analysis result</h2>
        <strong>{result.health_score}/100</strong><p>Baseline health score</p>
        <p><b>Deficiencies:</b> {result.deficiencies.length ? result.deficiencies.join(", ") : "None detected"}</p>
        <p><b>Risks:</b> {result.degradation_risks.length ? result.degradation_risks.join(", ") : "None detected"}</p>
        <h3>Recommendations</h3>
        {result.recommendations.map((item) => <p key={item.title}><b>{item.title}:</b> {item.guidance}</p>)}
        <h3>Crop suitability</h3>
        <p>{result.crop_suitability.map((item) => `${item.crop}: ${item.suitable ? "suitable" : "review conditions"}`).join(" · ")}</p>
        {result.image_classification && <p className="note"><b>Image model:</b> {result.image_classification.status}. {result.image_classification.explanation}</p>}
      </section>}
      <p className="note">This baseline is not a trained model or a substitute for agronomic advice.</p>
    </main>
  );
}
