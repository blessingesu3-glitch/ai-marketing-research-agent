"use client";

import { FormEvent, useState } from "react";

type ResearchReport = {
  question: string;
  summary: string;
  key_findings: string[];
  opportunities: string[];
  recommendations: string[];
  sources: string[];
};

export default function Home() {
  const [question, setQuestion] = useState("");
  const [report, setReport] = useState<ResearchReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleResearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!question.trim()) {
      setError("Enter a research question first.");
      return;
    }

    setLoading(true);
    setError("");
    setReport(null);

    try {
      const response = await fetch("http://localhost:8000/research", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: question.trim(),
        }),
      });

      if (!response.ok) {
        throw new Error("The research request failed.");
      }

      const data: ResearchReport = await response.json();

      setReport(data);
    } catch (err) {
      console.error(err);
      setError(
        "We couldn't connect to the research agent. Make sure the FastAPI server is running."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-[#f5f5f2] text-[#111111]">
      <div className="mx-auto max-w-6xl px-6 py-8 md:px-10 lg:px-12">
        {/* Header */}
        <header className="flex items-center justify-between border-b border-black/10 pb-6">
          <div>
            <p className="text-sm font-semibold tracking-[0.18em] uppercase">
              Research Agent
            </p>
            <p className="mt-1 text-sm text-black/50">
              AI-powered marketing intelligence
            </p>
          </div>

          <div className="flex items-center gap-2 text-sm text-black/60">
            <span className="h-2 w-2 rounded-full bg-green-500" />
            Agent online
          </div>
        </header>

        {/* Hero */}
        <section className="py-20 md:py-28">
          <div className="max-w-4xl">
            <p className="mb-5 text-sm font-semibold tracking-[0.2em] text-black/45 uppercase">
              LangGraph + Tavily + OpenAI
            </p>

            <h1 className="max-w-4xl text-5xl font-medium tracking-[-0.04em] md:text-7xl">
              Turn a question into
              <span className="block text-black/45">
                useful marketing intelligence.
              </span>
            </h1>

            <p className="mt-7 max-w-2xl text-lg leading-8 text-black/60">
              Ask a question. The agent searches the web, analyzes the
              evidence, and turns the research into a structured report.
            </p>
          </div>

          {/* Research Form */}
          <form
            onSubmit={handleResearch}
            className="mt-12 max-w-4xl"
          >
            <div className="rounded-2xl border border-black/10 bg-white p-3 shadow-sm">
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="What do you want to research?"
                rows={4}
                className="w-full resize-none border-0 bg-transparent px-4 py-3 text-lg outline-none placeholder:text-black/30"
              />

              <div className="flex flex-col gap-3 border-t border-black/10 pt-3 sm:flex-row sm:items-center sm:justify-between">
                <p className="px-4 text-sm text-black/40">
                  Try: SEO opportunities for Nigerian beauty businesses
                </p>

                <button
                  type="submit"
                  disabled={loading}
                  className="rounded-xl bg-black px-6 py-3 text-sm font-medium text-white transition hover:bg-black/80 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {loading ? "Researching..." : "Research →"}
                </button>
              </div>
            </div>
          </form>

          {/* Error */}
          {error && (
            <div className="mt-5 max-w-4xl rounded-xl border border-red-200 bg-red-50 px-5 py-4 text-sm text-red-700">
              {error}
            </div>
          )}
        </section>

        {/* Loading */}
        {loading && (
          <section className="border-t border-black/10 py-12">
            <div className="flex items-center gap-4">
              <div className="h-3 w-3 animate-pulse rounded-full bg-black" />

              <div>
                <p className="font-medium">
                  Researching your question...
                </p>

                <p className="mt-1 text-sm text-black/50">
                  Searching the web and analyzing the findings.
                </p>
              </div>
            </div>
          </section>
        )}

        {/* Report */}
        {report && !loading && (
          <section className="border-t border-black/10 pb-20">
            {/* Report Header */}
            <div className="py-12">
              <p className="text-sm font-semibold tracking-[0.18em] text-black/40 uppercase">
                Research Report
              </p>

              <h2 className="mt-4 max-w-4xl text-3xl font-medium tracking-[-0.03em] md:text-5xl">
                {report.question}
              </h2>
            </div>

            {/* Summary */}
            <div className="grid gap-10 border-t border-black/10 py-10 md:grid-cols-[180px_1fr]">
              <p className="text-sm font-semibold tracking-[0.12em] text-black/40 uppercase">
                Summary
              </p>

              <p className="max-w-3xl text-xl leading-9">
                {report.summary}
              </p>
            </div>

            {/* Key Findings */}
            <div className="grid gap-10 border-t border-black/10 py-10 md:grid-cols-[180px_1fr]">
              <p className="text-sm font-semibold tracking-[0.12em] text-black/40 uppercase">
                Key Findings
              </p>

              <div className="space-y-6">
                {report.key_findings.map((finding, index) => (
                  <div
                    key={index}
                    className="grid gap-4 sm:grid-cols-[40px_1fr]"
                  >
                    <span className="text-sm text-black/35">
                      {String(index + 1).padStart(2, "0")}
                    </span>

                    <p className="text-lg leading-8 text-black/75">
                      {finding}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Opportunities */}
            <div className="grid gap-10 border-t border-black/10 py-10 md:grid-cols-[180px_1fr]">
              <p className="text-sm font-semibold tracking-[0.12em] text-black/40 uppercase">
                Opportunities
              </p>

              <div className="space-y-5">
                {report.opportunities.map((opportunity, index) => (
                  <div
                    key={index}
                    className="rounded-2xl border border-black/10 bg-white p-6"
                  >
                    <p className="text-sm text-black/35">
                      Opportunity {String(index + 1).padStart(2, "0")}
                    </p>

                    <p className="mt-3 text-lg leading-8">
                      {opportunity}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Recommendations */}
            <div className="grid gap-10 border-t border-black/10 py-10 md:grid-cols-[180px_1fr]">
              <p className="text-sm font-semibold tracking-[0.12em] text-black/40 uppercase">
                Recommendations
              </p>

              <div className="space-y-5">
                {report.recommendations.map((recommendation, index) => (
                  <div
                    key={index}
                    className="flex gap-5 border-b border-black/10 pb-5"
                  >
                    <span className="text-sm text-black/35">
                      {String(index + 1).padStart(2, "0")}
                    </span>

                    <p className="text-lg leading-8 text-black/75">
                      {recommendation}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Sources */}
            <div className="grid gap-10 border-t border-black/10 py-10 md:grid-cols-[180px_1fr]">
              <p className="text-sm font-semibold tracking-[0.12em] text-black/40 uppercase">
                Sources
              </p>

              <div className="space-y-3">
                {report.sources.map((source, index) => (
                  <a
                    key={index}
                    href={source}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block break-all text-sm text-black/55 underline decoration-black/20 underline-offset-4 hover:text-black"
                  >
                    {source}
                  </a>
                ))}
              </div>
            </div>

            {/* New Research */}
            <div className="border-t border-black/10 pt-10">
              <button
                onClick={() => {
                  setReport(null);
                  setQuestion("");
                  window.scrollTo({
                    top: 0,
                    behavior: "smooth",
                  });
                }}
                className="rounded-xl border border-black/15 bg-white px-6 py-3 text-sm font-medium transition hover:bg-black hover:text-white"
              >
                Start new research
              </button>
            </div>
          </section>
        )}

        {/* Footer */}
        <footer className="border-t border-black/10 py-8 text-sm text-black/40">
          Built with LangGraph, Tavily and OpenAI.
        </footer>
      </div>
    </main>
  );
}
