"use client";

import { useState } from "react";

type Report = {
  question: string;
  summary: string;
  key_findings: string[];
  opportunities: string[];
  recommendations: string[];
  sources: string[];
};

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  "http://localhost:8000";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [report, setReport] = useState<Report | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleResearch() {
    if (loading) {
      return;
    }

    if (!question.trim()) {
      setError("Please enter a research question.");
      return;
    }

    setLoading(true);
    setError("");
    setReport(null);

    try {
      const response = await fetch(`${API_URL}/research`, {
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

      const data = await response.json();
      setReport(data);
    } catch (err) {
      console.error(err);
      setError(
        "Unable to complete the research right now. Please try again."
      );
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(
    event: React.KeyboardEvent<HTMLTextAreaElement>
  ) {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      handleResearch();
    }
  }

  return (
    <main className="min-h-screen bg-[#f7f7f5] text-[#111111]">
      <div className="mx-auto max-w-6xl px-6 py-10 md:px-10 md:py-14">
        <header className="mb-16">
          <div className="mb-8 flex items-center justify-between">
            <div className="text-sm font-semibold tracking-tight">
              AI Marketing Research Agent
            </div>

            <div className="rounded-full border border-black/10 bg-white px-4 py-2 text-xs font-medium">
              Powered by LangGraph
            </div>
          </div>

          <div className="max-w-4xl">
            <p className="mb-5 text-sm font-medium uppercase tracking-[0.2em] text-black/50">
              AI-powered research
            </p>

            <h1 className="text-5xl font-semibold leading-[0.95] tracking-[-0.04em] md:text-7xl">
              Turn a marketing question into actionable research.
            </h1>

            <p className="mt-7 max-w-2xl text-lg leading-8 text-black/60">
              Ask a question and the agent searches the live web, analyzes
              the findings, and produces a structured marketing research
              report.
            </p>
          </div>
        </header>

        <section className="rounded-3xl border border-black/10 bg-white p-5 shadow-sm md:p-7">
          <div className="mb-4 flex items-center justify-between">
            <label
              htmlFor="question"
              className="text-sm font-semibold"
            >
              Research question
            </label>

            <span className="text-xs text-black/40">
              ⌘ + Enter to research
            </span>
          </div>

          <textarea
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="e.g. What are the biggest SEO opportunities for Nigerian beauty businesses in 2026?"
            className="min-h-[170px] w-full resize-none rounded-2xl border border-black/10 bg-[#fafafa] p-5 text-base outline-none transition focus:border-black/30 focus:bg-white"
          />

          <div className="mt-5 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <p className="text-sm text-black/45">
              The agent uses web search and an LLM to research your question.
            </p>

            <button
              onClick={handleResearch}
              disabled={loading}
              className="rounded-full bg-black px-7 py-3.5 text-sm font-semibold text-white transition hover:bg-black/80 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? "Researching..." : "Research"}
            </button>
          </div>

          {error && (
            <div className="mt-5 rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
              {error}
            </div>
          )}
        </section>

        {loading && (
          <section className="mt-10 rounded-3xl border border-black/10 bg-white p-8">
            <div className="flex items-center gap-3">
              <div className="h-3 w-3 animate-pulse rounded-full bg-black" />
              <p className="text-sm text-black/60">
                Searching the web and analyzing the research...
              </p>
            </div>
          </section>
        )}

        {report && !loading && (
          <section className="mt-10 space-y-6">
            <div className="rounded-3xl border border-black/10 bg-white p-7 md:p-9">
              <p className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-black/40">
                Research question
              </p>

              <h2 className="text-2xl font-semibold tracking-tight">
                {report.question}
              </h2>
            </div>

            <div className="rounded-3xl border border-black/10 bg-white p-7 md:p-9">
              <p className="mb-4 text-xs font-semibold uppercase tracking-[0.18em] text-black/40">
                Summary
              </p>

              <div className="max-w-4xl space-y-5 text-lg leading-8 text-black/70">
                {report.summary.split(/\n\s*\n/).map((paragraph, index) => (
                  <p key={index}>{paragraph}</p>
                ))}
              </div>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
              <ReportSection
                title="Key Findings"
                items={report.key_findings}
              />

              <ReportSection
                title="Opportunities"
                items={report.opportunities}
              />

              <ReportSection
                title="Recommendations"
                items={report.recommendations}
              />

              <div className="rounded-3xl border border-black/10 bg-white p-7">
                <p className="mb-5 text-xs font-semibold uppercase tracking-[0.18em] text-black/40">
                  Sources
                </p>

                <ol className="space-y-4">
                  {report.sources.map((source, index) => (
                    <li key={`${source}-${index}`} className="flex gap-3 text-sm leading-6">
                      <span className="shrink-0 font-semibold text-black/50">
                        [{index + 1}]
                      </span>
                      <a
                        href={source}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="min-w-0 text-black/60 transition hover:text-black"
                      >
                        <span className="block font-medium text-black/80">
                          {siteName(source)}
                        </span>
                        <span className="block break-all text-xs text-black/40 underline decoration-black/20 underline-offset-4">
                          {source}
                        </span>
                      </a>
                    </li>
                  ))}
                </ol>
              </div>
            </div>

            <button
              onClick={() => {
                setReport(null);
                setQuestion("");
              }}
              className="rounded-full border border-black/10 bg-white px-6 py-3 text-sm font-semibold transition hover:border-black/30"
            >
              Start new research
            </button>
          </section>
        )}
      </div>
    </main>
  );
}

function ReportSection({
  title,
  items,
}: {
  title: string;
  items: string[];
}) {
  return (
    <div className="rounded-3xl border border-black/10 bg-white p-7">
      <p className="mb-5 text-xs font-semibold uppercase tracking-[0.18em] text-black/40">
        {title}
      </p>

      <ol className="space-y-5">
        {items.map((item, index) => (
          <li
            key={`${item}-${index}`}
            className="flex gap-4 text-sm leading-7 text-black/70"
          >
            <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-black text-xs font-semibold text-white">
              {index + 1}
            </span>

            <ReportItem text={item} />
          </li>
        ))}
      </ol>
    </div>
  );
}


function siteName(url: string) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

function ReportItem({ text }: { text: string }) {
  const match = text.match(/^\[([^\]]+)\]\s*/);

  if (!match) {
    return <span>{text}</span>;
  }

  return (
    <span>
      <span className="mr-2 inline-block rounded-full bg-black/[0.06] px-2.5 py-0.5 text-[11px] font-medium uppercase tracking-wide text-black/50">
        {match[1]}
      </span>
      {text.slice(match[0].length)}
    </span>
  );
}
