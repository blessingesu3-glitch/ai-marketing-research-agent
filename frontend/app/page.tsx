"use client";

import { useMemo, useState, useSyncExternalStore } from "react";

type Report = {
  question: string;
  summary: string;
  key_findings: string[];
  opportunities: string[];
  recommendations: string[];
  sources: string[];
};

type CompetitorCard = {
  name: string;
  website: string;
  offer: string;
  pricing: string;
  target_customer: string;
  positioning: string;
  channels: string;
  sources: string[];
  note: string;
};

type Teardown = {
  summary: string;
  competitors: CompetitorCard[];
  gaps: string[];
  recommendations: string[];
  sources: string[];
};

type Mode = "research" | "teardown";

const MAX_COMPETITORS = 3;

/** An error whose message is safe and useful to show the user (e.g. "limit reached"). */
class RequestError extends Error {}

function splitCompetitors(text: string): string[] {
  return text
    .split(/[,;\n]| and /)
    .map((part) => part.trim())
    .filter(Boolean)
    .slice(0, MAX_COMPETITORS);
}

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  "http://localhost:8000";

type Profile = {
  business_name: string;
  industry: string;
  offering: string;
  location: string;
  target_customer: string;
  goal: string;
  competitors: string;
  price_point: string;
};

const EMPTY_PROFILE: Profile = {
  business_name: "",
  industry: "",
  offering: "",
  location: "",
  target_customer: "",
  goal: "",
  competitors: "",
  price_point: "",
};

const PROFILE_FIELDS: {
  key: keyof Profile;
  label: string;
  placeholder: string;
  maxLength: number;
  wide?: boolean;
}[] = [
  { key: "business_name", label: "Business name", placeholder: "e.g. BrightSmile Dental", maxLength: 120 },
  { key: "industry", label: "Industry", placeholder: "e.g. Dental clinic", maxLength: 120 },
  { key: "offering", label: "What you sell", placeholder: "e.g. Teeth whitening and braces", maxLength: 300, wide: true },
  { key: "location", label: "Location / markets", placeholder: "e.g. Lekki, Lagos, Nigeria", maxLength: 120 },
  { key: "price_point", label: "Price point", placeholder: "e.g. Mid-range, from ₦50,000", maxLength: 120 },
  { key: "target_customer", label: "Target customer", placeholder: "e.g. Young professionals aged 25-40", maxLength: 300, wide: true },
  { key: "goal", label: "Marketing goal", placeholder: "e.g. More bookings from Instagram", maxLength: 300, wide: true },
  { key: "competitors", label: "Known competitors", placeholder: "Names or websites, separated by commas", maxLength: 500, wide: true },
];

const PROFILE_STORAGE_KEY = "marketing-agent-business-profile";
const PROFILE_CHANGED_EVENT = "marketing-agent-profile-changed";

// Read the saved profile through React's external-store hook (no setState in effects).
function subscribeToProfile(onChange: () => void) {
  window.addEventListener("storage", onChange);
  window.addEventListener(PROFILE_CHANGED_EVENT, onChange);
  return () => {
    window.removeEventListener("storage", onChange);
    window.removeEventListener(PROFILE_CHANGED_EVENT, onChange);
  };
}

function readStoredProfileText(): string | null {
  try {
    return window.localStorage.getItem(PROFILE_STORAGE_KEY);
  } catch {
    return null;
  }
}

function parseProfile(raw: string | null): Profile | null {
  if (!raw) {
    return null;
  }
  try {
    const parsed = JSON.parse(raw) as Partial<Profile>;
    const profile = { ...EMPTY_PROFILE };
    for (const field of PROFILE_FIELDS) {
      const value = parsed[field.key];
      profile[field.key] = typeof value === "string" ? value.slice(0, field.maxLength) : "";
    }
    return hasProfileData(profile) ? profile : null;
  } catch {
    return null;
  }
}

function hasProfileData(profile: Profile | null): profile is Profile {
  return !!profile && Object.values(profile).some((value) => value.trim() !== "");
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [report, setReport] = useState<Report | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Competitor teardown mode
  const [mode, setMode] = useState<Mode>("research");
  const [competitorInputs, setCompetitorInputs] = useState<string[]>(["", "", ""]);
  const [focus, setFocus] = useState("");
  const [teardown, setTeardown] = useState<Teardown | null>(null);

  // Business profile: saved in this browser only, used only when the box is ticked.
  const storedProfileText = useSyncExternalStore(
    subscribeToProfile,
    readStoredProfileText,
    () => null, // server render: no profile yet
  );
  // Used only if the browser blocks storage (e.g. some private windows).
  const [sessionProfile, setSessionProfile] = useState<Profile | null>(null);
  const savedProfile = useMemo(
    () => parseProfile(storedProfileText) ?? sessionProfile,
    [storedProfileText, sessionProfile],
  );

  const [profileDraft, setProfileDraft] = useState<Profile>(EMPTY_PROFILE);
  const [profileOpen, setProfileOpen] = useState(false);
  const [profileNotice, setProfileNotice] = useState("");
  const [useProfile, setUseProfile] = useState(false);
  const [tailoredTo, setTailoredTo] = useState("");

  function toggleProfilePanel() {
    if (!profileOpen) {
      setProfileDraft(savedProfile ?? EMPTY_PROFILE);
    }
    setProfileOpen(!profileOpen);
    setProfileNotice("");
  }

  function saveProfile() {
    if (!hasProfileData(profileDraft)) {
      setProfileNotice("Fill in at least one field before saving.");
      return;
    }
    setProfileOpen(false);
    try {
      window.localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify(profileDraft));
      setSessionProfile(null);
      window.dispatchEvent(new Event(PROFILE_CHANGED_EVENT));
      setProfileNotice("Profile saved in this browser.");
    } catch {
      setSessionProfile(profileDraft);
      setProfileNotice("Profile saved for this visit only (your browser blocked storage).");
    }
  }

  function clearProfile() {
    setProfileDraft(EMPTY_PROFILE);
    setSessionProfile(null);
    setUseProfile(false);
    try {
      window.localStorage.removeItem(PROFILE_STORAGE_KEY);
      window.dispatchEvent(new Event(PROFILE_CHANGED_EVENT));
    } catch {
      // nothing stored, nothing to remove
    }
    setProfileNotice("Profile cleared.");
  }

  const profileName =
    savedProfile?.business_name.trim() || savedProfile?.industry.trim() || "my business";

  async function handleResearch() {
    if (loading) {
      return;
    }

    const competitors = competitorInputs.map((c) => c.trim()).filter(Boolean);

    if (mode === "research" && !question.trim()) {
      setError("Please enter a research question.");
      return;
    }

    if (mode === "teardown" && competitors.length === 0) {
      setError("Enter at least one competitor name or website.");
      return;
    }

    setLoading(true);
    setError("");
    setReport(null);
    setTeardown(null);

    const sendProfile = useProfile && hasProfileData(savedProfile);
    setTailoredTo(sendProfile ? profileName : "");
    const profilePart = sendProfile ? { profile: savedProfile } : {};

    try {
      const response = await fetch(
        `${API_URL}/${mode === "teardown" ? "teardown" : "research"}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(
            mode === "teardown"
              ? { competitors, ...(focus.trim() ? { focus: focus.trim() } : {}), ...profilePart }
              : { question: question.trim(), ...profilePart },
          ),
        },
      );

      if (!response.ok) {
        // 429 = usage limit reached; 422 = input problem. Both carry a readable message.
        let detail = "";
        try {
          const body = await response.json();
          detail = typeof body?.detail === "string" ? body.detail : "";
        } catch {
          // no readable message in the response
        }
        throw new RequestError(
          response.status === 429 || response.status === 422 ? detail : "",
        );
      }

      const data = await response.json();
      if (mode === "teardown") {
        setTeardown(data);
      } else {
        setReport(data);
      }
    } catch (err) {
      console.error(err);
      setError(
        err instanceof RequestError && err.message
          ? err.message
          : "Unable to complete the research right now. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  }

  function fillCompetitorsFromProfile() {
    const names = splitCompetitors(savedProfile?.competitors ?? "");
    setCompetitorInputs([0, 1, 2].map((i) => names[i] ?? ""));
  }

  function switchMode(next: Mode) {
    setMode(next);
    setError("");
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

        <section className="mb-6 rounded-3xl border border-black/10 bg-white p-5 md:p-7">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-sm font-semibold">Business profile</p>
              <p className="mt-1 text-sm text-black/45">
                {savedProfile
                  ? `Saved: ${profileName}. Tick "Tailor" below to use it for a question.`
                  : "Optional. Add your business once to get research tailored to it."}
              </p>
            </div>

            <button
              type="button"
              onClick={toggleProfilePanel}
              className="shrink-0 rounded-full border border-black/10 bg-white px-5 py-2.5 text-sm font-semibold transition hover:border-black/30"
            >
              {profileOpen ? "Close" : savedProfile ? "Edit profile" : "Add profile"}
            </button>
          </div>

          {profileOpen && (
            <div className="mt-6">
              <div className="grid gap-4 md:grid-cols-2">
                {PROFILE_FIELDS.map((field) => (
                  <label
                    key={field.key}
                    className={`block ${field.wide ? "md:col-span-2" : ""}`}
                  >
                    <span className="mb-1.5 block text-xs font-semibold uppercase tracking-[0.12em] text-black/45">
                      {field.label}
                    </span>
                    <input
                      type="text"
                      value={profileDraft[field.key]}
                      maxLength={field.maxLength}
                      placeholder={field.placeholder}
                      onChange={(event) =>
                        setProfileDraft((draft) => ({
                          ...draft,
                          [field.key]: event.target.value,
                        }))
                      }
                      className="w-full rounded-xl border border-black/10 bg-[#fafafa] px-4 py-3 text-sm outline-none transition focus:border-black/30 focus:bg-white"
                    />
                  </label>
                ))}
              </div>

              <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <p className="text-xs text-black/40">
                  Saved only in this browser. It is sent with a question only when you tick
                  &ldquo;Tailor&rdquo;.
                </p>
                <div className="flex gap-3">
                  {savedProfile && (
                    <button
                      type="button"
                      onClick={clearProfile}
                      className="rounded-full border border-black/10 px-5 py-2.5 text-sm font-semibold text-black/60 transition hover:border-black/30"
                    >
                      Clear
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={saveProfile}
                    className="rounded-full bg-black px-6 py-2.5 text-sm font-semibold text-white transition hover:bg-black/80"
                  >
                    Save profile
                  </button>
                </div>
              </div>
            </div>
          )}

          {profileNotice && (
            <p className="mt-4 text-sm text-black/55">{profileNotice}</p>
          )}
        </section>

        <section className="rounded-3xl border border-black/10 bg-white p-5 shadow-sm md:p-7">
          <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div
              role="tablist"
              aria-label="What do you want to do?"
              className="inline-flex self-start rounded-full bg-black/[0.05] p-1"
            >
              {(
                [
                  ["research", "Research question"],
                  ["teardown", "Competitor teardown"],
                ] as [Mode, string][]
              ).map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  role="tab"
                  aria-selected={mode === value}
                  onClick={() => switchMode(value)}
                  className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                    mode === value ? "bg-white text-black shadow-sm" : "text-black/50 hover:text-black"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>

            {mode === "research" && (
              <span className="text-xs text-black/40">
                ⌘ + Enter to research
              </span>
            )}
          </div>

          {mode === "research" ? (
            <>
              <label htmlFor="question" className="sr-only">
                Research question
              </label>
              <textarea
                id="question"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={handleKeyDown}
                maxLength={2000}
                placeholder="e.g. What are the biggest SEO opportunities for Nigerian beauty businesses in 2026?"
                className="min-h-[170px] w-full resize-none rounded-2xl border border-black/10 bg-[#fafafa] p-5 text-base outline-none transition focus:border-black/30 focus:bg-white"
              />
            </>
          ) : (
            <div>
              <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <p className="text-sm text-black/55">
                  Up to {MAX_COMPETITORS} competitors. Paste a website if you know it, or just the name.
                </p>
                {savedProfile?.competitors.trim() && (
                  <button
                    type="button"
                    onClick={fillCompetitorsFromProfile}
                    className="self-start text-sm font-semibold text-black/70 underline decoration-black/20 underline-offset-4 hover:text-black"
                  >
                    Use competitors from my profile
                  </button>
                )}
              </div>
              <div className="grid gap-3 md:grid-cols-3">
                {competitorInputs.map((value, index) => (
                  <input
                    key={index}
                    type="text"
                    value={value}
                    maxLength={200}
                    aria-label={`Competitor ${index + 1}`}
                    placeholder={["e.g. acme.com", "e.g. Beta Clinic", "Optional third competitor"][index]}
                    onChange={(event) =>
                      setCompetitorInputs((inputs) =>
                        inputs.map((v, i) => (i === index ? event.target.value : v)),
                      )
                    }
                    className="w-full rounded-xl border border-black/10 bg-[#fafafa] px-4 py-3 text-sm outline-none transition focus:border-black/30 focus:bg-white"
                  />
                ))}
              </div>
              <input
                type="text"
                value={focus}
                maxLength={500}
                aria-label="What do you most want to know? (optional)"
                placeholder="Optional: what do you most want to know? e.g. how they price their packages"
                onChange={(event) => setFocus(event.target.value)}
                className="mt-3 w-full rounded-xl border border-black/10 bg-[#fafafa] px-4 py-3 text-sm outline-none transition focus:border-black/30 focus:bg-white"
              />
            </div>
          )}

          <div className="mt-5 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <label
              className={`flex items-center gap-3 text-sm ${
                savedProfile ? "cursor-pointer text-black/70" : "cursor-not-allowed text-black/35"
              }`}
            >
              <input
                type="checkbox"
                checked={useProfile && !!savedProfile}
                disabled={!savedProfile}
                onChange={(event) => setUseProfile(event.target.checked)}
                className="h-4 w-4 accent-black"
              />
              {savedProfile
                ? `Tailor this research to ${profileName}`
                : "Tailor to my business (add a profile first)"}
            </label>

            <button
              onClick={handleResearch}
              disabled={loading}
              className="rounded-full bg-black px-7 py-3.5 text-sm font-semibold text-white transition hover:bg-black/80 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading
                ? mode === "teardown"
                  ? "Comparing..."
                  : "Researching..."
                : mode === "teardown"
                  ? "Compare competitors"
                  : "Research"}
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
                {mode === "teardown"
                  ? "Finding each competitor's website and reading their pages..."
                  : "Searching the web and analyzing the research..."}
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

              {tailoredTo && (
                <p className="mt-3 text-sm text-black/50">
                  Tailored to: {tailoredTo}
                </p>
              )}
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

        {teardown && !loading && (
          <TeardownView
            teardown={teardown}
            tailoredTo={tailoredTo}
            onReset={() => {
              setTeardown(null);
              setCompetitorInputs(["", "", ""]);
              setFocus("");
            }}
          />
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

const TEARDOWN_ROWS: { key: keyof CompetitorCard; label: string }[] = [
  { key: "offer", label: "What they sell" },
  { key: "pricing", label: "Published pricing" },
  { key: "target_customer", label: "Target customer" },
  { key: "positioning", label: "Main message" },
  { key: "channels", label: "Channels" },
];

function TeardownView({
  teardown,
  tailoredTo,
  onReset,
}: {
  teardown: Teardown;
  tailoredTo: string;
  onReset: () => void;
}) {
  return (
    <section className="mt-10 space-y-6">
      <div className="rounded-3xl border border-black/10 bg-white p-7 md:p-9">
        <p className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-black/40">
          Competitor teardown
        </p>
        <h2 className="text-2xl font-semibold tracking-tight">
          {teardown.competitors.map((c) => c.name).join(" vs ")}
        </h2>
        {tailoredTo && (
          <p className="mt-3 text-sm text-black/50">Tailored to: {tailoredTo}</p>
        )}
        <div className="mt-6 max-w-4xl space-y-5 text-lg leading-8 text-black/70">
          {teardown.summary.split(/\n\s*\n/).map((paragraph, index) => (
            <p key={index}>{paragraph}</p>
          ))}
        </div>
      </div>

      <div className="rounded-3xl border border-black/10 bg-white p-5 md:p-7">
        <p className="mb-5 text-xs font-semibold uppercase tracking-[0.18em] text-black/40">
          Side by side
        </p>
        {/* Phones: one card per competitor */}
        <div className="space-y-4 md:hidden">
          {teardown.competitors.map((c) => (
            <div key={c.name} className="rounded-2xl border border-black/[0.07] p-4">
              <p className="text-base font-semibold">{c.name}</p>
              <p className="mb-3 text-xs text-black/45">{c.website || "Website not found"}</p>
              <dl className="space-y-3 text-sm">
                {TEARDOWN_ROWS.map((row) => {
                  const value = String(c[row.key]);
                  return (
                    <div key={row.key}>
                      <dt className="text-[11px] font-semibold uppercase tracking-[0.12em] text-black/40">
                        {row.label}
                      </dt>
                      <dd className={value.startsWith("Not found") ? "italic text-black/35" : "text-black/75"}>
                        {value}
                      </dd>
                    </div>
                  );
                })}
                {c.note && <p className="text-xs leading-5 text-black/45">{c.note}</p>}
              </dl>
            </div>
          ))}
        </div>

        {/* Larger screens: side-by-side table */}
        <div className="hidden overflow-x-auto md:block">
          <table className="w-full min-w-[640px] border-collapse text-left text-sm">
            <thead>
              <tr>
                <th className="w-40 pb-4 pr-4 align-bottom text-xs font-semibold uppercase tracking-[0.12em] text-black/40" />
                {teardown.competitors.map((c) => (
                  <th key={c.name} className="pb-4 pr-4 align-bottom">
                    <span className="block text-base font-semibold text-black">{c.name}</span>
                    {c.website ? (
                      <a
                        href={`https://${c.website}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs font-normal text-black/45 underline decoration-black/20 underline-offset-4 hover:text-black"
                      >
                        {c.website}
                      </a>
                    ) : (
                      <span className="text-xs font-normal text-black/40">Website not found</span>
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {TEARDOWN_ROWS.map((row) => (
                <tr key={row.key} className="border-t border-black/[0.07]">
                  <th className="py-4 pr-4 align-top text-xs font-semibold uppercase tracking-[0.12em] text-black/45">
                    {row.label}
                  </th>
                  {teardown.competitors.map((c) => {
                    const value = String(c[row.key]);
                    const missing = value.startsWith("Not found");
                    return (
                      <td
                        key={c.name}
                        className={`py-4 pr-4 align-top leading-6 ${missing ? "text-black/35 italic" : "text-black/75"}`}
                      >
                        {value}
                      </td>
                    );
                  })}
                </tr>
              ))}
              {teardown.competitors.some((c) => c.note) && (
                <tr className="border-t border-black/[0.07]">
                  <th className="py-4 pr-4 align-top text-xs font-semibold uppercase tracking-[0.12em] text-black/45">
                    Notes
                  </th>
                  {teardown.competitors.map((c) => (
                    <td key={c.name} className="py-4 pr-4 align-top text-xs leading-5 text-black/45">
                      {c.note}
                    </td>
                  ))}
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <ReportSection title="Openings you can use" items={teardown.gaps} />
        <ReportSection title="Recommendations" items={teardown.recommendations} />
      </div>

      {teardown.sources.length > 0 && (
        <div className="rounded-3xl border border-black/10 bg-white p-7">
          <p className="mb-5 text-xs font-semibold uppercase tracking-[0.18em] text-black/40">
            Pages read
          </p>
          <ol className="space-y-3">
            {teardown.sources.map((source, index) => (
              <li key={`${source}-${index}`} className="flex gap-3 text-sm leading-6">
                <span className="shrink-0 font-semibold text-black/50">[{index + 1}]</span>
                <a
                  href={source}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="min-w-0 break-all text-black/60 underline decoration-black/20 underline-offset-4 hover:text-black"
                >
                  {source}
                </a>
              </li>
            ))}
          </ol>
        </div>
      )}

      <button
        onClick={onReset}
        className="rounded-full border border-black/10 bg-white px-6 py-3 text-sm font-semibold transition hover:border-black/30"
      >
        Start a new teardown
      </button>
    </section>
  );
}
