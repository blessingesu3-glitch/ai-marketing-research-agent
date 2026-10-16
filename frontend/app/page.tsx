"use client";

import { useMemo, useRef, useState, useSyncExternalStore } from "react";

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

/** Starter questions that work for any sector. Uses the saved profile where it can. */
function starterQuestions(profile: Profile | null): { label: string; question: string }[] {
  const lower = (text: string) => (/^[A-Z][a-z]/.test(text) ? text[0].toLowerCase() + text.slice(1) : text);
  const offering = lower(profile?.offering.trim() || "[what you sell]");
  const industry = lower(profile?.industry.trim() || "[type of business]");
  const business = profile?.business_name.trim() || `a ${industry}`;
  const place = profile?.location.trim() || "[your city or market]";
  return [
    {
      label: "Competitor prices",
      question: `Who are the main competitors selling ${offering} in ${place}, and what do they charge?`,
    },
    {
      label: "Customer segments",
      question: `Which customer groups should ${business} in ${place} target first, and why?`,
    },
    {
      label: "Keyword ideas",
      question: `What do people search for online when looking for ${offering} in ${place}? Which keywords should we target?`,
    },
    {
      label: "Market entry",
      question: `Is there room for a new ${industry} in ${place}? What do demand and competition look like?`,
    },
  ];
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [report, setReport] = useState<Report | null>(null);
  const [reportDate, setReportDate] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const questionRef = useRef<HTMLTextAreaElement>(null);
  const deskRef = useRef<HTMLElement>(null);

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

  function openProfileFromNav() {
    if (!profileOpen) {
      toggleProfilePanel();
    }
    deskRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
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
      setReportDate(
        new Date().toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" }),
      );
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

  function applyStarter(text: string) {
    setMode("research");
    setError("");
    setQuestion(text);
    deskRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    // Focus after React renders the research textarea.
    window.setTimeout(() => {
      const box = questionRef.current;
      if (box) {
        box.focus({ preventScroll: true });
        const bracket = text.indexOf("[");
        if (bracket >= 0) {
          box.setSelectionRange(bracket, text.indexOf("]", bracket) + 1);
        }
      }
    }, 0);
  }

  function handleKeyDown(
    event: React.KeyboardEvent<HTMLTextAreaElement>
  ) {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      handleResearch();
    }
  }

  return (
    <div className="min-h-screen bg-page text-ink">
      {/* Masthead */}
      <div className="bg-ink text-page">
        <header className="mx-auto flex min-h-[72px] max-w-[1180px] flex-wrap items-center justify-between gap-x-8 gap-y-3 px-5 py-4 md:px-10">
          <a href="#top" className="flex items-baseline gap-3.5 text-page no-underline">
            <span className="font-serif text-[22px] font-semibold tracking-[0.01em] md:text-[28px]">
              AI Marketing Research Agent
            </span>
          </a>
          <nav aria-label="Main" className="flex flex-wrap items-center gap-x-8 gap-y-2 text-xs uppercase tracking-[0.2em]">
            <a href="#method" className="text-[#D9CBB4] transition hover:text-page">
              Method
            </a>
            <button
              type="button"
              onClick={openProfileFromNav}
              className="min-h-11 border-b border-gold pb-0.5 uppercase tracking-[0.2em] text-page"
            >
              Business profile
            </button>
          </nav>
        </header>
      </div>

      <main id="top" className="mx-auto w-full max-w-[1180px] px-5 pb-24 md:px-10">
        {/* Hero */}
        <section className="flex flex-wrap gap-x-[72px] gap-y-10 border-b border-rule pb-12 pt-14 md:pb-14 md:pt-[88px]">
          <div className="min-w-0 flex-[999_1_520px]">
            <p className="eyebrow mb-6 text-gold-deep md:mb-7">Research you can verify</p>
            <h1 className="font-serif text-[52px] font-medium leading-[0.98] tracking-[-0.015em] sm:text-[64px] md:text-[84px]">
              Know your market.
              <br />
              <span className="italic text-gold-text">Prove</span> every claim.
            </h1>
          </div>
          <div className="flex min-w-0 flex-[1_1_300px] flex-col justify-end gap-7">
            <p className="text-[17px] leading-[1.7] text-muted">
              Competitors, prices, customers and keywords, researched across the live web.
              Every finding is tied to its source and graded for how far it can be trusted.
            </p>
            <ol className="border-t border-rule">
              {[
                ["i.", "Research a question"],
                ["ii.", "Tear down up to three competitors"],
                ["iii.", "Turn findings into briefs, ads and plans"],
              ].map(([numeral, text]) => (
                <li key={numeral} className="flex gap-[18px] border-b border-rule-soft py-3 text-sm text-muted">
                  <span className="w-6 font-serif text-lg italic text-gold-text">{numeral}</span>
                  {text}
                </li>
              ))}
            </ol>
          </div>
        </section>

        {/* The writing surface */}
        <section ref={deskRef} aria-label="Start your research" className="scroll-mt-6 pb-6 pt-12 md:pt-14">
          <div className="flex flex-wrap items-baseline justify-between gap-4">
            <div role="tablist" aria-label="What do you want to do?" className="flex flex-wrap gap-x-6 gap-y-2 sm:gap-x-9">
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
                  className={`eyebrow min-h-11 border-b-2 text-[11px] tracking-[0.18em] transition sm:text-xs sm:tracking-[0.26em] ${
                    mode === value
                      ? "border-ink font-semibold text-ink"
                      : "border-transparent text-faint hover:text-ink"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>

            <button
              type="button"
              onClick={toggleProfilePanel}
              aria-expanded={profileOpen}
              className="min-h-11 text-left text-sm text-muted"
            >
              {savedProfile ? (
                <>
                  Business profile{" "}
                  <span className="font-serif text-xl italic text-ink">{profileName}</span>{" "}
                  <span className="link-quiet ml-1.5 text-xs uppercase tracking-[0.16em]">
                    {profileOpen ? "Close" : "Edit"}
                  </span>
                </>
              ) : (
                <span className="link-quiet text-xs uppercase tracking-[0.16em]">
                  {profileOpen ? "Close profile" : "Add your business profile"}
                </span>
              )}
            </button>
          </div>

          {profileOpen && (
            <div className="mt-6 border border-rule bg-paper p-6 md:p-10">
              <div className="flex flex-wrap items-baseline justify-between gap-3">
                <h2 className="font-serif text-[28px] font-medium leading-tight">Your business profile</h2>
                <p className="text-sm text-faint">Optional. Fill in only what you want to share.</p>
              </div>
              <div className="mt-6 grid gap-x-10 gap-y-6 md:grid-cols-2">
                {PROFILE_FIELDS.map((field) => (
                  <label
                    key={field.key}
                    className={`block ${field.wide ? "md:col-span-2" : ""}`}
                  >
                    <span className="eyebrow block text-[11px] text-faint">
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
                      className="field"
                    />
                  </label>
                ))}
              </div>

              <div className="mt-8 flex flex-col gap-4 border-t border-rule-soft pt-6 sm:flex-row sm:items-center sm:justify-between">
                <p className="text-sm text-faint">
                  Saved only in this browser. It is sent with a question only when you tick
                  &ldquo;Tailor&rdquo;.
                </p>
                <div className="flex gap-3">
                  {savedProfile && (
                    <button type="button" onClick={clearProfile} className="btn-secondary">
                      Clear
                    </button>
                  )}
                  <button type="button" onClick={saveProfile} className="btn-primary min-h-12">
                    Save profile
                  </button>
                </div>
              </div>
            </div>
          )}

          {profileNotice && (
            <p className="mt-4 text-sm text-muted" role="status">{profileNotice}</p>
          )}

          {mode === "research" ? (
            <>
              <label htmlFor="question" className="eyebrow mt-7 block text-faint">
                Your question
              </label>
              <textarea
                id="question"
                ref={questionRef}
                rows={2}
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={handleKeyDown}
                maxLength={2000}
                placeholder="What would you like to know about your market?"
                className="mt-2.5 block min-h-[96px] w-full resize-none border-0 border-b border-ink bg-transparent pb-4 pt-2 font-serif text-[26px] italic leading-[1.25] text-ink outline-none placeholder:text-[#a8967f] focus:border-gold-text md:text-[34px]"
              />
            </>
          ) : (
            <div className="mt-7">
              <div className="mb-2 flex flex-col gap-2 sm:flex-row sm:items-baseline sm:justify-between">
                <p className="eyebrow text-faint">
                  Up to {MAX_COMPETITORS} competitors
                </p>
                {savedProfile?.competitors.trim() && (
                  <button
                    type="button"
                    onClick={fillCompetitorsFromProfile}
                    className="link-quiet self-start text-sm"
                  >
                    Use competitors from my profile
                  </button>
                )}
              </div>
              <div className="grid gap-x-8 gap-y-2 md:grid-cols-3">
                {competitorInputs.map((value, index) => (
                  <div key={index} className="flex items-baseline gap-3">
                    <span className="font-serif text-xl italic text-gold-text">
                      {["i.", "ii.", "iii."][index]}
                    </span>
                    <input
                      type="text"
                      value={value}
                      maxLength={200}
                      aria-label={`Competitor ${index + 1}`}
                      placeholder={["Website or name", "Website or name", "Optional"][index]}
                      onChange={(event) =>
                        setCompetitorInputs((inputs) =>
                          inputs.map((v, i) => (i === index ? event.target.value : v)),
                        )
                      }
                      className="field border-ink font-serif text-[22px] italic"
                    />
                  </div>
                ))}
              </div>
              <label className="mt-6 block">
                <span className="eyebrow block text-[11px] text-faint">
                  What do you most want to know? (optional)
                </span>
                <input
                  type="text"
                  value={focus}
                  maxLength={500}
                  placeholder="e.g. how they price their packages"
                  onChange={(event) => setFocus(event.target.value)}
                  className="field"
                />
              </label>
            </div>
          )}

          <div className="mt-6 flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between">
            <div className="space-y-2">
              <label
                className={`flex min-h-11 items-center gap-3 text-sm ${
                  savedProfile ? "cursor-pointer text-muted" : "cursor-not-allowed text-faint"
                }`}
              >
                <input
                  type="checkbox"
                  checked={useProfile && !!savedProfile}
                  disabled={!savedProfile}
                  onChange={(event) => setUseProfile(event.target.checked)}
                  className="h-4 w-4 rounded-none accent-[#2a1c13]"
                />
                {savedProfile ? (
                  <span>
                    Tailor this to{" "}
                    <span className="font-serif text-lg italic text-ink">{profileName}</span>
                  </span>
                ) : (
                  "Tailor to my business (add a profile first)"
                )}
              </label>
              <p className="text-sm text-faint">
                {mode === "teardown"
                  ? "Reads each competitor's own website. Nothing is invented."
                  : "Several focused searches and a careful read of each page. Nothing is invented."}
                {mode === "research" && (
                  <span className="hidden md:inline"> Press ⌘/Ctrl + Enter to start.</span>
                )}
              </p>
            </div>

            <button
              type="button"
              onClick={handleResearch}
              disabled={loading}
              className="btn-primary min-h-14 px-9 text-[13px]"
            >
              {loading
                ? mode === "teardown"
                  ? "Comparing…"
                  : "Researching…"
                : mode === "teardown"
                  ? "Compare competitors"
                  : "Begin research"}
              {!loading && <span aria-hidden="true" className="text-gold">→</span>}
            </button>
          </div>

          {error && <ErrorNote message={error} />}
        </section>

        {/* Or begin with */}
        <section
          aria-label="Or begin with"
          className="mt-4 flex flex-wrap items-stretch border-y border-rule"
        >
          <p className="w-full py-4 font-serif text-xl italic text-gold-deep md:w-auto md:flex-none md:py-[22px] md:pr-7">
            Or begin with
          </p>
          {starterQuestions(savedProfile).map((starter) => (
            <button
              key={starter.label}
              type="button"
              onClick={() => applyStarter(starter.question)}
              className="flex min-h-11 flex-[1_1_200px] items-center justify-between gap-3 border-t border-rule-soft px-0 py-4 text-left text-[15px] text-ink transition hover:bg-paper md:border-l md:border-t-0 md:px-6 md:py-[22px]"
            >
              {starter.label} <span aria-hidden="true" className="text-gold-text">→</span>
            </button>
          ))}
        </section>

        {loading && (
          <section aria-live="polite" className="mt-16 border border-rule bg-paper p-8 md:p-12">
            <p className="eyebrow font-semibold text-ink">Research in progress</p>
            <p className="mt-4 font-serif text-[28px] italic leading-snug text-ink md:text-[34px]">
              {mode === "teardown"
                ? "Finding each competitor’s website and reading their pages…"
                : "Searching the web, reading each page and checking every figure…"}
            </p>
            <div className="mt-8 h-px w-full overflow-hidden bg-rule">
              <div className="progress-bar h-px w-2/5 bg-gold-text" />
            </div>
            <p className="mt-4 text-sm text-faint">This usually takes a minute or two.</p>
          </section>
        )}

        {report && !loading && (
          <section aria-label="Research memorandum" className="mt-16">
            <Memorandum
              kind="Research memorandum"
              title={report.question}
              preparedFor={tailoredTo}
              date={reportDate}
              sourceCount={report.sources.length}
            >
              <EvidenceLedger findings={report.key_findings} />

              <MemoSection title="Summary">
                <Summary text={report.summary} />
              </MemoSection>

              <MemoSection title="Findings">
                <Ledger items={report.key_findings} />
              </MemoSection>

              {report.opportunities.length > 0 && (
                <MemoSection title="Opportunities">
                  <Ledger items={report.opportunities} />
                </MemoSection>
              )}

              {report.recommendations.length > 0 && (
                <MemoSection title="Recommendations">
                  <Ledger items={report.recommendations} />
                </MemoSection>
              )}

              <DeliverablePanel
                source={{
                  question: report.question,
                  summary: report.summary,
                  key_findings: report.key_findings,
                  opportunities: report.opportunities,
                  recommendations: report.recommendations,
                }}
                profile={tailoredTo ? savedProfile : null}
              />

              <SourceList title="Sources" sources={report.sources} />
            </Memorandum>

            <button
              type="button"
              onClick={() => {
                setReport(null);
                setQuestion("");
                deskRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
              }}
              className="btn-secondary mt-8"
            >
              Start new research
            </button>
          </section>
        )}

        {teardown && !loading && (
          <TeardownView
            teardown={teardown}
            tailoredTo={tailoredTo}
            date={reportDate}
            profile={tailoredTo ? savedProfile : null}
            onReset={() => {
              setTeardown(null);
              setCompetitorInputs(["", "", ""]);
              setFocus("");
              deskRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
            }}
          />
        )}

        {/* Method */}
        <section id="method" aria-label="Method" className="mt-24 flex scroll-mt-6 flex-wrap gap-x-16 gap-y-10">
          <div className="flex-[1_1_260px]">
            <p className="eyebrow mb-4 text-gold-deep">The method</p>
            <h2 className="font-serif text-[40px] font-medium leading-[1.05] md:text-[44px]">
              Three steps, <span className="italic">no shortcuts.</span>
            </h2>
          </div>
          <div className="grid flex-[999_1_600px] gap-8 [grid-template-columns:repeat(auto-fit,minmax(200px,1fr))]">
            {[
              ["I.", "Search", "Focused searches for every part of your question, worded the way real pages are written."],
              ["II.", "Read", "The pages themselves, not search snippets. Competitors’ own sites included."],
              ["III.", "Verify", "Every figure checked against its source. What cannot be established is said plainly."],
            ].map(([numeral, name, text]) => (
              <div key={numeral} className="border-t border-gold pt-5">
                <p className="font-serif text-[22px] italic text-gold-text">{numeral}</p>
                <p className="mt-2.5 font-serif text-[26px] font-semibold">{name}</p>
                <p className="mt-2 text-[15px] leading-[1.65] text-muted">{text}</p>
              </div>
            ))}
          </div>
        </section>

        <footer className="mt-20 flex flex-wrap justify-between gap-x-8 gap-y-3 border-t border-rule pt-6 text-xs uppercase tracking-[0.12em] text-faint">
          <span className="font-serif text-xl normal-case tracking-[0.01em] text-ink">
            AI Marketing Research Agent
          </span>
          <span>Evidence · Vendor claim · Inference · Not established</span>
        </footer>
      </main>
    </div>
  );
}

/* ---------- Report building blocks ---------- */

function ErrorNote({ message }: { message: string }) {
  return (
    <div role="alert" className="mt-6 border-l-2 border-[#8a2a1a] bg-[#fbf0ea] px-5 py-4 text-sm text-[#7a2416]">
      {message}
    </div>
  );
}

function Memorandum({
  kind,
  title,
  preparedFor,
  date,
  sourceCount,
  sourceLabel = "Sources cited",
  children,
}: {
  kind: string;
  title: string;
  preparedFor: string;
  date: string;
  sourceCount: number;
  sourceLabel?: string;
  children: React.ReactNode;
}) {
  return (
    <article className="border border-rule bg-paper px-5 pb-10 pt-8 sm:px-8 md:px-[72px] md:pb-14 md:pt-16">
      <header className="flex flex-wrap justify-between gap-x-10 gap-y-4 border-b border-ink pb-7">
        <p className="eyebrow font-semibold text-ink">{kind}</p>
        <dl className="flex flex-wrap gap-x-8 gap-y-1.5 text-[13px] text-muted">
          {preparedFor && <MetaItem term="Prepared for" value={preparedFor} />}
          {date && <MetaItem term="Date" value={date} />}
          <MetaItem term={sourceLabel} value={String(sourceCount)} />
        </dl>
      </header>
      <h2 className="mt-8 max-w-[900px] break-words font-serif text-[32px] font-medium leading-[1.1] md:mt-10 md:text-[46px] md:leading-[1.08]">
        {title}
      </h2>
      {children}
    </article>
  );
}

function MetaItem({ term, value }: { term: string; value: string }) {
  return (
    <div className="flex gap-2">
      <dt className="pt-0.5 text-[11px] uppercase tracking-[0.16em] text-faint">{term}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function MemoSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mt-12 flex flex-wrap gap-x-16 gap-y-5 md:mt-[52px]">
      <h3 className="eyebrow flex-[1_1_180px] text-gold-deep">{title}</h3>
      <div className="min-w-0 flex-[999_1_560px]">{children}</div>
    </div>
  );
}

/** Turn footnote marks like [3] or [1][2] into superscript links to the source list. */
function withFootnotes(text: string): React.ReactNode[] {
  return text.split(/((?:\[\d+\])+)/g).map((part, index) => {
    const numbers = part.match(/^(?:\[\d+\])+$/) ? part.match(/\d+/g) ?? [] : [];
    if (numbers.length === 0) {
      return part;
    }
    return (
      <sup key={index} className="ml-0.5 text-[0.7em] font-semibold">
        {numbers.map((n, i) => (
          <span key={n + i}>
            {i > 0 && ","}
            <a href={`#source-${n}`} className="text-gold-text no-underline hover:text-ink">
              {n}
            </a>
          </span>
        ))}
      </sup>
    );
  });
}

function Summary({ text }: { text: string }) {
  return (
    <div className="space-y-4 text-[17px] leading-[1.75] text-[#3a2a1e] md:text-lg">
      {text.split(/\n\s*\n/).map((paragraph, index) => (
        <p
          key={index}
          className={
            index === 0 && paragraph.length > 160
              ? "first-letter:float-left first-letter:mr-3 first-letter:mt-1.5 first-letter:font-serif first-letter:text-[72px] first-letter:leading-[0.78] first-letter:text-gold-text"
              : undefined
          }
        >
          {withFootnotes(paragraph)}
        </p>
      ))}
    </div>
  );
}

type Marker = "solid" | "wide" | "outline" | "soft" | "dashed" | "none";

/** How each backend label is marked. Labels can carry a qualifier after " – ". */
function markerFor(label: string): Marker {
  const main = label.split(" – ")[0].toLowerCase();
  if (main.startsWith("evidence")) {
    return label.includes(" – ") ? "wide" : "solid";
  }
  if (main.startsWith("vendor claim")) return "outline";
  if (main.startsWith("inference")) return "soft";
  if (main.startsWith("social post") || main.startsWith("uploaded document") || main.startsWith("weakly supported")) {
    return "outline";
  }
  if (main.startsWith("not established") || main.startsWith("untested idea")) return "dashed";
  return "none";
}

function MarkerSquare({ marker }: { marker: Marker }) {
  const style: Record<Marker, string> = {
    solid: "h-2 w-2 bg-ink",
    wide: "h-2 w-2 bg-gold-text",
    outline: "h-1.5 w-1.5 border border-gold-text",
    soft: "h-2 w-2 bg-tan",
    dashed: "h-1.5 w-1.5 border border-dashed border-quiet",
    none: "hidden",
  };
  return <span aria-hidden="true" className={`mr-2 inline-block align-[1px] ${style[marker]}`} />;
}

function splitLabel(text: string): { label: string; body: string } {
  const match = text.match(/^\[([^\]]+)\]\s*/);
  return match ? { label: match[1], body: text.slice(match[0].length) } : { label: "", body: text };
}

function Ledger({ items }: { items: string[] }) {
  const parsed = items.map(splitLabel);
  const anyLabel = parsed.some((item) => item.label);
  return (
    <ol className="border-t border-ink">
      {parsed.map(({ label, body }, index) => {
        const [main, qualifier] = label.split(" – ");
        const marker = markerFor(label);
        const muted = marker === "dashed";
        return (
          <li
            key={`${body}-${index}`}
            className="flex flex-wrap gap-x-6 gap-y-2 border-b border-rule-soft py-5"
          >
            <span className="w-8 font-serif text-[22px] leading-7 text-gold-text lining-nums">
              {String(index + 1).padStart(2, "0")}
            </span>
            {anyLabel && (
              <span className="w-[150px] pt-1">
                {label && (
                  <span
                    className={`block text-[11px] font-semibold uppercase leading-5 tracking-[0.18em] ${
                      marker === "solid" ? "text-ink" : marker === "outline" || marker === "wide" ? "text-gold-deep" : "text-quiet"
                    }`}
                  >
                    <MarkerSquare marker={marker} />
                    {main}
                  </span>
                )}
                {qualifier && (
                  <span className="mt-1 block text-xs leading-4 text-faint">{qualifier}</span>
                )}
              </span>
            )}
            <span
              className={`min-w-0 flex-[1_1_280px] break-words leading-[1.65] ${muted ? "text-muted" : "text-ink"}`}
            >
              {withFootnotes(body)}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

function EvidenceLedger({ findings }: { findings: string[] }) {
  const labels = findings.map((f) => splitLabel(f).label.toLowerCase());
  const count = (prefix: string) => labels.filter((l) => l.startsWith(prefix)).length;
  const cells: { value: number; label: string; marker: Marker }[] = [
    { value: findings.length, label: "Findings", marker: "none" },
    { value: count("evidence"), label: "Independent evidence", marker: "solid" },
    { value: count("vendor claim"), label: "Vendor claims", marker: "outline" },
    { value: count("not established"), label: "Not established", marker: "dashed" },
  ];
  return (
    <div
      aria-label="Evidence overview"
      className="mt-10 grid grid-cols-2 border-y border-rule md:grid-cols-4"
    >
      {cells.map((cell, index) => (
        <div
          key={cell.label}
          className={`py-5 pr-4 ${index % 2 === 1 ? "border-l border-rule-soft pl-4 md:pl-5" : ""} ${
            index === 2 ? "border-t border-rule-soft md:border-l md:border-t-0 md:pl-5" : ""
          } ${index === 3 ? "border-t border-rule-soft md:border-t-0" : ""}`}
        >
          <p className="font-serif text-[44px] leading-none lining-nums">{cell.value}</p>
          <p className="mt-2 text-[11px] uppercase tracking-[0.2em] text-faint">
            <MarkerSquare marker={cell.marker} />
            {cell.label}
          </p>
        </div>
      ))}
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

function SourceList({ title, sources }: { title: string; sources: string[] }) {
  if (sources.length === 0) {
    return null;
  }
  return (
    <MemoSection title={title}>
      <ol className="space-y-3 text-[13px] leading-6 text-quiet">
        {sources.map((source, index) => (
          <li
            key={`${source}-${index}`}
            id={`source-${index + 1}`}
            className="flex scroll-mt-6 gap-3"
          >
            <span className="w-6 shrink-0 font-serif text-base text-gold-text lining-nums">{index + 1}.</span>
            <a
              href={source}
              target="_blank"
              rel="noopener noreferrer"
              className="group min-w-0"
            >
              <span className="block font-medium text-ink group-hover:text-gold-deep">
                {siteName(source)}
              </span>
              <span className="block break-all text-xs text-faint underline decoration-rule underline-offset-4">
                {source}
              </span>
            </a>
          </li>
        ))}
      </ol>
    </MemoSection>
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
  date,
  profile,
  onReset,
}: {
  teardown: Teardown;
  tailoredTo: string;
  date: string;
  profile: Profile | null;
  onReset: () => void;
}) {
  return (
    <section aria-label="Competitor teardown" className="mt-16">
      <Memorandum
        kind="Competitor teardown"
        title={teardown.competitors.map((c) => c.name).join(" vs ")}
        preparedFor={tailoredTo}
        date={date}
        sourceCount={teardown.sources.length}
        sourceLabel="Pages read"
      >
        <MemoSection title="Summary">
          <Summary text={teardown.summary} />
        </MemoSection>

        <MemoSection title="Side by side">
          {/* Phones: one block per competitor */}
          <div className="space-y-8 md:hidden">
            {teardown.competitors.map((c) => (
              <div key={c.name} className="border-t border-ink pt-4">
                <p className="font-serif text-2xl font-semibold">{c.name}</p>
                <p className="mb-3 text-xs text-faint">{c.website || "Website not found"}</p>
                <dl className="text-sm">
                  {TEARDOWN_ROWS.map((row) => {
                    const value = String(c[row.key]);
                    return (
                      <div key={row.key} className="border-b border-rule-soft py-3">
                        <dt className="text-[11px] font-semibold uppercase tracking-[0.16em] text-faint">
                          {row.label}
                        </dt>
                        <dd className={`mt-1 leading-6 ${value.startsWith("Not found") ? "italic text-faint" : "text-ink"}`}>
                          {value}
                        </dd>
                      </div>
                    );
                  })}
                </dl>
                {c.note && <p className="mt-3 text-xs leading-5 text-faint">{c.note}</p>}
              </div>
            ))}
          </div>

          {/* Larger screens: side-by-side table */}
          <div className="hidden overflow-x-auto md:block">
            <table className="w-full min-w-[600px] border-collapse text-left text-sm">
              <thead>
                <tr className="border-b border-ink">
                  <th className="w-36 pb-4 pr-4" />
                  {teardown.competitors.map((c) => (
                    <th key={c.name} className="pb-4 pr-6 align-bottom font-normal">
                      <span className="block font-serif text-2xl font-semibold leading-tight text-ink">{c.name}</span>
                      {c.website ? (
                        <a
                          href={`https://${c.website}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-xs text-faint underline decoration-rule underline-offset-4 hover:text-ink"
                        >
                          {c.website}
                        </a>
                      ) : (
                        <span className="text-xs text-faint">Website not found</span>
                      )}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {TEARDOWN_ROWS.map((row) => (
                  <tr key={row.key} className="border-b border-rule-soft">
                    <th className="py-4 pr-4 align-top text-[11px] font-semibold uppercase tracking-[0.16em] text-faint">
                      {row.label}
                    </th>
                    {teardown.competitors.map((c) => {
                      const value = String(c[row.key]);
                      const missing = value.startsWith("Not found");
                      return (
                        <td
                          key={c.name}
                          className={`py-4 pr-6 align-top leading-6 ${missing ? "italic text-faint" : "text-ink"}`}
                        >
                          {value}
                        </td>
                      );
                    })}
                  </tr>
                ))}
                {teardown.competitors.some((c) => c.note) && (
                  <tr className="border-b border-rule-soft">
                    <th className="py-4 pr-4 align-top text-[11px] font-semibold uppercase tracking-[0.16em] text-faint">
                      Notes
                    </th>
                    {teardown.competitors.map((c) => (
                      <td key={c.name} className="py-4 pr-6 align-top text-xs leading-5 text-quiet">
                        {c.note}
                      </td>
                    ))}
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </MemoSection>

        {teardown.gaps.length > 0 && (
          <MemoSection title="Openings you can use">
            <Ledger items={teardown.gaps} />
          </MemoSection>
        )}

        {teardown.recommendations.length > 0 && (
          <MemoSection title="Recommendations">
            <Ledger items={teardown.recommendations} />
          </MemoSection>
        )}

        <DeliverablePanel
          source={{
            question: `Competitor teardown: ${teardown.competitors.map((c) => c.name).join(" vs ")}`,
            summary: teardown.summary,
            key_findings: teardown.competitors.map(
              (c) =>
                `${c.name}: sells ${c.offer} Pricing: ${c.pricing}. Target customer: ${c.target_customer}. ` +
                `Main message: ${c.positioning} Channels: ${c.channels}.`,
            ),
            opportunities: teardown.gaps,
            recommendations: teardown.recommendations,
          }}
          profile={profile}
        />

        <SourceList title="Pages read" sources={teardown.sources} />
      </Memorandum>

      <button type="button" onClick={onReset} className="btn-secondary mt-8">
        Start a new teardown
      </button>
    </section>
  );
}

type SourceReport = {
  question: string;
  summary: string;
  key_findings: string[];
  opportunities: string[];
  recommendations: string[];
};

type DeliverableKind = "campaign_brief" | "ad_copy" | "content_calendar";

type Deliverable = {
  kind: DeliverableKind;
  title: string;
  sections: { heading: string; items: string[] }[];
  text: string;
};

const DELIVERABLE_BUTTONS: [DeliverableKind, string][] = [
  ["campaign_brief", "Campaign brief"],
  ["ad_copy", "Ad copy"],
  ["content_calendar", "Content calendar"],
];

function DeliverablePanel({
  source,
  profile,
}: {
  source: SourceReport;
  profile: Profile | null;
}) {
  const [working, setWorking] = useState<DeliverableKind | null>(null);
  const [result, setResult] = useState<Deliverable | null>(null);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);

  async function create(kind: DeliverableKind) {
    if (working) {
      return;
    }
    setWorking(kind);
    setError("");
    setCopied(false);
    try {
      const response = await fetch(`${API_URL}/deliverable`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          kind,
          report: {
            ...source,
            key_findings: source.key_findings.slice(0, 40),
            opportunities: source.opportunities.slice(0, 20),
            recommendations: source.recommendations.slice(0, 20),
          },
          ...(profile ? { profile } : {}),
        }),
      });
      if (!response.ok) {
        let detail = "";
        try {
          const body = await response.json();
          detail = typeof body?.detail === "string" ? body.detail : "";
        } catch {
          // no readable message
        }
        throw new RequestError(response.status === 429 || response.status === 422 ? detail : "");
      }
      setResult(await response.json());
    } catch (err) {
      console.error(err);
      setError(
        err instanceof RequestError && err.message
          ? err.message
          : "Could not create this right now. Please try again.",
      );
    } finally {
      setWorking(null);
    }
  }

  async function copy() {
    if (!result) {
      return;
    }
    try {
      await navigator.clipboard.writeText(result.text);
      setCopied(true);
    } catch {
      setError("Your browser blocked copying. Select the text and copy it instead.");
    }
  }

  return (
    <div className="mt-14 border-t border-ink pt-7">
      <div className="flex flex-wrap items-center justify-between gap-5">
        <div>
          <p className="font-serif text-[26px] italic leading-tight">Turn this memorandum into —</p>
          <p className="mt-1.5 text-sm text-faint">
            Built only from this report{profile ? " and your business profile" : ""}. No new searches.
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          {DELIVERABLE_BUTTONS.map(([kind, label]) => (
            <button
              key={kind}
              type="button"
              onClick={() => create(kind)}
              disabled={working !== null}
              aria-pressed={result?.kind === kind}
              className={result?.kind === kind ? "btn-primary min-h-12 px-6" : "btn-secondary"}
            >
              {working === kind ? "Writing…" : label}
            </button>
          ))}
        </div>
      </div>

      {error && <ErrorNote message={error} />}

      {result && working === null && (
        <div className="mt-8 border border-rule bg-page/60 p-6 md:p-10">
          <div className="mb-6 flex flex-wrap items-baseline justify-between gap-4 border-b border-rule pb-5">
            <h3 className="font-serif text-[30px] font-medium leading-tight">{result.title}</h3>
            <button type="button" onClick={copy} className="link-quiet text-xs uppercase tracking-[0.18em]">
              {copied ? "Copied" : "Copy all"}
            </button>
          </div>
          <div className="space-y-7">
            {result.sections.map((section, index) => (
              <div key={`${section.heading}-${index}`}>
                <p className="eyebrow mb-3 text-[11px] font-semibold text-gold-deep">{section.heading}</p>
                <ul className="space-y-2.5 text-[15px] leading-[1.65] text-ink">
                  {section.items.map((item, i) => (
                    <li key={i} className="flex gap-3">
                      <span aria-hidden="true" className="mt-[11px] h-px w-3 shrink-0 bg-gold" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
