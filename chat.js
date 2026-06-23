/*
 * chat.js — "AllenBot", an in-browser AI assistant for Allen Tran's portfolio.
 *
 * Runs ENTIRELY in the visitor's browser using WebLLM (https://github.com/mlc-ai/web-llm)
 * + WebGPU. There is no backend and no API key — the model is downloaded and cached by
 * the browser on first use, then all inference happens locally on the visitor's device.
 *
 * To use a different model, change MODEL_F16 / MODEL_F32 below to any id from WebLLM's
 * prebuilt list (see https://mlc.ai/models). Smaller = faster first load, larger = smarter.
 */

import { CreateMLCEngine } from "https://esm.run/@mlc-ai/web-llm";

// Smaller/faster build (needs the "shader-f16" GPU feature) and a broader-compatibility
// fallback. We auto-pick based on what the visitor's GPU supports.
const MODEL_F16 = "Llama-3.2-1B-Instruct-q4f16_1-MLC";
const MODEL_F32 = "Llama-3.2-1B-Instruct-q4f32_1-MLC";

// Everything the bot is allowed to know about Allen (kept factual, from his resume + site).
const ALLEN_FACTS = `
Allen Tran is a student at the University of Southern California (USC).

EDUCATION:
- M.S. in Computer Science, USC Viterbi School of Engineering — January 2025 to May 2026.
- B.S. in Computer Science and Business Administration (dual degree across USC Viterbi and the
  Marshall School of Business) — August 2022 to May 2026.
- GPA: 3.65. Dean's List every semester from Fall 2022 through Spring 2025.
- Codepath Intermediate Technical Interview certificate.
- Relevant coursework: Data Structures & OOP, Advanced Algorithms, Machine Learning,
  Information Retrieval.

He describes himself as a "Builder, Coder, and Dreamer" who works at the intersection of computer
science and business — using AI and full-stack development to turn ideas into real products.

EXPERIENCE:
- ServiceNow — Machine Learning Engineer Intern (May 2025 – August 2025, Santa Clara, CA).
  Built scalable infrastructure for evaluating agentic router prompts across multiple LLMs
  (GPT-4o, Claude, Gemini, LLaMA). Built an evaluation pipeline in Python + PyTorch to measure
  routing accuracy, tool-invocation success, and task-completion rates across tens of thousands
  of model outputs. Designed AutoChat and Judge LLM systems to simulate and evaluate full
  multi-turn conversations for automated conversational QA. Collaborated with linguists to deploy
  labeled datasets into production workflows. Merged 6 pull requests.
- USC Interaction Lab — Undergraduate Research Assistant (August 2023 – May 2026, Los Angeles).
  Uses pre-trained LLMs (CLIP) to generate image embeddings from the MuFaSAA dataset (165 robots,
  1,200+ entries); reduced computational time by 87%. Hyperparameter-tunes SVR and transformers
  with Optuna to predict social and functional expectations of robots, hitting R² > 0.7 on
  warmth, discomfort, and competency. Works with a PhD mentor on Vision-Language Models that
  predict explanatory metaphors for robot designs.
- ServiceNow — Software Engineer Intern (May 2024 – August 2024, Santa Clara, CA). Automated
  story/regression tests in Java + Selenium across ServiceNow's internal lists, forms, and
  streams (5 merged PRs). Built an AI-driven extension in TypeScript + Node using the GitHub and
  OpenAI APIs that auto-fills build requests and summarizes PRs. Deployed a Mattermost bot for
  live defect/story updates (JavaScript, JSON, Jenkins).
- CodePath — Tech Fellow, Advanced Interview Prep (May 2024 – August 2024, Remote). TA for the
  TIP 103 cohort, teaching data structures and algorithms. Used UMPIRE and cold-calling to coach
  interview skills. Mentored groups of 6 students at a time.

PROJECTS:
- Adaptive RAG Research Paper, titled "Token-Efficient Question Answering with Adaptive RAG via
  Uncertainty Estimation" (Python, PyTorch). Co-authored a research paper proposing a novel RAG
  pipeline that selectively triggers retrieval based on an LLM's uncertainty estimation, reducing
  token usage and improving efficiency. Developed an inference-time controller that computes
  entropy-based thresholds to decide when external context is needed vs. when parametric knowledge
  suffices. Benchmarked on SQuAD, Natural Questions, and TriviaQA — measuring accuracy, latency,
  and token savings vs. standard RAG baselines. (August 2024 – May 2025.) Read it here:
  https://drive.google.com/file/d/1s6fPVSWtlVpQMpadnJKEMUJUJ3u1KFus/view
- BizLang (JavaScript, React.js, Flask) — A Duolingo-style app that teaches industry-specific
  language through interactive learning. Includes gamified elements like Wordle, news feeds from
  the GNews API, and chat-style flashcards, backed by 1,000+ jargon terms. 100+ users, with
  active beta testing among 30 healthcare professionals in Ethiopia. Live at trybizlang.com.
- DNA Foundation Models Paper, titled "Probing and Lightweight Fine-Tuning of DNA Foundation
  Models" (PyTorch). Co-authored with Runpeng He, Jiaxuan Cao, Nan Huang, Natalie Kao, and
  Zuge Li (May 2026). Evaluates pretrained DNA foundation models (DNABERT-2, Nucleotide
  Transformer, and HyenaDNA) on three genomic classification tasks from the Genomic Benchmarks
  collection: human non-TATA promoter prediction, human enhancer prediction (Cohn), and
  Drosophila enhancer prediction (Stark). Compares probing (frozen backbone with lightweight
  classifiers) against fine-tuning strategies including unfreezing the last 4 layers, progressive
  unfreeze, and LoRA, combined with three pooling functions (mean, max, CLS). Measures accuracy,
  F1, and Matthews Correlation Coefficient (MCC) to understand how architecture, scale, and
  adaptation strategy jointly affect performance. (January 2026 – May 2026.) Read it here:
  images/DNA_Foundation_Models_Paper.pdf

SKILLS:
- Languages: Python, C/C++, SQL, JavaScript, Java, TypeScript, PHP, R.
- Frameworks & tools: React, Flask, Django, PyTorch, TensorFlow, Jenkins, Docker, GCP, AWS, Git.
- Libraries: Hugging Face, Pandas, NumPy, scikit-learn, Firebase.
- Core skills: ML Ops, prompt engineering, NLP, data analysis, distributed systems, CI/CD,
  REST APIs.

CONTACT / LINKS:
- LinkedIn: linkedin.com/in/allenvtran
- GitHub: github.com/allenvtran
- Email: avtran@usc.edu
- More: bit.ly/allentran
`.trim();

const SYSTEM_PROMPT = `You are "AllenBot," a friendly, concise assistant embedded on Allen Tran's personal portfolio website. Your job is to answer visitors' questions about Allen — his background, education, skills, experience, and projects.

Rules:
- Use ONLY the information in the ALLEN FACTS section below. Never invent employers, dates, numbers, or details.
- If a question is not covered by the facts, say you don't have that detail and suggest reaching out to Allen on LinkedIn or by email.
- Keep answers short and conversational — usually 1 to 3 sentences. Use bullet points only when listing several things.
- Refer to Allen in the third person ("Allen..."). Be warm and professional, like a helpful representative.
- If a question is unrelated to Allen, gently steer back to topics about Allen.

ALLEN FACTS:
${ALLEN_FACTS}`;

const STARTERS = [
  "What does Allen study?",
  "Tell me about his projects",
  "What are his technical skills?",
  "Where has he interned?",
];

// ---------------------------------------------------------------- state
let engine = null;
let loadingPromise = null;
let generating = false;
const history = [{ role: "system", content: SYSTEM_PROMPT }];

// ---------------------------------------------------------------- helpers
const ICON_CHAT =
  '<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg>';

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

// Escape first, then turn URLs, emails and bare domains into safe links in a single pass.
function formatMessage(text) {
  const safe = escapeHtml(text);
  const linkRe =
    /(https?:\/\/[^\s<]+)|([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})|((?:[a-z0-9-]+\.)+(?:com|co|io|org|net|edu|ai|dev|ly|app|me)(?:\/[^\s<]*)?)/gi;
  return safe
    .replace(linkRe, (m, url, email, domain) => {
      if (url) return `<a href="${url}" target="_blank" rel="noopener">${url}</a>`;
      if (email) return `<a href="mailto:${email}">${email}</a>`;
      if (domain) return `<a href="https://${domain}" target="_blank" rel="noopener">${domain}</a>`;
      return m;
    })
    .replace(/\n/g, "<br>");
}

// ---------------------------------------------------------------- UI
const root = document.createElement("div");
root.className = "allenbot";
root.innerHTML = `
  <button class="allenbot-toggle" type="button" aria-label="Chat with AllenBot">${ICON_CHAT}</button>
  <section class="allenbot-panel" role="dialog" aria-label="AllenBot chat" hidden>
    <header class="allenbot-header">
      <span class="allenbot-title"><span class="allenbot-dot"></span> AllenBot</span>
      <button class="allenbot-close" type="button" aria-label="Close chat">&times;</button>
    </header>
    <div class="allenbot-messages" aria-live="polite"></div>
    <div class="allenbot-status" hidden><div class="allenbot-bar"><i></i></div><span></span></div>
    <form class="allenbot-form">
      <input class="allenbot-input" type="text" autocomplete="off"
             placeholder="Ask me anything about Allen…" aria-label="Your message" />
      <button class="allenbot-send" type="submit" aria-label="Send">Send</button>
    </form>
  </section>
`;
document.body.appendChild(root);

const $ = (sel) => root.querySelector(sel);
const toggleBtn = $(".allenbot-toggle");
const panel = $(".allenbot-panel");
const closeBtn = $(".allenbot-close");
const messagesEl = $(".allenbot-messages");
const statusEl = $(".allenbot-status");
const statusBar = $(".allenbot-bar i");
const statusText = $(".allenbot-status span");
const form = $(".allenbot-form");
const input = $(".allenbot-input");
const sendBtn = $(".allenbot-send");

let started = false; // whether we've shown the welcome / checked WebGPU

function addMessage(role, html) {
  const el = document.createElement("div");
  el.className = `allenbot-msg allenbot-${role}`;
  el.innerHTML = html;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

function addSuggestions() {
  const wrap = document.createElement("div");
  wrap.className = "allenbot-suggestions";
  STARTERS.forEach((q) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "allenbot-chip";
    chip.textContent = q;
    chip.addEventListener("click", () => {
      wrap.remove();
      submitQuestion(q);
    });
    wrap.appendChild(chip);
  });
  messagesEl.appendChild(wrap);
}

function setStatus(text, progress) {
  statusEl.hidden = false;
  statusText.textContent = text;
  if (typeof progress === "number") statusBar.style.width = `${Math.round(progress * 100)}%`;
}
function clearStatus() {
  statusEl.hidden = true;
  statusBar.style.width = "0%";
}

function setBusy(busy) {
  generating = busy;
  input.disabled = busy;
  sendBtn.disabled = busy;
}

// Lazily download + spin up the model the first time it's actually needed.
async function ensureEngine() {
  if (engine) return engine;
  if (!("gpu" in navigator)) {
    const err = new Error("no-webgpu");
    err.code = "no-webgpu";
    throw err;
  }
  if (loadingPromise) return loadingPromise;

  let modelId = MODEL_F32;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter && adapter.features.has("shader-f16")) modelId = MODEL_F16;
  } catch (_) {
    /* fall back to the f32 build */
  }

  setStatus("Loading AllenBot — one-time download…", 0);
  loadingPromise = CreateMLCEngine(modelId, {
    initProgressCallback: (r) => setStatus(r.text || "Loading…", r.progress ?? 0),
  });
  engine = await loadingPromise;
  clearStatus();
  return engine;
}

async function streamAnswer() {
  const bubble = addMessage("bot", "");
  bubble.innerHTML = '<span class="allenbot-typing"><i></i><i></i><i></i></span>';

  let full = "";
  try {
    const stream = await engine.chat.completions.create({
      messages: history,
      stream: true,
      temperature: 0.4,
      max_tokens: 400,
    });
    for await (const chunk of stream) {
      const delta = chunk.choices?.[0]?.delta?.content ?? "";
      if (!delta) continue;
      full += delta;
      bubble.innerHTML = formatMessage(full);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  } catch (e) {
    console.error("AllenBot generation error:", e);
  }

  if (full.trim()) {
    history.push({ role: "assistant", content: full });
  } else {
    bubble.innerHTML = formatMessage(
      "Sorry — I had trouble answering that. Please try again, or reach Allen at linkedin.com/in/allenvtran."
    );
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function submitQuestion(text) {
  const q = text.trim();
  if (!q || generating) return;

  addMessage("user", formatMessage(q));
  history.push({ role: "user", content: q });
  setBusy(true);

  try {
    await ensureEngine();
    await streamAnswer();
  } catch (e) {
    if (e.code === "no-webgpu") {
      addMessage(
        "bot",
        formatMessage(
          "This in-browser assistant needs WebGPU, which your current browser doesn't support. " +
            "Try the latest Chrome or Edge on a computer. In the meantime you can reach Allen here:\n" +
            "• LinkedIn: linkedin.com/in/allenvtran\n• Email: avtran@gmail.com"
        )
      );
    } else {
      console.error("AllenBot error:", e);
      addMessage("bot", formatMessage("Sorry — something went wrong loading the model. Please try again."));
    }
    clearStatus();
  } finally {
    setBusy(false);
    input.focus();
  }
}

function openPanel() {
  panel.hidden = false;
  // allow the [hidden] removal to apply before transitioning
  requestAnimationFrame(() => root.classList.add("allenbot-open"));
  toggleBtn.setAttribute("aria-expanded", "true");

  if (!started) {
    started = true;
    addMessage(
      "bot",
      "Hi! I'm <strong>AllenBot</strong> 🤖 — ask me anything about Allen: his projects, experience, skills, or background."
    );
    if (!("gpu" in navigator)) {
      addMessage(
        "bot",
        formatMessage(
          "Heads up: running the AI needs WebGPU (latest Chrome/Edge on desktop, or Safari 17+). " +
            "It may not work in this browser — but you can always reach Allen at linkedin.com/in/allenvtran."
        )
      );
    } else {
      addSuggestions();
    }
  }
  setTimeout(() => input.focus(), 150);
}

function closePanel() {
  root.classList.remove("allenbot-open");
  toggleBtn.setAttribute("aria-expanded", "false");
  setTimeout(() => {
    panel.hidden = true;
  }, 200);
}

// ---------------------------------------------------------------- events
toggleBtn.addEventListener("click", () => {
  if (root.classList.contains("allenbot-open")) closePanel();
  else openPanel();
});
closeBtn.addEventListener("click", closePanel);
form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value;
  input.value = "";
  submitQuestion(q);
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && root.classList.contains("allenbot-open")) closePanel();
});
