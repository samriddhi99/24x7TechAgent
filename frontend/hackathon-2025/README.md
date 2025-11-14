# hackathon-2025

Cards & fields (friendlier wording + IDs)
0) Welcome (new, #welcome)

Quick Start Checklist (4 check items with small descriptions).

Button “Fill with a preset” (scrolls to Presets).

1) Personality (#branding)

Agent name agent_name (required). Help: “Customers will hear/see this name.”

[TODO PREVIEW] Avatar upload avatar_file + round preview avatar_preview.

Tone & feel tone (Friendly, Professional, Playful, Empathetic) + helper text.

Response speed response_speed (Instant, Fast, Balanced).

Greeting greeting and Sign-off signoff with soft examples.

Languages: primary_lang select + checkboxes langs[] (EN, FR, ES, DE, RU).

<details><summary>Advanced</summary> brand colors brand_primary, brand_accent.

2) Capabilities (#capabilities)

Friendly groups with toggles (each row has a short hint and an optional “Ask first” switch):

Automatic: Password resets, Basic troubleshooting, Account info, Service status.

Conditional: Schedule appointments, Service plan change, Process refunds, Cancel service, Create quote (devis). Each shows “Ask before doing this?” (default ON).

Human-only note: Billing disputes, Legal requests (info text, no toggles).

Per-action spend/impact limit impact_cap_eur (default 0).

3) Quotes / Devis (#quotes)

Enable switch quotes_enabled.

Currency quotes_currency, VAT % quotes_vat.

Items mini-repeater (name, unit, price).

Email template subject/body ({{customer_name}}, {{total}}).

“Require human confirmation before sending” (default ON).

Button “Preview total” quotes_preview_btn.

4) Appointments (#appointments)

Enable switch appts_enabled.

Provider appts_provider (Google default).

Connect button gcal_connect_btn (placeholder).

Calendar id appts_calendar, duration appts_len, buffer appts_buffer, reminder minutes appts_reminder.

Double-booking guard (toggle).

Confirmation message templates (SMS/email).

5) Technical Assistance (#tech)

Issue checkboxes (Internet down, Slow, Device setup, Password reset, Custom text).

Troubleshooting steps textarea (markdown allowed).

Diagnostics endpoints JSON textarea diag_tools_json (name, url, method, auth).

Auto-create ticket toggle + target URL ticket_api.

6) Knowledge & FAQs (#knowledge)

FAQ repeater (Question, Short answer, Tags).

KB search endpoint kb_url, Top-K, “Include citations” (checkbox).

Upload docs input multiple kb_upload.

Fallback answer when unknown (warm tone).

7) Feedback (#feedback)

Collect post-call rating (1–5) + comments (toggle).

Owner email feedback_email.

“Create follow-ups from low ratings” (toggle).

Export CSV button feedback_export_btn.

8) Safeguards (#safeguards)

Empathetic copy at top: “We keep you in control.”

Global “Ask before risky actions” (ON).

Max monetary impact without approval risk_cap_eur (default 0).

“Escalate when unsure” slider escalate_uncertainty (0–100, default 60).

Actions whitelist multi-select.

Blocked phrases/topics textarea.

9) Data & Privacy (RGPD/GDPR) (#privacy)

Consent prompt (toggle) + editable consent line.

Retention days retention_days (0 = don’t keep).

PII redaction (toggle) for transcripts.

Data residency select (EU, US).

Links: DPA/Privacy Policy fields.

Buttons (disabled placeholders): Export my data, Delete my data.

10) Storage & Conversation Snaps (#storage)

“Store parts of conversations on demand” (toggle).

Options: last N customer questions, unresolved questions, intents only.

Button: Export top questions snaps_export_btn with mini preview list.

11) Escalation (#escalation)

Reasons (Unknown issue, Customer asks, Complex billing, Special case, Detect frustration).

Business hours start/end; Weekend/Holiday toggles; After-hours auto-reply textarea.

Queue target select + emails list.

Sliders: “Auto-escalate after X minutes”, “Satisfaction threshold”.

12) Analytics (#analytics)

KPI cards (Total calls, Resolved by AI %, Escalated %, Avg time).

Time range pills (7d/30d/90d).

Table of top questions (Question, Count, Resolution rate).

“Use insights to improve your FAQs and flows.” note.

13) Industry Presets (more general, covers most SMBs) (#presets)

Cards with a short description and Apply button (data-preset):

Services (generic) – hairdressers, salons, repair shops

Tone: Friendly, Appointments: ON, Quotes: OFF, Hours: 9–18, Safeguards: Ask first (ON).

Retail (store/online)

Tone: Helpful, Order status/returns: Conditional (Ask first), Appointments: OFF, Quotes: OFF.

Restaurant / Café

Tone: Friendly, Reservations: ON, Hours include weekends, Quotes: OFF.

Healthcare Clinic (basic)

Tone: Empathetic, Appointments: ON with buffers, Strict privacy text, No payments.

Home Services (plumbers/electricians)

Tone: Professional, Quotes: ON (required confirmation), Appointments: ON, Tech steps enabled.

Education / Tutoring

Tone: Encouraging, Appointments: ON, FAQs for programs, Quotes (optional).

IT & SaaS Support

Tone: Professional, Troubleshooting: ON, Service status check: ON, Escalate on outage keywords.

Real Estate / Rentals

Tone: Professional & warm, Appointments: ON (viewings), Lead capture & follow-up.

Hospitality (B&B/Hotel)

Tone: Warm, Reservations: ON, FAQs for amenities, After-hours reply active.

(Each preset sets a handful of fields; users can tweak after.)

14) Integrations (#integrations)

Webhooks in/out + secret.

Email “from” address, optional SMS provider id.

Google Calendar connect (as in Appointments).

Ticketing endpoint (reused from Tech).

15) Sandbox (#sandbox)

Button “Open Call Simulator” → record.html (new tab). Short tip: “Try a quick scenario to hear your agent in action.”