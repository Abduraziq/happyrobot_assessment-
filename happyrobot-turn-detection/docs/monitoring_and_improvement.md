# EoT Production Monitoring and Continuous Improvement

## Scope and current baseline

The serving bundle is Logistic Regression, feature schema `acoustic-v1`, with
threshold `0.3691340835714273`. Its held-out test results were F1 0.6538,
precision 0.5121, recall 0.9040, 323 false positives, and 36 false negatives
on 750 balanced examples. A false positive predicts the user has finished
while they are still speaking and can interrupt them. Treat reducing and
understanding these errors as a first-class quality goal.

The current FastAPI response includes prediction/probability/threshold, audio
duration, sample rate, channels, upload-inspection time, feature-extraction
time, model-inference time, total inference time, and server-processing time. It
does not currently expose a request ID, separate decoding from feature
calculation, or measure queue/network time. The `/health` and `/ready`
endpoints support probes, but application metrics and alerting are not yet
implemented. Add these as a small instrumentation layer rather than changing
model behavior.

The original warm-cache offline benchmark measured end-to-end mean/median/P95/
P99 of 17.17/16.23/28.45/35.77 ms and model-only P95 of 2.18 ms. Its original
HTTP benchmark used a 0.5-second synthetic clip and measured 111.52 ms P95 at
concurrency 8. A separate follow-up used eight Uvicorn workers and measured
89.19 ms P95 for the 0.5-second clip and 146.41 ms for a 6.75-second FLAC at
concurrency 8. These are development-machine references, not production
baselines or SLAs; the longer clip remains above the ideal target. Establish
real-traffic baselines before enabling the proposed alerts.

## Monitoring architecture

At approximately one million calls per month (about 33,000 per day or 0.4
requests/second on average), a modest metrics stack is sufficient; size it for
bursts, not just the monthly average:

1. Instrument the FastAPI process and ingress/gateway to emit structured JSON
   logs and Prometheus counters, gauges, and latency histograms.
2. Scrape service and host/container metrics with the existing platform's
   Prometheus-compatible collector. If none exists, one Prometheus instance
   and Grafana dashboards are a reasonable starting point; use Alertmanager
   or the existing on-call tool for notifications.
3. Keep delayed labels and audited examples in a separate access-controlled
   evaluation store. A scheduled job joins labels to prediction records and
   publishes cohort metrics; it must not change the live model or threshold.

Use bounded metric labels such as route, status class, model version, and
coarse duration band. Do not label metrics with request IDs, user IDs, audio
paths, or unbounded values. Structured logs can contain a short-lived,
pseudonymous request ID for incident correlation, but it should not become a
metric label. Record the model artifact hash, model/config version, feature
version, and threshold on startup and with each evaluation window.

## Metrics

| Category | Recommended signals |
|---|---|
| Service health | Request count; success/failure count; HTTP status and error class; `/health` and `/ready` availability; client/server timeouts; active/in-flight requests; CPU, memory, restarts, and worker saturation. |
| Latency/performance | Client/ingress end-to-end latency and server latency histograms; P50/P95/P99; upload, audio decode, feature extraction, and model inference spans; requests/second; active concurrency; latency by duration band and concurrency. The current feature timer combines decode/resampling and feature calculation, so split it if diagnosing those costs is useful. |
| Model behavior | Positive/negative prediction counts and rates; probability histogram and quantiles; share of probabilities near the fixed threshold; duration, sample-rate, channel, and rejection distributions; feature-distribution summaries; changes by model/config version. |
| Model quality | Confusion matrix, precision, recall, F1, false-positive rate, and false-negative rate on labeled cohorts, overall and by sufficiently large, privacy-safe cohorts. Report sample counts, label coverage/delay, and confidence intervals with each score. |

Use `endpoint_bool=True` as the predicted positive (end of turn). For labeled
examples, let TP be correctly predicted end-of-turn, FP be an incorrect
end-of-turn prediction, TN be correctly predicted continuing speech, and FN
be a missed end-of-turn:

- Precision = TP / (TP + FP): among predicted endpoints, how many were correct.
- Recall = TP / (TP + FN): among actual endpoints, how many were detected.
- F1 = 2 * precision * recall / (precision + recall).
- False-positive rate = FP / (FP + TN); false-negative rate = FN / (FN + TP)
  (equivalently, 1 - recall).
- Confusion matrix = `[[TN, FP], [FN, TP]]`.

The current API's `model_inference_ms` and `feature_extraction_ms` provide
useful server-side spans. `upload_ms` starts after FastAPI parses the multipart
body; neither it nor `server_processing_ms` measures the complete upload or
client round trip. Measure end-to-end at the ingress/client boundary and
include gateway queueing, network transfer, and response delivery there.
Collect CPU/memory and concurrency from the process/container runtime, not
from the model response.

## Labels and quality measurement

Production labels will often arrive later than predictions. Create an
evaluation record keyed by a pseudonymous request ID and attach a label when
available from:

- Explicit human review of sampled interactions.
- Agent/user corrections or downstream outcomes that have been validated as
  meaningful signals.
- A delayed annotation workflow with label source, timestamp, and reviewer
  confidence.

Downstream behavior is a proxy unless its relationship to turn completion has
been validated; do not silently treat every agent action as ground truth.
Sample both a random slice of traffic and targeted slices: predicted
endpoints, probabilities near the threshold, model/user disagreement, long
silences, suspected interruptions, and underrepresented languages/audio
conditions. Keep the random slice so estimates are not biased toward uncertain
or already-flagged examples. Where feasible, blind reviewers to the model
prediction and use a second reviewer for disputed/high-impact cases.

Compute quality on rolling, labeled cohorts (for example, weekly and monthly),
with minimum sample counts and label-coverage reporting. Compare against the
fixed offline evaluation and prior production versions; include confidence
intervals and cohort mix. Prioritize false positives: inspect cases where
speech continues after an endpoint decision and measure precision/FPR as
labels mature. Do not tune the threshold on the same live cohort used to
claim performance.

## Data drift and concept drift

Maintain reference distributions from versioned train/validation data and
compare them with recent production windows. Monitor:

- Audio duration, sample rate, channels, decoding/rejection rate, and any
  available upstream audio-quality indicators.
- Language and real/synthetic/source mix only when trusted upstream metadata is
  available. The current prediction API does not accept or infer these fields.
- Fixed acoustic feature summaries and probability/prediction distributions.
- Labeled precision, recall, F1, FPR, and FNR when delayed ground truth is
  available.

Use duration/rate histograms and robust feature summaries; for distribution
comparisons, PSI or Jensen-Shannon distance can be useful. Establish reference
bins, minimum sample sizes, and alert windows before interpreting changes.
Feature and score drift are investigation signals, not proof of quality loss.

**Data drift** means the incoming input distribution changes (for example,
longer clips, new microphones, or a language-mix shift). **Concept drift**
means the relationship between input/context and the correct end-of-turn label
changes. Data drift can occur without worse predictions; concept drift can
occur without a large marginal feature shift. Confirm suspected impact using
representative labels and cohort evaluation rather than retraining
automatically.

## Starting alert thresholds

These are operational starting assumptions, not established SLAs. Calibrate
them against real traffic and error budgets. Require a minimum request volume
for rate alerts and persistence across more than one short interval to avoid
noise.

| Signal | Initial investigation trigger |
|---|---|
| Availability/readiness | `/ready` unavailable for 2 minutes, or successful requests below a proposed 99.9% over a rolling 30-minute window with adequate volume. |
| HTTP errors/timeouts | 5xx above 1% for 5 minutes, or client/server timeouts above 0.5% for 5 minutes; page sooner for a sustained outage. |
| Latency | End-to-end P95 above 100 ms for 10 minutes or P99 above 250 ms for 10 minutes. These proposed targets need validation with representative audio; local short-clip benchmarks are not an SLA. |
| Resource/concurrency | CPU above 80% for 15 minutes, memory above 85%, or sustained in-flight requests/queueing near configured worker capacity. Alert on saturation together with latency rather than CPU alone. |
| Rejected/missing audio | Missing, unreadable, or invalid upload rate over 2x its established baseline or +1 percentage point for 30 minutes, after at least 500 requests. |
| Positive prediction rate | Relative change over 20% or absolute change over 5 percentage points from a time-of-week baseline for 30 minutes, with at least 1,000 requests. Investigate mix changes before attributing it to the model. |
| Feature/probability drift | PSI over 0.2 for a material feature or score distribution in two consecutive windows with at least 1,000 examples. Treat as a review trigger, not a rollback by itself. |
| Labeled quality | After a sufficiently representative labeled cohort (initially at least 500 examples), investigate precision below 0.45 or a drop of 5 percentage points from the version's measured baseline; separately investigate recall below 0.85 or a 5-point decline. Use confidence intervals and label-mix checks before action. |

The precision/recall values are proposed guardrails, not promised service
levels. Given the test precision of 0.512, a precision alert should trigger
review and improved labels, not an automatic model swap.

## Continuous improvement and rollout

1. Collect only representative prediction metadata and, where lawful and
   consented, a carefully sampled set of production audio plus necessary
   conversational context.
2. Sample uncertain, disagreement, interruption-suspected, and randomly
   selected examples; retain sampling probabilities so metrics can account
   for targeted sampling.
3. Obtain human labels or validated downstream labels; adjudicate ambiguous
   cases and distinguish endpoint errors from transcription/audio failures.
4. Create a versioned, de-duplicated dataset. Preserve immutable train,
   validation, and test partitions and add temporal and speaker-grouped
   evaluation sets where identifiers and policy allow.
5. Add difficult cases: continuing speech after pauses, interruptions,
   fillers, long/short turns, microphones/noise conditions, real examples,
   and multilingual speech. Track coverage and label quality by cohort.
6. Train candidate models offline with the same feature/versioned preprocessing
   contract; use training data for fitting and validation data for selection
   and threshold choice.
7. Evaluate on fixed held-out sets, including a recent temporal set and
   speaker-separated sets where possible. Do not repeatedly tune against the
   current test set.
8. Compare candidates with the current production version on overall and
   high-risk metrics, especially false-positive interruption rate, calibration,
   latency, and subgroup coverage. Require a documented review and rollback
   threshold.
9. Run a shadow phase or small canary. Shadow predictions must not control
   live turn-taking; canaries should have a kill switch and limited exposure.
10. Expand only if quality, latency, and error rates remain acceptable. Keep
    the previous artifact/config available and roll back on confirmed
    regressions; continue collecting delayed labels after rollout.

Drift alerts should start investigation and targeted labeling, not launch
automatic retraining or threshold changes.

## Dataset limits and privacy

The current training/evaluation base is only 5,000 samples with a balanced
class distribution. Dataset inspection reported 4,892 of 5,000 examples as
synthetic, broad language coverage with uneven per-language counts, and no
speaker-group identifier. The test set contains 750 examples and has low
precision (0.512) with 323 false positives. These results do not establish
production, speaker-independent, or real-world language/microphone
generalization. Collect representative real production evidence and keep
speaker/temporal leakage in mind before claiming improvement.

The current API writes uploads to temporary files and removes them after
inference; preserve that no-retention default. Do not log audio, transcripts,
raw feature vectors, or identifying metadata by default. For any approved
human-audit audio collection:

- Obtain required consent/legal approval and document purpose and scope.
- Minimize captured audio/context, encrypt in transit and at rest, restrict
  access by role, audit access, and use short, explicit retention periods with
  deletion verification.
- Pseudonymize linkage IDs, separate identity mappings, and provide deletion
  and opt-out handling as required by policy.
- Aggregate monitoring metrics and apply minimum cohort sizes so dashboards
  cannot expose individual conversations.

Version every model artifact, threshold/config, feature schema, code revision,
and training/evaluation dataset snapshot. A deployment record should make it
possible to reproduce which version produced a prediction without retaining
the raw audio.
