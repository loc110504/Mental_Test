# AgentMental-ToM Proposed Method Specification

## 1. Purpose

This specification describes how to extend the current `src/` implementation of AgentMental into a more flexible Theory-of-Mind-guided multi-agent assessment framework. 

The proposed method should preserve the useful parts of the current system:

- adaptive questioning;
- topic-based assessment;
- simulated participant responses in automated mode;
- tree-structured memory;
- scale-based scoring;
- final explainable reporting.

The extension adds three complementary capabilities:

1. **Question Agent with Theory of Mind awareness**
   - Cognitive ToM: beliefs, goals, desires, intentions, and interpretations.
   - Affective ToM: emotions, emotional causes, mixed emotions, and emotional reactions.

2. **Answer Agent with grounded contextual reasoning**
   - It answers consistently with the original interview.
   - It may combine related evidence from different turns.
   - It must not invent unsupported symptoms, events, relationships, or mental states.

3. **Judge Agent with decoding and reasoning**
   - Decoding: extract explicit and implicit cues from the latest response.
   - Reasoning: decide whether the interaction has enough useful information, what remains uncertain, and what the next best conversational objective should be.

The implementation must remain **flexible**. Agents should not be forced into an overly detailed checklist on every turn. Structured outputs are used only to make orchestration reliable, while the actual reasoning and questioning policy should remain open-ended and context-sensitive.

---

## 2. Design Principles

### 2.1 Flexible capability-based agents

Each agent is defined by its capabilities and responsibilities, not by a rigid sequence of mandatory sub-questions.

For example, the Question Agent should be able to explore beliefs, emotions, intentions, frequency, impact, contradictions, or contextual details, but it should not be required to ask all of them for every topic.

The orchestration layer should communicate goals such as:

- clarify symptom frequency;
- understand the user's interpretation of a social event;
- identify the emotional cause;
- resolve a contradiction;
- check whether another plausible interpretation exists.

The agent decides how to formulate a natural question.

### 2.2 Minimal stable schemas, extensible payloads

Agent outputs should contain a small set of required fields for reliable code execution and an optional open-ended `details` object for future extension.

Avoid large hard-coded schemas containing every possible clinical or ToM category.

Example:

```json
{
  "text": "What made you think your colleague was disappointed?",
  "focus": "supporting_evidence",
  "reason": "The user's interpretation is important but currently unsupported.",
  "details": {
    "related_topic": "low_self_worth",
    "possible_tom_dimension": "cognitive"
  }
}
```

The required contract is stable. The `details` field remains flexible.

### 2.3 ToM is a conversational reasoning layer, not a separate diagnostic scale

ToM should help the system:

- discover richer contextual information;
- understand social interpretations;
- identify possible negative interpretation bias;
- improve question selection;
- explain interpersonal difficulty;
- estimate uncertainty.

ToM must not be directly added to PHQ-8 or another clinical score unless a future validated model explicitly supports that operation.

### 2.4 Scale scoring remains standards-based

Scale scores must be determined from the scale's official criteria.

For PHQ-8, item scores primarily represent symptom frequency. Functional impact, emotional intensity, and ToM observations should be stored separately and should not automatically increase a frequency score.

### 2.5 Evidence grounding over fluent fabrication

In automated mode, the Answer Agent must prefer:

1. direct evidence from the original interview;
2. conservative integration of compatible evidence;
3. an honest uncertain answer when evidence is insufficient.

Natural fluency is useful, but factual fidelity is more important.

### 2.6 Adaptive depth, not fixed depth

The current fixed `max_depth` loop should be replaced or supplemented by a configurable interaction budget controlled by the Judge Agent.

The system may stop early when information is sufficient or continue when a high-value gap remains.

The budget is a safety and efficiency bound, not the main reasoning policy.

### 2.7 Preserve uncertainty

The framework should explicitly represent uncertainty instead of forcing every agent to produce a definitive conclusion.

Uncertainty may come from:

- ambiguous responses;
- incomplete source interview;
- inconsistent statements;
- weak support for inferred mental states;
- low-confidence decoding;
- unclear scale frequency.

---

## 3. Scope

### 3.1 In scope

- Extend the current code under `src/`.
- Keep support for manual and automated modes.
- Add ToM-aware question generation.
- Replace the simple necessity score with a richer Judge decision.
- Improve grounded simulated responses.
- Extend memory to store general evidence and optional ToM observations.
- Preserve existing scale scoring and report generation.
- Add configuration switches so the proposed method can be enabled incrementally.
- Keep backward compatibility where practical.

### 3.2 Out of scope

- Clinical diagnosis.
- Treatment recommendation generation beyond the existing screening disclaimer.
- Direct use of ToM as a validated diagnostic score.
- Mandatory audio or video processing.
- Full redesign of the dataset format.
- Training a new neural model.
- Requiring a fixed ontology for every possible belief, emotion, or intention.

---

## 4. Current Code Mapping

The proposed method should be implemented by evolving the current modules rather than rewriting the project from scratch.

### `src/agents.py`

Current role:

- creates Question Agent;
- creates Necessity Agent;
- creates Scoring Agent;
- creates Summary Agent;
- creates User Proxy.

Proposed changes:

- keep Question Agent but replace its prompt with a flexible ToM-aware prompt;
- replace or rename Necessity Agent as Judge Agent;
- optionally add an Answer Agent abstraction for automated mode;
- keep Scoring Agent;
- update Summary Agent so ToM observations are reported separately from scale scores;
- preserve User Proxy if still required by AutoGen orchestration.

### `src/assessment.py`

Current role:

- controls the topic loop;
- asks questions;
- obtains simulated or real responses;
- calls the necessity agent;
- scores topics;
- updates memory;
- creates the final report.

Proposed changes:

- introduce a state-driven interaction loop;
- let Judge Agent select the next conversational objective;
- remove dependence on only a `0/1/2` necessity score;
- support early stop, clarification, ToM exploration, and contradiction resolution;
- store structured turn records;
- keep the existing topic scoring stage;
- keep final reporting and result saving.

### `src/generate_response.py`

Current role:

- simulates participant responses from the original interview.

Proposed changes:

- refactor into a grounded Answer Agent service;
- add explicit evidence support metadata;
- avoid inventing demographics when not available unless legacy mode is enabled;
- support conservative reasoning across related interview turns;
- return uncertainty when information is absent;
- optionally maintain the existing function name as a compatibility wrapper.

### `src/memory.py`

Current role:

- stores topic nodes and statement nodes;
- extracts emotion, frequency, symptom, duration, and impact;
- stores completed topic scores;
- performs holistic reassessment.

Proposed changes:

- replace the narrow entity extractor with a general evidence extractor;
- support flexible observations such as beliefs, intentions, interpretations, emotional causes, contradictions, and alternative explanations;
- keep topic and statement nodes;
- add optional social episode and ToM observation nodes;
- retain provenance to source turns;
- avoid letting memory reassessment silently modify official scale scores.

### `src/utils.py`

Proposed changes:

- add parsers for structured Question and Judge outputs;
- add safe JSON extraction and schema validation;
- add fallback logic when an agent returns malformed output;
- replace `is_necessary()` with a general decision interpreter;
- keep score parsing and report helpers.

### `src/config.py`

Proposed changes:

- expose proposed-method flags and budgets;
- support agent-specific temperature or model configuration;
- default Judge and Scoring Agents to low temperature;
- allow Question and Answer Agents slightly more flexibility if desired.

### `src/data_load.py`

Proposed changes:

- preserve the current input structure;
- optionally build normalized interview evidence records;
- support missing demographics without forced fabrication.

### `src/main.py`

Proposed changes:

- add a method selector or configuration flag:
  - `baseline`;
  - `tom`;
- preserve the current manual and automated mode selection.

---

## 5. Proposed Architecture

```text
Original Interview / Real User
            |
            v
   Evidence Context Builder
            |
            v
+---------------------------------------------+
| Adaptive Interaction Loop                   |
|                                             |
| Question Agent -> Answer/User -> Judge Agent|
|       ^                         |            |
|       |                         v            |
|       +------ next objective ---+            |
+---------------------------------------------+
            |
            v
   Flexible Memory Graph
            |
            v
     Scale Scoring Agent
            |
            v
Summary + ToM Observations + Uncertainty
```

The core loop should not assume that every topic follows the same path.

Possible paths include:

```text
initial symptom question
-> sufficient answer
-> score topic
```

```text
initial symptom question
-> ambiguous answer
-> clarification
-> score topic
```

```text
initial symptom question
-> relevant social event
-> cognitive ToM question
-> affective ToM question
-> evidence or alternative interpretation question
-> score topic
```

```text
initial symptom question
-> contradiction with memory
-> clarification
-> score topic
```

The Judge Agent chooses the path based on expected value, not a fixed flowchart.

---

## 6. Core Data Models

Use Python dataclasses or Pydantic models. Pydantic is preferred if it is already acceptable for the project dependencies.

### 6.1 TurnRecord

```python
class TurnRecord:
    turn_id: int
    topic: str
    question: str
    answer: str
    question_focus: str | None
    judge_decision: dict | None
    evidence: list[dict]
    metadata: dict
```

### 6.2 QuestionRequest

```python
class QuestionRequest:
    topic: str
    interaction_goal: str
    user_context: dict
    topic_history: list[dict]
    memory_context: dict
    previous_judge_decision: dict | None
    remaining_budget: int
```

`interaction_goal` is intentionally open-ended. Examples:

- `"start_topic"`
- `"clarify_frequency"`
- `"explore_social_interpretation"`
- `"understand_emotional_cause"`
- `"resolve_contradiction"`
- `"seek_supporting_evidence"`
- `"explore_alternative_explanation"`

The code should not reject unknown goals.

### 6.3 QuestionOutput

Minimal required fields:

```python
class QuestionOutput:
    text: str
    focus: str
    reason: str | None
    details: dict
```

The Question Agent may populate `details` with any useful metadata.

### 6.4 AnswerOutput

```python
class AnswerOutput:
    text: str
    support_level: str
    evidence_ids: list[str]
    uncertainty: float | None
    details: dict
```

Suggested `support_level` values:

- `direct`;
- `integrated`;
- `uncertain`;
- `unsupported`;
- `real_user`.

Do not make this an inflexible enum unless needed by the implementation. Unknown values should be accepted and logged.

### 6.5 JudgeOutput

```python
class JudgeOutput:
    continue_topic: bool
    next_goal: str | None
    rationale: str
    confidence: float | None
    extracted_observations: list[dict]
    missing_information: list[str]
    warnings: list[str]
    details: dict
```

This replaces the current single integer necessity score.

The Judge Agent should be free to express a new `next_goal` rather than being restricted to a short fixed action enum.

The orchestration layer only needs to understand:

- continue or stop;
- the next high-level goal;
- warnings;
- observations to store.

### 6.6 TopicAssessment

```python
class TopicAssessment:
    topic: str
    score: int
    basis: str
    confidence: float | None
    scale_evidence: list[dict]
    contextual_observations: list[dict]
```

Scale evidence and contextual observations must remain separate.

---

## 7. Agent Specifications

## 7.1 Question Agent

### Responsibility

Generate the next most useful, natural, non-leading interview question.

### Capabilities

The agent may use:

- symptom clarification;
- frequency, duration, context, and impact exploration;
- cognitive ToM;
- affective ToM;
- evidence checking;
- alternative interpretation exploration;
- contradiction clarification;
- uncertainty clarification;
- user-centered empathetic phrasing.

### Flexible ToM behavior

Cognitive ToM may involve:

- what another person believed;
- what another person wanted;
- what another person intended;
- how another person interpreted the situation;
- what the user believed another person thought about them.

Affective ToM may involve:

- what another person felt;
- why that person may have felt it;
- whether multiple emotions were possible;
- how the user's own emotion related to their interpretation.

The agent should not mechanically ask all of these.

It should use ToM only when:

- a social event is relevant;
- the user's interpretation may help explain distress or withdrawal;
- the Judge identifies missing social-cognitive information;
- ToM exploration is likely to improve understanding.

### Prompt guidance

The prompt should emphasize broad judgment:

```text
You are an adaptive mental health interview agent.

Your goal is to ask the single next question that is most likely to improve
the accuracy and depth of the current assessment.

You may explore symptoms, context, impact, beliefs, intentions, emotions,
emotional causes, evidence, alternative explanations, or contradictions.

Theory of Mind is available as a reasoning capability, not a mandatory
checklist. Use it when social interpretation is clinically or contextually
relevant.

Prefer:
- one clear idea per question;
- neutral and non-leading language;
- natural follow-up based on the user's own wording;
- questions that reduce uncertainty;
- questions that avoid repeating known information.

Do not diagnose.
Do not assume an interpretation is true.
Do not ask about unrelated topics.
Return JSON with text, focus, reason, and optional details.
```

### Flexibility requirements

- Do not hard-code a separate prompt for every ToM subtype.
- Do not require a specific sequence such as belief -> intention -> emotion.
- Do not force multiple-choice questions unless useful.
- Do not limit the agent to PHQ language.
- Do not require the question to mention ToM terminology.

---

## 7.2 Answer Agent

### Responsibility

In automated mode, simulate a participant response that is faithful to the original interview while allowing careful contextual integration.

In manual mode, the real user replaces this agent.

### Grounding policy

The Answer Agent may:

- restate direct interview evidence;
- combine compatible statements from different turns;
- express uncertainty in a natural way;
- answer in the participant's approximate style;
- infer only what is strongly supported by the source context.

The Answer Agent must not:

- invent new symptoms;
- invent frequency values;
- invent traumatic events;
- invent relationships;
- invent direct quotes from other people;
- invent a specific belief or emotion with no source support;
- use hidden scale labels to optimize the answer.

### Recommended evidence preparation

Build a simple evidence store before the interaction:

```python
{
    "E001": {
        "text": "...",
        "role": "Participant",
        "source_turn": 12,
        "tags": [],
        "metadata": {}
    }
}
```

Tags are optional and may be generated lazily.

The first implementation does not need a complex retrieval system. It may provide the full interview when short enough and use keyword or embedding retrieval later.

### Prompt guidance

```text
You are simulating the participant represented by the original interview.

Answer naturally in the first person.

Use only information supported by the interview and prior simulated answers.
You may combine compatible evidence to answer a follow-up more clearly, but
do not invent new events, symptoms, frequency, relationships, or mental states.

When the source is insufficient, express uncertainty naturally rather than
filling the gap.

Preserve ambiguity when the participant was ambiguous.
Do not mention the dataset, source interview, evidence store, or simulation.

Return a natural answer plus hidden grounding metadata.
```

### Required implementation change

Replace the current behavior that infers missing demographic fields by default.

Add a flag:

```python
ALLOW_DEMOGRAPHIC_GUESSING = False
```

When disabled, missing fields should be represented as unknown.

---

## 7.3 Judge Agent

### Responsibility

Evaluate the latest interaction and choose the next conversational objective.

### Two complementary processes

#### Decoding

Extract useful cues from the response, such as:

- explicitly named emotion;
- uncertainty or hesitation;
- strong certainty;
- absolute negative language;
- social attribution;
- contradiction;
- avoidance;
- symptom frequency;
- duration;
- functional effect;
- evidence offered for an interpretation.

Decoding should be modality-aware. If only text exists, it must not invent vocal or visual cues.

#### Reasoning

Integrate:

- the latest question and answer;
- current topic history;
- relevant long-term memory;
- scale requirements;
- available source support;
- uncertainty;
- potential information value of another question.

### Main decision

The Judge should answer:

1. Is there enough information to end the current topic?
2. If not, what is the next best high-level goal?
3. What observations should be stored?
4. Are there contradictions, grounding problems, or safety concerns?

### Prompt guidance

```text
You are an adaptive interaction judge.

Evaluate the latest question-answer pair in the context of the current topic,
conversation history, memory, and scoring requirements.

Use two complementary processes:

1. Decoding:
   identify explicit and implicit cues available in the provided modality.

2. Reasoning:
   assess relevance, sufficiency, consistency, uncertainty, source support,
   and the expected value of another question.

Do not follow a rigid checklist. Select the most useful next objective.

Possible objectives include symptom clarification, contextual exploration,
cognitive or affective Theory of Mind exploration, evidence checking,
alternative interpretation, contradiction resolution, or topic completion.

Return:
- whether to continue;
- the next goal in concise natural language;
- rationale;
- observations;
- missing information;
- warnings;
- confidence;
- optional details.
```

### Safety behavior

Judge may emit a warning such as:

```json
{
  "warnings": [
    "possible_self_harm_signal"
  ]
}
```

The orchestration layer should route safety warnings to existing or future safety handling.

Do not tightly couple the Judge prompt to a fixed list of all possible warning types.

---

## 7.4 Scoring Agent

### Responsibility

Score the current scale topic using official scoring standards.

### Rules

- Use the topic dialogue and relevant memory evidence.
- Distinguish official scale evidence from contextual ToM observations.
- Do not raise a PHQ frequency score solely because the impact is severe.
- Report uncertainty when frequency is unclear.
- Do not use ToM difficulty as direct evidence of a depressive symptom unless the scale item itself supports it.

### Proposed output

```json
{
  "score": 1,
  "basis": "The participant reported the symptom on several days.",
  "confidence": 0.84,
  "scale_evidence": [
    {
      "turn_id": 5,
      "evidence": "several days"
    }
  ],
  "contextual_observations": [
    {
      "type": "negative_social_interpretation",
      "summary": "The participant assumed rejection without direct evidence."
    }
  ]
}
```

The current parser may initially ignore the additional fields, but the new parser should preserve them.

---

## 7.5 Summary Agent

### Responsibility

Produce a clear final summary without conflating clinical score and ToM observations.

### Required report structure

```text
1. Scale results
2. Evidence supporting the scale results
3. Contextual and interpersonal observations
4. ToM-related observations, if enough evidence exists
5. Uncertainty and unresolved information
6. Screening disclaimer
```

Avoid definitive labels such as "the user has impaired ToM" unless the system has a validated assessment protocol.

Prefer wording such as:

- "The interaction showed a repeated tendency to interpret ambiguous social situations negatively."
- "The participant had difficulty providing evidence for some social interpretations."
- "The available evidence is insufficient to determine whether this pattern is stable."

### Score adjustment policy

The current Summary Agent is allowed to modify topic scores. This should become conservative.

Preferred options:

1. Disable final score adjustment by default.
2. Allow adjustment only when the Summary Agent cites explicit scale-relevant evidence.
3. Never adjust a scale score because of ToM observations alone.

Configuration:

```python
ALLOW_SUMMARY_SCORE_ADJUSTMENT = False
```

---

## 8. Memory Design

## 8.1 Preserve the existing graph

Keep:

```text
User -> Topic -> Statement
```

Add optional nodes only when useful:

```text
User
  -> Topic
      -> Statement
      -> SocialEpisode
          -> Observation
```

The implementation should not require every response to create every node type.

## 8.2 General evidence extraction

Replace the current fixed extractor with a flexible observation extractor.

Minimal output:

```json
{
  "summary": "The participant avoids friends after losing work.",
  "observations": [
    {
      "type": "social_withdrawal",
      "value": "avoids friends",
      "source_text": "I stopped seeing them",
      "confidence": 0.94,
      "details": {}
    },
    {
      "type": "interpreted_belief",
      "value": "friends may see the participant as a failure",
      "confidence": 0.73,
      "details": {
        "tom_dimension": "cognitive"
      }
    }
  ]
}
```

The `type` field is open-ended.

The memory layer should store unknown types instead of rejecting them.

## 8.3 Provenance

Every memory observation should preserve:

- conversation turn;
- source interview turn when available;
- whether it is direct or inferred;
- confidence;
- the agent that produced it.

Example:

```python
{
    "source_turn": 8,
    "source_evidence_ids": ["E012", "E018"],
    "support_level": "integrated",
    "created_by": "JudgeAgent"
}
```

## 8.4 Contradictions

Do not immediately overwrite conflicting memories.

Store both observations and add a relation:

```text
Observation A --contradicts--> Observation B
```

The Judge may later request clarification.

## 8.5 Context retrieval

`get_context_for_prompt()` should return a compact, relevant context rather than all graph content.

Recommended sections:

```json
{
  "current_topic": {},
  "relevant_completed_topics": [],
  "recent_observations": [],
  "open_questions": [],
  "contradictions": [],
  "tom_context": [],
  "uncertainties": []
}
```

Use simple heuristics first:

- current topic;
- recent turns;
- connected social episode;
- unresolved contradiction;
- observations referenced by the Judge.

Do not require an embedding database in the first implementation.

---

## 9. Interaction Orchestration

## 9.1 New loop

Replace the fixed necessity score loop with the following state machine.

```python
for topic in topics:
    initialize_topic_state(topic)

    next_goal = "start_topic"
    remaining_budget = config.max_questions_per_topic

    while remaining_budget > 0:
        question_request = build_question_request(
            topic=topic,
            interaction_goal=next_goal,
            state=state,
            memory=memory,
            remaining_budget=remaining_budget,
        )

        question_output = question_agent.generate(question_request)

        answer_output = get_answer(
            question=question_output.text,
            automated=automated,
            evidence_store=evidence_store,
            history=topic_history,
        )

        turn = record_turn(question_output, answer_output)

        judge_output = judge_agent.evaluate(
            topic=topic,
            turn=turn,
            topic_history=topic_history,
            memory=memory,
            scoring_standard=current_standard,
        )

        memory.update_from_turn(turn, judge_output)
        remaining_budget -= 1

        if judge_output.warnings:
            handle_warnings(judge_output.warnings)

        if not judge_output.continue_topic:
            break

        next_goal = judge_output.next_goal or "deepen_understanding"

    topic_assessment = scoring_agent.score(...)
    memory.complete_topic(topic_assessment)
```

## 9.2 Budget configuration

Use soft and hard limits.

```python
MIN_QUESTIONS_PER_TOPIC = 1
SOFT_MAX_QUESTIONS_PER_TOPIC = 3
HARD_MAX_QUESTIONS_PER_TOPIC = 5
```

Behavior:

- Judge may stop after one question.
- Judge may continue beyond the soft maximum only when it explains that a high-value gap, contradiction, or safety issue remains.
- Never exceed the hard maximum except for a dedicated safety flow.

## 9.3 Fallback behavior

If Question Agent output is invalid:

- use the raw text as `text`;
- set `focus="general_followup"`;
- log the schema error.

If Judge Agent output is invalid:

- use a conservative heuristic:
  - continue if only one question has been asked and the answer is short;
  - otherwise stop;
- log the malformed response.

If Answer Agent fails:

- return a natural uncertainty response;
- mark support as `unsupported`;
- do not fabricate.

---

## 10. Configuration

Add a central method configuration, for example in `src/config.py` or a new `src/method_config.py`.

```python
METHOD_CONFIG = {
    "method": "tom",
    "enable_tom_questioning": True,
    "enable_grounded_answer_agent": True,
    "enable_judge_agent": True,
    "enable_flexible_memory": True,
    "enable_summary_score_adjustment": False,
    "allow_demographic_guessing": False,
    "min_questions_per_topic": 1,
    "soft_max_questions_per_topic": 3,
    "hard_max_questions_per_topic": 5,
    "question_temperature": 0.3,
    "answer_temperature": 0.2,
    "judge_temperature": 0.0,
    "scoring_temperature": 0.0,
}
```

The implementation should support:

```python
method="baseline"
```

to preserve the current behavior for comparison.

---

## 11. Suggested File Changes

### Modify

```text
src/agents.py
src/assessment.py
src/generate_response.py
src/memory.py
src/utils.py
src/config.py
src/data_load.py
src/main.py
```

### Add

```text
src/models.py
src/judge.py                 # optional wrapper
src/evidence.py              # optional evidence-store helper
src/method_config.py         # optional
src/prompts/
    question_agent.txt
    answer_agent.txt
    judge_agent.txt
    scoring_agent.txt
    summary_agent.txt
tests/
    test_question_output.py
    test_judge_output.py
    test_grounded_answer.py
    test_memory_observations.py
    test_assessment_loop.py
```

External prompt files are recommended because they are easier to iterate on than large inline strings.

---

## 12. Backward Compatibility

Maintain the current public functions where possible.

Example:

```python
def generate_mock_response(...):
    result = grounded_answer_agent.generate(...)
    return result.text
```

`setup_agents()` may continue returning a tuple during migration, but a named container is preferred:

```python
@dataclass
class AgentBundle:
    question: Any
    judge: Any
    scoring: Any
    summary: Any
    user_proxy: Any
```

To avoid breaking existing code immediately:

```python
def setup_agents(chatprompt, method="baseline"):
    if method == "baseline":
        ...
    return AgentBundle(...)
```

---

## 13. Implementation Phases

## Phase 1: Reliable orchestration

- Add structured output parsers.
- Add `JudgeOutput`.
- Replace the necessity score with `continue_topic` and `next_goal`.
- Keep current memory extraction.
- Keep current simulated answer behavior.
- Add method flag.

Success condition:

- Baseline still runs.
- Proposed method completes an assessment without changing score semantics.

## Phase 2: Flexible ToM-aware Question Agent

- Update Question Agent prompt.
- Pass Judge's next goal to Question Agent.
- Add ToM context to memory.
- Ensure ToM is optional and relevant, not mandatory.

Success condition:

- The system can ask natural belief, intention, emotion, evidence, and alternative-interpretation questions when appropriate.
- It does not ask ToM questions for every topic.

## Phase 3: Grounded Answer Agent

- Build evidence IDs from original interview turns.
- Return support metadata.
- Remove default demographic fabrication.
- Add unsupported-claim tests.

Success condition:

- Simulated answers remain consistent with source interview.
- Missing information produces uncertainty rather than invented detail.

## Phase 4: Flexible Memory

- Store open-ended observations.
- Preserve provenance and contradictions.
- Implement compact relevant context retrieval.

Success condition:

- New observation types can be stored without code changes.
- Question Agent does not repeatedly ask already answered questions.

## Phase 5: Reporting and evaluation

- Separate score evidence and ToM observations.
- Disable score adjustment by default.
- Add uncertainty section.
- Add experiment logging.

Success condition:

- Final report clearly distinguishes official scale score from contextual interpretation.

---

## 14. Testing Requirements

## 14.1 Unit tests

### Question Agent parser

Test:

- valid JSON;
- JSON inside markdown fences;
- raw text fallback;
- missing optional fields;
- unknown `focus` value.

### Judge Agent parser

Test:

- continue with next goal;
- stop topic;
- unknown next goal;
- missing confidence;
- malformed JSON fallback;
- warning propagation.

### Grounded Answer Agent

Test:

- direct answer from one source turn;
- answer combining two compatible turns;
- uncertain answer when no evidence exists;
- no invented symptom frequency;
- no invented demographic information;
- no hidden scale-score leakage.

### Memory

Test:

- store unknown observation type;
- preserve provenance;
- retrieve relevant topic context;
- store contradiction without overwriting;
- serialize graph context.

### Scoring

Test:

- frequency and impact remain separate;
- ToM observation does not increase PHQ score;
- uncertain frequency yields low confidence;
- valid score range enforcement.

## 14.2 Integration tests

Create small deterministic fixtures.

### Case A: Sufficient direct answer

Expected:

- one question;
- Judge stops;
- correct score.

### Case B: Ambiguous frequency

Expected:

- Judge requests frequency clarification;
- Question Agent asks a natural follow-up;
- score uses clarified frequency.

### Case C: Relevant social interpretation

Expected:

- Judge identifies a useful social-cognitive gap;
- Question Agent asks a ToM-related question;
- memory stores interpretation separately;
- scale score remains standards-based.

### Case D: Unsupported simulated answer

Expected:

- Answer Agent expresses uncertainty;
- Judge marks low grounding;
- no false memory is stored.

### Case E: Contradiction

Expected:

- both statements are stored;
- Judge requests clarification;
- contradiction is resolved or remains explicitly unresolved.

---

## 15. Logging and Experiment Outputs

Log one JSON record per turn.

```json
{
  "participant_id": "P001",
  "topic": "Low self-worth",
  "turn_id": 4,
  "question": "...",
  "question_focus": "supporting_evidence",
  "answer": "...",
  "answer_support_level": "integrated",
  "answer_evidence_ids": ["E004", "E009"],
  "judge_continue": true,
  "judge_next_goal": "explore_alternative_interpretation",
  "judge_confidence": 0.82,
  "memory_observation_ids": ["O017"],
  "remaining_budget": 1
}
```

This log should support:

- debugging;
- agent behavior analysis;
- question-count evaluation;
- unsupported-claim evaluation;
- ablation experiments;
- reproducibility.

---

## 16. Evaluation Plan

Compare:

```text
M0: Current AgentMental baseline
M1: Baseline + flexible Judge
M2: M1 + ToM-aware Question Agent
M3: M2 + grounded Answer Agent
M4: Full proposed method + flexible memory
```

### Clinical metrics

- item-level MAE;
- total-score MAE;
- F1;
- precision;
- recall;
- Cohen's Kappa;
- calibration if probabilities are available.

### Interaction metrics

- average questions per topic;
- repeated-question rate;
- topic completion rate;
- contradiction-resolution rate;
- useful-information yield;
- malformed-agent-output rate.

### Grounding metrics

- unsupported claim rate;
- contradiction with source interview;
- source evidence coverage;
- human-rated participant fidelity.

### ToM interaction metrics

Do not treat them as clinical diagnostic metrics.

Evaluate:

- relevance of ToM question;
- whether the question reduced uncertainty;
- whether the response was supported;
- whether evidence or alternative explanations were elicited;
- expert-rated usefulness of the social interpretation summary.

---

## 17. Acceptance Criteria

The proposed implementation is acceptable when all of the following are true:

1. The current baseline can still run.
2. Proposed mode uses a Judge decision rather than only a `0/1/2` necessity score.
3. Question Agent can ask ToM-aware questions but is not forced to do so for every topic.
4. Answer Agent does not fabricate unsupported interview content.
5. Memory accepts open-ended observation types.
6. Scale scoring remains based on official standards.
7. ToM observations remain separate from scale scores.
8. The interaction can stop early when sufficient.
9. The interaction has a hard question budget.
10. Agent output failures have safe fallbacks.
11. Every stored inference has provenance and uncertainty where possible.
12. Final reports distinguish:
    - symptom score;
    - contextual evidence;
    - social-cognitive observations;
    - uncertainty.
13. Tests cover grounding, flexible Judge goals, contradiction handling, and score separation.

---

## 18. Coding Guidance for the Implementation Agent

When implementing this specification:

- Prefer incremental refactoring over a full rewrite.
- Keep functions small and testable.
- Introduce typed models before changing the main loop.
- Preserve legacy wrappers until integration tests pass.
- Do not hard-code every possible ToM concept.
- Do not turn agent prompts into long questionnaires.
- Use broad capabilities and adaptive goals.
- Use schemas for transport reliability, not to constrain reasoning.
- Store unknown observation types safely.
- Keep official clinical scoring deterministic and conservative.
- Treat LLM-derived interpretations as observations with uncertainty, not facts.
- Make all major behaviors configurable.
- Avoid coupling the implementation to one model provider.
- Keep prompts external or centrally defined.
- Add clear logs for every agent decision.

---

## 19. Recommended First Pull Request

The first implementation PR should be limited to:

1. Add `models.py` with `QuestionOutput`, `AnswerOutput`, and `JudgeOutput`.
2. Add robust structured-output parsing.
3. Convert Necessity Agent into Judge Agent.
4. Update `assessment.py` to use `continue_topic` and `next_goal`.
5. Add a flexible ToM-aware Question Agent prompt.
6. Keep the existing Answer Agent and memory implementation temporarily.
7. Add baseline/proposed method flag.
8. Add unit tests for parser and orchestration fallback.

This produces the core proposed interaction behavior with low migration risk.

The grounded Answer Agent and flexible memory can be implemented in later pull requests.

---

## 20. Final Method Summary

The proposed method is not a rigid ToM questionnaire.

It is an adaptive AgentMental extension in which:

- the **Question Agent** uses symptom reasoning, Cognitive ToM, and Affective ToM as optional tools for choosing the most informative next question;
- the **Answer Agent** uses grounded contextual reasoning to remain faithful to the original interview;
- the **Judge Agent** combines decoding and reasoning to evaluate sufficiency, uncertainty, contradictions, and the next conversational objective;
- the **Memory Graph** stores flexible evidence and social-cognitive observations with provenance;
- the **Scoring Agent** applies official scale standards independently from ToM observations;
- the **Summary Agent** reports symptoms, context, ToM-related patterns, and uncertainty as separate components.

The key implementation principle is:

> Keep orchestration structured, but keep agent reasoning flexible.
