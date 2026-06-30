# -----------------------------------------------------------------------------
# Reasoning evaluation (Eval/Reasoning_Evaluation.py & Training)
# -----------------------------------------------------------------------------

DECOMPOSE_REASONING_PROMPT = """You are a medical reasoning verification module.

### Task
You will be given:
- Image: A medical image (provided separately)
- Problem: A medical VQA question about the image
- Ground_Truth: The correct answer to the problem
- Gold_Reasoning: Gold reasoning steps (reference standard)
- Reasoning_Sentences: Generated reasoning sentences to evaluate

For EACH generated sentence, output:
- Alignment: 0/1
- Contribution: 0/1


### General Rules (Apply First)

**Automatic 0 for BOTH Alignment and Contribution:**
- Meta-Commentary: "I will identify...", "Consider the possibilities..."
- Empty or meaningless statements
- Pure repetition of previous steps without new content

Check these rules FIRST before evaluating Alignment and Contribution.


### Evaluation Criteria

**First, extract Key Elements from Gold_Reasoning:**
- Modality/Context, Key Findings, Anatomical Location (including laterality), Diagnostic Direction

---

#### Alignment (Gold_Reasoning Consistency)

"Is this step consistent with Gold_Reasoning?"

Alignment ONLY checks whether the step matches Gold_Reasoning, regardless of contribution to answer.

**Step Roles:**

| Position | Role | Alignment = 1 if |
|----------|------|------------------|
| Early (First 1-2 steps) | Context-setting | Correctly identifies modality (e.g., "X-ray", "CT", "MRI", "electron microscopy"), tissue type (e.g., "histological section", "gross specimen"), or staining method (e.g., "H&E", "immunostain") that matches Gold_Reasoning |
| Middle (Middle steps) | Observation | Identifies **specific findings** mentioned in Gold_Reasoning, including: abnormalities, pathological features, key structures, AND correct anatomical location/laterality. Must match Gold's level of specificity. |
| Later (Last 1-2 steps) | Inference | Reaches or clearly approaches the **same diagnostic conclusion** as Gold_Reasoning. Must demonstrate diagnostic reasoning toward Ground_Truth, not just restate observations. |

**Alignment = 0 if:**
- Wrong Location/Laterality: Gold says "LEFT" but generated says "right"
- Contradiction: Directly contradicts Gold_Reasoning
- Misdirection: Different diagnostic direction than Gold
- Content not mentioned or supported by Gold_Reasoning
- Missing Critical Findings: Gold identifies pathological/abnormal findings but generated only describes normal or generic features
- Specificity Mismatch: Gold is specific (e.g., "lymphoma", "abnormal features") but generated is generic (e.g., "cellular structure", "tissue")

**Important:** 
- If Gold identifies specific pathology (e.g., "granuloma", "infarction", "metastasis", "fibrosis") but generated only describes generic features (e.g., "tissue changes", "some abnormality", "lesion") → Alignment = 0
- If Gold_Reasoning contains diagnostic conclusions (e.g., "tuberculosis", "adenocarcinoma", "hemorrhage", "fracture"), generated reasoning MUST progress toward that diagnosis to get Alignment = 1
- Describing only normal-appearing structures when Gold identifies abnormalities = Alignment 0

---

#### Contribution (Ground_Truth Derivation)

"Does this step directly help reach Ground_Truth: '{solution}'?"

**Contribution = 1 if:**
- Directly mentions Ground_Truth or semantically equivalent terms
- Identifies a finding that is explicitly required to derive Ground_Truth
- States the specific diagnosis, location, or structure that matches Ground_Truth

**Contribution = 0 if:**
- No direct relevance to Ground_Truth
- Generic observation that applies to any image of this type
- Describes features not connected to Ground_Truth
- Evasion: "Unknown", "cannot determine", "None"
- Context-only: states modality/setting without advancing toward Ground_Truth


### Example

Problem: "Where is the mass located?"
Ground_Truth: "lower left lung"
Gold_Reasoning: "The mass is identified in the lower left lung by examining the X-ray image, showing increased opacity."

Key Elements: X-ray, mass, lower LEFT lung, increased opacity

Reasoning_Sentences:
1. "The image is a chest X-ray."
2. "Signs of infection such as consolidation would be abnormal."
3. "There is a mass in the lower left lung field."
4. "There is opacification in the right upper lung."
5. "I will now analyze the findings."

Output:
{{
  "Reasoning_Check": {{
    "step1": {{"Alignment": 1, "Contribution": 0}},
    "step2": {{"Alignment": 0, "Contribution": 0}},
    "step3": {{"Alignment": 1, "Contribution": 1}},
    "step4": {{"Alignment": 0, "Contribution": 0}},
    "step5": {{"Alignment": 0, "Contribution": 0}}
  }}
}}

Why:
- step1: Alignment=1 (matches Gold's X-ray), Contribution=0 (context-only)
- step2: Alignment=0 (misdirection to infection), Contribution=0 (wrong direction)
- step3: Alignment=1 (matches Gold), Contribution=1 (directly states Ground_Truth)
- step4: Alignment=0 (wrong location), Contribution=0 (wrong location)
- step5: Alignment=0, Contribution=0 (General Rule: meta-commentary)


### Output Format
Return JSON only:
{{
  "Reasoning_Check": {{
    "step1": {{"Alignment": 1, "Contribution": 0}},
    "step2": {{"Alignment": 0, "Contribution": 1}},
    ...
  }}
}}


### Rules
- Apply General Rules FIRST (meta-commentary → both 0)
- Evaluate Alignment and Contribution INDEPENDENTLY
- Both values must be 0 or 1
- Do NOT output explanations outside JSON


### Inputs
- Problem: {problem}
- Ground_Truth: {solution}
- Gold_Reasoning: {gold_reasoning}
- Reasoning_Sentences: {sentences}

"""


# -----------------------------------------------------------------------------
# Answer evaluation (Eval/Answer_Evaluation.py)
# -----------------------------------------------------------------------------

ANSWER_EVAL_SYSTEM_PROMPT = (
    "Given a question about a medical image, there is a correct answer to the "
    "question and an answer to be determined. If the answer to be determined "
    "matches the correct answer or is a good enough answer to the question, "
    "output 'O'; otherwise output 'X'. Respond with a single character: "
    "'O' (correct) or 'X' (incorrect)."
)

ANSWER_EVAL_USER_PROMPT = """Question:
- question about the medical image: {problem}

Answers:
- correct answer (ground truth): {solution}
- answer to be determined: {generated_answer}

Your response must be a single character: 'O' (correct) or 'X' (incorrect)."""
