import networkx as nx
import uuid
from openai import OpenAI
import json
import logging
from config import get_openai_settings

logger = logging.getLogger(__name__)

class MemoryGraph:
    def __init__(self, user_identification):
        self.graph = nx.DiGraph()
        self.user_node = "User"
        self.graph.add_node(self.user_node, type="User", info=user_identification)
        logger.info(f"MemoryGraph initialized for user: {user_identification}")
        settings = get_openai_settings()
        self.model_name = settings["model"]
        self.client = OpenAI(
            base_url=settings["base_url"],
            api_key=settings["api_key"]
        )

    def add_topic(self, topic_name):
        if not self.graph.has_node(topic_name):
            self.graph.add_node(topic_name, type="Topic", status="ongoing")
            self.graph.add_edge(self.user_node, topic_name, relation="assessed_on")
            logger.info(f"[MemoryGraph] Added new topic: {topic_name}")

    def extract_key_info_with_api(self, user_response, topic):
        prompt = f"""
You are a psychological assessment assistant. Extract key information from the user's response related to the topic '{topic}' and return it in JSON format.

**Input**:
User response: "{user_response}"

**Task**:
1. Extract key entities in these categories (only include information explicitly mentioned):
   - Emotion (e.g., sadness, anxiety, happiness)
   - Frequency (e.g., often, sometimes, always)
   - Symptom (e.g., insomnia, fatigue, appetite loss)
   - Duration (e.g., days, weeks, one month)
   - Impact (e.g., work, life, social)
2. Do not infer or add information not present in the response.
3. If no entities are extracted, provide a brief summary (max 20 characters).

**Output Format**:
```json
{{
  "entities": {{
    "Emotion": ["<entity1>", "<entity2>"],
    "Frequency": ["<entity1>"],
    "Symptom": ["<entity1>", "<entity2>"],
    "Duration": ["<entity1>"],
    "Impact": ["<entity1>"]
  }},
  "summary": "<brief summary if no entities>"
}}
```

**Example**:
Input: "I often feel sad, have insomnia for a month, and it affects my work."
Output:
```json
{{
  "entities": {{
    "Emotion": ["sadness"],
    "Frequency": ["often"],
    "Symptom": ["insomnia"],
    "Duration": ["one month"],
    "Impact": ["work"]
  }},
  "summary": ""
}}
```
"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a psychological assessment assistant. Extract key information strictly as instructed and return JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=512
            )
            response = completion.choices[0].message.content.strip()
            logger.info(f"[MemoryGraph] API extraction response: {response}")

            if response.startswith("```json"):
                response = response[len("```json"):].strip()
            if response.startswith("```"):
                response = response[len("```"):].strip()
            if response.endswith("```"):
                response = response[:-len("```")].strip()

            try:
                data = json.loads(response)
                entities = data.get("entities", {})
                summary = data.get("summary", "")

                extracted_info = []
                for category, values in entities.items():
                    if values: 
                        extracted_info.append(f"{category}: {', '.join(values)}")

                if not extracted_info and summary:
                    extracted_info.append(f"Summary: {summary}")
                elif not extracted_info:
                    summary = user_response[:20] + ("..." if len(user_response) > 20 else "")
                    extracted_info.append(f"Summary: {summary}")

                key_info = "; ".join(extracted_info)
                logger.info(f"[MemoryGraph] Extracted key info for topic '{topic}': {key_info}")
                return key_info

            except json.JSONDecodeError:
                logger.error(f"[MemoryGraph] Failed to parse API response as JSON: {response}")
                summary = user_response[:20] + ("..." if len(user_response) > 20 else "")
                logger.info(f"[MemoryGraph] Falling back to default summary: {summary}")
                return f"Summary: {summary}"

        except Exception as e:
            logger.exception(f"[MemoryGraph] Error calling API for key info extraction: {e}")
            summary = user_response[:20] + ("..." if len(user_response) > 20 else "")
            logger.info(f"[MemoryGraph] Falling back to default summary: {summary}")
            return f"Summary: {summary}"
    
    def add_short_term_memory(self, topic_name, user_response, turn_id):
        key_info = self.extract_key_info_with_api(user_response, topic_name)

        statement_id = f"statement_{uuid.uuid4().hex}"
        
        self.graph.add_node(
            statement_id, 
            type="Statement", 
            content=key_info, 
            raw_text=user_response,
            source_turn=turn_id 
        )
        
        self.graph.add_edge(topic_name, statement_id, relation="has_statement")
        logger.info(f"[MemoryGraph] Added STM for '{topic_name}' (Node ID: {statement_id}, Key Info: {key_info})")
        return statement_id

    def _heuristic_tom_hypotheses(self, user_response, statement_id):
        hypotheses = []
        lowered = user_response.lower()

        if any(term in lowered for term in ["don't want to bother", "don't want to trouble", "burden", "make things worse for them"]):
            hypotheses.append({
                "hypothesis_type": "social_belief",
                "label": "perceived_burdensomeness",
                "hypothesis_text": "The participant may believe their needs would burden other people.",
                "evidence_span_ids": [statement_id],
                "confidence": 0.65,
                "alternative_explanations": ["The participant may simply prefer privacy."],
                "missing_information": ["Whether the participant avoids support because they feel like a burden."],
                "verification_status": "unverified",
            })

        if any(term in lowered for term in ["i'm okay though", "probably okay", "not that bad", "fine i guess"]):
            hypotheses.append({
                "hypothesis_type": "communicative_intent",
                "label": "possible_symptom_minimization",
                "hypothesis_text": "The participant may be minimizing the severity or frequency of symptoms.",
                "evidence_span_ids": [statement_id],
                "confidence": 0.6,
                "alternative_explanations": ["The participant may genuinely see the symptom as mild."],
                "missing_information": ["How often the symptom occurs over the past two weeks."],
                "verification_status": "unverified",
            })

        if any(term in lowered for term in ["worthless", "failure", "useless", "not useful anymore"]):
            hypotheses.append({
                "hypothesis_type": "self_belief",
                "label": "negative_self_belief",
                "hypothesis_text": "The participant may hold a negative belief about their self-worth.",
                "evidence_span_ids": [statement_id],
                "confidence": 0.75,
                "alternative_explanations": ["The statement may reflect temporary frustration rather than a persistent belief."],
                "missing_information": ["Whether this belief is recurrent during the last two weeks."],
                "verification_status": "unverified",
            })

        return hypotheses[:3]

    def add_tom_hypotheses(self, topic_name, hypotheses, fallback_statement_id=None, user_response=""):
        if not hypotheses and fallback_statement_id:
            hypotheses = self._heuristic_tom_hypotheses(user_response, fallback_statement_id)

        created_ids = []
        for item in hypotheses[:3]:
            hypothesis_id = f"tom_{uuid.uuid4().hex}"
            evidence_ids = item.get("evidence_span_ids") or ([fallback_statement_id] if fallback_statement_id else [])
            self.graph.add_node(
                hypothesis_id,
                type="ToMHypothesis",
                hypothesis_type=item.get("hypothesis_type", "unknown"),
                label=item.get("label", "unknown"),
                hypothesis_text=item.get("hypothesis_text", ""),
                evidence_span_ids=evidence_ids,
                confidence=item.get("confidence", 0.0),
                alternative_explanations=item.get("alternative_explanations", []),
                missing_information=item.get("missing_information", []),
                verification_status=item.get("verification_status", "unverified"),
            )
            self.graph.add_edge(topic_name, hypothesis_id, relation="has_hypothesis")
            self.graph.add_edge(hypothesis_id, topic_name, relation="relevant_to_topic")
            self.graph.add_edge(hypothesis_id, topic_name, relation="requires_verification")
            for evidence_id in evidence_ids:
                if evidence_id and self.graph.has_node(evidence_id):
                    self.graph.add_edge(hypothesis_id, evidence_id, relation="supported_by")
            created_ids.append(hypothesis_id)
            logger.info(f"[MemoryGraph] Added ToM hypothesis for '{topic_name}' (Node ID: {hypothesis_id})")

        return created_ids

    def update_hypothesis_status(self, hypothesis_ids, status):
        for hypothesis_id in hypothesis_ids:
            if self.graph.has_node(hypothesis_id) and self.graph.nodes[hypothesis_id].get("type") == "ToMHypothesis":
                self.graph.nodes[hypothesis_id]["verification_status"] = status
                logger.info(f"[MemoryGraph] Updated hypothesis '{hypothesis_id}' to status '{status}'.")

    def _trigger_holistic_reassessment(self, new_topic_name, new_topic_score, new_topic_summary, new_topic_statements_str):
        past_completed_topics = [
            n for n, d in self.graph.nodes(data=True) 
            if d.get('type') == 'Topic' 
            and d.get('status') == 'completed' 
            and n != new_topic_name
        ]

        if not past_completed_topics:
            logger.info(f"[MemoryGraph] No past topics to reassess. '{new_topic_name}' is the first completed topic.")
            return

        logger.info(f"[MemoryGraph] Triggering HOLISTIC reassessment of {len(past_completed_topics)} past topics based on new info from '{new_topic_name}'.")

        past_assessments_context = []
        for topic_name in past_completed_topics:
            node_data = self.graph.nodes[topic_name]
            past_assessments_context.append({
                "topic_name": topic_name,
                "current_score": node_data.get('score'),
                "current_basis": node_data.get('summary')
            })
        
        new_evidence_context = {
            "topic_name": new_topic_name,
            "score": new_topic_score,
            "summary": new_topic_summary,
            "supporting_statements": new_topic_statements_str
        }

        holistic_reassessment_prompt = f"""
You are a clinical psychologist reviewing a patient's assessment. A new topic has been completed. Your task is to perform a consistency check on past assessments and update the `basis` **only if necessary**

【NEW EVIDENCE】(The newly completed topic)
{json.dumps(new_evidence_context, indent=2, ensure_ascii=False)}

【PAST ASSESSMENTS TO REVIEW】
{json.dumps(past_assessments_context, indent=2, ensure_ascii=False)}

【TASK】
Your primary goal is to ensure consistency. Do not update the `basis` without a strong reason.
1.  **Determine Necessity:** For each past topic, ask: Does the "NEW EVIDENCE" directly contradict, supplement, or significantly alter the understanding of this past topic?
2.  **If NO UPDATE is needed:** If the new evidence is irrelevant or doesn't change the assessment, set "update_required": false. This should be your default action.
3.  **If UPDATE IS NEEDED:**
    - Update the `basis` to reflect how the "NEW EVIDENCE" supplements or corrects the past assessment.
    - Set `"update_required": true`.

【OUTPUT INSTRUCTIONS】
Provide a strict JSON response with a single key "results", which is a list of objects.

**JSON Schema:**
```json
{{
  "results": [
    {{
      "topic_name": "<Topic name>",
      "update_required": true,
      "new_basis": "<Updated basis for the assessment>"
    }},
    {{
      "topic_name": "<Topic name>",
      "update_required": false
    }}
  ]
}}
```
"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a senior clinical psychologist performing a file review. Output your findings in the specified JSON format only."},
                    {"role": "user", "content": holistic_reassessment_prompt}
                ],
                temperature=0,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            response_str = completion.choices[0].message.content.strip()
            response_data = json.loads(response_str)
            logger.info(f"[MemoryGraph] Holistic reassessment API response: {json.dumps(response_data, indent=2)}")

            results = response_data.get("results", [])
            
            if not results:
                logger.info("[MemoryGraph] Holistic reassessment complete. No results returned by API.")
                return

            for result_item in results:
                topic_to_update = result_item.get("topic_name")
                if not self.graph.has_node(topic_to_update):
                    logger.error(f"[MemoryGraph] API suggested action for non-existent topic '{topic_to_update}'. Skipping.")
                    continue
                update_flag = result_item.get("update_required")
                if str(update_flag).lower() == 'true':
                    new_basis = result_item.get("new_basis")
                    if new_basis:
                        nx.set_node_attributes(self.graph, {
                            topic_to_update: {
                                "summary": new_basis
                            }
                        })
                        logger.info(f"[MemoryGraph] Updated basis for topic '{topic_to_update}': {new_basis}")
                    else:
                        logger.warning(f"[MemoryGraph] Update required for '{topic_to_update}', but new_basis was not provided. Skipping update.")
                else:
                    logger.info(f"[MemoryGraph] No update required for topic '{topic_to_update}'. Skipping.")

        except Exception as e:
            logger.error(f"[MemoryGraph] FAILED during holistic reassessment. Error: {e}")

    def convert_topic_to_long_term(self, topic_name, score, summary):
        if self.graph.has_node(topic_name):
            nx.set_node_attributes(self.graph, {
                topic_name: {
                    "status": "completed",
                    "score": score,
                    "summary": summary
                }
            })
            logger.info(f"[MemoryGraph] Converted topic '{topic_name}' to LTM with score {score}.")
        
        statements = [f"- {self.graph.nodes[n].get('content')}" for n in self.graph.successors(topic_name) if self.graph.nodes[n].get('type') == 'Statement']
        statements_str = "\n".join(statements)

        self._trigger_holistic_reassessment(topic_name, score, summary, statements_str)

    def update_topic_score(self, topic_name, updated_score, reason):
        if self.graph.has_node(topic_name) and self.graph.nodes[topic_name].get('status') == 'completed':
            nx.set_node_attributes(self.graph, {
                topic_name: {
                    "updated_score": updated_score,
                    "update_reason": reason
                }
            })
            logger.info(f"[MemoryGraph] Updated score for topic '{topic_name}' to {updated_score} with reason: '{reason}'.")
        else:
            logger.warning(f"[MemoryGraph] Attempted to update non-existent or non-completed topic '{topic_name}'.")

    def get_context_for_prompt(self, current_topic):
        memory_data = {
            "long_term_memory": [],
            "short_term_memory": [],
            "tom_hypotheses": []
        }

        completed_topics = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'Topic' and d.get('status') == 'completed']
        for topic in completed_topics:
            attrs = self.graph.nodes[topic]
            memory_data["long_term_memory"].append({
                "topic": topic,
                "score": attrs.get('score', 'N/A'),
                "summary": attrs.get('summary', 'N/A'),
                "updated_score": attrs.get('updated_score', None),
                "update_reason": attrs.get('update_reason', None)
            })

        if self.graph.has_node(current_topic):
            statements = []
            for neighbor in self.graph.successors(current_topic):
                if self.graph.nodes[neighbor].get('type') == 'Statement':
                    statements.append({
                        "node_id": neighbor,
                        "content": self.graph.nodes[neighbor].get('content'),
                        "raw_text": self.graph.nodes[neighbor].get('raw_text', ''),
                        "source_turn": self.graph.nodes[neighbor].get('source_turn')
                    })
            memory_data["short_term_memory"] = statements

            hypotheses = []
            for neighbor in self.graph.successors(current_topic):
                if self.graph.nodes[neighbor].get('type') == 'ToMHypothesis':
                    hypotheses.append({
                        "node_id": neighbor,
                        "hypothesis_type": self.graph.nodes[neighbor].get('hypothesis_type'),
                        "label": self.graph.nodes[neighbor].get('label'),
                        "hypothesis_text": self.graph.nodes[neighbor].get('hypothesis_text'),
                        "evidence_span_ids": self.graph.nodes[neighbor].get('evidence_span_ids', []),
                        "confidence": self.graph.nodes[neighbor].get('confidence', 0.0),
                        "alternative_explanations": self.graph.nodes[neighbor].get('alternative_explanations', []),
                        "missing_information": self.graph.nodes[neighbor].get('missing_information', []),
                        "verification_status": self.graph.nodes[neighbor].get('verification_status', 'unverified')
                    })
            memory_data["tom_hypotheses"] = hypotheses

        context = json.dumps(memory_data, ensure_ascii=False, indent=2)
        logger.info(f"[MemoryGraph] Generated JSON context for topic '{current_topic}': {context}")
        return context
