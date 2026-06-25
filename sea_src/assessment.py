import pandas as pd
import logging
import autogen
import json
import os
from utils import get_valid_input, categorize_score, extract_score_and_summary, extract_summary_and_updated_scores, generate_score_table, is_necessary, makerequest, parse_personal_info, extract_score, save_assessment_results, is_file_already_evaluated, custom_speaker_selection_func, generate_report, extract_json_object
from agents import setup_agents
from data_load import load_real_data
from logging_setup import dialog_print
from config import get_llm_config
from memory import MemoryGraph
from generate_response import generate_mock_response
from self_evolution import SelfEvolutionManager

logger = logging.getLogger(__name__)


def perform_assessment(topics, chatprompt, agents, scale_name, scoring_standards, real_interview, scale_scores, case_identifier="unknown_case", automated=False):
    try:
        llm_config = get_llm_config()
        logger.info("Starting psychological assessment task.")
        question_agent, tom_agent, scoring_agent, necessity_agent, summary_agent, user_proxy = agents
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self_evolution = SelfEvolutionManager(logger, automated=automated, repo_root=repo_root)
        groupchat = autogen.GroupChat(
            agents=[question_agent, tom_agent, scoring_agent, necessity_agent, summary_agent, user_proxy],
            messages=[],
            max_round=2,
            speaker_selection_method=custom_speaker_selection_func,
        )
        group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        total_history = []
        scores = []
        identification = ""
        last_topic = "Basic Information Collection"
        last_question = ""
        last_response = ""
        qa_count = 0
        groupchat.reset()  

        initial_message = (
            f"Hello, I am your dedicated psychological assistant. I will conduct an interview with you based on {scale_name} to assess the severity of related symptoms. Please note that this is only a preliminary screening and cannot replace formal psychiatric diagnosis and treatment. "
            "First, for the accuracy of the assessment, I would like to collect your basic information: age, gender, occupation. If you're ready, let's begin."
        )
        dialog_print(f"Question: {initial_message}")
        logger.info(f"Initial message sent: {initial_message}")
        logger.info("Initial message has been delivered to the user.")

        if automated:
            user_response = generate_mock_response(initial_message, topic=None, identification="", real_interview=real_interview, scale_scores=scale_scores, 
                                                   scoring_standard=None, current_topic_history=None, scale_name=scale_name)
            dialog_print(f"Simulated answer: {user_response}")
            age, gender, occupation = parse_personal_info(user_response)
            if age is None or gender is None or occupation is None:
                logger.error("Unable to parse basic information in automated mode.")
                return "Unable to parse basic information in automated mode.", 0, 0, {}
        else:
            while True:
                user_response = get_valid_input("Your response (e.g., 25, male, engineer or age:25, gender:male, occupation:engineer): ")
                age, gender, occupation = parse_personal_info(user_response)
                if age is not None and gender is not None and occupation is not None:
                    logger.info(f"Successfully parsed user basic information: age={age}, gender={gender}, occupation={occupation}")
                    break
                else:
                    missing_fields = []
                    if age is None:
                        missing_fields.append("Age")
                    if gender is None:
                        missing_fields.append("Gender")
                    if occupation is None:
                        missing_fields.append("Occupation")
                    dialog_print(f"Unable to parse your input. Please ensure it includes the following information: {', '.join(missing_fields)}. Separate items with commas, commas in Chinese enumeration, spaces, or keywords (e.g., 25, male, engineer or age:25, gender:male, occupation:engineer or 25 male engineer).")
        identification = f"Age: {age}, Gender: {gender}, Occupation: {occupation}"
        print(f"\nBasic information: {identification}")

        memory_graph = MemoryGraph(identification)
        total_history.append({"role": "system", "content": identification})
        last_question = initial_message
        last_response = user_response

        for idx, topic in enumerate(topics, 1):
            dialog_print("\n")
            logger.info(f"Starting topic {idx}/{len(topics)}: {topic}")
            dialog_print(f"{'-'*20}Current Topic: {topic}")

            memory_graph.add_topic(topic)
            asked_questions = 0
            depth = 0
            max_depth = 3
            current_topic_history = []
            fidelity_records = []
            necessity_records = []
            topic_retrieved_lessons = {}

            while depth < max_depth:
                question_type = "initial" if depth == 0 else "followup"
                memory_context_str = memory_graph.get_context_for_prompt(topic)
                memory_context = extract_json_object(memory_context_str, {})
                active_tom_hypotheses = [
                    item for item in memory_context.get("tom_hypotheses", [])
                    if item.get("verification_status", "unverified") == "unverified"
                ]
                pattern_tags = self_evolution.derive_pattern_tags(topic, memory_context)
                question_lessons, question_lesson_ids = self_evolution.retrieve_lessons(
                    case_identifier,
                    topic,
                    "QuestionAgent",
                    pattern_tags,
                )
                topic_retrieved_lessons.setdefault("QuestionAgent", [])
                topic_retrieved_lessons["QuestionAgent"] = sorted(
                    set(topic_retrieved_lessons["QuestionAgent"] + question_lesson_ids)
                )
                other_topics_str = ", ".join([t for t in topics if t != topic])
                question_payload = (
                    f"Type: {question_type}\n"
                    f"Topic: {topic}\n"
                    f"Identification: {identification}\n"
                    f"Last Question: {last_question}\n"
                    f"Last Response: {last_response}\n"
                    f"Memory: {memory_context_str}\n"
                    f"Active ToM Hypotheses: {json.dumps(active_tom_hypotheses, ensure_ascii=False)}\n"
                    f"Retrieved Lessons: {json.dumps(question_lessons, ensure_ascii=False)}\n"
                    f"Other Topics: {other_topics_str}\n"
                )
                question = makerequest(group_chat_manager, user_proxy, question_agent, question_payload)
                # print(groupchat.messages)
                if question is None:
                    question = "Sorry, I cannot generate a question at the moment."
                dialog_print(f"Question: {question}")
                logger.info(f"Question: {question}")
                
                if automated:
                    current_scoring_standard = scoring_standards[scale_name][topic]
                    response_lessons, response_lesson_ids = self_evolution.retrieve_lessons(
                        case_identifier,
                        topic,
                        "ResponseAgent",
                        pattern_tags,
                    )
                    topic_retrieved_lessons.setdefault("ResponseAgent", [])
                    topic_retrieved_lessons["ResponseAgent"] = sorted(
                        set(topic_retrieved_lessons["ResponseAgent"] + response_lesson_ids)
                    )
                    response = generate_mock_response(question, topic, identification, real_interview, scale_scores, scoring_standard=current_scoring_standard, 
                                                      retrieved_lessons=response_lessons, current_topic_history=current_topic_history, depth=depth, scale_name=scale_name)
                    dialog_print(f"\nSimulated answer: {response}")
                else:
                    response = get_valid_input("\nAnswer: ")
                logger.info(f"User's response: {response}")
                scores.append({"topic": topic, "question": question, "response": response})

                group_chat_manager.groupchat.messages.append({
                    "content": response,
                    "role": "user",
                    "name": "UserProxy"
                })

                qa_count += 1
                total_history.append({"role": "assistant", "content": question})
                total_history.append({"role": "user", "content": response})
                current_topic_history.append({"question": question, "response": response})
                last_question = question
                last_response = response
                statement_id = memory_graph.add_short_term_memory(topic, response, turn_id=qa_count)
                if automated:
                    fidelity = self_evolution.audit_response_fidelity(
                        case_identifier,
                        topic,
                        question,
                        response,
                        current_topic_history,
                        real_interview,
                    )
                    if fidelity:
                        fidelity_records.append(fidelity)

                if active_tom_hypotheses:
                    memory_graph.update_hypothesis_status(
                        [item.get("node_id") for item in active_tom_hypotheses if item.get("node_id")],
                        "queried",
                    )

                statement_node = memory_graph.graph.nodes[statement_id]
                latest_statement = {
                    "statement_id": statement_id,
                    "raw_text": statement_node.get("raw_text", response),
                    "extracted_key_info": statement_node.get("content", ""),
                    "source_turn": statement_node.get("source_turn"),
                }
                updated_memory_context_str = memory_graph.get_context_for_prompt(topic)
                updated_memory_context = extract_json_object(updated_memory_context_str, {})
                topic_history_str = "\n".join([f"Q: {qa['question']}\nA: {qa['response']}" for qa in current_topic_history])
                tom_pattern_tags = self_evolution.derive_pattern_tags(topic, updated_memory_context)
                tom_lessons, tom_lesson_ids = self_evolution.retrieve_lessons(
                    case_identifier,
                    topic,
                    "ToMAgent",
                    tom_pattern_tags,
                )
                topic_retrieved_lessons.setdefault("ToMAgent", [])
                topic_retrieved_lessons["ToMAgent"] = sorted(
                    set(topic_retrieved_lessons["ToMAgent"] + tom_lesson_ids)
                )
                tom_payload = (
                    f"Topic: {topic}\n"
                    f"Latest Statement: {json.dumps(latest_statement, ensure_ascii=False)}\n"
                    f"History:\n{topic_history_str}\n"
                    f"Memory: {updated_memory_context_str}\n"
                    f"Retrieved Lessons: {json.dumps(tom_lessons, ensure_ascii=False)}\n"
                )
                tom_output_text = makerequest(group_chat_manager, user_proxy, tom_agent, tom_payload)
                tom_output = extract_json_object(tom_output_text, {"hypotheses": []})
                memory_graph.add_tom_hypotheses(
                    topic,
                    tom_output.get("hypotheses", []),
                    fallback_statement_id=statement_id,
                    user_response=response,
                )

                post_tom_memory_context_str = memory_graph.get_context_for_prompt(topic)
                post_tom_memory_context = extract_json_object(post_tom_memory_context_str, {})
                necessity_pattern_tags = self_evolution.derive_pattern_tags(topic, post_tom_memory_context)
                necessity_lessons, necessity_lesson_ids = self_evolution.retrieve_lessons(
                    case_identifier,
                    topic,
                    "NecessityAgent",
                    necessity_pattern_tags,
                )
                topic_retrieved_lessons.setdefault("NecessityAgent", [])
                topic_retrieved_lessons["NecessityAgent"] = sorted(
                    set(topic_retrieved_lessons["NecessityAgent"] + necessity_lesson_ids)
                )
                necessity_payload = (
                    f"Topic: {topic}\n"
                    f"History:\n{topic_history_str}\n"
                    f"Retrieved Lessons: {json.dumps(necessity_lessons, ensure_ascii=False)}\n"
                )
                necessity_score_text = makerequest(group_chat_manager, user_proxy, necessity_agent, necessity_payload)

                if necessity_score_text is not None:
                    necessity_score = extract_score(necessity_score_text)
                else:
                    necessity_score = 0
                necessity_records.append(
                    {
                        "turn_index": qa_count,
                        "necessity_score": necessity_score,
                        "history_size": len(current_topic_history),
                    }
                )
                asked_questions += 1
                depth += 1
                if is_necessary(necessity_score, asked_questions):
                    pass
                else:
                    break

            topic_history_str = "\n".join([f"Q: {qa['question']}\nA: {qa['response']}" for qa in current_topic_history])
            scoring_standard_str = json.dumps(scoring_standards[scale_name][topic], ensure_ascii=False, indent=2)
            final_topic_memory_context_str = memory_graph.get_context_for_prompt(topic)
            final_topic_memory_context = extract_json_object(final_topic_memory_context_str, {})
            scoring_pattern_tags = self_evolution.derive_pattern_tags(topic, final_topic_memory_context)
            scoring_lessons, scoring_lesson_ids = self_evolution.retrieve_lessons(
                case_identifier,
                topic,
                "ScoringAgent",
                scoring_pattern_tags,
            )
            topic_retrieved_lessons.setdefault("ScoringAgent", [])
            topic_retrieved_lessons["ScoringAgent"] = sorted(
                set(topic_retrieved_lessons["ScoringAgent"] + scoring_lesson_ids)
            )
            scoring_payload = (
                f"Topic: {topic}\n"
                f"History:\n{topic_history_str}\n"
                f"Memory: {final_topic_memory_context_str}\n"
                f"Standard:\n{scoring_standard_str}\n"
                f"Retrieved Lessons: {json.dumps(scoring_lessons, ensure_ascii=False)}\n"
            )
            total_score_text = makerequest(group_chat_manager, user_proxy, scoring_agent, scoring_payload)

            scoring_output = extract_json_object(total_score_text, {})
            if total_score_text is not None:
                total_score, summary = extract_score_and_summary(total_score_text, scale_name)
            else:
                total_score, summary = 0, ""
            scoring_output.setdefault("score", total_score)
            scoring_output.setdefault("summary", summary)

            dialog_print(f"\nTotal score for topic '{topic}': {total_score} points")
            if summary:
                dialog_print(f"Scoring basis: {summary}\n")
            else:
                print()
            scores.append({"topic": topic, "question": "Total score", "response": "", "score": total_score})
            memory_graph.convert_topic_to_long_term(topic, total_score, summary)
            ground_truth_score = self_evolution.get_ground_truth_score(topic, scale_scores)
            self_evolution.run_topic_self_evolution(
                case_id=case_identifier,
                topic_name=topic,
                predicted_score=total_score,
                scoring_payload=scoring_output,
                scoring_rubric=scoring_standards[scale_name][topic],
                topic_history=current_topic_history,
                memory_context=final_topic_memory_context,
                retrieved_lesson_ids=topic_retrieved_lessons,
                pattern_tags=scoring_pattern_tags,
                ground_truth_score=ground_truth_score,
                fidelity_records=fidelity_records,
                necessity_records=necessity_records,
            )

        total_history_str = "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in total_history
        )
        initial_scores_from_graph = {}
        for topic in topics:
            if memory_graph.graph.has_node(topic):
                node_data = memory_graph.graph.nodes[topic]
                if node_data.get('status') == 'completed':
                    initial_scores_from_graph[topic] = node_data.get('score', 0)
        scores_str_for_summary_agent = ", ".join([f"{topic}:{score}" for topic, score in initial_scores_from_graph.items()])

        final_memory_str = memory_graph.get_context_for_prompt("Overall Summary")
        summary_payload = f"Full History:\n{total_history_str}\n\nInitial Scores:\n{scores_str_for_summary_agent}\n\nMemory:\n{final_memory_str}"

        summary_output = makerequest(group_chat_manager, user_proxy, summary_agent, summary_payload)
        if summary_output is not None:
            summary, updated_scores = extract_summary_and_updated_scores(summary_output, scale_name)
        else:
            summary, updated_scores = "", {}

        if updated_scores:
            dialog_print("\n--- Score Adjustments ---")
            for topic, details in updated_scores.items():
                score = details["score"]
                reason = details["reason"]
                memory_graph.update_topic_score(topic, score, reason)
                dialog_print(f"Score for topic '{topic}' updated to: {score} points (reason: {reason})")
                logger.info(f"Score for topic '{topic}' updated: -> {score} points (reason: {reason})")
        else:
            logger.warning("No updated scores received.")
            dialog_print("\nNo topic scores were adjusted.")

        overall_score = 0
        for topic in topics:
            if memory_graph.graph.has_node(topic):
                attrs = memory_graph.graph.nodes[topic]
                if attrs.get('status') == 'completed':
                    score_to_use = attrs.get('updated_score') if attrs.get('updated_score') is not None else attrs.get('score', 0)
                    if score_to_use is None:
                        logger.warning(f"Score_to_use for topic '{topic}' is None; defaulting to 0")
                        score_to_use = 0
                    overall_score += score_to_use
                    
        symptom_level = categorize_score(overall_score, scale_name)
        logger.info(f"Overall score: {overall_score}, symptom level: {symptom_level}")
        report_table = generate_score_table(memory_graph, topics, symptom_level)
        logger.info("Score table generation complete.")

        final_report = generate_report(report_table, summary, scale_name)
        if final_report is None:
            final_report = "Sorry, the report could not be generated."
        logger.info("Assessment task completed.")
        dialog_print(f"\nNumber of Q&As in this session: {qa_count}")
        return final_report, overall_score, symptom_level, updated_scores
    except Exception as e:
        logger.exception("An unknown error occurred while executing the assessment task: %s", e)
        return "Sorry, an error occurred during the assessment.", 0, 0, {}


def process_single_file(
    file_path,
    scoring_standards,
    chatprompt,
    selected_scale,
    mode_choice,
    csv_file_path,
    automated=False,
    skip_if_evaluated=True,
):
    try:
        identifier = os.path.splitext(os.path.basename(file_path))[0] 
        if skip_if_evaluated and is_file_already_evaluated(identifier, csv_file_path):
            logger.info(f"File {file_path} has already been evaluated—skipped.")
            dialog_print(f"File {file_path} has already been evaluated—skipped.")
            return 
        logger.info(f"Starting to process file: {file_path}")
        dialog_print(f"\n{'='*50}\nStarting to process file: {file_path}\n{'='*50}\n")
        identifier, real_interview, scores = load_real_data(file_path, selected_scale)
        agents = setup_agents(chatprompt)
        if automated:
            clear_memory_response = generate_mock_response("", topic=None, identification="", real_interview=[], scale_scores={}, clear_memory=True, scoring_standard=None, current_topic_history=None)
            logger.info(f"API clear-memory response: {clear_memory_response}")
            dialog_print("\nConversation memory cleared; ready to process current file.\n")

        final_report, overall_score, symptom_level, updated_scores = perform_assessment(
            topics=list(chatprompt.keys()),
            chatprompt=chatprompt,
            agents=agents,
            scale_name=selected_scale,
            scoring_standards=scoring_standards,
            real_interview=real_interview,
            scale_scores=scores,
            case_identifier=identifier,
            automated=automated
        )

        save_assessment_results(
            identifier=identifier,
            overall_score=overall_score,
            symptom_level=symptom_level,
            updated_scores=updated_scores,
            csv_file=csv_file_path,
            scale_name=selected_scale
        )
        logger.info(f"Assessment results saved to {csv_file_path}")
        dialog_print(f"Assessment results saved to {csv_file_path}")

        if mode_choice == "2":
            dialog_print("\nAutomated test completed. Psychological screening report:\n")
        else:
            dialog_print("\nThank you for your candid participation. Below is your preliminary psychological screening report. "
                "The results are for reference only and do not constitute a medical diagnosis. Generating the report may take a while—please wait.\n")
        dialog_print(final_report)
        logger.info("Psychological assessment report delivered; program ended.")

    except Exception as e:
        logger.exception(f"Error processing file {file_path}: {e}")
        dialog_print(f"Error processing file {file_path}; check logs for details.")
