import logging
import json
import re
from config import get_openai_settings

logger = logging.getLogger(__name__)
REMOTE_SIMULATOR_ENABLED = True

TOPIC_TO_ITEM_KEY = {
    "Loss of Interest": "PHQ8_NoInterest",
    "Depressed Mood": "PHQ8_Depressed",
    "Sleep Problems": "PHQ8_Sleep",
    "Fatigue or Low Energy": "PHQ8_Tired",
    "Appetite or Weight Changes": "PHQ8_Appetite",
    "Low Self-Worth": "PHQ8_Failure",
    "Concentration Difficulties": "PHQ8_Concentrating",
    "Psychomotor Changes": "PHQ8_Moving",
}

TOPIC_RESPONSES = {
    "Loss of Interest": {
        0: [
            "I still enjoy the things I care about, especially music and training.",
            "My interest is mostly there. I still care a lot about my goals and what I love doing.",
        ],
        1: [
            "I still care about music, but there have been a few days when it was harder to feel motivated.",
            "I have still enjoyed some things, but for several days I felt less interested than usual.",
        ],
        2: [
            "I still love music, but for more than half the days lately I have had trouble feeling interested in much else.",
            "A lot of days I feel less engaged, even with things I normally care about.",
        ],
        3: [
            "Nearly every day lately I have felt disconnected and had trouble taking pleasure in things I usually enjoy.",
            "Most days it feels hard to get interested in anything, even things that matter to me.",
        ],
    },
    "Depressed Mood": {
        0: [
            "I have not really felt down or hopeless lately.",
            "My mood has been mostly okay overall.",
        ],
        1: [
            "I have felt down on a few days, mostly when I think about work and where I am in life.",
            "There have been several days where I felt low, but not constantly.",
        ],
        2: [
            "I have been feeling down pretty often, especially about work and feeling stuck.",
            "More than half the days lately I have felt pretty low and frustrated.",
        ],
        3: [
            "Nearly every day lately I have felt depressed and it has been hard to feel hopeful.",
            "My mood has been low almost every day for the past couple of weeks.",
        ],
    },
    "Sleep Problems": {
        0: [
            "My sleep has been mostly normal.",
            "I have not noticed much of a sleep problem lately.",
        ],
        1: [
            "I have had a few nights where sleep was off, but not all the time.",
            "Some days I do not sleep well, though it is not constant.",
        ],
        2: [
            "My sleep has been pretty inconsistent for more than half the days lately.",
            "I have been having trouble sleeping a lot of the time recently.",
        ],
        3: [
            "Sleep has been a big problem nearly every day. I do not sleep enough and I cannot really turn my mind off.",
            "Almost every day lately my sleep has been poor and my mind keeps going when I try to rest.",
        ],
    },
    "Fatigue or Low Energy": {
        0: [
            "My energy has been okay overall.",
            "I have not really felt unusually tired.",
        ],
        1: [
            "I have felt tired on several days, but I can still get through what I need to do.",
            "There have been some low-energy days lately.",
        ],
        2: [
            "I have been feeling pretty tired a lot of the time lately.",
            "More than half the days I feel low on energy and it slows me down.",
        ],
        3: [
            "I have felt worn out nearly every day lately, even when I keep pushing through work and training.",
            "Most days I feel exhausted and low on energy.",
        ],
    },
    "Appetite or Weight Changes": {
        0: [
            "My appetite has been about the same.",
            "I have not noticed much change in eating.",
        ],
        1: [
            "My appetite has been off on a few days lately.",
            "There have been several days where I either did not feel like eating much or ate irregularly.",
        ],
        2: [
            "My appetite has been off more than half the days lately.",
            "I have noticed a pretty clear change in how I am eating recently.",
        ],
        3: [
            "Nearly every day lately my appetite has been off in a noticeable way.",
            "Eating has felt irregular almost every day over the last couple of weeks.",
        ],
    },
    "Low Self-Worth": {
        0: [
            "I have not really been beating myself up lately.",
            "I do not feel especially worthless or like a failure.",
        ],
        1: [
            "I have been hard on myself on a few days, especially about career progress.",
            "There have been some days where I felt like I should be doing better.",
        ],
        2: [
            "I do feel like I should be further along and I can be pretty hard on myself about that.",
            "More than half the days lately I have felt like I am not doing enough or falling short.",
        ],
        3: [
            "Nearly every day lately I have felt bad about myself and like I am not measuring up.",
            "I have been carrying a strong sense of failure almost every day.",
        ],
    },
    "Concentration Difficulties": {
        0: [
            "My concentration has been mostly fine.",
            "I have not noticed much trouble focusing.",
        ],
        1: [
            "On some days my mind is busy and it is harder to focus.",
            "I have had a little trouble concentrating here and there.",
        ],
        2: [
            "I have been having trouble focusing a lot of the time lately.",
            "More than half the days it feels harder to concentrate on what I need to do.",
        ],
        3: [
            "Nearly every day lately my concentration has been poor and my mind feels overloaded.",
            "I have been struggling to focus almost every day.",
        ],
    },
    "Psychomotor Changes": {
        0: [
            "I have not really noticed moving slower or feeling unusually restless.",
            "No, I have not noticed much change in my movement or speech.",
        ],
        1: [
            "A few days I have felt a little more restless, but nothing major.",
            "There have been some minor days where I felt off physically, but not often.",
        ],
        2: [
            "I have felt more restless than usual a lot of the time lately.",
            "More than half the days I have noticed some agitation or physical slowing.",
        ],
        3: [
            "Nearly every day lately I have either felt restless or slowed down in a noticeable way.",
            "There has been a strong physical change almost every day lately.",
        ],
    },
}


def _extract_participant_lines(real_interview):
    return [
        para.get("content", "").strip()
        for para in real_interview
        if para.get("roleName") == "Participant" and para.get("content", "").strip()
    ]


def _infer_basic_info(real_interview):
    participant_lines = _extract_participant_lines(real_interview)
    joined = " ".join(participant_lines).lower()

    occupation = "musician"
    if "school teacher" in joined or "working with kids" in joined:
        occupation = "teacher"
    elif "education" in joined:
        occupation = "educator"
    elif "music" in joined or "opera" in joined or "singing" in joined:
        occupation = "musician"
    elif "engineer" in joined:
        occupation = "engineer"
    elif "grad school" in joined or "school" in joined:
        occupation = "graduate student"

    gender = "other"
    if re.search(r"\bmy wife\b|\bhusband and i\b", joined):
        gender = "male"
    elif re.search(r"\bmy husband\b|\bwife and i\b", joined):
        gender = "female"

    age = 28
    if "grad school" in joined:
        age = 27
    elif "retired" in joined:
        age = 65

    return f"age:{age}, gender:{gender}, occupation:{occupation}"


def _score_for_topic(topic, scale_scores):
    items = scale_scores.get("items", {})
    item_key = TOPIC_TO_ITEM_KEY.get(topic)
    if not item_key:
        return 0
    score = items.get(item_key, 0)
    return score if isinstance(score, int) and 0 <= score <= 3 else 0


def _offline_topic_response(topic, scale_scores, depth):
    score = _score_for_topic(topic, scale_scores)
    variants = TOPIC_RESPONSES.get(topic, {}).get(score)
    if not variants:
        return "I'm not completely sure, but that has not stood out much lately."
    return variants[min(depth, len(variants) - 1)]


def _offline_response(question, topic, identification, real_interview, scale_scores, clear_memory=False, depth=0):
    if clear_memory:
        return "Memory cleared."

    if identification == "":
        return _infer_basic_info(real_interview)

    if topic:
        return _offline_topic_response(topic, scale_scores, depth)

    return "I'm not sure how to answer that."


def _build_interview_history(real_interview):
    return "\n".join(
        f"{para.get('roleName', 'Unknown role')}: {para.get('content', '')}"
        for para in real_interview
    )


def _remote_simulator_configured():
    settings = get_openai_settings()
    return (
        REMOTE_SIMULATOR_ENABLED
        and bool(settings["api_key"])
        and settings["base_url"].startswith(("http://", "https://"))
    )


def generate_mock_response(
    question,
    topic,
    identification,
    real_interview,
    scale_scores,
    retrieved_lessons=None,
    scoring_standard=None,
    current_topic_history=None,
    depth=0,
    clear_memory=False,
    scale_name="PHQ-8",
):
    global REMOTE_SIMULATOR_ENABLED
    interview_history = _build_interview_history(real_interview)

    if clear_memory:
        logger.info("Clearing API's conversation memory...")
        return _offline_response(question, topic, identification, real_interview, scale_scores, clear_memory=True, depth=depth)

    if identification == "":
        prompt = (
            f"Please answer the following question:\n{question}\n\n"
            "Please provide your basic information (age, gender, occupation) in the format "
            "'age:<age>, gender:<gender>, occupation:<occupation>' without adding anything else."
        )
    elif depth == 0:
        prompt = (
            f"Please answer the following question:\n{question}\n"
            "Please provide a truthful and reasonable answer based on your real interview dialogue and your profile, with no more than 50 words."
        )
    else:
        topic_history_str = ""
        if current_topic_history:
            for qa in current_topic_history:
                topic_history_str += f"Question: {qa['question']}\nUser's Response: {qa['response']}\n"

        prompt = (
            f"Please answer the following in-depth question:\n{question}\n\n"
            "Please provide a truthful and reasonable answer based on the interview dialogue and all your previous responses. "
            f"Your previous responses:\n{topic_history_str}\n"
            "Limit your response to 50 words."
        )

    if retrieved_lessons:
        prompt += f"\n\nRetrieved lessons for the simulated patient:\n{json.dumps(retrieved_lessons, ensure_ascii=False)}"

    logger.info(f"Prompt for generating simulated reply: {prompt}")

    if not _remote_simulator_configured():
        return _offline_response(
            question,
            topic,
            identification,
            real_interview,
            scale_scores,
            clear_memory=False,
            depth=depth,
        )

    try:
        from openai import OpenAI

        settings = get_openai_settings()
        client = OpenAI(base_url=settings["base_url"], api_key=settings["api_key"])
        system_prompt = (
            "You are speaking with a psychological assistant, and you are the client in this conversation with the following interview dialogue:\n"
            f"{interview_history}\n\n"
            "You should follow the provided information to act as a client in the conversation. "
            "Your responses should be coherent and avoid repeating previous utterances. "
            "Your response should ONLY include what the Client should say, in a natural, first-person tone."
        )
        completion = client.chat.completions.create(
            model=settings["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        response = completion.choices[0].message.content
        logger.info(f"Generated simulated reply: {response}")

        think_index = response.find("</think>")
        if think_index != -1:
            response = response[think_index + len("</think>"):].strip()

        return response
    except Exception as e:
        REMOTE_SIMULATOR_ENABLED = False
        logger.exception(f"Error while calling API to generate reply: {e}")
        fallback = _offline_response(
            question,
            topic,
            identification,
            real_interview,
            scale_scores,
            clear_memory=False,
            depth=depth,
        )
        logger.info(f"Using offline fallback simulated reply: {fallback}")
        return fallback
