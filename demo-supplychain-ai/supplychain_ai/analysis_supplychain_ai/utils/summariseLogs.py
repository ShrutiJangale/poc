import openai
import json
import re
# Load environment variables only once
from dotenv import load_dotenv
import os
from analysis_supplychain_ai.config import questions_file, prompts_file
# Rely on Django settings to load .env at startup; no local load here

openai.api_key = os.getenv("OPENAI_API_KEY")
print("****************OpenAI API key loaded:", "Yes" if openai.api_key else "No"   )

def summarize_categorized_logs(categorized_logs):
    """
    Summarizes categorized supply chain logs using OpenAI GPT-4o with a system-level supply chain summarization prompt.
    Args:
        categorized_logs (dict): { "High": [...], "Medium": [...], "Low": [...] }
    Returns:
        dict: { "High": summary, "Medium": summary, "Low": summary }
    """
    if not openai.api_key:
        print("OpenAI API key not found in environment variables")
        return {"Error": "API key not configured"}

    system_prompt = """
        You are SupplyChain Analyst — specialized in supply chain summarization and risk analysis.

        Your job is to read categorized incident logs (High, Medium, Low priority) and produce concise, executive-style summaries that inherently answer operational and strategic questions about:
        - End-to-End orchestration, supplier risk, inventory, logistics, financial and predictive performance.

        Behavior Rules:
        - Be factual, structured, and insight-driven.
        - Infer insights like delay causes, supplier lead time volatility, inventory risks, logistics bottlenecks, compliance, or cost implications.

         Output Requirements:
        - Output strictly in **valid JSON** format — no markdown, no text outside JSON.
        - Use exactly this schema:
        {
            "High": "summary of high priority incidents",
            "Medium": "summary of medium priority incidents",
            "Low": "summary of low priority incidents"
        }


        Guidelines:
        - Always produce **strict JSON** — no markdown, comments, or extra prose.
        - Fill missing or unavailable sections with `"insufficient data"`.
        - Keep each summary section concise (2–4 sentences maximum).
        - Ensure every section in the schema is present, even if empty.
"""


    summaries = {}

    all_comments_text = []
    for priority, comments in categorized_logs.items():
        if comments:
            joined = "\n".join(comments)
            all_comments_text.append(f"{priority} Priority Comments:\n{joined}\n")

    combined_logs = "\n".join(all_comments_text)

    user_prompt = f"""
Below are categorized supply chain comments. Generate a full summary and analysis according to your system instructions.

{combined_logs}
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=700,
            temperature=0
        )
        full_summary = response.choices[0].message.content.strip()

        
        cleaned = re.sub(r"^```json|^```|```$", "", full_summary, flags=re.MULTILINE).strip()
        try:
            summary_json = json.loads(cleaned)
        except Exception:
            summary_json = {"Error": "Could not parse summary as JSON", "raw": full_summary}

        with open("log_summaries.json", "w", encoding="utf-8") as f:
            json.dump(summary_json, f, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"Error summarizing categorized logs: {e}")
        summary_json = {"Error": str(e)}

    return summary_json

def generate_question_wise_summaries(categorized_logs, questions_file=questions_file, prompts_file=prompts_file):
    """Return all question-wise answers in one JSON payload (no streaming)."""
    if not openai.api_key:
        return {"Error": "OpenAI API key not configured."}

    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts = json.load(f)

        # Prepare logs
        all_comments_text = []
        for priority, comments in categorized_logs.items():
            if comments:
                joined = "\n".join(comments)
                all_comments_text.append(f"{priority} Priority Comments:\n{joined}\n")
        combined_logs = "\n".join(all_comments_text)

        question_wise_summaries = {}

        for question_key, question_text in questions.items():
            if question_key in prompts:
                system_prompt = (
                    "You are a Supply Chain Analyst. Your task is to answer the following "
                    "question based on the provided supply chain logs.\n\n"
                    f"Question: {question_text}\n\n"
                    "Instructions:\n"
                    "- Analyze the provided logs to answer the question\n"
                    "- Provide a concise, factual summary\n"
                    "- If specific data not found, mention 'Data not available in current logs'\n"
                    "- 2–3 sentences maximum.\n"
                )

                user_prompt = (
                    f"Based on the following logs, please answer: {question_text}\n\n"
                    f"{combined_logs}"
                )

                resp = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0
                )
                answer = resp.choices[0].message.content.strip()
                question_wise_summaries[question_key] = {
                    "question": question_text,
                    "summary": answer
                }
            else:
                question_wise_summaries[question_key] = {
                    "question": question_text,
                    "summary": "No corresponding prompt found"
                }

        output_data = {
            "question_wise_summaries": question_wise_summaries,
            "total_questions": len(questions),
            "processed_questions": len(question_wise_summaries)
        }

        with open("question_wise_summaries.json", "w", encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        return output_data
    except Exception as e:
        return {"Error": f"Error generating question-wise summaries: {e}"}


def generate_question_wise_summaries_stream(categorized_logs, questions_file="questions.json", prompts_file="prompts.json"):
    #print("*********",categorized_logs)
    """
    Streaming here token-by-token answers for each question.

    Yields events dicts in this sequence per question:
    - {"type": "start_question", "question_key": str, "question": str}
    - multiple {"type": "delta", "question_key": str, "text": str}
    - {"type": "end_question", "question_key": str, "answer": str}
    Finally yields {"type": "complete", "data": <final_output_json>}.

    NOTE: Implemented per user request to stream OpenAI answers and print as generated.
    """
    if not openai.api_key:
        yield {"type": "error", "error": "OpenAI API key not configured."}
        return

    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts = json.load(f)

        all_comments_text = []
        for priority, comments in categorized_logs.items():
            if comments:
                joined = "\n".join(comments)
                all_comments_text.append(f"{priority} Priority Comments:\n{joined}\n")
        combined_logs = "\n".join(all_comments_text)

        question_wise_summaries = {}

        for question_key, question_text in questions.items():
            if question_key not in prompts:
                question_wise_summaries[question_key] = {
                    "question": question_text,
                    "summary": "No corresponding prompt found"
                }
                yield {"type": "start_question", "question_key": question_key, "question": question_text}
                yield {"type": "delta", "question_key": question_key, "text": "No corresponding prompt found"}
                yield {"type": "end_question", "question_key": question_key, "answer": "No corresponding prompt found"}
                continue

            system_prompt = (
                "You are a Supply Chain Analyst. Your task is to answer the following "
                "question based on the provided supply chain logs.\n\n"
                f"Question: {question_text}\n\n"
                "Instructions:\n"
                "- Analyze the provided logs to answer the question\n"
                "- Provide a concise, factual summary\n"
                "- If specific data not found, mention 'Data not available in current logs'\n"
                "- 2–3 sentences maximum.\n"
            )

            user_prompt = (
                f"Based on the following logs, please answer: {question_text}\n\n"
                f"{combined_logs}"
            )

            yield {"type": "start_question", "question_key": question_key, "question": question_text}

            full_answer_parts = []
            try:
                stream = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0,
                    stream=True
                )

                for chunk in stream:
                    # Support both .delta.content and .message delta shapes
                    delta_text = None
                    try:
                        delta = chunk.choices[0].delta
                        if delta and hasattr(delta, 'content') and delta.content:
                            delta_text = delta.content
                    except Exception:
                        pass
                    if delta_text is None:
                        try:
                            delta_text = chunk.choices[0].delta.get('content')
                        except Exception:
                            delta_text = None

                    if delta_text:
                        full_answer_parts.append(delta_text)
                        yield {"type": "delta", "question_key": question_key, "text": delta_text}

                full_answer = "".join(full_answer_parts).strip()
            except Exception as e:
                full_answer = f"Error while streaming answer: {e}"

            question_wise_summaries[question_key] = {
                "question": question_text,
                "summary": full_answer
            }
            yield {"type": "end_question", "question_key": question_key, "answer": full_answer}

        output_data = {
            "question_wise_summaries": question_wise_summaries,
            "total_questions": len(questions),
            "processed_questions": len(question_wise_summaries)
        }

        try:
            with open("question_wise_summaries.json", "w", encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
        except Exception:
            pass

        yield {"type": "complete", "data": output_data}

    except Exception as e:
        yield {"type": "error", "error": f"Error generating question-wise summaries: {e}"}