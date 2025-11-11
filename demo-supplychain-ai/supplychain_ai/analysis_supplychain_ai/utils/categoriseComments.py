import openai
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import time
from dotenv import load_dotenv
import os
 
# Rely on Django settings to load .env at startup; no local load here
openai.api_key = os.getenv("OPENAI_API_KEY")
print("****************OpenAI API key loaded:", "Yes" if openai.api_key else "No"   )

def _clean_json_content(raw_text):
    """
     extract a valid JSON array/object from model output.
    - Strips ```json fences
    - Trims whitespace
    - Fallback: find first top-level JSON array with a simple regex
    """
    if raw_text is None:
        return None

    text = raw_text.strip()
    text = re.sub(r"^```json\s*|^```\s*|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    array_match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if array_match:
        candidate = array_match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    obj_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if obj_match:
        candidate = obj_match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


def categorize_batch(batch, batch_index=None):
    """
    Categorize a batch of comments reliably using structured JSON output.
    """
    system_prompt = """
    You are an expert supply chain assistant. 
    Classify each comment individually into one of: "Low", "Medium", or "High".
    
    Rules:
    - Low: Routine or minor issues (no shipment delay or customer impact)
    - Medium: Moderate impact (short delays, coordination required)
    - High: Severe issue (operations stopped, customer escalation, major delay)

    Return your answer ONLY as valid JSON like:
    [
      {"comment": "Shipment delayed by 2 days", "priority": "Medium"},
      {"comment": "Production line stopped", "priority": "High"}
    ]
    """
    
    user_prompt = "Here are the comments:\n" + "\n".join([f"- {c}" for c in batch])
    
    for attempt in range(3): 
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",  
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0
            )
            content = (response.choices[0].message.content or "").strip()

            cleaned = _clean_json_content(content)
            if cleaned is not None:
                return cleaned

            preview = content[:400].replace("\n", " ")
            where = f" (batch {batch_index})" if batch_index is not None else ""
            print(f" JSON parse failed{where} attempt {attempt+1}: preview=\"{preview}\"")
            time.sleep(1)
        except Exception as e:
            where = f" (batch {batch_index})" if batch_index is not None else ""
            print(f" API error{where} attempt {attempt+1}: {e}")
            time.sleep(1)
    return []

def categoriseComments(comments, batch_size=5, max_workers=3):
    print(f"Categorizing {len(comments)} comments in batches of {batch_size} with {max_workers} workers.")
    try:
        print("******* Input preview:", comments[:min(3, len(comments))])
    except Exception:
        pass
  
    priority_dict = defaultdict(list)

    batches = [comments[i:i + batch_size] for i in range(0, len(comments), batch_size)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(categorize_batch, batch, idx): batch for idx, batch in enumerate(batches)}

        for future in as_completed(futures):
            try:
                results = future.result()
                for item in results:
                    comment = item.get("comment")
                    priority = item.get("priority")
                    if comment and priority:
                        priority_dict[priority].append(comment)
            except Exception as e:
                print(f" Error in batch processing: {e}")

    return dict(priority_dict)
