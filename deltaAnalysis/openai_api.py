# def generate_procurement_trueup_delta_analysis(procurement_json: Dict[str, List], trueup_json: Dict[str, List]) -> List[Dict[str, Any]]:
#     # Prepare only necessary structured data for GPT
#     procurement_entries = [
#         {
#             "original_desc": k,
#             "expanded_desc": v[0],
#             "quantity": v[1],
#             "unit_cost": v[2]
#         }
#         for k, v in procurement_json.items()
#     ]

#     trueup_entries = [
#         {
#             "original_desc": k,
#             "expanded_desc": v[0],
#             "quantity": v[1]
#         }
#         for k, v in trueup_json.items()
#     ]

#     system_message = """
#         You are a semantic matcher for comparing two datasets: procurement and true-up.

#         Each dataset contains entries with the following fields:
#         - `original_desc`: the raw short-form item description (for reference only)
#         - `expanded_desc`: the detailed normalized item name (for matching)
#         - `quantity`: the count of items
#         - `unit_cost`: (only in procurement) the cost per unit

#         Your task:
#         - Match items **only** by their `expanded_desc` using semantic understanding.
#         - Ignore `original_desc`.
#         - For each true-up entry, find the best semantic match in procurement.
#         - If similarity > 85%, output:

#         {
#             "procurement_description": <original_desc from procurement>,
#             "trueup_description": <original_desc from true-up>,
#             "matched_procurement_expanded_description": <expanded_desc from procurement>,
#             "matched_trueup_expanded_description": <expanded_desc from true-up>,
#             "quantity_delta": <procurement quantity - true-up quantity>,
#             "unit_cost": <unit cost from procurement>,
#             "cost_delta": <quantity_delta * unit_cost>
#         }

#         Skip unmatched entries.

#         Examples of equivalent phrases you should match:
#         - 'elbow 2"' = '2 inch elbow'
#         - '8 1/2"' = '8.5 inches'
#         - '4\'' = '4 feet'
#         - 'cs' = 'carbon steel'
#         """

#     user_message = {
#         "role": "user",
#         "content": (
#             f"Procurement Entries:\n{json.dumps(procurement_entries, indent=2)}\n\n"
#             f"True-Up Entries:\n{json.dumps(trueup_entries, indent=2)}\n\n"
#             "Match using only the expanded descriptions. Return list of matched items with quantity delta and cost delta."
#         )
#     }

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": system_message},
#                 user_message
#             ],
#             temperature=0
#         )

#         content = response.choices[0].message.content.strip()

#         # Extract JSON list using regex
#         match = re.search(r"\[\s*{.*?}\s*]", content, re.DOTALL)
#         if match:
#             clean_json = match.group(0)
#             return json.loads(clean_json)
#         else:
#             print("No valid JSON list found in response.")
#             return []

#     except Exception as e:
#         print("OpenAI JSON parsing failed:", str(e))
#         return []


# def expand_asme_descriptions(descriptions: list[str], model="gpt-4o-mini") -> dict[str, str]:
#     system_prompt = (
#         "You are a materials and mechanical engineering expert. For each item description, "
#         "extract and list the core technical attributes concisely for semantic matching. "
#         "Focus on material, standard, dimensions, manufacturing, and end finish. "
#         "Avoid conversational language, definitions, or general explanations. "
#         "Output only the key attributes separated by commas or short phrases."
#     )

#     few_shot_examples = [
#         {
#             "input": "PIPE, CS, ASTM A53B, ASME B36.10, SCH. 40, ERW, BOExPOE",
#             "expanded": "Carbon Steel Pipe, ASTM A53 Grade B, ASME B36.10, Schedule 40, ERW, Beveled One End x Plain One End",
#         },
#         {
#             "input": "FLANGE, WN, RF, ASTM A105, ANSI B16.5, 150LB",
#             "expanded": "Weld Neck Flange, Raised Face, Carbon Steel ASTM A105, ANSI B16.5, Class 150LB",
#         },
#         {
#             "input": "VALVE, GATE, OS&Y, FLG, CS, ASTM A216 WCB, API 600, 3 IN, 300LB",
#             "expanded": "Gate Valve, OS&Y, Flanged, Carbon Steel ASTM A216 WCB, API 600, 3 inch, Class 300LB",
#         }
#     ]

#     # Build conversation
#     messages = [{"role": "system", "content": system_prompt}]
#     for ex in few_shot_examples:
#         messages.append({"role": "user", "content": ex["input"]})
#         messages.append({"role": "assistant", "content": ex["expanded"]})

#     # Combine all user inputs into one message
#     joined_descriptions = "\n".join(
#         [f"{i+1}. {desc}" for i, desc in enumerate(descriptions)]
#     )
#     user_prompt = (
#         "Expand the following item descriptions into concise technical attributes as per the above format:\n\n"
#         f"{joined_descriptions}\n\n"
#         "Return the outputs in numbered list format corresponding to the input order."
#     )
#     messages.append({"role": "user", "content": user_prompt})

#     # Send single API call
#     try:
#         response = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             temperature=0.1
#         )
#         output_text = response.choices[0].message.content.strip()

#         # Parse the numbered list response
#         expanded_lines = output_text.split("\n")
#         expanded_map = {}
#         current_idx = 0
#         for line in expanded_lines:
#             if not line.strip():
#                 continue
#             if line.lstrip().startswith(f"{current_idx+1}."):
#                 expanded_map[descriptions[current_idx]] = line.split(".", 1)[1].strip()
#                 current_idx += 1
#             if current_idx >= len(descriptions):
#                 break

#         # Fill any missing descriptions with a fallback
#         for desc in descriptions:
#             if desc not in expanded_map:
#                 expanded_map[desc] = "Error: No output returned"

#         return expanded_map

#     except Exception as e:
#         print(f"Error in batch processing: {e}")
#         return {desc: f"Error: {e}" for desc in descriptions}


