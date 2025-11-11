import openai
import pandas as pd
import json
import os
import re
from django.shortcuts import render
from .forms import UploadForm
from rapidfuzz import fuzz, process
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# API Key setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is not configured. Please set the OPENAI_API_KEY environment variable.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)


def upload_and_analyze_statement(request):
    context = {}
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                procurement_file = request.FILES['procurement_sheet']
                trueup_file = request.FILES['trueup_sheet']

                df_procurement = pd.read_excel(procurement_file)
                df_trueup = pd.read_excel(trueup_file)

                print("--- Starting True-Up Data Processing ---")
                trueup_data = process_trueup_data_enhanced(df_trueup)
                print(f"--- Completed True-Up Data Processing. Found {len(trueup_data)} unique items. ---")

                print("\n--- Starting Procurement Data Processing ---")
                procurement_data = process_procurement_data_enhanced(df_procurement)
                print(f"--- Completed Procurement Data Processing. Found {len(procurement_data)} unique items. ---")

                print("\n--- Starting Enhanced Semantic Delta Analysis ---")
                delta_results = perform_enhanced_semantic_analysis(
                    procurement_data=procurement_data,
                    trueup_data=trueup_data,
                )
                print(f"--- Completed Enhanced Semantic Analysis. Found {len(delta_results)} matches. ---")

                procurement_html = df_procurement.to_html(classes='table table-bordered table-hover table-sm', index=False)
                trueup_html = df_trueup.to_html(classes='table table-bordered table-hover table-sm', index=False)

                context = {
                    'form': form,
                    'procurement_html': procurement_html,
                    'trueup_html': trueup_html,
                    'delta_json_str': json.dumps(delta_results, indent=4),
                    'delta_json': delta_results,
                    'uploaded': True,
                }

            except Exception as e:
                context['error'] = f"An error occurred during analysis: {str(e)}"
                context['form'] = UploadForm()
    else:
        form = UploadForm()
        context['form'] = form

    return render(request, 'analysis_app/upload.html', context)


# Enhanced preprocessing functions
def normalize_technical_description(desc: str) -> str:
    """Enhanced normalization for technical descriptions"""
    if pd.isna(desc) or not isinstance(desc, str):
        return ""
    
    # Convert to lowercase and strip
    desc = desc.lower().strip()
    
    # Standardize common abbreviations and units
    replacements = {
        '"': ' inch ',
        "'": ' foot ',
        ' in ': ' inch ',
        ' ft ': ' foot ',
        'sch.': 'schedule',
        'sch ': 'schedule ',
        'std': 'standard',
        'od': 'outer diameter',
        'id': 'inner diameter',
        'cs': 'carbon steel',
        'ss': 'stainless steel',
        'wn': 'weld neck',
        'rf': 'raised face',
        'ff': 'flat face',
        'rtj': 'ring type joint',
        'erw': 'electric resistance welded',
        'smls': 'seamless',
        'be': 'beveled end',
        'pe': 'plain end',
        'boe': 'beveled one end',
        'poe': 'plain one end',
        'bw': 'butt weld',
        'sw': 'socket weld',
        'thr': 'threaded',
        'npt': 'national pipe thread',
        'ansi': 'american national standards institute',
        'asme': 'american society of mechanical engineers',
        'astm': 'american society for testing and materials',
        'api': 'american petroleum institute',
        "a105": "ASTM A105 carbon steel",
        "a105n": "normalized ASTM A105 carbon steel",
        "a106b": "ASTM A106 grade B carbon steel",
        "a53": "ASTM A53 carbon steel",
        "a53b": "ASTM A53 grade B carbon steel",
        "a182": "ASTM A182 forged or rolled alloy/stainless steel pipe flanges",
        "a234": "ASTM A234 wrought carbon steel and alloy steel fittings",
        "a403": "ASTM A403 stainless steel pipe fittings",
        "a312": "ASTM A312 seamless and welded stainless steel pipe",
        "a351": "ASTM A351 castings for pressure-containing parts",
        "a193": "ASTM A193 alloy and stainless steel bolting material",
        "a194": "ASTM A194 carbon and alloy steel nuts",
        "a269": "ASTM A269 seamless and welded austenitic stainless steel tubing",
        "a395": "ASTM A395 ferritic ductile iron castings",
        "a536": "ASTM A536 ductile iron castings",
        "api": "American Petroleum Institute",
        "asme": "American Society of Mechanical Engineers",
        "astm": "American Society for Testing and Materials",
        "b16": "ASME B16 standards for fittings, flanges, and valves",
        "b36": "ASME B36 pipe and tubing standards",
        "bw": "butt weld",
        "sw": "socket weld",
        "npt": "national pipe thread",
        "fnpt": "female national pipe thread",
        "mnpt": "male national pipe thread",
        "smls": "seamless",
        "erw": "electric resistance welded",
        "od": "outer diameter",
        "id": "inner diameter",
        "cs": "carbon steel",
        "ss": "stainless steel",
        "wn": "weld neck",
        "rf": "raised face",
        "ff": "flat face",
        "rtj": "ring type joint",
        "be": "beveled end",
        "pe": "plain end",
        "boe": "beveled one end",
        "poe": "plain one end",
        "gr": "grade",
        "grb": "grade B",
        "cl": "class",
        "sch": "schedule",
        "std": "standard",
        "cap": "cap fitting",
        "tee": "tee fitting",
        "flg": "flange",
        "flange": "flange",
        "plug": "plug fitting",
        "valve": "valve",
        "gasket": "gasket",
        "ptfe": "polytetrafluoroethylene (Teflon)",
        "rptfe": "reinforced PTFE",
        "rtfe": "reinforced PTFE",
        "epdm": "ethylene propylene diene monomer rubber",
        "hex": "hexagonal",
        "thd": "threaded",
        "thk": "thickness",
        "cvpf": "check valve, plastic flanged",
        "wog": "water-oil-gas pressure rating",
        "lugged": "lugged end connection",
        "bolt": "bolt",
        "nut": "nut",
        "washer": "washer",
        "ecc": "eccentric",
        "con": "concentric",
        "elbow": "elbow fitting",
        "elb": "elbow",
        "sr": "short radius",
        "lr": "long radius",
        "tee": "tee fitting",
        "olet": "branch outlet fitting",
        "capf": "cap flange",
        "bv": "ball valve",
        "sm": "sheet metal",
        "std": "standard weight",
        "mss": "Manufacturers Standardization Society"
    }
    
    for old, new in replacements.items():
        desc = desc.replace(old, new)
    
    # Remove extra spaces and normalize spacing
    desc = re.sub(r'\s+', ' ', desc).strip()
    
    return desc


def extract_technical_features(desc: str) -> Dict[str, str]:
    """Extract key technical features from description for better matching"""
    features = {}
    
    # Extract size/diameter information
    size_pattern = r'(\d+(?:\.\d+)?)\s*(?:inch|in|"|mm|cm)'
    size_matches = re.findall(size_pattern, desc.lower())
    if size_matches:
        features['size'] = size_matches[0]
    
    # Extract schedule information
    schedule_pattern = r'schedule\s*(\d+(?:\.\d+)?|std|standard|xs|xxs)'
    schedule_match = re.search(schedule_pattern, desc.lower())
    if schedule_match:
        features['schedule'] = schedule_match.group(1)
    
    # Extract material grade
    grade_patterns = [
        r'a\d+[a-z]*',  # ASTM grades like A53, A106
        r'grade\s*[a-z]\d*',  # Grade B, Grade A
        r'type\s*\d+',  # Type 304, Type 316
    ]
    for pattern in grade_patterns:
        match = re.search(pattern, desc.lower())
        if match:
            features['grade'] = match.group(0)
            break
    
    # Extract pipe type
    pipe_types = ['seamless', 'welded', 'erw', 'electric resistance welded']
    for pipe_type in pipe_types:
        if pipe_type in desc.lower():
            features['pipe_type'] = pipe_type
            break
    
    # Extract end type
    end_types = ['beveled', 'plain', 'threaded', 'socket', 'butt weld', 'beveled both ends', 'beveled one end']
    for end_type in end_types:
        if end_type in desc.lower():
            features['end_type'] = end_type
            break

    return features


def combine_description_with_size(description: str, size: str) -> str:
    """Combine description with size information"""
    if pd.isna(size) or not str(size).strip():
        return str(description).strip()
    
    desc = str(description).strip()
    size_clean = str(size).strip()
    
    # Add size to description with separator
    combined = f"{desc} | {size_clean}"
    return combined


def group_trueup_by_combined_description(df_trueup: pd.DataFrame) -> Dict[str, float]:
    """Group true-up items by combined description (description + size) and sum quantities"""
    
    # Find the correct column names (handling newlines and spaces)
    def find_column(pattern_list, available_cols):
        for pattern in pattern_list:
            for col in available_cols:
                if pattern.lower().replace(' ', '').replace('\n', '') in col.lower().replace(' ', '').replace('\n', ''):
                    return col
        return None
    
    # Find correct column names
    type_col = find_column(['Type'], df_trueup.columns)
    desc_col = find_column(['Description'], df_trueup.columns)
    qty_col = find_column(['QTY', 'Quantity'], df_trueup.columns)
    size_col = find_column(['Size'], df_trueup.columns)
    
    print(f"Found columns - Type: '{type_col}', Description: '{desc_col}', QTY: '{qty_col}', Size: '{size_col}'")
    
    if not desc_col:
        raise ValueError(f"Could not find Description column. Available: {list(df_trueup.columns)}")
    if not qty_col:
        raise ValueError(f"Could not find QTY column. Available: {list(df_trueup.columns)}")
    
    # Filter for Flange type if Type column exists
    if type_col and df_trueup[type_col].notna().any():
        df_filtered = df_trueup[df_trueup[type_col].isin(['Flange'])].copy()
        print(f"Filtered for Flange type: {len(df_filtered)} rows from {len(df_trueup)} total rows")
    else:
        df_filtered = df_trueup.copy()
        print("No Type column filtering applied")
    
    # Remove rows with missing Description or QTY
    df_clean = df_filtered.dropna(subset=[desc_col, qty_col]).copy()
    
    # Convert QTY to numeric
    df_clean[qty_col] = pd.to_numeric(df_clean[qty_col], errors='coerce').fillna(0)
    
    # Remove zero quantities
    df_clean = df_clean[df_clean[qty_col] > 0].copy()
    
    print(f"After cleaning: {len(df_clean)} rows with valid data")
    
    # Combine description with size
    if size_col:
        df_clean['Combined_Description'] = df_clean.apply(
            lambda row: combine_description_with_size(row[desc_col], row[size_col]), 
            axis=1
        )
    else:
        df_clean['Combined_Description'] = df_clean[desc_col].astype(str)
        print("No Size column found, using description only")
    
    # Group by combined description and sum quantities
    grouped = df_clean.groupby('Combined_Description')[qty_col].sum().reset_index()
    
    # Convert to dictionary
    result_dict = dict(zip(grouped['Combined_Description'], grouped[qty_col]))
    
    print(f"Grouped into {len(result_dict)} unique combined descriptions")
    
    # Show some examples
    for i, (desc, qty) in enumerate(list(result_dict.items())[:3]):
        print(f"Example {i+1}: {desc} -> Qty: {qty}")
    
    return result_dict


def process_trueup_data_enhanced(df_trueup: pd.DataFrame) -> List[Dict[str, Any]]:
    """Enhanced true-up data processing with proper grouping"""
    
    print("Processing True-Up data...")
    print(f"Initial rows: {len(df_trueup)}")
    print(f"Columns: {list(df_trueup.columns)}")
    
    # Group by combined description and sum quantities
    grouped_data = group_trueup_by_combined_description(df_trueup)
    
    if not grouped_data:
        print("No valid true-up data found after grouping")
        return []
    
    # Get expanded descriptions in batches
    combined_descriptions = list(grouped_data.keys())
    try:
        expanded_map = expand_descriptions_with_features(combined_descriptions)
    except Exception as e:
        print(f"Error expanding True-Up descriptions: {e}")
        expanded_map = {k: k for k in combined_descriptions}  # Fallback to original
    
    # Build result list
    result = []
    for combined_desc, quantity in grouped_data.items():
        expanded_desc = expanded_map.get(combined_desc, combined_desc)
        technical_features = extract_technical_features(expanded_desc)
        
        result.append({
            'original_desc': combined_desc,
            'expanded_desc': expanded_desc,
            'normalized_desc': normalize_technical_description(combined_desc),
            'quantity': float(quantity),
            'technical_features': technical_features,
            'search_text': f"{expanded_desc} {' '.join(technical_features.values())}"
        })
    
    print(f"Processed {len(result)} unique true-up items")
    return result


def process_procurement_data_enhanced(df_proc: pd.DataFrame) -> List[Dict[str, Any]]:
    """Enhanced procurement data processing with correct column names"""
    
    print("Processing Procurement data...")
    print(f"Initial rows: {len(df_proc)}")
    print(f"Columns: {list(df_proc.columns)}")
    
    # Filter for Fittings & Flanges items if Item column exists
    if 'Item' in df_proc.columns and df_proc['Item'].notna().any():
        df_filtered = df_proc[df_proc['Item'] == 'Fittings & Flanges (EA)'].copy()
        print(f"Filtered for Fittings & Flanges (EA): {len(df_filtered)} rows from {len(df_proc)} total rows")
    else:
        df_filtered = df_proc.copy()
        print("No Item column filtering applied")
    
    # Find the correct column names (handling newlines and spaces)
    def find_column(pattern_list, available_cols):
        for pattern in pattern_list:
            for col in available_cols:
                if pattern.lower().replace(' ', '').replace('\n', '') in col.lower().replace(' ', '').replace('\n', ''):
                    return col
        return None
    
    # Find correct column names
    desc_col = find_column(['Description'], df_filtered.columns)
    cost_col = find_column(['Unit Cost', 'UnitCost'], df_filtered.columns)
    qty_col = find_column(['Quantity Ordered', 'QuantityOrdered'], df_filtered.columns)
    
    if not desc_col:
        raise ValueError(f"Could not find Description column. Available: {list(df_filtered.columns)}")
    if not cost_col:
        raise ValueError(f"Could not find Unit Cost column. Available: {list(df_filtered.columns)}")
    if not qty_col:
        raise ValueError(f"Could not find Quantity Ordered column. Available: {list(df_filtered.columns)}")
    
    print(f"Found columns - Description: '{desc_col}', Unit Cost: '{cost_col}', Quantity: '{qty_col}'")
    
    # Clean the dataframe
    required_cols = [desc_col, cost_col, qty_col]
    df_clean = df_filtered.dropna(subset=required_cols).copy()
    
    # Convert numeric columns
    df_clean[cost_col] = pd.to_numeric(df_clean[cost_col], errors='coerce').fillna(0.0)
    df_clean[qty_col] = pd.to_numeric(df_clean[qty_col], errors='coerce').fillna(0.0)
    
    # Remove zero quantities
    df_clean = df_clean[df_clean[qty_col] > 0].copy()
    
    print(f"After cleaning: {len(df_clean)} rows with valid data")
    
    # Get unique descriptions for expansion
    unique_descriptions = df_clean[desc_col].unique().tolist()
    
    try:
        expanded_map = expand_descriptions_with_features(unique_descriptions)
    except Exception as e:
        print(f"Error expanding Procurement descriptions: {e}")
        expanded_map = {k: k for k in unique_descriptions}
    
    # Build result list
    result = []
    for _, row in df_clean.iterrows():
        original_desc = row[desc_col]
        expanded_desc = expanded_map.get(original_desc, original_desc)
        technical_features = extract_technical_features(expanded_desc)
        
        result.append({
            'original_desc': original_desc,
            'expanded_desc': expanded_desc,
            'normalized_desc': normalize_technical_description(original_desc),
            'quantity': float(row[qty_col]),
            'unit_cost': float(row[cost_col]),
            'technical_features': technical_features,
            'search_text': f"{expanded_desc} {' '.join(technical_features.values())}"
        })
    
    print(f"Processed {len(result)} procurement items")
    return result


def expand_descriptions_with_features(descriptions: List[str], batch_size: int = 30) -> Dict[str, str]:
    """Enhanced expansion with technical feature emphasis"""
    all_expanded = {}
    all_input = 0
    all_output = 0
    all_total = 0
    
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i + batch_size]
        print(f"Expanding batch {i//batch_size + 1}/{(len(descriptions) + batch_size - 1)//batch_size}")
        
        try:
            batch_result, (input, output, total) = expand_single_batch_enhanced(batch)
            all_input += input
            all_output += output
            all_total += total
            all_expanded.update(batch_result)
        except Exception as e:
            print(f"Error in batch expansion: {e}")
            for desc in batch:
                all_expanded[desc] = desc  # Fallback
    
    print(f"Total tokens used for expansion - Input: {all_input}, Output: {all_output}, Total: {all_total}")
    return all_expanded


def expand_single_batch_enhanced(descriptions: List[str]) -> Dict[str, str]:
    """Enhanced single batch expansion with better prompting"""
    if not descriptions:
        return {}
    
    system_prompt = """You are an expert in mechanical engineering and industrial piping systems. 
    Your task is to expand abbreviated technical descriptions into comprehensive, standardized formats 
    that emphasize key matching attributes like material, size, schedule, grade, and manufacturing method.
    
    Focus on:
    - Material specifications (Carbon Steel, Stainless Steel, etc.)
    - Size and dimensional information (including sizes after | symbol)
    - Schedule and wall thickness
    - Manufacturing methods (ERW, Seamless, etc.)
    - End preparations (Beveled, Plain, etc.)
    - Standards and specifications (ASTM, ASME, ANSI, etc.)
    
    Return a JSON object where each key is the original description and the value is the expanded version."""
    
    examples = """
    Input: "PIPE, CS, ASTM A53B, ASME B36.10, SCH. STD, ERW, BBE | 16\""
    Output: "16 Inch Carbon Steel Pipe ASTM A53 Grade B ASME B36.10 Schedule Standard Electric Resistance Welded Beveled Both Ends"
    
    Input: "2\" STD OD PIPE A53 GRB ERW"
    Output: "2 Inch Standard Outer Diameter Carbon Steel Pipe ASTM A53 Grade B Electric Resistance Welded"
    
    Input: "FLANGE, WN, RF, A105, 150LB"  
    Output: "Weld Neck Flange Raised Face Carbon Steel ASTM A105 Class 150LB"
    """
    
    user_prompt = f"""Expand these technical descriptions following the examples:

Examples:
{examples}

Descriptions to expand:
{json.dumps(descriptions, indent=2)}

Provide the response as a JSON object with original descriptions as keys and expanded versions as values."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        usage = response.usage
        input, output, total = usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        result = json.loads(response.choices[0].message.content)

        return result, (input, output, total)

    except Exception as e:
        print(f"OpenAI expansion error: {e}")
        return {desc: desc for desc in descriptions}, (0, 0, 0)


def get_enhanced_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Get embeddings with better preprocessing"""
    if not texts:
        return []
    
    # Clean and prepare texts for embedding
    cleaned_texts = []
    for text in texts:
        # Remove excessive whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', str(text)).strip()
        cleaned_texts.append(cleaned)
    
    try:
        response = client.embeddings.create(
            input=cleaned_texts,
            model=model
        )
        usage = response.usage
        total_tokens = usage.total_tokens
        return [embedding.embedding for embedding in response.data], total_tokens
    except Exception as e:
        print(f"Embedding error: {e}")
        # Return zero vectors as fallback
        return [[0.0] * 1536 for _ in texts]


def calculate_feature_similarity(features1: Dict[str, str], features2: Dict[str, str]) -> float:
    """Calculate similarity based on technical features"""
    common_keys = set(features1.keys()) & set(features2.keys())
    if not common_keys:
        return 0.0
    
    matches = 0
    for key in common_keys:
        if features1[key].lower() == features2[key].lower():
            matches += 1
    
    return matches / len(common_keys)


def perform_enhanced_semantic_analysis(procurement_data: List[Dict[str, Any]], 
                                     trueup_data: List[Dict[str, Any]],
                                     similarity_threshold: float = 0.65,
                                     feature_weight: float = 0.3) -> List[Dict[str, Any]]:
    """Enhanced semantic analysis with feature-based scoring"""
    if not procurement_data or not trueup_data:
        return []
    
    print("Generating embeddings for enhanced semantic analysis...")
    
    # Use search_text for embeddings (includes expanded description + features)
    proc_texts = [item['search_text'] for item in procurement_data]
    trueup_texts = [item['search_text'] for item in trueup_data]

    proc_embeddings, proc_tokens = get_enhanced_embeddings(proc_texts)
    trueup_embeddings, trueup_tokens = get_enhanced_embeddings(trueup_texts)
    all_tokens_semantic_match = proc_tokens + trueup_tokens
    print("embedding generation total count:", all_tokens_semantic_match)

    if not proc_embeddings or not trueup_embeddings:
        print("Failed to generate embeddings")
        return []
    
    # Convert to numpy arrays
    proc_matrix = np.array(proc_embeddings)
    trueup_matrix = np.array(trueup_embeddings)
    
    print("Calculating similarity matrix with feature enhancement...")
    
    # Calculate base cosine similarity
    similarity_matrix = cosine_similarity(proc_matrix, trueup_matrix)
    
    results = []
    matched_trueup_indices = set()
    
    # Enhanced matching with feature similarity
    for i, proc_item in enumerate(procurement_data):
        best_match_idx = -1
        best_combined_score = 0
        
        for j, trueup_item in enumerate(trueup_data):
            if j in matched_trueup_indices:
                continue
            
            # Base semantic similarity
            semantic_score = similarity_matrix[i][j]
            
            # Feature-based similarity
            feature_score = calculate_feature_similarity(
                proc_item['technical_features'], 
                trueup_item['technical_features']
                
            )
            
            # Combined score
            combined_score = (1 - feature_weight) * semantic_score + feature_weight * feature_score
            
            if combined_score > best_combined_score and combined_score >= similarity_threshold:
                best_combined_score = combined_score
                best_match_idx = j
        
        if best_match_idx >= 0:
            trueup_item = trueup_data[best_match_idx]
            matched_trueup_indices.add(best_match_idx)
            
            # Calculate deltas
            quantity_delta = proc_item['quantity'] - trueup_item['quantity']
            cost_delta = quantity_delta * proc_item['unit_cost']
            
            results.append({
                "procurement_original_normalized_description": proc_item['original_desc'],
                "trueup_original_normalized_description": trueup_item['original_desc'],
                "matched_procurement_expanded_description": proc_item['expanded_desc'],
                "matched_trueup_expanded_description": trueup_item['expanded_desc'],
                "procurement_quantity": proc_item['quantity'],
                "trueup_quantity": trueup_item['quantity'],
                "quantity_delta": round(quantity_delta, 2),
                # "unit_cost": proc_item['unit_cost'],
                # "cost_delta": round(cost_delta, 2),
                "similarity_score": round(best_combined_score, 4),
                "technical_features_match": round(calculate_feature_similarity(
                    proc_item['technical_features'], 
                    trueup_item['technical_features']
                ), 4)
            })
    
    # Sort by similarity score descending
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    print(f"Found {len(results)} high-quality matches out of {len(procurement_data)} procurement items")
    
    return results


# import openai
# import pandas as pd
# import json
# import os
# from django.shortcuts import render
# from .forms import UploadForm
# from rapidfuzz import fuzz, process
# from typing import List, Dict, Any
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# # It's a good practice to handle the API key securely
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     raise ValueError("OpenAI API key is not configured. Please set the OPENAI_API_KEY environment variable.")

# client = openai.OpenAI(api_key=OPENAI_API_KEY)


# def upload_and_analyze_statement(request):
#     context = {}
#     if request.method == 'POST':
#         form = UploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             try:
#                 procurement_file = request.FILES['procurement_sheet']
#                 trueup_file = request.FILES['trueup_sheet']

#                 df_procurement = pd.read_excel(procurement_file)
#                 df_trueup = pd.read_excel(trueup_file)

#                 print("--- Starting True-Up Data Processing ---")
#                 trueup_json = get_normalized_item_quantity_dict(df_trueup)
#                 print(f"--- Completed True-Up Data Processing. Found {len(trueup_json)} unique items. ---")

#                 print("\n--- Starting Procurement Data Processing ---")
#                 procurement_json = get_procurement_grouped_data(df_procurement)
#                 print(f"--- Completed Procurement Data Processing. Found {len(procurement_json)} unique items. ---")

#                 print("\n--- Starting Final Delta Analysis (with Semantic Search) ---")
#                 # MODIFIED: Calling the new analysis function
#                 delta_json = perform_semantic_delta_analysis(
#                     procurement_json=procurement_json,
#                     trueup_json=trueup_json,
#                 )
#                 print(f"--- Completed Final Delta Analysis. Found {len(delta_json)} matches. ---")

#                 procurement_html = df_procurement.to_html(classes='table table-bordered table-hover table-sm', index=False)
#                 trueup_html = df_trueup.to_html(classes='table table-bordered table-hover table-sm', index=False)

#                 context = {
#                     'form': form,
#                     'procurement_html': procurement_html,
#                     'trueup_html': trueup_html,
#                     'delta_json_str': json.dumps(delta_json, indent=4),
#                     'delta_json': delta_json,
#                     'uploaded': True,
#                 }

#             except Exception as e:
#                 context['error'] = f"An error occurred during analysis: {str(e)}"
#                 context['form'] = UploadForm()
#     else:
#         form = UploadForm()
#         context['form'] = form

#     return render(request, 'analysis_app/upload.html', context)



# # --- NEW AND IMPROVED DELTA ANALYSIS ---

# def get_embeddings(texts: list[str], model="text-embedding-3-small") -> list[list[float]]:
#     """Generates embeddings for a list of texts."""
#     if not texts:
#         return []
#     # Replace newlines, which can affect performance
#     texts = [text.replace("\n", " ") for text in texts]
#     try:
#         response = client.embeddings.create(input=texts, model=model)
#         return [embedding.embedding for embedding in response.data]
#     except Exception as e:
#         print(f"Error while getting embeddings: {e}")
#         # Return a list of zero vectors if the API call fails
#         return [[0] * 1536 for _ in texts] # text-embedding-3-small has 1536 dimensions

# def perform_semantic_delta_analysis(
#     procurement_json: Dict[str, List], trueup_json: Dict[str, List], similarity_threshold: float = 0.7
# ) -> List[Dict[str, Any]]:
#     """
#     Performs delta analysis using semantic search with vector embeddings.
#     """
#     if not procurement_json or not trueup_json:
#         print("Skipping delta analysis due to empty procurement or true-up data.")
#         return []

#     # Prepare data into structured lists
#     procurement_items = procurement_json
#     trueup_items = [{"original_desc": k, "expanded_desc": v[0], "quantity": v[1]} for k, v in trueup_json.items()]
    
#     # Extract expanded descriptions for embedding
#     procurement_descs = [item['expanded_desc'] for item in procurement_items]
#     trueup_descs = [item['expanded_desc'] for item in trueup_items]

#     print("Generating embeddings for semantic search...")
#     procurement_embeddings = get_embeddings(procurement_descs)
#     trueup_embeddings = get_embeddings(trueup_descs)

#     # Convert to NumPy arrays for efficient calculation
#     proc_matrix = np.array(procurement_embeddings)
#     trueup_matrix = np.array(trueup_embeddings)

#     print("Calculating similarity matrix...")
#     # Calculate cosine similarity between all procurement and true-up items
#     similarity_matrix = cosine_similarity(proc_matrix, trueup_matrix)

#     delta_results = []
#     matched_trueup_indices = set() # To prevent matching one true-up item to multiple procurement items

#     print(f"Finding matches with similarity threshold > {similarity_threshold}...")
#     # Iterate through each procurement item and find its best match
#     for i, proc_item in enumerate(procurement_items):
#         similarity_scores = similarity_matrix[i]
        
#         best_match_index = np.argmax(similarity_scores)
#         best_match_score = similarity_scores[best_match_index]

#         if best_match_score >= similarity_threshold and best_match_index not in matched_trueup_indices:
#             trueup_item = trueup_items[best_match_index]
#             matched_trueup_indices.add(best_match_index)
            
#             # Calculate the delta
#             quantity_delta = proc_item['quantity'] - trueup_item['quantity']
#             cost_delta = quantity_delta * proc_item['unit_cost']

#             # Append the detailed result
#             delta_results.append({
#                 "procurement_original_normalized_description": proc_item['original_desc'],
#                 "trueup_original_normalized_description": trueup_item['original_desc'],
#                 "matched_procurement_expanded_description": proc_item['expanded_desc'],
#                 "matched_trueup_expanded_description": trueup_item['expanded_desc'],
#                 "procurement_quantity": proc_item['quantity'],
#                 "trueup_quantity": trueup_item['quantity'],
#                 "quantity_delta": quantity_delta,
#                 "unit_cost": proc_item['unit_cost'],
#                 "cost_delta": round(cost_delta, 2),
#                 "similarity_score": round(best_match_score, 4) # Add score for context
#             })

#     return delta_results

# # --- Data Processing Functions ---
# def get_normalized_item_quantity_dict(df_trueup, desc_col='Description', qty_col='QTY', fuzzy_threshold=98):
#     # Filter for relevant types if 'Type' column exists
#     if 'Type' in df_trueup.columns and df_trueup['Type'].notna().any():
#         df_trueup = df_trueup[df_trueup['Type'].isin(['Pipe'])].copy()

#     if 'Size' in df_trueup.columns:
#         df_trueup['Size'] = df_trueup['Size'].fillna('').astype(str).str.strip()
#         df_trueup[desc_col] = df_trueup[desc_col].astype(str).str.strip() + ' ' + df_trueup['Size']

#     df_trueup[desc_col] = (
#         df_trueup[desc_col]
#         .astype(str)
#         .str.lower()
#         .str.strip()
#         .str.replace('"', ' inches', regex=False)
#         .str.replace("'", ' feet', regex=False)
#     )

#     # Group by description and sum quantities
#     grouped = df_trueup.groupby(desc_col)[qty_col].sum().reset_index()
#     descriptions = grouped[desc_col].tolist()
#     visited = set()
#     result_dict = {}

#     # Fuzzy matching to combine similar descriptions
#     for i, base_desc in enumerate(descriptions):
#         if base_desc in visited:
#             continue
#         # Using a more robust scorer for matching
#         matches = process.extract(base_desc, descriptions, scorer=fuzz.token_set_ratio, limit=None)
#         similar = [desc for desc, score, _ in matches if score >= fuzzy_threshold and desc not in visited]
#         qty_sum = grouped[grouped[desc_col].isin(similar)][qty_col].sum()
#         # Use the base description as the canonical name for the group
#         result_dict[base_desc] = qty_sum
#         visited.update(similar)

#     print(f"True-Up: Found {len(result_dict)} unique items before expansion.")
#     if not result_dict:
#         return {}

#     try:
#         # Expand descriptions using OpenAI
#         expanded_map = expand_asme_descriptions_openai_batched(list(result_dict.keys()))
#     except Exception as e:
#         print(f"Error calling OpenAI for True-Up expansion: {e}")
#         # Provide a fallback value in case of API error
#         expanded_map = {k: "Expansion failed" for k in result_dict.keys()}

#     final_dict = {
#         k: [expanded_map.get(k, "Expansion failed"), int(v)] for k, v in result_dict.items()
#     }
#     return final_dict


# def get_procurement_grouped_data(df_proc, desc_col='Description', qty_col='Quantity_Ordered', cost_col='Unit_Cost'):
#     if 'Item' in df_proc.columns and df_proc['Item'].notna().any():
#         df_proc = df_proc[df_proc['Item'] == 'Pipe (FT)'].copy()

#     # Drop rows with missing key values
#     df_proc = df_proc.dropna(subset=[desc_col, qty_col, cost_col])

#     # Ensure cost is numeric
#     df_proc[cost_col] = pd.to_numeric(df_proc[cost_col], errors='coerce').fillna(0.0).astype(float).round(2)

#     # Optionally ensure quantity is numeric too
#     df_proc[qty_col] = pd.to_numeric(df_proc[qty_col], errors='coerce').fillna(0.0).astype(float)

#     # Get unique descriptions
#     unique_descriptions = df_proc[desc_col].unique().tolist()

#     # Call OpenAI to expand each description
#     try:
#         expanded_map = expand_asme_descriptions_openai_batched(unique_descriptions)
#     except Exception as e:
#         print(f"Error calling OpenAI for Procurement expansion: {e}")
#         expanded_map = {desc: "Expansion failed" for desc in unique_descriptions}

#     # Build the output rows
#     output = []
#     for _, row in df_proc.iterrows():
#         original_desc = row[desc_col]
#         expanded_desc = expanded_map.get(original_desc, "Expansion failed")
#         quantity = row[qty_col]
#         unit_cost = row[cost_col]

#         output.append({
#             "original_desc": original_desc,
#             "expanded_desc": expanded_desc,
#             "quantity": quantity,
#             "unit_cost": unit_cost
#         })

#     return output

# def generate_procurement_trueup_delta_analysis_openai(
#     procurement_json: Dict[str, List], trueup_json: Dict[str, List]
# ) -> List[Dict[str, Any]]:
#     if not procurement_json or not trueup_json:
#         print("Skipping delta analysis due to empty procurement or true-up data.")
#         return []

#     procurement_entries = [
#         {"original_desc": k, "expanded_desc": v[0], "quantity": v[1], "unit_cost": v[2]}
#         for k, v in procurement_json.items()
#     ]
#     trueup_entries = [
#         {"original_desc": k, "expanded_desc": v[0], "quantity": v[1]}
#         for k, v in trueup_json.items()
#     ]
    
#     # Updated system instruction to get detailed quantity analysis
#     system_instruction = """
#         You are a procurement analyst AI. Your task is to match items from a procurement list to a true-up list based on their expanded technical descriptions.
#         For each match you find, you must provide a detailed comparison.

#         The final output should be a JSON array of objects. Each object represents a matched pair and must contain the following fields:
#         - "procurement_original_normalized_description": The original description from the procurement list.
#         - "trueup_original_normalized_description": The original description from the true-up list.
#         - "matched_procurement_expanded_description": The expanded description from the procurement item.
#         - "matched_trueup_expanded_description": The expanded description from the true-up item.
#         - "procurement_quantity": The quantity from the procurement list.
#         - "trueup_quantity": The quantity from the true-up list.
#         - "quantity_delta": The result of (procurement_quantity - trueup_quantity).
#         - "unit_cost": The unit cost from the procurement list.
#         - "cost_delta": The result of (quantity_delta * unit_cost).

#         Ensure your response is only the JSON array, without any additional text or explanations.
#         """

#     user_prompt = (
#         f"Procurement Entries:\n{json.dumps(procurement_entries, indent=2)}\n\n"
#         f"True-Up Entries:\n{json.dumps(trueup_entries, indent=2)}"
#     )

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4-turbo-preview", # Using a recommended model for JSON mode
#             temperature=0,
#             messages=[
#                 {"role": "system", "content": system_instruction.strip()},
#                 {"role": "user", "content": user_prompt}
#             ],
#             # FIX: The response_format parameter should be an object, not a string.
#             response_format={"type": "json_object"}
#         )
#         # The response is a JSON object with a key that contains the array.
#         # It is better to look for the array in the response content.
#         response_data = json.loads(response.choices[0].message.content)

#         # The model might return a JSON object with a key like "matches" or "delta_analysis"
#         if isinstance(response_data, dict):
#             for key, value in response_data.items():
#                 if isinstance(value, list):
#                     return value # Return the first list found in the JSON object
#         elif isinstance(response_data, list):
#              return response_data # If the response is already a list
        
#         return [] # Return empty if no list is found

#     except Exception as e:
#         print(f"OpenAI delta analysis failed: {e}")
#         return []


# def _single_batch_expand_openai(descriptions_batch: list[str]) -> dict[str, str]:
#     if not descriptions_batch:
#         return {}

#     system_instruction = (
#         "You are a materials and mechanical engineering expert. For each item description provided, "
#         "expand it into its core technical attributes for semantic matching. "
#         "The output must be a single JSON object where each key is the original input description "
#         "and its value is the expanded, more detailed string."
#     )

#     few_shot_examples = """
#     Example Input:
#     [
#         "PIPE, CS, ASTM A53B, ASME B36.10, SCH. 40, ERW, BOExPOE",
#         "FLANGE, WN, RF, ASTM A105, ANSI B16.5, 150LB"
#     ]

#     Example Output:
#     {
#         "PIPE, CS, ASTM A53B, ASME B36.10, SCH. 40, ERW, BOExPOE": "Carbon Steel Pipe, ASTM A53 Grade B, ASME B36.10, Schedule 40, ERW, Beveled One End x Plain One End",
#         "FLANGE, WN, RF, ASTM A105, ANSI B16.5, 150LB": "Weld Neck Flange, Raised Face, Carbon Steel ASTM A105, ANSI B16.5, Class 150LB"
#     }
#     """

#     user_prompt = (
#         f"Please expand the following descriptions based on the examples provided. Return a single JSON object.\n\n"
#         f"Descriptions:\n{json.dumps(descriptions_batch, indent=2)}"
#     )

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4-turbo-preview", # Using a recommended model for JSON mode
#             temperature=0,
#             messages=[
#                 {"role": "system", "content": system_instruction.strip()},
#                 {"role": "user", "content": f"{few_shot_examples.strip()}\n\n{user_prompt}"}
#             ],
#             # FIX: The response_format parameter should be an object, not a string.
#             response_format={"type": "json_object"}
#         )
#         return json.loads(response.choices[0].message.content)

#     except Exception as e:
#         print(f"Error in OpenAI single batch processing: {e}")
#         return {desc: f"Error: API call failed ({e})" for desc in descriptions_batch}


# def expand_asme_descriptions_openai_batched(descriptions: list[str], batch_size: int = 50) -> dict[str, str]:
#     all_expanded_map = {}
#     for i in range(0, len(descriptions), batch_size):
#         batch = descriptions[i:i + batch_size]
#         print(f"Processing batch {int(i/batch_size) + 1}/{(len(descriptions) // batch_size) + 1} with {len(batch)} items.")
#         try:
#             batch_expanded_map = _single_batch_expand_openai(batch)
#             all_expanded_map.update(batch_expanded_map)
#         except Exception as e:
#             print(f"Error processing batch starting with '{batch[0]}': {e}")
#             for desc in batch:
#                 all_expanded_map[desc] = f"Error: Batch failed ({e})"
#     return all_expanded_map





# import pandas as pd
# import json
# # import openai
# # import re
# # import time
# import os

# from django.shortcuts import render
# from .forms import UploadForm
# from rapidfuzz import fuzz, process
# from typing import List, Dict, Any
# # from fuzzywuzzy import fuzz, process
# # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# # client = openai.OpenAI(api_key=OPENAI_API_KEY)
# import google.generativeai as genai
# gemini_key = os.getenv("GEMINI_API_KEY")

# # Configure the Gemini client with your API key
# genai.configure(api_key=gemini_key)

# def upload_and_analyze_statement(request):
#     context = {}
#     if request.method == 'POST':
#         form = UploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             file1 = request.FILES['procurement_sheet']
#             file2 = request.FILES['trueup_sheet']

#             df_procurement = pd.read_excel(file1)
#             df_trueup = pd.read_excel(file2, sheet_name=0)

#             trueup_json = get_normalized_item_quantity_dict(df_trueup)
#             procurement_json = get_procurement_grouped_data(df_procurement)

#             delta_json = generate_procurement_trueup_delta_analysis_gemini(
#                 procurement_json=procurement_json,
#                 trueup_json=trueup_json,
#             )

#             procurement_html = df_procurement.to_html(classes='table table-bordered table-hover table-sm', index=False)
#             trueup_html = df_trueup.to_html(classes='table table-bordered table-hover table-sm', index=False)

#             context = {
#                 'form': form,
#                 'procurement_html': procurement_html,
#                 'trueup_html': trueup_html,
#                 'delta_json': delta_json,
#                 'uploaded': True,
#             }
#     else:
#         form = UploadForm()
#         context['form'] = form

#     return render(request, 'analysis_app/upload.html', context)




# def get_normalized_item_quantity_dict(df_trueup, desc_col='Description', qty_col='QTY', fuzzy_threshold=98):
#     # Step 0: Append raw Size (if available) to Description
#     if 'Size' in df_trueup.columns:
#         df_trueup['Size'] = df_trueup['Size'].fillna('').astype(str).str.strip()
#         df_trueup[desc_col] = df_trueup[desc_col].astype(str).str.strip() + ' ' + df_trueup['Size']

#     # Step 1: Standardize description text
#     df_trueup[desc_col] = (
#         df_trueup[desc_col]
#         .astype(str)
#         .str.lower()
#         .str.strip()
#         .str.replace('"', ' inches', regex=False)
#         .str.replace("'", ' feet', regex=False)
#     )

#     # Step 2: Aggregate exact matches
#     grouped = df_trueup.groupby(desc_col)[qty_col].sum().reset_index()

#     # Step 3: Fuzzy deduplication
#     descriptions = grouped[desc_col].tolist()
#     visited = set()
#     result_dict = {}

#     for i, base_desc in enumerate(descriptions):
#         if base_desc in visited:
#             continue

#         matches = process.extract(
#             base_desc,
#             descriptions,
#             scorer=fuzz.token_sort_ratio,
#             limit=None
#         )

#         similar = [desc for desc, score, _ in matches if score >= fuzzy_threshold and desc not in visited]
#         qty_sum = grouped[grouped[desc_col].isin(similar)][qty_col].sum()
#         result_dict[base_desc] = qty_sum
#         visited.update(similar)

#     print(f"Final dictionary with {len(result_dict)} unique items:")

#     # Step 4: Expand descriptions using OpenAI
#     try:
#         expanded_map = expand_asme_descriptions_gemini(list(result_dict.keys()))
#     except Exception as e:
#         print(f"Error calling OpenAI: {e}")
#         expanded_map = {k: "" for k in result_dict.keys()}

#     # Step 5: Build final output with expanded description
#     final_dict = {
#         k: [expanded_map.get(k, ""), int(v)] for k, v in result_dict.items()
#     }

#     return final_dict


# def get_procurement_grouped_data(df_proc, desc_col='Description', qty_col='Quantity_Ordered', cost_col='Unit_Cost', fuzzy_threshold=98):
#     """
#     Returns:
#     {
#         "cleaned description": [expanded_description, total_qty, unit_cost]
#     }
#     """
#     # Step 1: Clean descriptions
#     df_proc[desc_col] = (
#         df_proc[desc_col]
#         .astype(str)
#         .str.lower()
#         .str.strip()
#         .str.replace('"', ' inches', regex=False)
#         .str.replace("'", ' feet', regex=False)
#     )

#     df_proc = df_proc.dropna(subset=[desc_col, qty_col, cost_col])
#     df_proc[cost_col] = df_proc[cost_col].astype(float).round(2)

#     # Step 2: Group by description and cost
#     grouped = df_proc.groupby([desc_col, cost_col])[qty_col].sum().reset_index()

#     results = {}
#     visited = set()

#     for i, row in grouped.iterrows():
#         base_desc = row[desc_col]
#         base_cost = row[cost_col]
#         base_key = (base_desc, base_cost)

#         if base_key in visited:
#             continue

#         # Fuzzy match with exact same unit cost
#         matches = process.extract(base_desc, grouped[desc_col], scorer=fuzz.token_sort_ratio, limit=None)
#         similar_keys = [
#             (grouped.iloc[j][desc_col], grouped.iloc[j][cost_col])
#             for j, (desc, score, _) in enumerate(matches)
#             if score >= fuzzy_threshold and grouped.iloc[j][cost_col] == base_cost
#         ]

#         total_qty = grouped[
#             grouped.apply(lambda r: (r[desc_col], r[cost_col]) in similar_keys, axis=1)
#         ][qty_col].sum()

#         results[base_desc] = [total_qty, base_cost]
#         visited.update(similar_keys)

#     print(f"Final dictionary PS with {len(results)} unique items:")

#     # Step 3: Call LLM to get expanded descriptions
#     try:
#         expanded_map = expand_asme_descriptions_gemini(list(results.keys()))
#         print(f"Expanded descriptions\n: {expanded_map}")
#     except Exception as e:
#         print(f"Error calling OpenAI: {e}")
#         expanded_map = {k: "" for k in results.keys()}

#     # Step 4: Construct final dictionary with expanded description
#     final_results = {
#         k: [expanded_map.get(k, ""), int(v[0]), float(v[1])] for k, v in results.items()
#     }
#     print(f"Final results with true up \n {final_results}")
#     return final_results

# def generate_procurement_trueup_delta_analysis_gemini(
#     procurement_json: Dict[str, List], trueup_json: Dict[str, List]
# ) -> List[Dict[str, Any]]:
#     """
#     Compares procurement and true-up data using Gemini 1.5 Flash with JSON output.
#     """
#     procurement_entries = [
#         {"original_desc": k, "expanded_desc": v[0], "quantity": v[1], "unit_cost": v[2]}
#         for k, v in procurement_json.items()
#     ]

#     trueup_entries = [
#         {"original_desc": k, "expanded_desc": v[0], "quantity": v[1]}
#         for k, v in trueup_json.items()
#     ]

#     system_instruction = """
#         You are a semantic matcher for comparing two datasets: procurement and true-up.
#         Each dataset contains entries with fields: `original_desc`, `expanded_desc`, `quantity`, and sometimes `unit_cost`.

#         Your task:
#         - Match items **only** by their `expanded_desc` using semantic understanding.
#         - Ignore `original_desc` for matching purposes.
#         - For each true-up entry, find the best semantic match in the procurement data.
#         - If similarity is high (approx > 85%), calculate the deltas.
#         - The final output MUST be a single JSON array of objects, where each object has the keys:
#           "procurement_description", "trueup_description", "matched_procurement_expanded_description",
#           "matched_trueup_expanded_description", "quantity_delta", "unit_cost", "cost_delta".
#         - Skip any true-up entries that do not have a confident match in the procurement data.

#         Examples of equivalent phrases to match:
#         - 'elbow 2"' is equivalent to '2 inch elbow'
#         - '8 1/2"' is equivalent to '8.5 inches'
#         - '4\'' is equivalent to '4 feet'
#         - 'cs' is equivalent to 'carbon steel'
#     """

#     user_prompt = (
#         f"Procurement Entries:\n{json.dumps(procurement_entries, indent=2)}\n\n"
#         f"True-Up Entries:\n{json.dumps(trueup_entries, indent=2)}\n\n"
#         "Match the entries based on their 'expanded_desc'. For each match, calculate the quantity_delta "
#         "(procurement quantity - true-up quantity) and the cost_delta (quantity_delta * unit_cost). "
#         "Return the result as a JSON array of objects."
#     )

#     try:
#         # Initialize the Gemini model with the system instruction
#         model = genai.GenerativeModel(
#             model_name='gemini-1.5-flash',
#             system_instruction=system_instruction
#         )

#         # Generate content with JSON mode enabled
#         response = model.generate_content(
#             user_prompt,
#             generation_config={"temperature": 0, "response_mime_type": "application/json"}
#         )

#         # The response text is a valid JSON string, so we can load it directly
#         return json.loads(response.text)

#     except Exception as e:
#         print(f"Gemini API call or JSON parsing failed: {e}")
#         return []
    


# def expand_asme_descriptions_gemini(descriptions: list[str]) -> dict[str, str]:
#     """
#     Expands short ASME descriptions using Gemini 1.5 Flash with few-shot examples and JSON output.
#     """
#     system_instruction = (
#         "You are a materials and mechanical engineering expert. For each item description, "
#         "expand it into its core technical attributes for semantic matching. "
#         "Focus on material, standard, dimensions, manufacturing process, and end finish. "
#         "The output must be a single JSON object where each key is the original input description "
#         "and its value is the expanded string."
#     )

#     # Few-shot examples are provided in a simple list for the chat history
#     few_shot_examples = [
#         # Example 1
#         "PIPE, CS, ASTM A53B, ASME B36.10, SCH. 40, ERW, BOExPOE",
#         "Carbon Steel Pipe, ASTM A53 Grade B, ASME B36.10, Schedule 40, ERW, Beveled One End x Plain One End",
#         # Example 2
#         "FLANGE, WN, RF, ASTM A105, ANSI B16.5, 150LB",
#         "Weld Neck Flange, Raised Face, Carbon Steel ASTM A105, ANSI B16.5, Class 150LB",
#         # Example 3
#         "VALVE, GATE, OS&Y, FLG, CS, ASTM A216 WCB, API 600, 3 IN, 300LB",
#         "Gate Valve, OS&Y, Flanged, Carbon Steel ASTM A216 WCB, API 600, 3 inch, Class 300LB",
#     ]

#     # The final user prompt with the new list of descriptions
#     user_prompt = (
#         "Following the examples, expand the following item descriptions.\n\n"
#         + "\n".join(descriptions)
#         + "\n\nReturn a single JSON object mapping each original description to its expanded form."
#     )

#     # The full prompt includes the few-shot history and the final user request
#     full_prompt = few_shot_examples + [user_prompt]

#     try:
#         model = genai.GenerativeModel(
#             model_name='gemini-1.5-flash',
#             system_instruction=system_instruction
#         )

#         response = model.generate_content(
#             full_prompt,
#             generation_config={"temperature": 0.1, "response_mime_type": "application/json"}
#         )

#         # Directly parse the JSON output
#         expanded_map = json.loads(response.text)

#         # Ensure all original descriptions were processed, adding a fallback if any are missing
#         for desc in descriptions:
#             if desc not in expanded_map:
#                 expanded_map[desc] = "Error: Model did not return an expansion for this item."

#         return expanded_map

#     except Exception as e:
#         print(f"Error in Gemini batch processing: {e}")
#         return {desc: f"Error: API call failed ({e})" for desc in descriptions}



# import openai
# import pandas as pd
# import json
# import os
# from django.shortcuts import render
# from .forms import UploadForm
# from rapidfuzz import fuzz, process
# from typing import List, Dict, Any
# # import google.generativeai as genai
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# client = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- Configuration ---
# Ensure your API key is set as an environment variable
# gemini_key = os.getenv("GEMINI_API_KEY")

# if gemini_key:
#     genai.configure(api_key=gemini_key)
# else:
#     print("GEMINI_API_KEY environment variable not set.")

# --- Django View ---

# def upload_and_analyze_statement(request):
#     context = {}
#     if request.method == 'POST':
#         form = UploadForm(request.POST, request.FILES)
#         print("here0")
#         if form.is_valid():
#             print("here0.1")
#             try:
#                 print("here1")
#                 # Check if OpenAI key is configured
#                 if not OPENAI_API_KEY:
#                     print("here2")
#                     raise ValueError("OpenAI API key is not configured.")

#                 file1 = request.FILES['procurement_sheet']
#                 file2 = request.FILES['trueup_sheet']
#                 print("here3")
#                 df_procurement = pd.read_excel(file1)
#                 print("here4")
#                 df_trueup = pd.read_excel(file2, sheet_name=0)

#                 print("--- Starting True-Up Data Processing ---")
#                 trueup_json = get_normalized_item_quantity_dict(df_trueup)
#                 print(f"--- Completed True-Up Data Processing. Found {len(trueup_json)} unique items. ---")

#                 print("\n--- Starting Procurement Data Processing ---")
#                 procurement_json = get_procurement_grouped_data(df_procurement)
#                 print(f"--- Completed Procurement Data Processing. Found {len(procurement_json)} unique items. ---")

#                 print("\n--- Starting Final Delta Analysis ---")
#                 delta_json = generate_procurement_trueup_delta_analysis_openai(
#                     procurement_json=procurement_json,
#                     trueup_json=trueup_json,
#                 )
#                 print(f"--- Completed Final Delta Analysis. Found {len(delta_json)} matches. ---")

#                 procurement_html = df_procurement.to_html(classes='table table-bordered table-hover table-sm', index=False)
#                 trueup_html = df_trueup.to_html(classes='table table-bordered table-hover table-sm', index=False)

#                 context = {
#                     'form': form,
#                     'procurement_html': procurement_html,
#                     'trueup_html': trueup_html,
#                     'delta_json_str': json.dumps(delta_json, indent=4),
#                     'delta_json': delta_json,
#                     'uploaded': True,
#                 }

#             except Exception as e:
#                 context['error'] = f"An error occurred during analysis: {str(e)}"
#                 context['form'] = UploadForm()
#     else:
#         print("here else case")
#         form = UploadForm()
#         context['form'] = form

#     return render(request, 'analysis_app/upload.html', context)

# # --- Data Processing Functions ---
# def get_normalized_item_quantity_dict(df_trueup, desc_col='Description', qty_col='QTY', fuzzy_threshold=98):
#     # ✅ Step 1: Filter only rows where 'Type' is FLANG or PIPE
#     df_trueup = df_trueup[df_trueup['Type'].isin(['Flange', 'Pipe'])]

#     if 'Size' in df_trueup.columns:
#         df_trueup['Size'] = df_trueup['Size'].fillna('').astype(str).str.strip()
#         df_trueup[desc_col] = df_trueup[desc_col].astype(str).str.strip() + ' ' + df_trueup['Size']

#     df_trueup[desc_col] = (
#         df_trueup[desc_col]
#         .astype(str)
#         .str.lower()
#         .str.strip()
#         .str.replace('"', ' inches', regex=False)
#         .str.replace("'", ' feet', regex=False)
#     )

#     grouped = df_trueup.groupby(desc_col)[qty_col].sum().reset_index()
#     descriptions = grouped[desc_col].tolist()
#     visited = set()
#     result_dict = {}

#     for i, base_desc in enumerate(descriptions):
#         if base_desc in visited:
#             continue
#         matches = process.extract(base_desc, descriptions, scorer=fuzz.token_sort_ratio, limit=None)
#         similar = [desc for desc, score, _ in matches if score >= fuzzy_threshold and desc not in visited]
#         qty_sum = grouped[grouped[desc_col].isin(similar)][qty_col].sum()
#         result_dict[base_desc] = qty_sum
#         visited.update(similar)

#     print(f"True-Up: Found {len(result_dict)} unique items before expansion.")
#     if not result_dict:
#         return {}

#     try:
#         expanded_map = expand_asme_descriptions_openai_batched(list(result_dict.keys()))
#     except Exception as e:
#         print(f"Error calling OpenAI for True-Up expansion: {e}")
#         expanded_map = {k: "Expansion failed" for k in result_dict.keys()}

#     final_dict = {
#         k: [expanded_map.get(k, ""), int(v)] for k, v in result_dict.items()
#     }
#     return final_dict


# # def get_normalized_item_quantity_dict(df_trueup, desc_col='Description', qty_col='QTY', fuzzy_threshold=98):
# #     if 'Size' in df_trueup.columns:
# #         df_trueup['Size'] = df_trueup['Size'].fillna('').astype(str).str.strip()
# #         df_trueup[desc_col] = df_trueup[desc_col].astype(str).str.strip() + ' ' + df_trueup['Size']

# #     df_trueup[desc_col] = (
# #         df_trueup[desc_col]
# #         .astype(str)
# #         .str.lower()
# #         .str.strip()
# #         .str.replace('"', ' inches', regex=False)
# #         .str.replace("'", ' feet', regex=False)
# #     )
# #     grouped = df_trueup.groupby(desc_col)[qty_col].sum().reset_index()
# #     descriptions = grouped[desc_col].tolist()
# #     visited = set()
# #     result_dict = {}
# #     for i, base_desc in enumerate(descriptions):
# #         if base_desc in visited:
# #             continue
# #         matches = process.extract(base_desc, descriptions, scorer=fuzz.token_sort_ratio, limit=None)
# #         similar = [desc for desc, score, _ in matches if score >= fuzzy_threshold and desc not in visited]
# #         qty_sum = grouped[grouped[desc_col].isin(similar)][qty_col].sum()
# #         result_dict[base_desc] = qty_sum
# #         visited.update(similar)

# #     print(f"True-Up: Found {len(result_dict)} unique items before expansion.")
# #     if not result_dict:
# #         return {}

# #     try:
# #         expanded_map = expand_asme_descriptions_openai_batched(list(result_dict.keys()))
# #     except Exception as e:
# #         print(f"Error calling Gemini for True-Up expansion: {e}")
# #         expanded_map = {k: "Expansion failed" for k in result_dict.keys()}

# #     final_dict = {
# #         k: [expanded_map.get(k, ""), int(v)] for k, v in result_dict.items()
# #     }
# #     return final_dict


# def get_procurement_grouped_data(df_proc, desc_col='Description', qty_col='Quantity_Ordered', cost_col='Unit_Cost', fuzzy_threshold=98):
#     df_proc[desc_col] = (
#         df_proc[desc_col]
#         .astype(str)
#         .str.lower()
#         .str.strip()
#         .str.replace('"', ' inches', regex=False)
#         .str.replace("'", ' feet', regex=False)
#     )
#     df_proc = df_proc.dropna(subset=[desc_col, qty_col, cost_col])
#     df_proc[cost_col] = pd.to_numeric(df_proc[cost_col], errors='coerce').fillna(0.0).astype(float).round(2)
#     grouped = df_proc.groupby([desc_col, cost_col])[qty_col].sum().reset_index()
#     results = {}
#     visited = set()
#     for i, row in grouped.iterrows():
#         base_desc = row[desc_col]
#         base_cost = row[cost_col]
#         base_key = (base_desc, base_cost)
#         if base_key in visited:
#             continue
#         matches = process.extract(base_desc, grouped[desc_col], scorer=fuzz.token_sort_ratio, limit=None)
#         similar_keys = [
#             (grouped.iloc[j][desc_col], grouped.iloc[j][cost_col])
#             for j, (desc, score, _) in enumerate(matches)
#             if score >= fuzzy_threshold and grouped.iloc[j][cost_col] == base_cost
#         ]
#         total_qty = grouped[
#             grouped.apply(lambda r: (r[desc_col], r[cost_col]) in similar_keys, axis=1)
#         ][qty_col].sum()
#         results[base_desc] = [total_qty, base_cost]
#         visited.update(similar_keys)

#     print(f"Procurement: Found {len(results)} unique items before expansion.")
#     if not results:
#         return {}

#     try:
#         expanded_map = expand_asme_descriptions_openai_batched(list(results.keys()))
#     except Exception as e:
#         print(f"Error calling Gemini for Procurement expansion: {e}")
#         expanded_map = {k: "Expansion failed" for k in results.keys()}

#     final_results = {
#         k: [expanded_map.get(k, ""), int(v[0]), float(v[1])] for k, v in results.items()
#     }
#     return final_results


# # from openai import OpenAI
# # import openai
# # import json
# # from typing import Dict, List, Any

# # openai_client = OpenAI()  # Assumes proper API key config

# def generate_procurement_trueup_delta_analysis_openai(
#     procurement_json: Dict[str, List], trueup_json: Dict[str, List]
# ) -> List[Dict[str, Any]]:
#     if not procurement_json or not trueup_json:
#         print("Skipping delta analysis due to empty procurement or true-up data.")
#         return []

#     procurement_entries = [
#         {"original_desc": k, "expanded_desc": v[0], "quantity": v[1], "unit_cost": v[2]}
#         for k, v in procurement_json.items()
#     ]
#     trueup_entries = [
#         {"original_desc": k, "expanded_desc": v[0], "quantity": v[1]}
#         for k, v in trueup_json.items()
#     ]

#     system_instruction = """
#         You are a procurement analyst AI. Match procurement entries with true-up entries using the 'expanded_desc' field. 
#         For each match, calculate:
#         - quantity_delta = procurement quantity - true-up quantity
#         - cost_delta = quantity_delta * unit_cost

#         Return results as a JSON array where each object has:
#         - procurement_original_normalized_description
#         - trueup_original_normalized_description
#         - matched_procurement_expanded_description
#         - matched_trueup_expanded_description
#         - quantity_delta
#         - unit_cost
#         - cost_delta
#         """

#     user_prompt = (
#         f"Procurement Entries:\n{json.dumps(procurement_entries, indent=2)}\n\n"
#         f"True-Up Entries:\n{json.dumps(trueup_entries, indent=2)}"
#     )

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4-1106-preview",
#             temperature=0,
#             messages=[
#                 {"role": "system", "content": system_instruction.strip()},
#                 {"role": "user", "content": user_prompt}
#             ],
#             response_format="json"
#         )
#         return json.loads(response.choices[0].message.content)

#     except Exception as e:
#         print(f"OpenAI delta analysis failed: {e}")
#         return []


# def _single_batch_expand_openai(descriptions_batch: list[str]) -> dict[str, str]:
#     if not descriptions_batch:
#         return {}

#     system_instruction = (
#         "You are a materials and mechanical engineering expert. For each item description, "
#         "expand it into its core technical attributes for semantic matching. "
#         "The output must be a single JSON object where each key is the original input description "
#         "and its value is the expanded string."
#     )

#     few_shot_examples = """
# PIPE, CS, ASTM A53B, ASME B36.10, SCH. 40, ERW, BOExPOE => Carbon Steel Pipe, ASTM A53 Grade B, ASME B36.10, Schedule 40, ERW, Beveled One End x Plain One End
# FLANGE, WN, RF, ASTM A105, ANSI B16.5, 150LB => Weld Neck Flange, Raised Face, Carbon Steel ASTM A105, ANSI B16.5, Class 150LB
# """

#     user_prompt = (
#         f"{few_shot_examples.strip()}\n\n"
#         f"Descriptions:\n" + "\n".join(descriptions_batch) +
#         "\n\nReturn a single JSON object mapping each original description to its expanded form."
#     )

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4-1106-preview",
#             temperature=0,
#             messages=[
#                 {"role": "system", "content": system_instruction.strip()},
#                 {"role": "user", "content": user_prompt}
#             ],
#             response_format="json"
#         )
#         return json.loads(response.choices[0].message.content)

#     except Exception as e:
#         print(f"Error in OpenAI single batch processing: {e}")
#         return {desc: f"Error: API call failed ({e})" for desc in descriptions_batch}


# def expand_asme_descriptions_openai_batched(descriptions: list[str], batch_size: int = 50) -> dict[str, str]:
#     all_expanded_map = {}
#     for i in range(0, len(descriptions), batch_size):
#         batch = descriptions[i:i + batch_size]
#         print(f"Processing batch {int(i/batch_size) + 1}/{(len(descriptions) // batch_size) + 1} with {len(batch)} items.")
#         try:
#             batch_expanded_map = _single_batch_expand_openai(batch)
#             all_expanded_map.update(batch_expanded_map)
#         except Exception as e:
#             print(f"Error processing batch starting with '{batch[0]}': {e}")
#             for desc in batch:
#                 all_expanded_map[desc] = f"Error: Batch failed ({e})"
#     return all_expanded_map

# # --- Gemini API Call Functions (Corrected) ---

# # def generate_procurement_trueup_delta_analysis_gemini(
# #     procurement_json: Dict[str, List], trueup_json: Dict[str, List]
# # ) -> List[Dict[str, Any]]:
# #     if not procurement_json or not trueup_json:
# #         print("Skipping delta analysis due to empty procurement or true-up data.")
# #         return []

# #     procurement_entries = [
# #         {"original_desc": k, "expanded_desc": v[0], "quantity": v[1], "unit_cost": v[2]}
# #         for k, v in procurement_json.items()
# #     ]
# #     trueup_entries = [
# #         {"original_desc": k, "expanded_desc": v[0], "quantity": v[1]}
# #         for k, v in trueup_json.items()
# #     ]
# #     # In generate_procurement_trueup_delta_analysis_gemini:
# #     system_instruction = """
# #         ...
# #         The final output MUST be a single JSON array of objects, where each object has the keys:
# #         "procurement_original_normalized_description", "trueup_original_normalized_description",
# #         "matched_procurement_expanded_description", "matched_trueup_expanded_description",
# #         "quantity_delta", "unit_cost", "cost_delta".
# #         ...
# #     """
# #     user_prompt = (
# #         f"Procurement Entries:\n{json.dumps(procurement_entries, indent=2)}\n\n"
# #         f"True-Up Entries:\n{json.dumps(trueup_entries, indent=2)}\n\n"
# #         "Match the entries based on their 'expanded_desc'. For each match, calculate the quantity_delta "
# #         "(procurement quantity - true-up quantity) and the cost_delta (quantity_delta * unit_cost). "
# #         "When returning the results, include the 'original_normalized_desc' from both matched procurement and true-up entries, "
# #         "labeling them as 'procurement_original_normalized_description' and 'trueup_original_normalized_description' respectively. "
# #         "Return the result as a JSON array of objects."
# #     )

# #     try:
# #         # **FIX**: Added safety settings to prevent blocking
# #         safety_settings = {
# #             'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
# #             'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
# #             'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
# #             'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
# #         }
# #         model = genai.GenerativeModel(
# #             model_name='gemini-1.5-flash',
# #             system_instruction=system_instruction,
# #             safety_settings=safety_settings
# #         )
# #         response = model.generate_content(
# #             user_prompt,
# #             generation_config={"temperature": 0, "response_mime_type": "application/json"}
# #         )

# #         # **FIX**: Check for valid response parts before parsing
# #         if response.parts:
# #             return json.loads(response.text)
# #         else:
# #             print("Gemini delta analysis response was empty. It might have been blocked by safety filters.")
# #             print(f"Prompt Feedback: {response.prompt_feedback}")
# #             return []

# #     except Exception as e:
# #         print(f"Gemini API call or JSON parsing failed during delta analysis: {e}")
# #         return []


# # # Example of how you might refactor for batching
# # def expand_asme_descriptions_gemini_batched(descriptions: list[str], batch_size: int = 50) -> dict[str, str]:
# #     all_expanded_map = {}
# #     for i in range(0, len(descriptions), batch_size):
# #         batch = descriptions[i:i + batch_size]
# #         print(f"Processing batch {int(i/batch_size) + 1}/{(len(descriptions) // batch_size) + 1} with {len(batch)} items.")
# #         try:
# #             # Call your original expand_asme_descriptions_gemini with the batch
# #             batch_expanded_map = _single_batch_expand_gemini(batch) # Create a helper function for the actual API call
# #             all_expanded_map.update(batch_expanded_map)
# #         except Exception as e:
# #             print(f"Error processing batch starting with '{batch[0]}': {e}")
# #             for desc in batch:
# #                 all_expanded_map[desc] = f"Error: Batch failed ({e})"
# #     return all_expanded_map

# # def _single_batch_expand_gemini(descriptions_batch: list[str]) -> dict[str, str]:
# #     # This function would contain your existing expand_asme_descriptions_gemini logic
# #     # but operates only on descriptions_batch
# #     if not descriptions_batch:
# #         return {}
# #     system_instruction = (
# #         "You are a materials and mechanical engineering expert. For each item description, "
# #         "expand it into its core technical attributes for semantic matching. "
# #         "The output must be a single JSON object where each key is the original input description "
# #         "and its value is the expanded string."
# #     )
# #     few_shot_examples = [
# #         "PIPE, CS, ASTM A53B, ASME B36.10, SCH. 40, ERW, BOExPOE",
# #         "Carbon Steel Pipe, ASTM A53 Grade B, ASME B36.10, Schedule 40, ERW, Beveled One End x Plain One End",
# #         "FLANGE, WN, RF, ASTM A105, ANSI B16.5, 150LB",
# #         "Weld Neck Flange, Raised Face, Carbon Steel ASTM A105, ANSI B16.5, Class 150LB",
# #     ]
# #     user_prompt = (
# #         "Following the examples, expand the following item descriptions.\n\n"
# #         + "\n".join(descriptions_batch)
# #         + "\n\nReturn a single JSON object mapping each original description to its expanded form."
# #     )
# #     full_prompt = few_shot_examples + [user_prompt]

# #     try:
# #         safety_settings = {
# #             'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
# #             'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
# #             'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
# #             'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
# #         }
# #         model = genai.GenerativeModel(
# #             model_name='gemini-1.5-flash',
# #             system_instruction=system_instruction,
# #             safety_settings=safety_settings
# #         )
# #         response = model.generate_content(
# #             full_prompt,
# #             generation_config={"temperature": 0.1, "response_mime_type": "application/json"}
# #         )

# #         if response.parts:
# #             return json.loads(response.text)
# #         else:
# #             print(f"Gemini batch response empty for: {descriptions_batch[0]}...")
# #             print(f"Prompt Feedback: {response.prompt_feedback}")
# #             return {desc: "Error: Blocked response from API" for desc in descriptions_batch}

# #     except Exception as e:
# #         print(f"Error in Gemini single batch processing: {e}")
# #         return {desc: f"Error: API call failed ({e})" for desc in descriptions_batch}

# # Replace calls to expand_asme_descriptions_gemini with:
# # expanded_map = expand_asme_descriptions_gemini_batched(list(result_dict.keys()))




