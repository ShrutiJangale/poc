import json
import os
import time
from django.shortcuts import render, redirect
from django.http import JsonResponse
from .utils.categoriseComments import categoriseComments
from .utils.summariseLogs import (
    summarize_categorized_logs,
    generate_question_wise_summaries,
)
from .utils.db_utils import create_file, update_file_status, insert_logs, upsert_summary, get_db_connection, get_all_files


def dashboard(request):
    """Main dashboard with sidebar"""
    return render(request, 'dashboard.html')


def file_selection(request):
    """Page to select a file_id and view its logs/summaries"""
    files = get_all_files()
    return render(request, 'file_selection.html', {'requests': files})


def _get_categorized_logs_from_file(file_id: int):
    """Helper function to reconstruct categorized logs from database"""
    conn = get_db_connection()
    rows = conn.execute(
        'SELECT category_name, log_text FROM logs WHERE request_id = ? ORDER BY log_id ASC',
        (file_id,)
    ).fetchall()
    conn.close()
    
    categorized = {"High": [], "Medium": [], "Low": []}
    for row in rows:
        category = row['category_name']
        log_text = row['log_text']
        if category in categorized:
            categorized[category].append(log_text)
    
    return categorized


def upload_file(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({'error': 'No file uploaded'}, status=400)

        # Create file record with file name
        file_id = create_file(status='uploaded', file_name=file.name)

        # Read file
        content = file.read().decode('utf-8').splitlines()
        comments = [line.strip() for line in content if line.strip()]

        update_file_status(file_id, 'categorizing')
        categorized = categoriseComments(comments)

        # Save logs
        logs = []
        for priority, comms in categorized.items():
            for c in comms:
                logs.append({'category_name': priority, 'log_text': c})
        insert_logs(file_id, logs)

        # Immediately generate and save summaries so they're ready
        update_file_status(file_id, 'summarizing')
        overall_summary = summarize_categorized_logs(categorized)
        upsert_summary(file_id, 'overall', json.dumps(overall_summary))

        questionwise_summary = generate_question_wise_summaries(categorized)
        upsert_summary(file_id, 'questionwise', json.dumps(questionwise_summary))

        update_file_status(file_id, 'completed')

        return redirect('results_page', file_id=file_id)

    return render(request, 'upload.html')


def results_page(request, file_id: int):
    conn = get_db_connection()
    rows = conn.execute(
        'SELECT log_id, log_text, category_name FROM logs WHERE request_id = ? ORDER BY log_id ASC',
        (file_id,)
    ).fetchall()
    conn.close()

    logs = [{'index': i + 1, 'log_text': r['log_text'], 'category_name': r['category_name']} for i, r in enumerate(rows)]
    return render(request, 'results.html', {
        'file_id': file_id,
        'logs': logs,
    })


def overall_summary_page(request, file_id: int):
    """Generate overall summary on-demand if not exists"""
    conn = get_db_connection()
    row = conn.execute(
        "SELECT summary_text FROM summaries WHERE request_id=? AND summary_type='overall' ORDER BY updated_at DESC LIMIT 1",
        (file_id,)
    ).fetchone()
    conn.close()

    # If summary doesn't exist, generate it
    if not row:
        categorized = _get_categorized_logs_from_file(file_id)
        update_file_status(file_id, 'summarizing')
        overall_summary = summarize_categorized_logs(categorized)
        upsert_summary(file_id, 'overall', json.dumps(overall_summary))
        update_file_status(file_id, 'completed')
        summary = overall_summary
    else:
        try:
            summary = json.loads(row['summary_text'])
        except Exception:
            summary = {'Error': 'Stored overall summary is not valid JSON', 'raw': row['summary_text']}

    return render(request, 'overall_summary.html', {
        'file_id': file_id,
        'summary': summary,
    })


def question_summary_page(request, file_id: int):
    """Display question-wise summary page; if missing, generate synchronously and render."""
    conn = get_db_connection()
    row = conn.execute(
        "SELECT summary_text FROM summaries WHERE request_id=? AND summary_type='questionwise' ORDER BY updated_at DESC LIMIT 1",
        (file_id,)
    ).fetchone()
    conn.close()

    if not row:
        categorized = _get_categorized_logs_from_file(file_id)
        update_file_status(file_id, 'summarizing')
        summary_data = generate_question_wise_summaries(categorized)
        upsert_summary(file_id, 'questionwise', json.dumps(summary_data))
        update_file_status(file_id, 'completed')
        summary = summary_data
    else:
        try:
            summary = json.loads(row['summary_text'])
        except Exception:
            summary = {"Error": "Stored question-wise summary is not valid JSON"}

    return render(request, 'question_summary.html', {
        'file_id': file_id,
        'summary': summary,
        'has_summary': True,
    })
