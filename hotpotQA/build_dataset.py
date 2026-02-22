from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict
from bs4 import BeautifulSoup

def clean_wikipedia_html(html_path: Path) -> str:
    with html_path.open('r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    main_content = soup.find('div', class_='mw-parser-output')
    
    if not main_content:
        main_content = soup.find('body') or soup
    
    unwanted_selectors = [
        'table', '.navbox', '.infobox', '.vertical-navbox', '.sidebar',
        '.reflist', '.references', 'sup.reference', '.mw-editsection',
        '.toc', '#toc', 'style', 'script', '.mw-references-wrap',
        '.shortdescription', '.hatnote', '.sister-inline-image',
        '.catlinks', '.printfooter', '.mw-jump-link',
    ]
    
    for selector in unwanted_selectors:
        for element in main_content.select(selector):
            element.decompose()
    
    text = main_content.get_text(separator=' ', strip=True)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[\s*edit\s*\]', '', text)
    text = re.sub(r'\[\s*\d+\s*\]', '', text)
    text = re.sub(r'\[\s*citation needed\s*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def extract_page_title(html_path: Path) -> str:
    filename = html_path.stem
    title = re.sub(r'^\d+_', '', filename)
    title = title.replace('-', ' ')
    return title

def create_corpus_from_hotpotqa(
    records_file: Path,
    wiki_base_dir: Path,
    output_file: Path,
    include_question_context: bool = False
) -> None:
    corpus_entries: Dict[str, Dict] = {}
    
    with records_file.open('r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            record_id = record.get('id', f'unknown-{line_num}')
            raw_files = record.get('raw_evidence_files', [])
            
            for file_path_str in raw_files:
                file_path = Path(file_path_str)
                if not file_path.exists():
                    continue
                
                corpus_name = f"{file_path.parent.name}/{file_path.name}"
                if corpus_name in corpus_entries:
                    continue
                
                try:
                    clean_text = clean_wikipedia_html(file_path)
                except Exception:
                    continue
                
                if not clean_text or len(clean_text) < 100:
                    continue
                
                corpus_entry = {
                    'corpus_name': corpus_name,
                    'context': clean_text
                }
                
                if include_question_context:
                    corpus_entry['page_title'] = extract_page_title(file_path)
                    corpus_entry['source_record'] = record_id
                
                corpus_entries[corpus_name] = corpus_entry
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', encoding='utf-8') as out:
        for corpus_entry in corpus_entries.values():
            out.write(json.dumps(corpus_entry, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--wiki_dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('hotpotqa_corpus.jsonl'))
    parser.add_argument('--include-metadata', action='store_true')
    args = parser.parse_args()
    
    create_corpus_from_hotpotqa(
        records_file=args.records,
        wiki_base_dir=args.wiki_dir,
        output_file=args.output,
        include_question_context=args.include_metadata
    )

if __name__ == '__main__':
    main()