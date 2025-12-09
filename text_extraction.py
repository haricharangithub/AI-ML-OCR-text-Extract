import re
import logging
from typing import List, Tuple, Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_target_line(text_lines: List[str], pattern: str = "_1_") -> Optional[str]:
    if not text_lines:
        return None
    for line in text_lines:
        if pattern in line:
            cleaned_line = line.strip()
            return cleaned_line
    pattern_lower = pattern.lower()
    for line in text_lines:
        if pattern_lower in line.lower():
            cleaned_line = line.strip()
            return cleaned_line
    pattern_variations = [
        pattern,
        pattern.replace('_', '-'),
        pattern.replace('_', ' '),
        pattern.replace('_', ''),
        '_1',
        '1_',
        ' 1 ',
    ]
    
    for line in text_lines:
        for var in pattern_variations:
            if var in line or var.lower() in line.lower():
                cleaned_line = line.strip()
                return cleaned_line
    regex_pattern = r'[\d\w]*[_\s-]?1[_\s-]?[\d\w]*'
    for line in text_lines:
        matches = re.findall(regex_pattern, line, re.IGNORECASE)
        if matches:
            cleaned_line = line.strip()
            return cleaned_line
    
    return None


def extract_target_text(ocr_results: Dict, pattern: str = "_1_") -> Dict:
    result = {
        'target_text': None,
        'source_engine': None,
        'all_matching_lines': [],
        'confidence': None
    }
    for engine_name, texts in ocr_results.items():
        if not texts:
            continue
        text_lines = [text for text, conf, bbox in texts]
        target_line = find_target_line(text_lines, pattern)
        
        if target_line:
            result['target_text'] = target_line
            result['source_engine'] = engine_name
            for text, conf, bbox in texts:
                if pattern in text or pattern.lower() in text.lower():
                    result['confidence'] = conf
                    break
            for line in text_lines:
                if pattern in line or pattern.lower() in line.lower():
                    result['all_matching_lines'].append(line)
            
            break  
    
   
    if result['target_text'] is None:
        all_text_lines = []
        for texts in ocr_results.values():
            all_text_lines.extend([text for text, conf, bbox in texts])
        
        target_line = find_target_line(all_text_lines, pattern)
        if target_line:
            result['target_text'] = target_line
            result['source_engine'] = 'combined'
    
    return result


def validate_extracted_text(text: str, pattern: str = "_1_") -> bool:
    if not text:
        return False
    
    return pattern in text or pattern.lower() in text.lower()


def clean_extracted_text(text: str) -> str:
    if not text:
        return ""
    text = ' '.join(text.split())
    return text.strip()


