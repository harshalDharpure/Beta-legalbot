"""
Prepare legal corpus for pretraining (Exp2 and Exp3).
Collects and preprocesses legal text data.

You can use code_mixed_posco_dataset/all_cases.txt as the legal corpus:
  python data/prepare_legal_corpus.py --use-all-cases
"""

import os
import json
import argparse
import shutil
from pathlib import Path

# Path to the POCSO legal cases corpus (optional)
ALL_CASES_PATH = 'code_mixed_posco_dataset/all_cases.txt'


def use_all_cases_corpus(corpus_dir):
    """
    Copy or link code_mixed_posco_dataset/all_cases.txt into the legal corpus dir
    so it can be used for Exp2/Exp3 pretraining. The file contains [case N] sections
    with legal judgment text (IPC, POCSO, courts) - ideal for domain pretraining.
    """
    src = Path(ALL_CASES_PATH)
    if not src.exists():
        print(f"⚠️  {ALL_CASES_PATH} not found; skipping.")
        return None
    dest = Path(corpus_dir) / 'all_cases.txt'
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"✅ Corpus already present: {dest}")
        return str(dest)
    try:
        shutil.copy2(src, dest)
        print(f"✅ Copied {src} -> {dest}")
    except Exception as e:
        print(f"⚠️  Copy failed: {e}")
        return None
    return str(dest)


def create_corpus_structure():
    """Create directory structure for legal corpus."""
    directories = [
        'experiments/exp2_pretraining_only/pretraining/legal_corpus',
        'experiments/exp3_pretraining_finetuning/pretraining/legal_corpus'
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created: {dir_path}")
    
    return directories[0]  # Return first directory


def create_pocso_act_template(corpus_dir):
    """Create a template file for POCSO Act text."""
    template_path = os.path.join(corpus_dir, 'pocso_act_template.txt')
    
    template_content = """# POCSO Act (Protection of Children from Sexual Offences Act, 2012)

## Instructions:
1. Copy the full text of the POCSO Act into this file
2. Include all sections, subsections, and provisions
3. Format: Plain text, one section per paragraph
4. Include definitions, penalties, procedures, etc.

## Example Format:
Section 1: Short title, extent and commencement
(1) This Act may be called the Protection of Children from Sexual Offences Act, 2012.
(2) It extends to the whole of India.
...

Section 2: Definitions
In this Act, unless the context otherwise requires,—
(a) "child" means any person below the age of eighteen years;
...

[Add all sections here]
"""
    
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(template_content)
    
    print(f"✅ Created template: {template_path}")
    return template_path


def create_legal_documents_template(corpus_dir):
    """Create template for legal documents."""
    docs_dir = os.path.join(corpus_dir, 'legal_documents')
    os.makedirs(docs_dir, exist_ok=True)
    
    readme_path = os.path.join(docs_dir, 'README.md')
    readme_content = """# Legal Documents for Pretraining

## Purpose
This directory contains legal documents, case summaries, and related legal text for domain-specific pretraining.

## Files to Add:
1. **pocso_act.txt** - Full text of POCSO Act (copy from parent directory)
2. **case_summaries.txt** - Summaries of POCSO-related cases
3. **legal_guidelines.txt** - Legal guidelines and procedures
4. **statutes.txt** - Related statutes and laws
5. **judicial_pronouncements.txt** - Relevant court judgments

## Format:
- Plain text files (.txt)
- One document per file or multiple documents separated by clear markers
- UTF-8 encoding
- Each file should be substantial (at least a few thousand words)

## Sources:
- Official government websites
- Legal databases
- Court judgments
- Legal textbooks
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✅ Created README: {readme_path}")
    return docs_dir


def create_sample_corpus(corpus_dir):
    """Create a sample corpus file with instructions."""
    sample_path = os.path.join(corpus_dir, 'SAMPLE_CORPUS.txt')
    
    sample_content = """# Sample Legal Corpus for POCSO Pretraining

## Instructions:
This is a sample file. Replace this with actual legal corpus data.

## Recommended Content:
1. **POCSO Act Full Text** - All sections and provisions
2. **Case Summaries** - Summaries of landmark POCSO cases
3. **Legal Procedures** - Investigation, trial, and appeal procedures
4. **Definitions** - Legal terms and definitions
5. **Penalties** - Punishments and sentencing guidelines

## Format:
- Plain text
- One paragraph per section/concept
- Clear section markers (e.g., "Section 1:", "Case: XYZ v. ABC")
- Minimum 50,000+ words recommended for effective pretraining

## Sources to Consider:
- https://www.indiacode.nic.in (Official legal database)
- Supreme Court judgments
- High Court judgments
- Legal textbooks on POCSO
- Government guidelines

## Example Entry:
Section 4: Penetrative sexual assault
Whoever commits penetrative sexual assault shall be punished with imprisonment of either description for a term which shall not be less than seven years but which may extend to imprisonment for life, and shall also be liable to fine.

[Continue with all sections and related content...]
"""
    
    with open(sample_path, 'w', encoding='utf-8') as f:
        f.write(sample_content)
    
    print(f"✅ Created sample corpus: {sample_path}")
    return sample_path


def main():
    """Main function to prepare legal corpus structure."""
    parser = argparse.ArgumentParser(description='Prepare legal corpus for Exp2/Exp3 pretraining')
    parser.add_argument('--use-all-cases', action='store_true',
                        help=f'Copy code_mixed_posco_dataset/all_cases.txt into legal corpus (recommended)')
    args = parser.parse_args()

    print("="*70)
    print("Preparing Legal Corpus Structure for Pretraining")
    print("="*70)
    print()

    # Create directory structure
    corpus_dir = create_corpus_structure()

    if args.use_all_cases:
        print("Using all_cases.txt as legal corpus...")
        use_all_cases_corpus(corpus_dir)
        # Also copy to exp3 if exists
        exp3_corpus = 'experiments/exp3_pretraining_finetuning/pretraining/legal_corpus'
        if os.path.exists(os.path.dirname(exp3_corpus)):
            use_all_cases_corpus(exp3_corpus)
        print()

    # Create templates and samples
    create_pocso_act_template(corpus_dir)
    create_legal_documents_template(corpus_dir)
    create_sample_corpus(corpus_dir)

    print()
    print("="*70)
    print("✅ Legal Corpus Structure Created")
    print("="*70)
    print()
    print("Next Steps:")
    if not args.use_all_cases:
        print("  Run with --use-all-cases to add code_mixed_posco_dataset/all_cases.txt")
    print("1. Add POCSO Act full text to: pocso_act_template.txt")
    print("2. Add legal documents to: legal_documents/ directory")
    print("3. Add case summaries and other legal text")
    print("4. Ensure total corpus is at least 50,000+ words")
    print()
    print(f"Corpus directory: {corpus_dir}")


if __name__ == '__main__':
    main()
