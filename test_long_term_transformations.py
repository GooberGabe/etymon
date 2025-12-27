#!/usr/bin/env python3
"""Comprehensive test of long-term linguistic transformations to evaluate system believability."""

import random
from pathlib import Path
from typing import List, Dict, Any
from src.linguistics.catalog_io import load_all_catalogs, format_catalog_summary
from src.linguistics.pipeline import transform_language
from src.linguistics.models import LanguageCatalog


def run_transformation_sequence(
    base_catalog: LanguageCatalog,
    transformations: List[Dict[str, Any]],
    seed: int = 42
) -> List[LanguageCatalog]:
    """Run a sequence of transformations on a catalog, returning all intermediate states."""
    rng = random.Random(seed)
    states = [base_catalog]  # Start with original
    current = base_catalog

    print(f"\n{'='*60}")
    print(f"TRANSFORMATION SEQUENCE FOR: {base_catalog.name}")
    print(f"{'='*60}")
    print(f"Original: {format_catalog_summary(base_catalog)}")
    print()

    for i, transform_spec in enumerate(transformations, 1):
        print(f"Step {i}: {transform_spec.get('description', 'Unknown transformation')}")

        # Apply transformation
        current = transform_language(
            current,
            transformation_type=transform_spec.get('type', 'time_evolution'),
            time_depth=transform_spec.get('time_depth', 1),
            substrate_typology=transform_spec.get('substrate_typology'),
            influence_strength=transform_spec.get('influence_strength', 0.3),
            peer_catalog=transform_spec.get('peer_catalog'),
            bidirectional_weight=transform_spec.get('bidirectional_weight', 0.5),
            rng=rng,
        )

        states.append(current)
        print(f"Result: {format_catalog_summary(current)}")
        print()

    return states


def evaluate_transformation_believability(states: List[LanguageCatalog]) -> Dict[str, Any]:
    """Evaluate how believable the transformation sequence appears."""
    evaluation = {
        'total_steps': len(states) - 1,
        'vocabulary_changes': [],
        'typology_evolution': [],
        'issues_found': [],
        'believability_score': 0,
        'recommendations': []
    }

    for i in range(1, len(states)):
        prev = states[i-1]
        curr = states[i]

        # Check vocabulary evolution
        prev_count = prev.word_count()
        curr_count = curr.word_count()
        vocab_change = curr_count - prev_count
        evaluation['vocabulary_changes'].append({
            'step': i,
            'from': prev_count,
            'to': curr_count,
            'change': vocab_change
        })

        # Check typology evolution
        prev_consonants = len(prev.typology.consonants)
        curr_consonants = len(curr.typology.consonants)
        consonant_change = curr_consonants - prev_consonants

        prev_vowels = len(prev.typology.vowels)
        curr_vowels = len(curr.typology.vowels)
        vowel_change = curr_vowels - prev_vowels

        evaluation['typology_evolution'].append({
            'step': i,
            'consonants': {'from': prev_consonants, 'to': curr_consonants, 'change': consonant_change},
            'vowels': {'from': prev_vowels, 'to': curr_vowels, 'change': vowel_change},
            'onset_complexity': {'from': prev.typology.max_onset_complexity, 'to': curr.typology.max_onset_complexity},
            'coda_complexity': {'from': prev.typology.max_coda_complexity, 'to': curr.typology.max_coda_complexity}
        })

    # Evaluate believability
    score = 100  # Start with perfect score

    # Check for vocabulary preservation (should generally stay same or grow slightly)
    for change in evaluation['vocabulary_changes']:
        if change['change'] < -5:  # Significant vocabulary loss
            score -= 20
            evaluation['issues_found'].append(f"Step {change['step']}: Excessive vocabulary loss ({change['change']} words)")
        elif change['change'] > 10:  # Unusual vocabulary growth
            score -= 10
            evaluation['issues_found'].append(f"Step {change['step']}: Unusual vocabulary growth ({change['change']} words)")

    # Check typology evolution (should be gradual)
    for i, evo in enumerate(evaluation['typology_evolution']):
        cons_change = abs(evo['consonants']['change'])
        if cons_change > 15:  # Too much consonant change at once
            score -= 15
            evaluation['issues_found'].append(f"Step {evo['step']}: Excessive consonant inventory change ({cons_change} types)")

        onset_change = abs(evo['onset_complexity']['to'] - evo['onset_complexity']['from'])
        if onset_change > 2:  # Too much syllable structure change
            score -= 10
            evaluation['issues_found'].append(f"Step {evo['step']}: Excessive syllable structure change")

    evaluation['believability_score'] = max(0, score)

    # Generate recommendations
    if evaluation['believability_score'] >= 90:
        evaluation['recommendations'].append("Excellent believability - system performs very well")
    elif evaluation['believability_score'] >= 75:
        evaluation['recommendations'].append("Good believability with minor issues to address")
    elif evaluation['believability_score'] >= 60:
        evaluation['recommendations'].append("Moderate believability - significant improvements needed")
    else:
        evaluation['recommendations'].append("Poor believability - major redesign required")

    return evaluation


def main():
    """Run comprehensive transformation tests."""
    print("COMPREHENSIVE LONG-TERM LINGUISTIC TRANSFORMATION TEST")
    print("=" * 70)

    # Load catalogs
    catalogs = load_all_catalogs(Path("init_catalogs"))
    if not catalogs:
        print("ERROR: No catalogs found!")
        return

    # Define transformation sequences for different scenarios
    test_scenarios = [
        {
            'name': 'Indo-European Evolution',
            'base': 'Proto-Indo-Iranian',
            'transformations': [
                {'type': 'time_evolution', 'time_depth': 2, 'description': 'Initial time evolution (2000 years)'},
                {'type': 'substrate', 'substrate_typology': catalogs['Proto-Dravidian'].typology,
                 'influence_strength': 0.4, 'description': 'Dravidian substrate influence'},
                {'type': 'time_evolution', 'time_depth': 1, 'description': 'Continued evolution (1000 years)'},
                {'type': 'adstratum', 'peer_catalog': catalogs['Proto-Greek'],
                 'bidirectional_weight': 0.6, 'description': 'Greek adstratum contact'},
                {'type': 'time_evolution', 'time_depth': 3, 'description': 'Extended evolution (3000 years)'},
                {'type': 'substrate', 'substrate_typology': catalogs['Proto-Semitic'].typology,
                 'influence_strength': 0.2, 'description': 'Minor Semitic substrate influence'},
            ]
        },
        {
            'name': 'African Language Contact',
            'base': 'Proto-Niger-Congo',
            'transformations': [
                {'type': 'time_evolution', 'time_depth': 3, 'description': 'Extended evolution (3000 years)'},
                {'type': 'adstratum', 'peer_catalog': catalogs['Proto-Semitic'],
                 'bidirectional_weight': 0.4, 'description': 'Semitic contact influence'},
                {'type': 'substrate', 'substrate_typology': catalogs['Proto-Sino-Tibetan'].typology,
                 'influence_strength': 0.3, 'description': 'Sino-Tibetan substrate influence'},
                {'type': 'time_evolution', 'time_depth': 2, 'description': 'Continued evolution (2000 years)'},
                {'type': 'adstratum', 'peer_catalog': catalogs['Proto-Indo-Iranian'],
                 'bidirectional_weight': 0.5, 'description': 'Indo-Iranian contact'},
            ]
        },
        {
            'name': 'East Asian Language Chain',
            'base': 'Proto-Sino-Tibetan',
            'transformations': [
                {'type': 'time_evolution', 'time_depth': 4, 'description': 'Long-term evolution (4000 years)'},
                {'type': 'substrate', 'substrate_typology': catalogs['Proto-Anatolian'].typology,
                 'influence_strength': 0.5, 'description': 'Anatolian substrate influence'},
                {'type': 'adstratum', 'peer_catalog': catalogs['Proto-Turkic'],
                 'bidirectional_weight': 0.7, 'description': 'Strong Turkic contact'},
                {'type': 'time_evolution', 'time_depth': 1, 'description': 'Recent evolution (1000 years)'},
                {'type': 'substrate', 'substrate_typology': catalogs['Proto-Dravidian'].typology,
                 'influence_strength': 0.2, 'description': 'Minor Dravidian influence'},
            ]
        }
    ]

    overall_results = []

    for scenario in test_scenarios:
        if scenario['base'] not in catalogs:
            print(f"WARNING: Base catalog '{scenario['base']}' not found. Skipping scenario.")
            continue

        base_catalog = catalogs[scenario['base']]

        # Run transformation sequence
        states = run_transformation_sequence(base_catalog, scenario['transformations'])

        # Evaluate believability
        evaluation = evaluate_transformation_believability(states)

        print(f"\n{'='*60}")
        print(f"EVALUATION: {scenario['name']}")
        print(f"{'='*60}")
        print(f"Believability Score: {evaluation['believability_score']}/100")
        print(f"Total Transformation Steps: {evaluation['total_steps']}")

        if evaluation['issues_found']:
            print(f"\nIssues Found ({len(evaluation['issues_found'])}):")
            for issue in evaluation['issues_found']:
                print(f"  - {issue}")
        else:
            print("\nNo significant issues detected!")

        print(f"\nRecommendations:")
        for rec in evaluation['recommendations']:
            print(f"  - {rec}")

        # Summary of changes
        print(f"\nTransformation Summary:")
        print(f"  Vocabulary: {states[0].word_count()} → {states[-1].word_count()} words")
        print(f"  Consonants: {len(states[0].typology.consonants)} → {len(states[-1].typology.consonants)} types")
        print(f"  Vowels: {len(states[0].typology.vowels)} → {len(states[-1].typology.vowels)} types")
        print(f"  Max Onset: {states[0].typology.max_onset_complexity} → {states[-1].typology.max_onset_complexity}")
        print(f"  Max Coda: {states[0].typology.max_coda_complexity} → {states[-1].typology.max_coda_complexity}")

        overall_results.append({
            'scenario': scenario['name'],
            'score': evaluation['believability_score'],
            'issues': len(evaluation['issues_found'])
        })

    # Overall assessment
    print(f"\n{'='*70}")
    print("OVERALL SYSTEM ASSESSMENT")
    print(f"{'='*70}")

    total_score = sum(r['score'] for r in overall_results)
    avg_score = total_score / len(overall_results) if overall_results else 0
    total_issues = sum(r['issues'] for r in overall_results)

    print(f"Average Believability Score: {avg_score:.1f}/100")
    print(f"Total Issues Across All Tests: {total_issues}")

    if avg_score >= 85:
        print("\n🎉 EXCELLENT: System demonstrates high linguistic believability!")
        print("   The transformation sequences produce realistic language evolution patterns.")
    elif avg_score >= 70:
        print("\n👍 GOOD: System shows reasonable linguistic believability with room for improvement.")
        print("   Most transformation sequences are plausible, but some refinements needed.")
    elif avg_score >= 55:
        print("\n🤔 MODERATE: System has moderate linguistic believability.")
        print("   Some transformation sequences work well, but significant issues exist.")
    else:
        print("\n❌ POOR: System needs major improvements for linguistic believability.")
        print("   Transformation sequences produce unrealistic language evolution patterns.")

    print(f"\nTested {len(overall_results)} scenarios with {sum(len(s['transformations']) for s in test_scenarios[:len(overall_results)])} total transformations.")


if __name__ == "__main__":
    main()