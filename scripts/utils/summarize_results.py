"""
Summarize evaluation results from all models.
Creates a comprehensive accuracy report and saves individual predictions.
"""

import os
import pandas as pd
import argparse
from glob import glob


def summarize_model_results(model_dir):
    """
    Summarize results for a single model.

    Args:
        model_dir: Path to model directory containing predictions/

    Returns:
        Dictionary with accuracy results per split
    """
    predictions_dir = os.path.join(model_dir, "predictions")

    if not os.path.exists(predictions_dir):
        print(f"⚠️  No predictions found in {predictions_dir}")
        return {}

    results = {}

    # Find all prediction CSV files
    pred_files = glob(os.path.join(predictions_dir, "*_predictions.csv"))

    for pred_file in pred_files:
        # Extract split name from filename
        # Format: {model_key}_{split_name}_predictions.csv
        basename = os.path.basename(pred_file)
        parts = basename.replace("_predictions.csv", "").split("_", 1)
        if len(parts) == 2:
            split_name = parts[1]
        else:
            split_name = basename.replace("_predictions.csv", "")

        # Load predictions
        df = pd.read_csv(pred_file)

        # Calculate accuracy
        if 'ground_truth' in df.columns and 'prediction' in df.columns:
            correct = (df['ground_truth'] == df['prediction']).sum()
            total = len(df)
            accuracy = correct / total if total > 0 else 0.0

            results[split_name] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'prediction_file': pred_file,
            }

    return results


def create_summary_table(model_results):
    """
    Create a summary table comparing all models.

    Args:
        model_results: Dict of {model_name: {split_name: results}}

    Returns:
        DataFrame with summary
    """
    rows = []

    for model_name, splits in model_results.items():
        for split_name, metrics in splits.items():
            rows.append({
                'Model': model_name,
                'Split': split_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Correct': metrics['correct'],
                'Total': metrics['total'],
            })

    df = pd.DataFrame(rows)

    # Pivot to show models as columns
    if len(df) > 0:
        pivot = df.pivot(index='Split', columns='Model', values='Accuracy')
        return pivot

    return df


def main(models_dir="trained_models", output_file="evaluation_summary.csv"):
    """
    Main function to summarize all model results.

    Args:
        models_dir: Directory containing trained models
        output_file: Output CSV file for summary
    """
    print("=" * 80)
    print("Evaluation Results Summary")
    print("=" * 80)

    # Find all model directories
    model_dirs = [
        d for d in glob(os.path.join(models_dir, "*"))
        if os.path.isdir(d)
    ]

    if not model_dirs:
        print(f"⚠️  No model directories found in {models_dir}")
        return

    print(f"\nFound {len(model_dirs)} model directories:")
    for d in model_dirs:
        print(f"  - {os.path.basename(d)}")

    # Collect results from all models
    all_results = {}

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        print(f"\n▶ Processing {model_name}...")

        results = summarize_model_results(model_dir)

        if results:
            all_results[model_name] = results
            print(f"  Found results for {len(results)} splits")

            # Print individual split accuracies
            for split_name, metrics in sorted(results.items()):
                print(f"    {split_name:<30}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
        else:
            print(f"  No results found")

    # Create summary table
    if all_results:
        print("\n" + "=" * 80)
        print("Summary Table")
        print("=" * 80)

        summary = create_summary_table(all_results)
        print(summary.to_string())

        # Save to CSV
        summary.to_csv(output_file)
        print(f"\n✅ Summary saved to: {output_file}")

        # Also save detailed results
        detailed_file = output_file.replace(".csv", "_detailed.csv")
        rows = []
        for model_name, splits in all_results.items():
            for split_name, metrics in splits.items():
                rows.append({
                    'Model': model_name,
                    'Split': split_name,
                    'Accuracy': metrics['accuracy'],
                    'Correct': metrics['correct'],
                    'Total': metrics['total'],
                    'Prediction_File': metrics['prediction_file'],
                })

        df_detailed = pd.DataFrame(rows)
        df_detailed.to_csv(detailed_file, index=False)
        print(f"✅ Detailed results saved to: {detailed_file}")

    else:
        print("\n⚠️  No evaluation results found!")

    print("\n" + "=" * 80)
    print("Summary complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize evaluation results")
    parser.add_argument("--models_dir", type=str, default="trained_models",
                        help="Directory containing trained models")
    parser.add_argument("--output", type=str, default="evaluation_summary.csv",
                        help="Output CSV file for summary")

    args = parser.parse_args()

    main(models_dir=args.models_dir, output_file=args.output)
