import sys
import torch
import json
import pickle
from model_seq2seq import testing_data, evaluate, MODELS, encoderRNN, decoderRNN, attention
from torch.utils.data import DataLoader
from bleu_eval import BLEU


def load_model(model_path):
    """Load the trained model from the given file."""
    return torch.load(model_path, map_location=lambda storage, loc: storage)


def prepare_data(filepath):
    """Prepare the test dataset and data loader."""
    dataset = testing_data(filepath)
    return DataLoader(dataset, batch_size=100, shuffle=True, num_workers=8)


def load_i2w_mapping(pickle_file):
    """Load the index-to-word mapping from a pickle file."""
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)


def generate_predictions(loader, model, i2w_mapping):
    """Generate predictions for the test data."""
    model = model.cuda()  # Move the model to GPU if available
    return evaluate(loader, model, i2w_mapping)


def save_predictions(predictions, output_file):
    """Save the generated predictions to a file."""
    with open(output_file, 'w') as f:
        for video_id, sentence in predictions:
            f.write(f'{video_id},{sentence}\n')


def load_ground_truth(json_file):
    """Load the ground truth test labels from a JSON file."""
    with open(json_file) as f:
        return json.load(f)


def load_predictions(output_file):
    """Load the model's predictions from a file."""
    results = {}
    with open(output_file, 'r') as f:
        for line in f:
            video_id, caption = line.strip().split(',', 1)
            results[video_id] = caption
    return results


def calculate_bleu_scores(ground_truth, predictions):
    """Calculate BLEU scores for the predictions against the ground truth."""
    bleu_scores = []
    for entry in ground_truth:
        reference_captions = [cap.rstrip('.') for cap in entry['caption']]
        score = BLEU(predictions[entry['id']], reference_captions, True)
        bleu_scores.append(score)
    return bleu_scores


def compute_average_bleu(bleu_scores):
    """Compute the average BLEU score."""
    return sum(bleu_scores) / len(bleu_scores)


if __name__ == "__main__":
    # Load the pre-trained model
    model = load_model('SavedModel/model0.h5')

    # Prepare the testing data
    test_loader = prepare_data(sys.argv[1])

    # Load index-to-word mapping
    i2w = load_i2w_mapping('i2w.pickle')

    # Generate predictions
    predictions = generate_predictions(test_loader, model, i2w)

    # Save predictions to a file
    save_predictions(predictions, sys.argv[2])

    # Load ground truth data and predictions
    ground_truth = load_ground_truth('/home/agolla/HW2/MLDS_hw2_1_data/testing_label.json')
    predictions = load_predictions(sys.argv[2])

    # Calculate BLEU scores
    bleu_scores = calculate_bleu_scores(ground_truth, predictions)

    # Compute and print the average BLEU score
    avg_bleu = compute_average_bleu(bleu_scores)
    print(f"Average BLEU score: {avg_bleu:.4f}")
