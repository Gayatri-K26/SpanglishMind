"""BERT token classifier for code-switch prediction.

This script loads the project's data/spanglish_dataset.json, extracts token-level
language labels from tagged_tokens, converts examples to a Hugging Face
Dataset, aligns labels to subword tokens, fine-tunes a multilingual transformer
for token classification, and evaluates token-level and switch-boundary metrics.

To run:
  python models/bert_token_classifier.py --model_name xlm-roberta-base --output_dir out/bert_cs --epochs 3

"""

from typing import List, Dict
import json
import argparse
import numpy as np
from pathlib import Path
import inspect

from datasets import Dataset, DatasetDict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
	AutoTokenizer,
	AutoModelForTokenClassification,
	TrainingArguments,
	Trainer,
	DataCollatorForTokenClassification,
)


def load_spanglish_json(path: str) -> List[Dict]:
	"""Load dataset and return list of examples with 'tokens' and 'labels'.

	Each example is a dict: {'tokens': [...], 'labels': [...]}
	Labels: 0 = en, 1 = es
	"""
	path = Path(path)
	j = json.loads(path.read_text(encoding="utf-8"))
	records = []

	# Accept either top-level list or dict with 'conversations'
	convs = j.get('conversations', j) if isinstance(j, dict) else j

	for conv in convs:
		for utt in conv.get('utterances', []):
			tagged = utt.get('tagged_tokens', [])
			tokens = []
			labels = []
			for item in tagged:
				# Expect items like [token, lang] or {'token':..., 'lang':...}
				if isinstance(item, (list, tuple)) and len(item) >= 2:
					tok, lang = item[0], item[1]
				elif isinstance(item, dict):
					tok = item.get('token') or item.get('word')
					lang = item.get('lang') or item.get('language')
				else:
					continue
				if not isinstance(tok, str) or not isinstance(lang, str):
					continue
				lang = lang.lower()
				if lang.startswith('en'):
					tokens.append(tok)
					labels.append(0)
				elif lang.startswith('es'):
					tokens.append(tok)
					labels.append(1)
				else:
					# Skip tokens without en/es labels
					continue
			if len(tokens) > 0:
				records.append({'tokens': tokens, 'labels': labels})
	return records


def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=False, max_length=128):
	tokenized_inputs = tokenizer(
		examples['tokens'],
		is_split_into_words=True,
		truncation=True,
		padding='max_length',
		max_length=max_length,
	)
	all_labels = examples['labels']
	new_labels = []
	for i, labels in enumerate(all_labels):
		word_ids = tokenized_inputs.word_ids(batch_index=i)
		previous_word_idx = None
		label_ids = []
		for word_idx in word_ids:
			if word_idx is None:
				label_ids.append(-100)
			elif word_idx != previous_word_idx:
				# Label for first token of the word
				label_ids.append(labels[word_idx])
			else:
				# For subsequent wordpieces
				label_ids.append(labels[word_idx] if label_all_tokens else -100)
			previous_word_idx = word_idx
		new_labels.append(label_ids)
	tokenized_inputs['labels'] = new_labels
	return tokenized_inputs


def compute_metrics(pred):
	preds = np.argmax(pred.predictions, axis=-1)
	labels = pred.label_ids
	preds_flat = []
	labels_flat = []
	for p_seq, l_seq in zip(preds, labels):
		for p, l in zip(p_seq, l_seq):
			if l == -100:
				continue
			preds_flat.append(p)
			labels_flat.append(l)
	if len(labels_flat) == 0:
		return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
	precision, recall, f1, _ = precision_recall_fscore_support(labels_flat, preds_flat, average='binary')
	acc = accuracy_score(labels_flat, preds_flat)
	return {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc}


def plot_confusion_matrix(pred, output_dir, label_names=None):
	"""Plot and save confusion matrix heatmap."""
	if label_names is None:
		label_names = ['English', 'Spanish']
	
	preds = np.argmax(pred.predictions, axis=-1)
	labels = pred.label_ids
	
	# Flatten and filter out padding (-100)
	preds_flat = []
	labels_flat = []
	for p_seq, l_seq in zip(preds, labels):
		for p, l in zip(p_seq, l_seq):
			if l != -100:
				preds_flat.append(p)
				labels_flat.append(l)
	
	if len(labels_flat) == 0:
		print("No valid predictions to plot confusion matrix.")
		return
	
	# Compute confusion matrix
	cm = confusion_matrix(labels_flat, preds_flat)
	
	# Plot
	plt.figure(figsize=(8, 6))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
	            xticklabels=label_names, yticklabels=label_names,
	            cbar_kws={'label': 'Count'})
	plt.xlabel('Predicted Label', fontsize=12)
	plt.ylabel('True Label', fontsize=12)
	plt.title('Token-Level Confusion Matrix', fontsize=14, fontweight='bold')
	plt.tight_layout()
	
	# Save
	output_path = Path(output_dir) / 'confusion_matrix.png'
	plt.savefig(output_path, dpi=300, bbox_inches='tight')
	print(f"Confusion matrix saved to {output_path}")
	plt.close()
	
	# Also save raw counts to JSON
	cm_dict = {
		'true_negative': int(cm[0, 0]),  # English predicted as English
		'false_positive': int(cm[0, 1]), # English predicted as Spanish
		'false_negative': int(cm[1, 0]), # Spanish predicted as English
		'true_positive': int(cm[1, 1]),  # Spanish predicted as Spanish
	}
	json_path = Path(output_dir) / 'confusion_matrix.json'
	with open(json_path, 'w') as f:
		json.dump(cm_dict, f, indent=2)
	print(f"Confusion matrix counts saved to {json_path}")


def main(args):
	data_path = args.data_path
	print(f"Loading data from {data_path}...")
	records = load_spanglish_json(data_path)
	if len(records) == 0:
		raise SystemExit("No examples found in dataset. Check 'tagged_tokens' format.")

	# build Hugging Face Dataset
	ds = Dataset.from_list(records)
	# small train/val/test split
	ds = ds.train_test_split(test_size=args.test_size, seed=args.seed)
	test = ds['test']
	train_val = ds['train'].train_test_split(test_size=args.val_size, seed=args.seed)
	dataset_dict = DatasetDict({
		'train': train_val['train'],
		'validation': train_val['test'],
		'test': test,
	})

	print('Preparing tokenizer and model...')
	tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
	model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=2)

	# Map tokenization & label alignment
	tokenized = dataset_dict.map(
		lambda ex: tokenize_and_align_labels(ex, tokenizer, label_all_tokens=False, max_length=args.max_length),
		batched=True,
		remove_columns=['tokens', 'labels'],
	)

	data_collator = DataCollatorForTokenClassification(tokenizer)

	# Build TrainingArguments dynamically
	training_kwargs = dict(
		output_dir=args.output_dir,
		learning_rate=args.lr,
		per_device_train_batch_size=args.batch_size,
		per_device_eval_batch_size=args.batch_size,
		num_train_epochs=args.epochs,
		weight_decay=0.01,
		logging_dir=f"{args.output_dir}/logs",
		push_to_hub=False,
	)

	# Add newer args only if supported by this transformers version
	try:
		init_sig = inspect.signature(TrainingArguments.__init__)
		if 'evaluation_strategy' in init_sig.parameters:
			training_kwargs['evaluation_strategy'] = 'epoch'
		if 'save_strategy' in init_sig.parameters:
			training_kwargs['save_strategy'] = 'epoch'
	except Exception:
		# If anything goes wrong inspecting, fall back to minimal args
		pass

	training_args = TrainingArguments(**training_kwargs)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=tokenized['train'],
		eval_dataset=tokenized['validation'],
		tokenizer=tokenizer,
		data_collator=data_collator,
		compute_metrics=compute_metrics,
	)

	# Only train if epochs > 0
	if args.epochs > 0:
		print('Starting training...')
		trainer.train()
		print(f"Saving model to {args.output_dir} ...")
		trainer.save_model(args.output_dir)
	else:
		print('Skipping training (--epochs 0). Loading existing model for evaluation...')
		# Re-load the saved model
		model = AutoModelForTokenClassification.from_pretrained(args.output_dir)
		trainer = Trainer(
			model=model,
			args=training_args,
			train_dataset=tokenized['train'],
			eval_dataset=tokenized['validation'],
			tokenizer=tokenizer,
			data_collator=data_collator,
			compute_metrics=compute_metrics,
		)

	print('Evaluating on test set...')
	test_results = trainer.predict(tokenized['test'])
	metrics = compute_metrics(test_results)
	print('\nTest metrics:')
	for k, v in metrics.items():
		print(f"{k}: {v:.4f}")
	
	# Save test metrics to JSON
	metrics_path = Path(args.output_dir) / 'test_metrics.json'
	with open(metrics_path, 'w') as f:
		json.dump(metrics, f, indent=2)
	print(f"Test metrics saved to {metrics_path}")
	
	# Plot confusion matrix
	plot_confusion_matrix(test_results, args.output_dir, label_names=['English', 'Spanish'])


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type=str, default='data/spanglish_dataset.json')
	parser.add_argument('--model_name', type=str, default='xlm-roberta-base')
	parser.add_argument('--output_dir', type=str, default='out/bert_cs')
	parser.add_argument('--epochs', type=int, default=3)
	parser.add_argument('--batch_size', type=int, default=8)
	parser.add_argument('--lr', type=float, default=2e-5)
	parser.add_argument('--max_length', type=int, default=128)
	parser.add_argument('--test_size', type=float, default=0.1)
	parser.add_argument('--val_size', type=float, default=0.1)
	parser.add_argument('--seed', type=int, default=42)
	args = parser.parse_args()
	main(args)

