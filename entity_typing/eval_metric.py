

def f1(p, r):
  if r == 0.:
    return 0.
  return 2 * p * r / float(p + r)

def strict(true_and_prediction):
  num_entities = len(true_and_prediction)
  correct_num = 0.
  for true_labels, predicted_labels in true_and_prediction:
    correct_num += set(true_labels) == set(predicted_labels)
  precision = recall = correct_num / num_entities
  return precision, recall, f1(precision, recall)

def macro(true_and_prediction):
  num_examples = len(true_and_prediction)
  p = 0.
  r = 0.
  pred_example_count = 0.
  pred_label_count = 0.
  gold_label_count = 0.
  for true_labels, predicted_labels in true_and_prediction:
    if predicted_labels:
      pred_example_count += 1
      pred_label_count += len(predicted_labels)
      per_p = len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
      p += per_p
    if len(true_labels):
      gold_label_count += 1
      per_r = len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
      r += per_r
  if pred_example_count > 0:
    precision = p / pred_example_count
  if gold_label_count > 0:
    recall = r / gold_label_count
  avg_elem_per_pred = pred_label_count / pred_example_count
  return num_examples, pred_example_count, avg_elem_per_pred, precision, recall, f1(precision, recall)

def micro(true_and_prediction):
  num_examples = len(true_and_prediction)
  num_predicted_labels = 0.
  num_true_labels = 0.
  num_correct_labels = 0.
  pred_example_count = 0.
  for true_labels, predicted_labels in true_and_prediction:
    if predicted_labels:
      pred_example_count += 1
    num_predicted_labels += len(predicted_labels)
    num_true_labels += len(true_labels)
    num_correct_labels += len(set(predicted_labels).intersection(set(true_labels)))
  if pred_example_count == 0:
    return num_examples, 0, 0, 0, 0, 0
  precision = num_correct_labels / num_predicted_labels
  recall = num_correct_labels / num_true_labels
  avg_elem_per_pred = num_predicted_labels / pred_example_count
  return num_examples, pred_example_count, avg_elem_per_pred, precision, recall, f1(precision, recall)