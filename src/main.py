import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_model_strategy_embeddings(path):
    if os.path.exists(path):
        print("Loading model/strategy embeddings...")
        return torch.load(path, weights_only=False)
    else:
        raise FileNotFoundError(f"File not found: {path}")

def prepare_data(csv_path, question_emb_path, model_strategy_emb_path):
    df = pd.read_csv(csv_path)
    question_embedding_tensor = torch.load(question_emb_path, weights_only=True)
    question_embeddings_np = question_embedding_tensor.numpy()
    model_strategy_embeddings = load_model_strategy_embeddings(model_strategy_emb_path)

    # Create ID mappings for models and strategies
    model_names = sorted(set(df["model"].values))
    strategy_types = sorted(set(df["strategy"].values))
    model2id = {name: idx for idx, name in enumerate(model_names)}
    strategy2id = {stype: idx for idx, stype in enumerate(strategy_types)}

    X, y_cls, y_reg, meta, model_ids, strategy_ids = [], [], [], [], [], []
    for _, row in df.iterrows():
        qid = row["question_id"]
        question_emb = question_embeddings_np[qid]
        model_name = row["model"]
        strategy_type = row["strategy"]
        model_emb = model_strategy_embeddings["model"].get(model_name)
        strategy_emb = model_strategy_embeddings["strategy"].get(strategy_type)
        if model_emb is None or strategy_emb is None:
            print(f"Warning: Missing embedding for {model_name} or {strategy_type}")
            continue

        combined_emb = np.concatenate([model_emb, strategy_emb, question_emb])
        X.append(combined_emb)
        y_cls.append(row["label"])
        y_reg.append(row["ans_length"])
        meta.append((model_name, strategy_type))
        model_ids.append(model2id[model_name])
        strategy_ids.append(strategy2id[strategy_type])

    return (
        np.array(X),
        np.array(y_cls),
        np.array(y_reg),
        np.array(meta),
        np.array(model_ids),
        np.array(strategy_ids),
        model2id,
        strategy2id,
        len(model_names),
        len(strategy_types)
    )


class DualTargetDataset(Dataset):
    def __init__(self, X, y_cls, y_reg, model_ids, strategy_ids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)
        self.model_ids = torch.tensor(model_ids, dtype=torch.long)
        self.strategy_ids = torch.tensor(strategy_ids, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y_cls[idx],
            self.y_reg[idx],
            self.model_ids[idx],
            self.strategy_ids[idx]
        )


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_models, num_strategies, emb_dim=64, hidden_dim=100, num_classes=2):
        super().__init__()
        self.model_emb = nn.Embedding(num_models, emb_dim)
        self.strategy_emb = nn.Embedding(num_strategies, emb_dim)
        self.model = nn.Sequential(
            nn.Linear(input_dim + 2 * emb_dim, hidden_dim),  # Adjust input_dim for trainable embeddings
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, model_ids, strategy_ids):
        model_emb = self.model_emb(model_ids)  # Shape: (batch_size, emb_dim)
        strategy_emb = self.strategy_emb(strategy_ids)  # Shape: (batch_size, emb_dim)
        x = torch.cat([x, model_emb, strategy_emb], dim=-1)  # Concatenate along feature dimension
        return self.model(x)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, num_models, num_strategies, emb_dim=64, hidden_dim=100):
        super().__init__()
        self.model_emb = nn.Embedding(num_models, emb_dim)
        self.strategy_emb = nn.Embedding(num_strategies, emb_dim)
        self.model = nn.Sequential(
            nn.Linear(input_dim + 2 * emb_dim, hidden_dim),  # Adjust input_dim for trainable embeddings
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def forward(self, x, model_ids, strategy_ids):
        model_emb = self.model_emb(model_ids)  # Shape: (batch_size, emb_dim)
        strategy_emb = self.strategy_emb(strategy_ids)  # Shape: (batch_size, emb_dim)
        x = torch.cat([x, model_emb, strategy_emb], dim=-1)  # Concatenate along feature dimension
        return self.model(x).squeeze(1)


def evaluate_classifier(model, val_loader, device, meta_val, batch_size):
    model.eval()
    correct = 0
    total = 0
    results = []
    group_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    with torch.no_grad():
        for i, (xb, yb_cls, _) in enumerate(val_loader):
            xb, yb_cls = xb.to(device), yb_cls.to(device)
            preds = model(xb)
            predicted = torch.argmax(preds, dim=1)
            correct += (predicted == yb_cls).sum().item()
            total += yb_cls.size(0)

            for j in range(len(yb_cls)):
                idx = i * batch_size + j
                if idx >= len(meta_val):
                    continue
                model_type, strategy_type = meta_val[idx]
                group_stats[(model_type, strategy_type)]["total"] += 1
                if predicted[j] == yb_cls[j]:
                    group_stats[(model_type, strategy_type)]["correct"] += 1
                results.append({
                    "model": model_type,
                    "strategy": strategy_type,
                    "true_label": yb_cls[j].item(),
                    "pred_label": predicted[j].item(),
                    "correct": int(predicted[j] == yb_cls[j])
                })
    
    pd.DataFrame(results).to_csv("./results/classifier_predictions.csv", index=False)
    print(f"Classifier predictions saved to ./results/classifier_predictions.csv")

    overall_acc = correct / total if total > 0 else 0.0
    print(f"\nClassifier Accuracy: {overall_acc:.2%}")
    print("\nAccuracy by Model and Strategy Type:")
    for (model_type, strategy_type), stats in sorted(group_stats.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  - {model_type:<30} | {strategy_type:<10} | Acc: {acc:.2%} ({stats['correct']}/{stats['total']})")
    return overall_acc


def evaluate_classifier(model, val_loader, device, meta_val, batch_size, num_models, num_strategies):
    model.eval()
    correct = 0
    total = 0
    results = []
    group_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    with torch.no_grad():
        for i, (xb, yb_cls, _, model_ids, strategy_ids) in enumerate(val_loader):
            xb, yb_cls = xb.to(device), yb_cls.to(device)
            model_ids, strategy_ids = model_ids.to(device), strategy_ids.to(device)
            preds = model(xb, model_ids, strategy_ids)
            predicted = torch.argmax(preds, dim=1)
            correct += (predicted == yb_cls).sum().item()
            total += yb_cls.size(0)

            for j in range(len(yb_cls)):
                idx = i * batch_size + j
                if idx >= len(meta_val):
                    continue
                model_type, strategy_type = meta_val[idx]
                group_stats[(model_type, strategy_type)]["total"] += 1
                if predicted[j] == yb_cls[j]:
                    group_stats[(model_type, strategy_type)]["correct"] += 1
                results.append({
                    "model": model_type,
                    "strategy": strategy_type,
                    "true_label": yb_cls[j].item(),
                    "pred_label": predicted[j].item(),
                    "correct": int(predicted[j] == yb_cls[j])
                })
    
    pd.DataFrame(results).to_csv("./results/classifier_predictions_encoder_embedding.csv", index=False)
    print(f"Classifier predictions saved to ./results/classifier_predictions_encoder_embedding.csv")

    overall_acc = correct / total if total > 0 else 0.0
    print(f"\nClassifier Accuracy: {overall_acc:.2%}")
    print("\nAccuracy by Model and Strategy Type:")
    for (model_type, strategy_type), stats in sorted(group_stats.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  - {model_type:<30} | {strategy_type:<10} | Acc: {acc:.2%} ({stats['correct']}/{stats['total']})")
    return overall_acc


def evaluate_regressor(model, val_loader, device, meta_val, num_models, num_strategies, tolerances=[50, 100, 150, 200, 250, 500, 800]):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, _, yb_reg, model_ids, strategy_ids in val_loader:
            xb = xb.to(device)
            model_ids, strategy_ids = model_ids.to(device), strategy_ids.to(device)
            pred = model(xb, model_ids, strategy_ids).cpu()
            preds.append(pred)
            targets.append(yb_reg)
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    mse = nn.MSELoss()(torch.tensor(preds), torch.tensor(targets))

    print(f"Regressor MSE: {mse.item():.4f}")

    if meta_val is None:
        return mse.item()

    model_names = [m[0] for m in meta_val]
    strategy_types = [m[1] for m in meta_val]
    results = []

    print("\nAccuracy analysis under different tolerance:")
    for tol in tolerances:
        within_tol = np.abs(preds - targets) <= tol
        acc = np.mean(within_tol)
        correct = int(np.sum(within_tol))
        total = len(targets)

        print(f"\n ±{tol} tolerance:")
        print(f"  - overall accuracy: {acc:.2%} ({correct}/{total})")
        results.append({
            "tolerance": tol,
            "group_type": "overall",
            "group_value": "all",
            "accuracy": acc,
            "correct": correct,
            "total": total
        })

        model_accs = defaultdict(list)
        for i, name in enumerate(model_names):
            model_accs[name].append(within_tol[i])
        print("Accuracy by model:")
        for name, acc_list in sorted(model_accs.items()):
            group_acc = np.mean(acc_list)
            correct = sum(acc_list)
            total = len(acc_list)
            print(f"    - {name:<30}: {group_acc:.2%} ({correct}/{total})")
            results.append({
                "tolerance": tol,
                "group_type": "model",
                "group_value": name,
                "accuracy": group_acc,
                "correct": correct,
                "total": total
            })

    os.makedirs(os.path.dirname("./results/regressor_accuracy_encoder_embedding.csv"), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("./results/regressor_accuracy_encoder_embedding.csv", index=False)
    print(f"\nSaved tolerance accuracy breakdown to ./results/regressor_accuracy_encoder_embedding.csv")

    return mse.item()


def train_dual_models(X_train, y_cls_train, y_reg_train, meta_train, model_ids_train, strategy_ids_train,
                      X_val, y_cls_val, y_reg_val, meta_val, model_ids_val, strategy_ids_val,
                      num_models, num_strategies,
                      epochs=50, batch_size=64, lr=1e-3,
                      patience=5, save_cls="best_model_cls1.pth", save_reg="best_model_reg1.pth",
                      cls_hidden_dim=768, reg_hidden_dim=768, emb_dim=64,
                      device=None):

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = DualTargetDataset(X_train, y_cls_train, y_reg_train, model_ids_train, strategy_ids_train)
    val_dataset = DualTargetDataset(X_val, y_cls_val, y_reg_val, model_ids_val, strategy_ids_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    input_dim = X_train.shape[1]
    num_classes = len(set(y_cls_train.tolist()))

    model_cls = MLPClassifier(
        input_dim, num_models, num_strategies, emb_dim=emb_dim, hidden_dim=cls_hidden_dim, num_classes=num_classes
    ).to(device)
    model_reg = MLPRegressor(
        input_dim, num_models, num_strategies, emb_dim=emb_dim, hidden_dim=reg_hidden_dim
    ).to(device)

    optimizer_cls = torch.optim.Adam(model_cls.parameters(), lr=lr)
    optimizer_reg = torch.optim.Adam(model_reg.parameters(), lr=lr)

    loss_cls_fn = nn.CrossEntropyLoss()
    loss_reg_fn = nn.MSELoss()

    best_acc, best_mse = 0, float("inf")
    patience_counter_cls = 0
    patience_counter_reg = 0
    stop_cls, stop_reg = False, False

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        total_loss_cls, total_loss_reg = 0, 0
        model_cls.train()
        model_reg.train()

        for xb, yb_cls, yb_reg, model_ids, strategy_ids in train_loader:
            xb, yb_cls, yb_reg = xb.to(device), yb_cls.to(device), yb_reg.to(device)
            model_ids, strategy_ids = model_ids.to(device), strategy_ids.to(device)

            if not stop_cls:
                optimizer_cls.zero_grad()
                pred_cls = model_cls(xb, model_ids, strategy_ids)
                loss_cls = loss_cls_fn(pred_cls, yb_cls)
                loss_cls.backward()
                optimizer_cls.step()
                total_loss_cls += loss_cls.item()

            if not stop_reg:
                optimizer_reg.zero_grad()
                pred_reg = model_reg(xb, model_ids, strategy_ids)
                loss_reg = loss_reg_fn(pred_reg, yb_reg)
                loss_reg.backward()
                optimizer_reg.step()
                total_loss_reg += loss_reg.item()

        avg_loss_cls = total_loss_cls / len(train_loader)
        avg_loss_reg = total_loss_reg / len(train_loader)
        print(f"  - Avg Loss_cls: {avg_loss_cls:.4f} | Avg Loss_reg: {avg_loss_reg:.4f}")

        if epoch % 3 == 0:
            if not stop_cls:
                acc = evaluate_classifier(model_cls, val_loader, device, meta_val, batch_size, num_models, num_strategies)
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model_cls.state_dict(), save_cls)
                    print(f"Saved best classifier model with accuracy {best_acc:.2%}")
                    patience_counter_cls = 0
                else:
                    patience_counter_cls += 1
                    print(f"Classifier patience {patience_counter_cls}/{patience}")
                    if patience_counter_cls >= patience:
                        print("Classifier early stopping triggered.")
                        stop_cls = True

            if not stop_reg:
                mse = evaluate_regressor(model_reg, val_loader, device, meta_val, num_models, num_strategies)
                if mse < best_mse:
                    best_mse = mse
                    torch.save(model_reg.state_dict(), save_reg)
                    print(f"Saved best regressor model with MSE {best_mse:.4f}")
                    patience_counter_reg = 0
                else:
                    patience_counter_reg += 1
                    print(f"Regressor patience {patience_counter_reg}/{patience}")
                    if patience_counter_reg >= patience:
                        print("Regressor early stopping triggered.")
                        stop_reg = True

        if stop_cls and stop_reg:
            print("Both models early stopped. Ending training loop.")
            break



def predict(question_path, model_strategy_embedding_path, 
            cls_model_path, reg_model_path, encoder_model_path=None, 
            device=None, cls_hidden_dim=768, reg_hidden_dim=768, emb_dim=768):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(question_path)
    questions = df["question"].unique().tolist()

    encoder = SentenceTransformer(encoder_model_path, local_files_only=True, device=device)
    question_embeddings = encoder.encode(questions, show_progress_bar=True)

    model_strategy_embeddings = load_model_strategy_embeddings(model_strategy_embedding_path)
    models = model_strategy_embeddings["model"]
    strategies = model_strategy_embeddings["strategy"]

    # Create ID mappings
    model_names = sorted(models.keys())
    strategy_types = sorted(strategies.keys())
    model2id = {name: idx for idx, name in enumerate(model_names)}
    strategy2id = {stype: idx for idx, stype in enumerate(strategy_types)}
    num_models = len(model_names)
    num_strategies = len(strategy_types)

    example_input = np.concatenate([
        next(iter(models.values())),
        next(iter(strategies.values())),
        question_embeddings[0]
    ])
    input_dim = example_input.shape[0]

    cls_model = MLPClassifier(
        input_dim=input_dim, num_models=num_models, num_strategies=num_strategies,
        emb_dim=emb_dim, hidden_dim=cls_hidden_dim, num_classes=2
    ).to(device)
    reg_model = MLPRegressor(
        input_dim=input_dim, num_models=num_models, num_strategies=num_strategies,
        emb_dim=emb_dim, hidden_dim=reg_hidden_dim
    ).to(device)
    cls_model.load_state_dict(torch.load(cls_model_path, map_location=device))
    reg_model.load_state_dict(torch.load(reg_model_path, map_location=device))
    cls_model.eval()
    reg_model.eval()

    def process_single_prediction(args):
        model_name, model_emb, strategy_name, strategy_emb, question, q_emb, question_idx = args
        input_vector = np.concatenate([model_emb, strategy_emb, q_emb])
        input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0).to(device)
        model_id = torch.tensor([model2id[model_name]], dtype=torch.long).to(device)
        strategy_id = torch.tensor([strategy2id[strategy_name]], dtype=torch.long).to(device)

        with torch.no_grad():
            out_cls = cls_model(input_tensor, model_id, strategy_id)
            probs = torch.softmax(out_cls, dim=-1)
            pred_label = torch.argmax(out_cls, dim=1).item()
            out_reg = reg_model(input_tensor, model_id, strategy_id)
            pred_length = out_reg.item()

        question_id = df[df["question"] == question]["question_id"].values[0]
        source = df[df["question_id"] == question_id]["source"].values[0]
        ground_truth_label = df[(df["question_id"] == question_id) & (df["model"] == model_name) & (df["strategy"] == strategy_name)]["label"].values
        ground_truth_length = df[(df["question_id"] == question_id) & (df["model"] == model_name) & (df["strategy"] == strategy_name)]["ans_length"].values
        pred_prob_1 = probs[0, 1].item()

        return (question_id, source, question, model_name, strategy_name), {
            "label": ground_truth_label.tolist() if len(ground_truth_label) > 0 else None,
            "pred_label": pred_label,
            "pred_cls_prob": pred_prob_1,
            "ans_length": ground_truth_length.tolist() if len(ground_truth_length) > 0 else None,
            "pred_length": pred_length
        }

    tasks = []
    for model_name, model_emb in models.items():
        for strategy_name, strategy_emb in strategies.items():
            for i, question in enumerate(questions):
                tasks.append((model_name, model_emb, strategy_name, strategy_emb, question, question_embeddings[i], i))

    results = {}
    with ThreadPoolExecutor() as executor:
        future_to_task = {executor.submit(process_single_prediction, task): task for task in tasks}
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Processing predictions"):
            key, result = future.result()
            results[key] = result

    results_list = []
    for (question_id, source, question, model_name, strategy_name), res in sorted(results.items()):
        print(f"  - Question: {question[:20]:<20} | Model: {model_name:<10} | Strategy: {strategy_name:<10} | "
              f"GT Label: {res['label']} | Pred Label: {res['pred_label']} | "
              f"GT Len: {res['ans_length']} | Pred Len: {res['pred_length']:.2f} | "
              f"Pred Prob: {res['pred_cls_prob']}")

        results_list.append({
            "question_id": question_id,
            "question": question,
            "source": source,
            "model": model_name,
            "strategy": strategy_name,
            "gt_label": res["label"][0] if res["label"] else None,
            "pred_label": res["pred_label"],
            "pred_cls_prob": res["pred_cls_prob"] if res["pred_cls_prob"] else None,
            "gt_ans_length": res["ans_length"][0] if res["ans_length"] else None,
            "pred_ans_length": res["pred_length"]
        })

    df_preds = pd.DataFrame(results_list)
    df_preds.to_csv("./results/prediction_results_encoder_embedding.csv", index=False)
    print("\nSaved prediction results to 'prediction_results_encoder_embedding.csv'.")

    return results

def main():
    parser = argparse.ArgumentParser(description="Train MLP classifier and regressor on model/strategy/question embeddings.")
    parser.add_argument("--model_strategy_emb_path", type=str, default="./data/model_strategy_embeddings.pth")
    parser.add_argument("--question_emb_path", type=str, default="./data/all_question_embeddings.pth")
    parser.add_argument("--train_csv_path", type=str, default="./data/split_data/train.csv")
    parser.add_argument("--test_csv_path", type=str, default="./data/split_data/test.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cls_hidden_dim", type=int, default=768)
    parser.add_argument("--reg_hidden_dim", type=int, default=768)
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--save_cls_path", type=str, default="./data/best_model_cls_encoder_embedding.pth")
    parser.add_argument("--save_reg_path", type=str, default="./data/best_model_reg_encoder_embedding.pth")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--encoder_model_path", type=str, default="") # your path to the encoder model
    parser.add_argument("--predict", type=bool, default=False)
    parser.add_argument("--predict_question_path", type=str, default="./data/split_data/test.csv")

    args = parser.parse_args()

    print("Loading training and validation data...")
    (
        X_train, y_cls_train, y_reg_train, meta_train,
        model_ids_train, strategy_ids_train, model2id_train, strategy2id_train,
        num_models_train, num_strategies_train
    ) = prepare_data(
        args.train_csv_path, args.question_emb_path, args.model_strategy_emb_path
    )
    (
        X_val, y_cls_val, y_reg_val, meta_val,
        model_ids_val, strategy_ids_val, model2id_val, strategy2id_val,
        num_models_val, num_strategies_val
    ) = prepare_data(
        args.test_csv_path, args.question_emb_path, args.model_strategy_emb_path
    )

    print(f"Training data: X: {X_train.shape}, y_cls: {y_cls_train.shape}, y_reg: {y_reg_train.shape}, "
          f"model_ids: {model_ids_train.shape}, strategy_ids: {strategy_ids_train.shape}")
    print(f"Validation data: X: {X_val.shape}, y_cls: {y_cls_val.shape}, y_reg: {y_reg_val.shape}, "
          f"model_ids: {model_ids_val.shape}, strategy_ids: {strategy_ids_val.shape}")

    if args.predict:
        print("Predicting...")
        predict(
            args.predict_question_path, args.model_strategy_emb_path,
            args.save_cls_path, args.save_reg_path, args.encoder_model_path,
            args.device, args.cls_hidden_dim, args.reg_hidden_dim, args.emb_dim
        )
    else:
        train_dual_models(
            X_train, y_cls_train, y_reg_train, meta_train, model_ids_train, strategy_ids_train,
            X_val, y_cls_val, y_reg_val, meta_val, model_ids_val, strategy_ids_val,
            num_models_train, num_strategies_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            cls_hidden_dim=args.cls_hidden_dim,
            reg_hidden_dim=args.reg_hidden_dim,
            emb_dim=args.emb_dim,
            patience=args.patience,
            save_cls=args.save_cls_path,
            save_reg=args.save_reg_path,
            device=args.device
        )


if __name__ == "__main__":
    main()