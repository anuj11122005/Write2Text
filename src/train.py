import torch

def train(model, loader, optimizer, criterion, device="cpu"):
    model.train()

    total_loss = 0

    for imgs, labels in loader:

        # move to device
        imgs = imgs.to(device)
        labels = [l.to(device) for l in labels]

        # forward
        preds = model(imgs)

        # CTC expects (T, N, C)
        preds = preds.log_softmax(2)
        preds = preds.permute(1, 0, 2)

        # =========================
        # ✅ CTC LABEL HANDLING
        # =========================

        targets = torch.cat(labels)

        target_lengths = torch.tensor(
            [len(l) for l in labels],
            dtype=torch.long
        ).to(device)

        input_lengths = torch.full(
            size=(imgs.size(0),),
            fill_value=preds.size(0),
            dtype=torch.long
        ).to(device)

        # =========================

        loss = criterion(preds, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()

        # 🔥 IMPORTANT (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print("Loss:", avg_loss)