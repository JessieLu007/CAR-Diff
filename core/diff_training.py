import torch

def train_loop(model, dataloader, train_cfg):
    #optimizer = torch.optim.AdamW(model._denoise_fn.parameters(), lr=train_cfg.lr, weight_decay=1e-4)

    den = model._denoise_fn
    optimizer = torch.optim.AdamW(
        [
            {'params': den.time_embed.parameters(),  'lr': train_cfg.lr},
            {'params': den.label_emb.parameters(),   'lr': train_cfg.lr},
            {'params': den.proj_in.parameters(),     'lr': train_cfg.lr},
            {'params': den.layers.parameters(),      'lr': train_cfg.lr},
            {'params': den.out_num.parameters(),     'lr': train_cfg.lr},
            # 分类头：lr 稍大
            {'params': den.out_cat.parameters(),     'lr': train_cfg.lr * 2.0},
        ],
        weight_decay=1e-4,
    )

    best_loss = float('inf')
    patience_counter = 0

    print(f"\n⏳ 开始训练 (Epochs: {train_cfg.epochs})...")
    model._denoise_fn.train()

    stop_training = False
    for epoch in range(train_cfg.epochs):
        if stop_training:
            break
        epoch_loss = 0.0
        epoch_loss_num = 0.0
        epoch_loss_cat = 0.0
        valid_batches = 0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(model.device), batch_y.to(model.device)
            optimizer.zero_grad()
            
            loss, loss_num, loss_cat, _ = model.compute_loss(batch_x, batch_y)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"跳过 NaN 或 Inf 批次 (Epoch {epoch})")
                stop_training = True
                break
            
            loss.backward()
            # 检查反向传播梯度中是否含有NaN或Inf
            has_invalid_grad = False
            for param in model._denoise_fn.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    has_invalid_grad = True
                    break

            if has_invalid_grad:
                optimizer.zero_grad()
                continue
            torch.nn.utils.clip_grad_norm_(model._denoise_fn.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_loss_num += loss_num.item()
            epoch_loss_cat += loss_cat.item()
            valid_batches += 1
            
        if valid_batches == 0: continue
        avg_loss = epoch_loss / valid_batches
        
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if (epoch + 1) % 50 == 0 or epoch == 0:
            avg_loss = epoch_loss / valid_batches
            avg_num = epoch_loss_num / valid_batches
            avg_cat = epoch_loss_cat / valid_batches
            
            print(f"Epoch [{epoch+1}/{train_cfg.epochs}], Tot: {avg_loss:.4f} (Num: {avg_num:.4f}, Cat: {avg_cat:.4f}), Best: {best_loss:.4f}")
            
        if patience_counter >= train_cfg.patience:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            break