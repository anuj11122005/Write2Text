chars = sorted(list(set("".join(df['label']))))

char_to_idx = {c: i+1 for i, c in enumerate(chars)}
idx_to_char = {i+1: c for i, c in enumerate(chars)}

num_classes = len(chars) + 1  # +1 for CTC blank