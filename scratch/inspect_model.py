
import torch
import os
import json

def inspect_model():
    checkpoint_path = 'checkpoints/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No checkpoint found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("--- Checkpoint Info ---")
    print(f"Epoch: {checkpoint.get('epoch')}")
    print(f"Val Loss: {checkpoint.get('val_loss')}")
    print(f"Num Classes: {checkpoint.get('num_classes')}")
    
    char_to_idx = checkpoint.get('char_to_idx', {})
    print(f"Vocab Size: {len(char_to_idx)}")
    
    # Check weights to infer architecture
    state_dict = checkpoint['model_state_dict']
    print("\n--- Architecture Inference ---")
    for key in state_dict.keys():
        if 'rnn.1.linear.weight' in key:
            print(f"Last linear layer shape: {state_dict[key].shape}")
            print(f"Does it match Num Classes ({checkpoint.get('num_classes')})?")
            if state_dict[key].shape[0] == checkpoint.get('num_classes'):
                print("  ✓ Yes!")
            else:
                print("  ✖ NO! Mismatch detected.")

    # Check for character 'a' (index 51)
    idx_to_char = checkpoint.get('idx_to_char', {})
    print(f"\n--- Probable Index for 'a': {checkpoint['char_to_idx'].get('a')} ---")

if __name__ == "__main__":
    inspect_model()
