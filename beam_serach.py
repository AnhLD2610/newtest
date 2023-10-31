import torch
import torch.nn.functional as F

def beam_search(model, src, max_len, trg_sos_idx, trg_eos_idx, beam_width):
    model.eval()

    with torch.no_grad():
        src_mask = model.make_src_mask(src)
        enc_src = model.encoder(src, src_mask)

        # Initialize the beam search candidates
        beam = [([], 0)]  # List of (predicted sequence, log probability)

        for _ in range(max_len):
            candidates = []

            for seq, score in beam:
                if len(seq) > 0 and seq[-1] == trg_eos_idx:
                    # If the sequence ends with the <eos> token, keep it in the candidates
                    candidates.append((seq, score))
                else:
                    trg = torch.tensor(seq, dtype=torch.long, device=model.device).unsqueeze(0)
                    trg_mask = model.make_trg_mask(trg)

                    output = model.decoder(trg, enc_src, trg_mask, src_mask)

                    # Get the last predicted token probabilities
                    last_token_probs = output[0, -1, :]

                    # Get the top-k tokens and their log probabilities
                    topk_probs, topk_tokens = torch.topk(last_token_probs, beam_width)

                    for i in range(beam_width):
                        new_seq = seq + [topk_tokens[i].item()]
                        new_score = score + topk_probs[i].item()
                        candidates.append((new_seq, new_score))

            # Sort candidates by score and keep the top beam_width sequences
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:beam_width]

        # Return the top sequence with the highest score
        best_seq, best_score = beam[0]
        return best_seq

# Example usage:
# src = ...  # Your source input
# max_len = ...  # Maximum sequence length
# trg_sos_idx = ...  # Index of the <sos> token
# trg_eos_idx = ...  # Index of the <eos> token
# beam_width = ...  # Beam width for beam search
# result = beam_search(model, src, max_len, trg_sos_idx, trg_eos_idx, beam_width)
